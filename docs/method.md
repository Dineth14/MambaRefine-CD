# Method Notes

## 1. Baseline FPN Decoder

Standard Feature Pyramid Network with Siamese feature fusion.

**Fusion at scale $i$:**

$$F_i = W_i \bigl[ |f_A^i - f_B^i|\ \|\ f_A^i + f_B^i \bigr]$$

**Top-down path:**

$$T_i = \text{Smooth}\bigl(\text{Up}(T_{i+1}) + F_i\bigr)$$

**Output:**

$$P = \text{Conv}_{1\times1}(\text{Conv}_{3\times3}(T_0))$$

Simple, fast, and a strong reference for all other decoders.

---

## 2. Adaptive RF Decoder

Replaces each FPN stage's $3\times3$ convolution with a **softmax-gated mixture of dilated branches**.

**Per-stage block:**

$$Y = \sum_{r \in \mathcal{R}} w_r(\mathbf{x}) \cdot \text{DilConv}_r(\mathbf{x})$$

where $w_r(\mathbf{x}) = \text{softmax}(\text{FC}(\text{GAP}(\mathbf{x})))$ are per-image channel-attention weights.

**Effect:** The model dynamically selects the receptive field radius best suited for each feature map, avoiding the need to pick a single dilation rate manually.

**Stability:** Using softmax (rather than sigmoid) ensures the weights sum to 1, preventing branch collapse.

---

## 3. Localization → Refinement Decoder *(MERCon contribution)*

### Motivation

Standard FPN decoders treat all pixels equally. Changed building boundaries — thin, high-frequency structures — are often blurred or missed.

The Refinement Decoder addresses this with an explicit two-stage pipeline:

1. **Localise** change regions coarsely (where is the change?)
2. **Refine** boundaries using high-resolution shallow features (exactly where?)

### Stage 1: Coarse Localisation

Standard lightweight FPN over all 4 encoder scales:

$$P_c = \text{CoarseFPN}(\{F_i\}_{i=0}^3)$$

$P_c \in \mathbb{R}^{B \times 1 \times H \times W}$ is the coarse logit map.

### Stage 2: Boundary Extraction

We extract the **boundary uncertainty map** from $P_c$ using a differentiable Sobel filter applied to the sigmoid probability:

$$\sigma_c = \sigma(P_c)$$

$$E = \|\nabla \sigma_c\|_2 = \sqrt{(k_x * \sigma_c)^2 + (k_y * \sigma_c)^2}$$

where $k_x, k_y$ are fixed horizontal and vertical Sobel kernels.

$E \in [0,1]$ is large near predicted change boundaries and near zero in the interior — precisely the regions that need sharper correction.

### Stage 3: Residual Refinement

The refinement block takes:
- $P_c$ — coarse prediction context
- $E$ — boundary uncertainty guide
- $f_0, f_1$ — the two shallowest encoder difference features (highest resolution)

$$\Delta = \text{RefinementBlock}(P_c,\ E,\ D_0,\ D_1)$$

where $D_i = W_i [|f_A^i - f_B^i|\ \|\ f_A^i + f_B^i]$.

The final head's weight is initialised to **zero**, so at the start of training:

$$P_f = P_c + \underbrace{\Delta}_{=0} = P_c$$

This makes the refinement purely incremental — the coarse prediction is always the baseline, and the refinement adds only what it learns to be useful.

### Final Output

$$P_f = P_c + \Delta$$

Both $P_f$ (final) and $P_c$ (coarse) are returned. $P_c$ can be used for **auxiliary supervision**:

$$\mathcal{L} = \mathcal{L}_{\text{final}}(P_f, y) + \lambda \cdot \mathcal{L}_{\text{coarse}}(P_c, y)$$

with $\lambda = 0.4$ by default (`decoder.aux_weight`).

---

## 4. Global-Local Decoder

Explicitly separates semantic understanding (global, deep features) from spatial detail (local, shallow features) and blends them via a learned sigmoid gate.

$$G = \sigma\bigl(\text{Conv}([T_{\text{local}}\ \|\ T_{\text{global}}])\bigr)$$

$$\text{Output} = G \cdot T_{\text{local}} + (1 - G) \cdot T_{\text{global}}$$

The global branch injects a global-average-pooled context vector at the deepest stage to provide scene-level priors.

---

## 5. Differential Region–Boundary Interaction (D-RBI)

### Motivation

Standard abs-diff fusion loses sign information and conflates region change
(large, coherent area differences) with boundary change (thin, high-frequency
edges). D-RBI decomposes the difference into two complementary signals via
soft-gating.

### Input Concatenation

For a pair of encoder features $F_1, F_2 \in \mathbb{R}^{B \times C \times H \times W}$:

$$\mathbf{X} = \bigl[F_1 \;\|\; F_2 \;\|\; |F_2 - F_1| \;\|\; F_1 \odot F_2\bigr] \in \mathbb{R}^{B \times 4C \times H \times W}$$

Ablation switches `use_absdiff` and `use_product` can drop the last two terms,
reducing the input to $2C$.

### Bottleneck Compression

$$D = \phi(\mathbf{X}) = \text{GELU}(\text{GN}(\text{DW-Conv}(\text{Conv}_{1\times1}^{4C \to C_\text{out}}(\mathbf{X}))))$$

where DW-Conv is an optional depthwise $3\times3$ (`use_depthwise=True`).
$D \in \mathbb{R}^{B \times C_\text{out} \times H \times W}$.

### Region Gate

A lightweight bottleneck MLP $\psi_r$ (two $1\times1$ convolutions with GELU):

$$G_r = \gamma_r^\text{min} + (\gamma_r^\text{max} - \gamma_r^\text{min}) \cdot \sigma(\psi_r(D))$$

$$R = G_r \odot D$$

Bounded gates $(\gamma_r^\text{min} = 0.2,\ \gamma_r^\text{max} = 1.0)$ prevent saturation and ensure a minimum pass-through.

### Boundary Gate

A fixed Sobel filter (no learned parameters) extracts spatial gradients from $D$:

$$|\nabla D| = \sqrt{(k_x * D)^2 + (k_y * D)^2 + \varepsilon}$$

The boundary gate is conditioned on this gradient magnitude:

$$G_b = \gamma_b^\text{min} + (\gamma_b^\text{max} - \gamma_b^\text{min}) \cdot \sigma(\psi_b(|\nabla D|))$$

$$B = G_b \odot D$$

Defaults: $\gamma_b^\text{min} = 0.0,\ \gamma_b^\text{max} = 0.7$ — the boundary stream is
more selective than the region stream, only activating near true edges.

### Output

Each D-RBI module returns `{"diff": D, "region": R, "boundary": B}`.  At inference
the decoder uses $R$ for the coarse ARF-FPN prediction and $B$ for the residual
boundary correction.

---

## 6. D-RBI + Adaptive RF Decoder (Full Pipeline)

The two-stage pipeline when `difference.enabled: true`:

### Stage 1 — Coarse ARF-FPN (region features)

$$\{R_i\}_{i=0}^3 = \{\text{D-RBI}_i(F_1^i, F_2^i)\text{["region"]}\}$$

$$T_i = \text{ARF}_i(W_i^{\text{proj}}(R_i))$$

$$P_c = \text{Upsample}(\text{CoarseHead}(\text{FPN}(\{T_i\})))$$

### Stage 2 — Boundary Residual Correction

The finest-scale boundary feature is upsampled to $H \times W$:

$$B_0^\uparrow = \text{Bilinear}(B_0, (H, W))$$

The Sobel edge of the coarse sigmoid probability:

$$E = \|\nabla \sigma(P_c)\|_2$$

Residual correction:

$$\Delta = \text{BoundaryRefineHead}([B_0^\uparrow \;\|\; P_c \;\|\; E])$$

$$P_f = P_c + \delta \cdot \tanh(\Delta)$$

where $\delta = 0.1$ (`decoder.residual_scale`). $\tanh$ bounds the correction
to $(-\delta, +\delta)$ logits, preventing large deviations from the coarse
prediction.

The `BoundaryRefineHead` final convolution is zero-initialised, so at
epoch 0: $P_f = P_c$ (pure identity). Training gradually adds corrections.

### Loss

$$\mathcal{L} = \mathcal{L}(P_f, y) + \lambda \cdot \mathcal{L}(P_c, y), \qquad \lambda = 0.4$$

---

## 7. Ablation Switches

| Switch | Default | Effect when disabled |
|---|---|---|
| `difference.enabled` | `true` | Falls back to abs-diff + sum decoder (old baseline) |
| `difference.use_absdiff` | `true` | Drops $|F_2 - F_1|$ from input concat |
| `difference.use_product` | `true` | Drops $F_1 \odot F_2$ from input concat |
| `difference.use_region_gate` | `true` | Region output = $D$ (no gate) |
| `difference.use_boundary_gate` | `true` | Boundary output = $D$ (no Sobel gate) |
| `decoder.use_boundary_residual` | `true` | Uses only coarse head; no $\Delta$ correction |

