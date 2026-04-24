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
