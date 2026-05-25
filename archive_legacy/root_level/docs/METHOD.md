# MambaRefine-CD Method

## 1. Overview

MambaRefine-CD is a binary remote-sensing change detection model for bi-temporal image pairs. The model class is `DRBISiameseMambaCD` (`src/models/cd_model.py`), built via the `build_model()` factory which enforces `model.mode: dual`. The pipeline is:

1. Shared MambaVision encoder extracts 4-scale feature pyramids for both images.
2. Per-scale D-RBI modules fuse temporal feature pairs into region and boundary streams.
3. CRAMLite spatial attention is optionally applied to region features.
4. The ARF-FPN decoder aggregates region features into a coarse change map.
5. A boundary residual head applies a learned logit-space correction guided by the D-RBI boundary stream.
6. Final output is a single binary change logit map.

The temporal Mamba mode (`model.mode: temporal_mamba`) has been permanently disabled due to training instability. Only `model.mode: dual` is active.

---

## 2. MambaVision Shared Backbone

The pre-change image `I1` and post-change image `I2` are encoded independently with **shared weights** through a MambaVisionFeatureExtractor (`src/models/backbone/mambavision_builder.py`):

```
F1 = E(I1)   →  [F1_0, F1_1, F1_2, F1_3]
F2 = E(I2)   →  [F2_0, F2_1, F2_2, F2_3]
```

Each `F_i` is a feature map at spatial resolution H/2^(i+2) × W/2^(i+2). Available backbone variants and their channel widths:

| Alias  | Registry Name     | Stem dim | Channel widths            | Params (encoder only) |
|--------|-------------------|----------|---------------------------|-----------------------|
| tiny   | mamba_vision_T    | 80       | [80, 160, 320, 640]       | ~31M                  |
| tiny2  | mamba_vision_T2   | 80       | [80, 160, 320, 640]       | ~31M                  |
| small  | mamba_vision_S    | 96       | [96, 192, 384, 768]       | ~49M                  |
| base   | mamba_vision_B    | 128      | [128, 256, 512, 1024]     | ~97M                  |
| large  | mamba_vision_L    | 196      | [196, 392, 784, 1568]     | —                     |

All backbone variants load ImageNet-pretrained weights by default (`model.pretrained: true`).

For ablation A0 only, a lightweight 4-stage SimpleCNN is substituted (channels [64, 128, 256, 512], ~7.8M total params, no pretrained weights).

---

## 3. Differential Region-Boundary Interaction (D-RBI)

**Source:** `src/models/modules/differential_region_boundary.py` — `DifferentialRegionBoundaryInteraction`

One D-RBI module is instantiated per encoder scale (4 total). Each module fuses `F1_i` and `F2_i` into separate region and boundary feature streams.

### 3.1 Pre-normalisation

When `pre_norm=True` (default), each branch is independently normalised before any arithmetic:

```
F1n = GroupNorm(F1_i)
F2n = GroupNorm(F2_i)
```

This prevents magnitude explosion in difference and product terms.

### 3.2 Input construction

The input tensor is built by concatenating the enabled streams:

```
parts = [F1n, F2n]                         # always included (raw pair)
if use_absdiff:  parts += |F2n - F1n|       # absolute temporal difference
if use_signed_diff: parts += (F2n - F1n)    # signed temporal difference (direction)
if use_product:  parts += product_scale * F1n * F2n   # co-activation (disabled by default)

X = cat(parts)   →  [B, n_streams × C_in, H, W]
```

The full model uses `use_absdiff=True` and `use_signed_diff=True`, giving 4 streams: `[F1n, F2n, |diff|, signed_diff]`.

### 3.3 Compression and spatial refinement

```
D = GELU(GroupNorm(Conv1x1(X)))            # bottleneck: n*C_in → C_out
D = DW3x3 + PW1x1 + GroupNorm + GELU(D)   # spatial refinement (depthwise-separable)
```

### 3.4 Region gate G_r

A lightweight 1×1 → GELU → 1×1 gate MLP produces a soft mask, bounded to `[region_gate_min, region_gate_max]`:

```
G_r_logits = clamp(psi_r(D), -8, 8)
G_r = region_gate_min + (region_gate_max - region_gate_min) * sigmoid(G_r_logits)
R   = G_r * D
```

Default bounds: `region_gate_min=0.2`, `region_gate_max=0.8`.

### 3.5 Boundary gate G_b (Sobel-conditioned)

The boundary stream is conditioned on the spatial gradient magnitude of `D`, extracted by a fixed (non-learnable) depthwise Sobel operator. The gradient magnitude is clamped to `[0, 10]`:

```
grad_mag = clamp(Sobel(D), 0, 10)
G_b_logits = clamp(psi_b(grad_mag), -8, 8)
G_b = boundary_gate_min + (boundary_gate_max - boundary_gate_min) * sigmoid(G_b_logits)
B   = G_b * D
```

Default bounds: `boundary_gate_min=0.0`, `boundary_gate_max=0.4`. The lower maximum prevents the boundary gate from becoming a full pass-through, which could cause gradient instability.

### 3.6 D-RBI outputs

Each D-RBI module returns:
- `region`   → `R = G_r * D`
- `boundary` → `B = G_b * D`
- `diff`     → `D` (compressed fused representation)

---

## 4. CRAMLite Spatial Attention

**Source:** `src/models/modules/cram_lite.py` — `CRAMLite`, `CRAMLiteBank`

An optional lightweight spatial attention applied to the D-RBI region features. Enabled via `model.cram_lite.enabled: true` in config.

Architecture per stage:

```
A = sigmoid( PW1x1(GELU(GroupNorm(PW1x1(GELU(GroupNorm(DW3x3(F_region))))))) )   # → [B,1,H,W]
F_out = F_region * (1 + alpha * A)
```

`alpha` is a learnable scalar initialized to `0.5` (from `model.cram_lite.alpha`). The residual formulation ensures the module is near-identity at initialization. Applied at stages `[0, 1, 2]` by default (3 of 4 encoder scales).

---

## 5. Adaptive Receptive Field Decoder (ARF-FPN)

**Source:** `src/models/decoders/adaptive_rf_decoder.py` — `AdaptiveRFDecoder`

Activated when `model.decoder: adaptive_rf`. Each encoder scale gets an `_AdaptiveDilationBlock`:

### 5.1 Adaptive Dilation Block

```
branches = [Conv3x3(d=r)(proj(region_feat)) for r in dilation_rates]   # parallel dilated convs
w = softmax(FC(GAP(proj(region_feat))))                                   # per-image attention weights
fused_scale = sum(w_i * branch_i)
```

Default dilation rates: `[1, 2, 4, 8]`. The attention weights `w` are predicted per image via global average pooling + FC, making the effective receptive field data-dependent with no deformable operations.

### 5.2 Top-down FPN aggregation

```
top = smooth(top-scale)
for each lower scale (top-down):
    top = smooth(upsample(top) + proj(scale_feat))
```

### 5.3 Coarse prediction head

```
P_c = bilinear_upsample(Conv1x1(Conv3x3(top)), out_size)   # [B, 1, H, W]
```

---

## 6. Boundary Residual Refinement

**Source:** `src/models/decoders/adaptive_rf_decoder.py` — `_BoundaryRefineHead`

Activated when `decoder.use_boundary_residual: true` and boundary features are provided by D-RBI.

### 6.1 Stage 1 — Coarse prediction

The ARF-FPN produces the coarse change logit map `P_c` from region features.

### 6.2 Stage 2 — Boundary residual correction

The finest-scale D-RBI boundary feature `B_0` is upsampled to full resolution. A boundary uncertainty map `E` is extracted from `P_c` using a fixed Sobel operator:

```
E = Sobel(sigmoid(P_c))   # [B, 1, H, W]  — high at logit uncertainty boundaries
```

The refinement head takes:
```
delta = BoundaryRefineHead(cat[B_0, P_c, E])   # DW3x3→PW1x1→Conv1x1
P_f   = P_c + residual_scale * tanh(delta)      # logit-space additive correction
```

`residual_scale=0.1` (from `decoder.residual_scale`). The final convolution in the head is zero-initialized, so the correction starts at zero and grows only as training progresses.

`P_f` is returned as the main output; `P_c` is returned as the auxiliary output for the coarse loss.

---

## 7. Loss Function

**Source:** `src/training/losses.py`

The full model uses the `bce_dice` loss family. The total loss combines:

```
L = L_bce + L_dice + w_coarse * L_coarse + w_boundary * L_boundary
```

| Term | Description | Default weight |
|------|-------------|----------------|
| `L_bce`      | Binary cross-entropy on `P_f` | 1.0 |
| `L_dice`     | Dice loss on `P_f` | 1.0 |
| `L_coarse`   | BCE+Dice on coarse output `P_c` (aux loss) | 0.4 |
| `L_boundary` | L1 loss on Sobel-extracted edges of `P_f` vs GT | 0.1 |

The boundary loss target is derived from the GT mask using the same Sobel operator (`target_type: sobel`). All losses use masked mean reduction to handle valid pixel masks.

For ablations A0–A5, `L_coarse` and `L_boundary` are both disabled (weight 0).

---

## 8. Training Details

**Source:** `src/training/trainer.py`, `src/training/pipeline.py`

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-5 |
| LR schedule | Cosine decay |
| Warmup | 2500 iterations |
| Weight decay | 0.01 |
| Gradient clip | 0.5 |
| Batch size | 8 |
| Mixed precision | AMP (torch.amp) |
| EMA decay | 0.999 |
| Checkpoint metric | F1 (max) |

Full model runs: 50 000 iterations. Ablation runs A1–A5: 30 000 iterations (except A0: 50 000).

**Validation:** every 5 000 iterations. Threshold is swept over `[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]` at validation; the best-F1 threshold from val is applied at test time.

**EMA:** all evaluation (val and test) uses the EMA checkpoint.

**Augmentation:** horizontal flip + vertical flip only (both datasets).

**Inference mode:** patch-based with `crop_size=256`, `overlap=0.25`.

---

## 9. Binary Change Detection Output

The final change probability is:

```
P = sigmoid(P_f)
```

Pixels with `P >= threshold` are predicted as changed. Threshold is selected at validation and applied at test time.

---

## 10. Active Metrics

Binary change detection is evaluated with:

| Metric | Description |
|--------|-------------|
| Pre | Precision (changed class) |
| Rec | Recall (changed class) |
| F1  | Harmonic mean of Pre and Rec |
| IoU | Intersection-over-Union (changed class) |
| OA  | Overall accuracy |

All metrics are computed in **global** (pooled) mode across all test tiles.
