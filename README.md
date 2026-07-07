# 🛰️ MambaRefine-CD

**MambaVision with Region-Boundary Temporal Refinement for Binary Remote-Sensing Change Detection**

### 🔬 Separate the Region. Preserve the Boundary. Don't Let One Ruin the Other. 🔬

[![📄 Paper](https://img.shields.io/badge/📄_MERCon-Accepted-brightgreen)](#citation)
[![🏛️ Venue](https://img.shields.io/badge/🏛️_MERCon_2026-University_of_Moratuwa-blue)](#citation)
[![🐍 Python](https://img.shields.io/badge/🐍_Python-3.9+-3776AB)](https://www.python.org/)
[![🔥 PyTorch](https://img.shields.io/badge/🔥_PyTorch-2.0+-EE4C2C)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Config-Driven](https://img.shields.io/badge/No_CLI-Config_Driven-brightgreen)](#config-system)

One shared MambaVision encoder. Four scales of D-RBI decomposition. Signed and absolute temporal evidence. An ARF-FPN region decoder. A bounded boundary residual head that refines, not overwrites. **95.67% F1 on DSIFN-CD. 95.34% F1 on WHU-CD.**

---

[🔥 Updates](#-updates) • [🔭 Overview](#-overview) • [🧠 Architecture](#-architecture) • [⚡ Quick Start](#-quick-start) • [🗂 Datasets](#-datasets) • [🚀 Train & Eval](#-train--evaluation) • [🔬 Ablations](#-ablations) • [📊 Results](#-results) • [🔧 Config System](#-config-system) • [📐 Metrics](#-metrics) • [🙏 Acknowledgements](#-acknowledgements) • [📜 Citation](#-citation)

---

## 🔥 Updates

| Date | Update |
|---|---|
| **2026** | MambaRefine-CD has been accepted at MERCon |
| **2026** | Codebase released — full config-driven pipeline, zero CLI arguments |

**Status:** Accepted at MERCon.

---

## 🔭 Overview

Change detection models have two jobs: find **where** change happened, and get the **boundary** right. Most models treat these as the same problem and optimize one objective for both. They are not the same problem.

Region-level changes — land cover transitions, building footprints, vegetation patches — are broad and respond well to global feature differencing. Boundaries are thin, spatially precise, and easily destroyed by pooling, upsampling, and standard BCE loss. A model that maximizes region F1 will systematically blur or fragment boundaries to do it.

**MambaRefine-CD separates these two problems from the first feature interaction:**

- A **shared MambaVision encoder** processes both images in the same feature space — no encoder mismatch, no spurious differences
- **D-RBI** builds temporal evidence from four streams — raw paired features, absolute difference, and signed difference — then decomposes it into a gated **region stream** and a Sobel-conditioned **boundary stream** at every scale
- **CRAM-lite** applies a lightweight learned spatial modulation to region features before decoding
- **ARF-FPN** decodes the region stream with parallel dilated branches (d ∈ {1, 2, 4, 8}) for multi-scale context
- A **bounded boundary residual head** refines the coarse logit map with boundary cues via a constrained update `Pf = Pc + 0.1·tanh(ΔP)` — it corrects the boundary without overwriting the region prediction

The result is a +10.58 BF1 improvement over a MambaVision-FPN baseline (61.36 → 71.94%) and competitive changed-class F1 and IoU against recent Mamba-based methods.

---

## 🧠 Architecture

<p align="center">
  <img src="/storage2/ChangeDetection/MV/MambaRefine-CD/figures/Architecture.png" alt="MambaRefine-CD Architecture" width="100%">
</p>
<p align="center"><i>Overall architecture of MambaRefine-CD. Two bi-temporal images pass through a shared MambaVision encoder. Same-scale features are fused by D-RBI modules. Region features are decoded by ARF-FPN. The finest boundary stream guides a bounded residual refinement of the coarse prediction.</i></p>


**Loss function:**

```
L = L_BCE + L_Dice + 0.4 · L_coarse + 0.1 · L_boundary
     └──────────────┘   └──────────┘   └──────────────┘
       final prediction   region         Sobel-edge ℓ₁
       (Pf, main loss)    decoder        contour penalty
                          stability
```

### Module Summary

| Module | Role | Key Mechanism |
|---|---|---|
| **MambaVision Encoder** | Shared bitemporal feature extraction | Hybrid Mamba + self-attention, 4-stage pyramid |
| **D-RBI** | Temporal evidence construction + decomposition | `[F₁, F₂, │ΔF│, ΔF]` → gated region + Sobel boundary streams |
| **CRAM-lite** | Lightweight spatial modulation of region features | `Reˡ = Rˡ·(1 + α·Aˡ)`, α=0.5 init, scales 1–3 only |
| **ARF-FPN** | Multi-scale region feature decoding | Dilated depthwise branches d∈{1,2,4,8}, top-down fusion |
| **Boundary Residual Head** | Bounded boundary correction | `Pf = Pc + 0.1·tanh(ΔP)` — refines, does not overwrite |

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/MambaRefine-CD
cd MambaRefine-CD
pip install -r requirements.txt
```

For VMamba encoder support:

```bash
python tools/setup_vmamba.py    # prints install instructions and checks import
```

### 2. Download Released MambaRefine-CD Weights

Released model checkpoints are hosted on Hugging Face:
[dineth18/MambaRefine-CD](https://huggingface.co/dineth18/MambaRefine-CD).

| Dataset | Checkpoint | Config | Threshold | Pre (%) | Rec (%) | F1 (%) | IoU (%) | OA (%) | Iter | Size | SHA256 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| WHU-CD | [mambarefine_cd_whu_cd_best.pth](https://huggingface.co/dineth18/MambaRefine-CD/resolve/main/checkpoints/mambarefine_cd_whu_cd_best.pth) | [whu_cd_run_config.yaml](https://huggingface.co/dineth18/MambaRefine-CD/blob/main/configs/whu_cd_run_config.yaml) | 0.55 | 96.0072 | 95.0623 | 95.5324 | 91.4469 | 99.5715 | 45k | 986.65 MiB | `cd3b176a483d5311c02251d2a72ac399de4aecec89ad2808df02805c89d33758` |
| DSIFN-CD | [mambarefine_cd_dsifn_cd_best.pth](https://huggingface.co/dineth18/MambaRefine-CD/resolve/main/checkpoints/mambarefine_cd_dsifn_cd_best.pth) | [dsifn_cd_run_config.yaml](https://huggingface.co/dineth18/MambaRefine-CD/blob/main/configs/dsifn_cd_run_config.yaml) | 0.60 | 96.2591 | 96.5340 | 96.3963 | 93.0434 | 97.4721 | 50k | 987.97 MiB | `6b56becbba10ad6e67e772db939339f9040de61c73dc049e3c752d04eb0bdf6f` |

Download with `huggingface_hub`:

```bash
pip install huggingface_hub
```

```python
from huggingface_hub import hf_hub_download

whu_ckpt = hf_hub_download(
    repo_id="dineth18/MambaRefine-CD",
    filename="checkpoints/mambarefine_cd_whu_cd_best.pth",
)

dsifn_ckpt = hf_hub_download(
    repo_id="dineth18/MambaRefine-CD",
    filename="checkpoints/mambarefine_cd_dsifn_cd_best.pth",
)
```

These are trained MambaRefine-CD model checkpoints. The ImageNet-pretrained
encoder weights below are separate and are only needed when training from
scratch or initializing a new run.

### 3. Place Pretrained Encoder Weights

| Encoder | Source | Default path |
|---|---|---|
| MambaVision-S *(default)* | [NVIDIA MambaVision](https://github.com/NVlabs/MambaVision) | `pretrained_weights/mambavision_small_1k.pth` |
| MambaVision-T / B | NVIDIA MambaVision | `pretrained_weights/` |

Set `encoder_pretrained: true` in `configs/active.yaml` and weights are loaded automatically.

### 4. Prepare Your Dataset

```
datasets/
└── DSIFN-CD/
    ├── train/
    │   ├── A/          ← pre-change RGB images
    │   ├── B/          ← post-change RGB images
    │   └── Mask/       ← binary change masks (0 / 255)
    ├── val/
    │   ├── A/  B/  Mask/
    └── test/
        ├── A/  B/  Mask/
```

Files are matched by **filename stem** — `A/00123.png` pairs with `B/00123.png` and `Mask/00123.png`.

### 5. Verify, Check, Train

```bash
python tools/verify_dataset.py       # check A/B/Mask consistency, counts, overlap
python tools/check_model.py          # build model, print shapes, params, FLOPs
python tools/check_training_step.py  # one batch → forward → loss → backward → PASS

python train.py                      # start training — no arguments needed
python val.py                        # validate with threshold sweep
python test.py                       # test with best validation threshold
python infer.py                      # save prediction masks for new images
```

**Everything is controlled by `configs/active.yaml`. No command-line arguments required anywhere.**

---

## 🗂 Datasets

Two datasets are used in the paper. Both use the standard `A / B / Mask` format.

### DSIFN-CD

High-resolution bi-temporal aerial images from six Chinese cities. Multi-class land-cover change.

| Split | Images | Evaluation |
|---|---|---|
| Train | 2 758 | — |
| Validation | 394 | Threshold selection (best = **0.50**) |
| Test | 789 (→ 3 156 tiles) | Deterministic tiling, val threshold |

### WHU-CD

Building change detection dataset from aerial imagery.

| Split | Patches | Evaluation |
|---|---|---|
| Train | 6 096 | — |
| Validation | 762 | Threshold selection (best = **0.55**) |
| Test | 762 | Val threshold applied |

Pre-written configs: `configs/datasets/dsifn.yaml`, `configs/datasets/whu.yaml`.

Point to your dataset:

```yaml
# configs/active.yaml
data:
  dataset_name: DSIFN-CD
  root: datasets/DSIFN-CD
  image_size: 256
```

---

## 🚀 Train & Evaluation

### Training

```bash
python train.py
```

Training uses **AdamW**, **cosine LR schedule**, **2 500 warmup iterations**, **batch size 8**, **256×256 crops**, **gradient clipping at 0.5**, **AMP**, and **EMA inference**. Augmentations: random horizontal and vertical flips only.

All outputs are saved to:

```
outputs/run_YYYYMMDD_HHMMSS_{ablation_id}/
├── config.yaml            ← frozen config for exact reproduction
├── train.log
├── val.log
├── model_summary.txt      ← params, GFLOPs, encoder channels
├── checkpoints/
│   └── best_iter_050000_F1_0.9567.pth
├── tensorboard/
├── metrics/
│   ├── train_history.json
│   ├── val_history.json
│   └── test_metrics.json
└── predictions/           ← binary mask PNGs (if save_predictions: true)
```

Console output during training:

```
[iter 05000/50000] loss=0.1823 | lr=8.4e-05 | mem=5.1GB | val_F1=0.9421 | best_F1=0.9421 ✓
```

### Validation

```bash
python val.py
```

Sweeps thresholds from 0.05 to 0.95 on the validation split. Saves the best validation threshold to the checkpoint for use at test time.

### Testing

```bash
python test.py
```

Loads the best checkpoint. Applies the saved validation threshold to the test split. Saves predictions and `test_metrics.json`.

### Inference

```bash
python infer.py
```

Runs on the folder specified in `active.yaml`. Saves binary prediction masks.

### Switching Encoders

```yaml
# configs/active.yaml
model:
  encoder_family: mambavision   # default
  encoder_variant: small

# Switch to VMamba:
  encoder_family: vmamba
  encoder_variant: small
```

D-RBI input channels adapt automatically. No other changes needed.

---

## 🔬 Ablations

### Ablation A: Component Contributions (DSIFN-CD, 50k iterations)

Each row adds one component to the previous. Results averaged across available runs.

| ID | Variant | Added Component | Params (M) | F1 (%) | IoU (%) | OA (%) |
|---|---|---|---:|---:|---:|---:|
| A0 | FPN Baseline | SimpleCNN encoder | 7.84 | 76.63 | 62.12 | 83.80 |
| A1 | + MambaVision-S | MambaVision encoder | 53.54 | 93.21 | 87.28 | 95.72 |
| A2 | + D-RBI (unsigned) | Absolute-difference D-RBI | 54.98 | 93.33 | 87.50 | 95.35 |
| A3 | + Signed Diff | Signed temporal stream | 55.34 | 94.28 | 89.19 | 95.99 |
| A4 | + ARF-FPN | Adaptive receptive-field decoder | 65.12 | 94.36 | 89.32 | 96.05 |
| A5 | + Boundary Residual | Bounded residual head | 65.19 | 93.59 | 87.94 | 95.53 |
| **A6** | **Full Model** | **CRAM-lite + aux/bnd. loss** | **65.40** | **95.67** | **91.71** | **96.98** |

**Key findings:**
- The MambaVision encoder is the largest single contributor (+16.58 F1 over the CNN baseline)
- Signed temporal evidence adds +0.95 F1 over unsigned D-RBI — direction of change matters
- Boundary residual refinement alone (A5) does not improve over A4 — it needs CRAM-lite, auxiliary supervision, and boundary loss to be effective
- The full pipeline (A6) improves +1.31 F1 over ARF-FPN alone

To run any ablation variant, update `active.yaml` and set `ablation.id` accordingly.

### Ablation B: Boundary-Aware Evaluation (Table IV from paper)

Boundary metrics evaluated with 3-pixel tolerance.

| Variant | F1 (%) | IoU (%) | BF1 (%) | BIoU (%) | Trimap F1₃ₚₓ (%) |
|---|---:|---:|---:|---:|---:|
| A1 — MambaVision-FPN | 93.21 | 87.28 | 61.36 | 38.41 | 68.02 |
| A4 — + ARF-FPN | 93.71 | 88.16 | 65.94 | 42.06 | 70.27 |
| A5 — + Boundary Residual | 93.59 | 87.94 | 65.14 | 41.37 | 69.98 |
| **A6 — Full Model (DSIFN)** | **95.67** | **91.71** | **71.94** | **47.39** | **72.93** |
| **Full Model (WHU-CD)** | **95.15** | **90.76** | **89.49** | **71.37** | **87.01** |

**The full pipeline improves BF1 by +10.58 points and BIoU by +8.98 points over the MambaVision-FPN baseline.** Boundary residual refinement is most effective when combined with signed temporal modeling, ARF-FPN decoding, CRAM-lite modulation, and boundary-aware supervision.

### Ablation C: Backbone Scaling (Table V from paper)

Parameters and GFLOPs reported for a 256×256 input.

| Backbone | Params (M) | GFLOPs | F1 (%) | IoU (%) | OA (%) |
|---|---:|---:|---:|---:|---:|
| MambaVision-T | 46.80 | 50.05 | 94.58 | 89.72 | 96.24 |
| **MambaVision-S** *(default)* | **65.40** | **64.08** | **95.67** | **91.71** | **96.98** |
| MambaVision-B | 113.44 | 109.15 | 95.77 | 91.89 | 97.07 |

MambaVision-B gains only +0.10 F1 over MambaVision-S while using 1.74× more parameters and 1.70× more GFLOPs. **MambaVision-S is the default** — it sits at the accuracy-efficiency sweet spot.

To switch backbone variant:

```yaml
model:
  encoder_variant: tiny   # or small (default) or base
```

---

## 📊 Results

### DSIFN-CD

<p align="center">
  <img src="/storage2/ChangeDetection/MV/MambaRefine-CD/figures/qualitative_dsifn_final.png" alt="DSIFN-CD Qualitative Results" width="90%">
</p>
<p align="center"><i>Qualitative results on DSIFN-CD. Each column shows the bi-temporal input pair (I₁, I₂), ground truth, a baseline prediction, MambaRefine-CD prediction, error map, and boundary overlay. White = TP, black = TN, red = FP, green = FN.</i></p>

> ⚠️ Literature values are taken from Peng et al.'s compiled comparison table. DSIFN-CD split protocols differ across papers — this comparison is **contextual**, not a strict same-split SOTA ranking.

| Model | Pre (%) | Rec (%) | F1 (%) | IoU (%) | OA (%) |
|---|---:|---:|---:|---:|---:|
| FC-Siam-Conc | 66.45 | 54.21 | 59.71 | 42.56 | 87.57 |
| SNUNet | 60.60 | 72.89 | 66.18 | 49.45 | 87.34 |
| ChangeFormer | 88.48 | 84.94 | 86.67 | 76.48 | 95.56 |
| BiFA | 73.99 | 68.87 | 71.34 | 55.45 | 90.80 |
| FTAN | 90.54 | 88.61 | 89.56 | 81.10 | — |
| ADSFNet | **94.79** | 95.24 | 95.01 | 90.50 | **98.30** |
| Mamba-CD | 95.60 | 95.61 | 95.61 | 91.69 | **98.51** |
| **MambaRefine-CD** *(ours)* | 95.47 | **95.87** | **95.67** | **91.71** | 96.98 |

### WHU-CD

<p align="center">
  <img src="/storage2/ChangeDetection/MV/MambaRefine-CD/figures/qualitative_whu_refined.png" alt="WHU-CD Qualitative Results" width="90%">
</p>
<p align="center"><i>Qualitative results on WHU-CD. Building boundaries are preserved with high precision. The D-RBI boundary gate suppresses false positives in shadow and road regions adjacent to buildings.</i></p>

> Results report the **average of three full-model runs using EMA checkpoints**. Literature values from Peng et al.

| Method | Pre (%) | Rec (%) | F1 (%) | IoU (%) | OA (%) |
|---|---:|---:|---:|---:|---:|
| FC-EF | 71.63 | 67.25 | 69.37 | 53.11 | 97.61 |
| STANet | 79.37 | 85.50 | 82.32 | 69.95 | 98.52 |
| SNUNet | 85.60 | 81.49 | 83.50 | 71.67 | 98.71 |
| IFNet | **96.91** | 73.19 | 83.40 | 71.52 | 98.83 |
| ChangeFormer | 91.83 | 88.02 | 89.88 | 81.63 | 99.12 |
| BiFA | 95.15 | 93.60 | 94.37 | 89.34 | 99.56 |
| RSM-CD | 93.37 | 90.42 | 91.87 | 84.96 | — |
| SChanger | 94.62 | 91.83 | 93.20 | 87.27 | — |
| CDMamba | 95.58 | 92.01 | 93.76 | 88.26 | 99.51 |
| Mamba-CD | **96.52** | 93.91 | 95.20 | 90.83 | **99.62** |
| **MambaRefine-CD** *(ours)* | 95.79 | **94.90** | **95.34** | **91.10** | 99.56 |

### Key Takeaways

🔬 **The full region-boundary pipeline is necessary.** No single module improves performance in isolation — the combination of signed temporal evidence, ARF-FPN, CRAM-lite, and boundary-aware supervision is what drives the gain.

📏 **Signed temporal difference matters.** Absolute differencing (`|F₂−F₁|`) discards the direction of change — appearance and disappearance become indistinguishable. Adding the signed stream (`F₂−F₁`) improves F1 by +0.95 at the same parameter count.

🧱 **Boundary refinement requires a stable coarse prediction.** A5 (boundary residual alone) underperforms A4 (ARF-FPN without refinement). The bounded residual head corrects boundaries — it cannot recover a poor region prediction.

📐 **Scaling saturates under a fixed decoder.** MambaVision-B adds 48M parameters and 45 GFLOPs for +0.10 F1 over MambaVision-S. The efficiency sweet spot is MambaVision-S.

---

## 🔧 Config System

All behavior is controlled by `configs/active.yaml`. Zero command-line arguments. Every run saves a frozen copy of the config for exact reproduction.

```yaml
project:
  name: MambaRefine-CD
  output_root: outputs
  seed: 42

data:
  dataset_name: DSIFN-CD
  root: datasets/DSIFN-CD
  image_size: 256
  train_dir: train
  val_dir: val
  test_dir: test
  a_folder: A
  b_folder: B
  mask_folder: Mask
  binary_threshold: 127

model:
  name: MambaRefineCD
  encoder_family: mambavision   # mambavision | vmamba
  encoder_variant: small        # tiny | small | base
  encoder_pretrained: true
  freeze_encoder: false
  decoder_channels: 128

ablation:
  id: original
  temporal_input_mode: abs_signed   # abs_only | signed_only | abs_signed

train:
  device: cuda:0
  iterations: 50000
  batch_size: 8
  num_workers: 8
  lr: 0.0001
  weight_decay: 0.01
  optimizer: adamw
  scheduler: cosine
  warmup_iters: 2500
  amp: true
  grad_clip_norm: 0.5
  log_interval: 50
  val_interval: 5000
  save_best_only: true
  best_metric: F1

loss:
  bce_weight: 1.0
  dice_weight: 1.0
  coarse_weight: 0.4
  boundary_weight: 0.1

eval:
  threshold: 0.5                    # starting threshold; val sweep overrides this
  sweep_thresholds_on_val: true
  use_val_threshold_for_test: true
  threshold_min: 0.05
  threshold_max: 0.95
  threshold_step: 0.05
  save_predictions: true

resume:
  enabled: false
  path: null
  resume_optimizer: true
  resume_scheduler: true
  resume_iteration: true
```

---

## 🗂 Repository Structure

```
MambaRefine-CD/
├── configs/
│   ├── active.yaml                      ← THE config — edit this to run experiments
│   ├── datasets/
│   │   ├── dsifn.yaml
│   │   ├── whu.yaml
│   │   ├── levir.yaml
│   │   └── sysu.yaml
│   ├── encoders/
│   │   ├── mambavision_small.yaml
│   │   └── vmamba_small.yaml
│   └── ablations/
│       └── group_a_temporal/
│           ├── A0_abs_only.yaml
│           ├── A1_signed_only.yaml
│           └── A2_abs_signed.yaml
│
├── datasets/                            ← place datasets here (see datasets/README.md)
├── pretrained_weights/                  ← encoder weights (see pretrained_weights/README.md)
├── third_party/                         ← VMamba if needed (see tools/setup_vmamba.py)
│
├── src/
│   ├── datasets/
│   │   ├── cd_dataset.py               ← ChangeDetectionDataset (A/B/Mask format)
│   │   ├── transforms.py               ← synchronized transforms for A, B, Mask
│   │   └── verify.py                   ← dataset verification logic
│   │
│   ├── models/
│   │   ├── build.py                    ← build_model(cfg) entry point
│   │   ├── mambarefine_cd.py           ← MambaRefineCD main model class
│   │   ├── encoders/
│   │   │   ├── registry.py             ← build_encoder(cfg)
│   │   │   ├── mambavision_adapter.py  ← MambaVision wrapper
│   │   │   └── vmamba_adapter.py       ← VMamba wrapper
│   │   └── modules/
│   │       ├── temporal_difference.py  ← TemporalDifference (abs_only/signed/abs_signed)
│   │       ├── drbi.py                 ← D-RBI module
│   │       ├── arf_fpn.py              ← ARF-FPN decoder
│   │       ├── boundary_refinement.py  ← bounded residual head
│   │       └── heads.py               ← prediction heads
│   │
│   └── engine/
│       ├── trainer.py                  ← iteration-based trainer
│       ├── evaluator.py               ← evaluate() with threshold sweep
│       ├── losses.py                  ← BCE + Dice + coarse + boundary loss
│       ├── metrics.py                 ← changed-class binary metrics + BF1/BIoU
│       ├── checkpoint.py              ← save / load / resume
│       └── logger.py                  ← console + file + TensorBoard logging
│
├── tools/
│   ├── verify_dataset.py              ← full verification report
│   ├── check_model.py                 ← build + dummy forward + shapes + FLOPs
│   ├── check_dataset.py               ← counts + sample names
│   ├── check_training_step.py         ← one batch end-to-end sanity check
│   ├── summarize_results.py           ← scan outputs/ → results.csv
│   ├── prepare_dataset.py             ← dataset format converters
│   └── setup_vmamba.py                ← VMamba install helper
│
├── experiments/
│   ├── README.md
│   └── group_a_temporal_findings.md
│
├── train.py                           ← python train.py
├── val.py                             ← python val.py
├── test.py                            ← python test.py
├── infer.py                           ← python infer.py
└── requirements.txt
```

---

## 📐 Metrics

All metrics are **changed-class binary metrics** — computed on the changed/foreground class only, consistent with the standard binary change detection evaluation protocol.

| Metric | Formula | What it measures |
|---|---|---|
| **F1** *(primary)* | `2·P·R / (P+R)` | Harmonic mean of precision and recall for the changed class |
| **IoU** | `TP / (TP+FP+FN)` | Changed-pixel intersection over union |
| **Precision** | `TP / (TP+FP)` | Of predicted change pixels, how many are actually changed |
| **Recall** | `TP / (TP+FN)` | Of truly changed pixels, how many are detected |
| **OA** | `(TP+TN) / total` | Overall accuracy — dominated by the unchanged class, reported for completeness |
| **BF1** | boundary F1 at 3px | Contour-level prediction quality (3-pixel tolerance) |
| **BIoU** | boundary IoU at 3px | IoU between boundary bands at 3-pixel tolerance |
| **Trimap F1₃ₚₓ** | changed-class F1 inside 3px band | Performance specifically within the boundary region |

> ⚠️ **F1 is changed-class F1, not macro-F1.** Macro-F1 is inflated by the large unchanged-class accuracy and is not a meaningful measure of change detection performance. All comparisons in this repo and paper use changed-class F1.

The validation threshold is swept from 0.05 to 0.95. The best validation threshold is saved in the checkpoint and automatically applied at test time.

---

## 🔄 Reproducibility

Every run is fully reproducible:

- `seed: 42` seeds Python random, NumPy, PyTorch, and CUDA
- A frozen copy of `configs/active.yaml` is saved to the run folder **before** training starts
- Checkpoints include the config, iteration, best metric value, and best validation threshold
- `train_history.json`, `val_history.json`, and `test_metrics.json` are written for every run
- To reproduce any past run: copy its saved `config.yaml` into `configs/active.yaml`, then run `python train.py`

**Resuming an interrupted run:**

```yaml
resume:
  enabled: true
  path: outputs/run_20260101_120000_original/checkpoints/best_iter_030000_F1_0.9421.pth
  resume_optimizer: true
  resume_scheduler: true
  resume_iteration: true
```

---

## 🔧 Troubleshooting

**`FileNotFoundError: Missing directory: datasets/DSIFN-CD/train/A`**
→ Dataset not in expected format. Run `python tools/verify_dataset.py` — it prints exactly what is missing.

**`ImportError: No module named 'mamba_ssm'`**
→ MambaVision requires `mamba-ssm`. Install from the MambaVision repo. Alternatively set `encoder_family: vmamba` if VMamba is installed.

**`RuntimeError: Cannot resume — checkpoint not found`**
→ The path in `resume.path` does not exist. Set `resume.enabled: false` to train from scratch.

**D-RBI input channel mismatch on checkpoint load**
→ The checkpoint was trained under a different `temporal_input_mode`. Do not mix checkpoints across ablation modes — the D-RBI projection layer channel count changes with the mode.

**Loss is NaN from iteration 1**
→ Run `python tools/check_training_step.py`. Most common causes: masks not binarized (values 0–255 instead of 0–1), or FP16 overflow under `amp: true`. Try `amp: false` temporarily to isolate.

**Out of memory**
→ Reduce `batch_size` in `active.yaml`. MambaVision-S at 256×256 with batch 8 uses approximately 10–12 GB. Drop to batch 4 or switch to MambaVision-T (46.80M, 50.05 GFLOPs).

---

## 🙏 Acknowledgements

This work builds on the following:

- **[MambaVision](https://github.com/NVlabs/MambaVision)** — Hatamizadeh & Kautz, CVPR 2025: hybrid Mamba-Transformer backbone
- **[VMamba](https://github.com/MzeroMiko/VMamba)** — Liu et al., NeurIPS 2024: visual state space model with 2D selective scanning
- **[Mamba-CD](https://ieeexplore.ieee.org/document/PLACEHOLDER)** — Peng et al., IEEE JSTARS 2026: closest related Mamba-based CD method
- **[DSIFN-CD](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection)** — Shi et al., Information Fusion 2022
- **[WHU-CD](http://gpcv.whu.edu.cn/data/building_dataset.html)** — Ji et al., IEEE TGRS 2019
- **[Mamba-Segmentation](https://arxiv.org/abs/2604.18721)** — Wasalathilaka et al., arXiv 2026: the controlled SSM benchmark that motivates the boundary-aware design

---

## 📜 Citation

If MambaRefine-CD is useful for your research, please cite:

```bibtex
@misc{perera2026mambarefinecdmambavisionregionboundarytemporal,
      title={MambaRefine-CD: MambaVision with Region-Boundary Temporal Refinement}, 
      author={Dineth Perera and Thaariq Firdous and Oshadha Samarakoon and Roshan Godaliyadda and Parakrama Ekanayake and Vijitha Herath},
      year={2026},
      eprint={2607.04403},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2607.04403}, 
}
```

If you use the Mamba-Segmentation benchmark findings that motivate this work:

```bibtex
@misc{wasalathilaka2026controlledbenchmarkvisualstatespace,
  title   = {A Controlled Benchmark of Visual State-Space Backbones with
             Domain-Shift and Boundary Analysis for Remote-Sensing Segmentation},
  author  = {Wasalathilaka, Nichula and Perera, Dineth and Samarakoon, Oshadha
             and Wijenayake, Buddhi and Godaliyadda, Roshan and Herath, Vijitha
             and Ekanayake, Parakrama},
  year    = {2026},
  eprint  = {2604.18721},
  url     = {https://arxiv.org/abs/2604.18721}
}
```

---

🌍🛰️ Built at the **University of Peradeniya**. Found it useful? Give us a ⭐
