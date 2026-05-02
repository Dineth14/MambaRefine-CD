"""Generate qualitative_boundary_examples.pdf for MambaRefine-CD paper.

Layout per row (4 rows, 3-4 test samples each):
    I1 | I2 | GT | A6 Prediction | A6 Error Map | Boundary Overlay

Error map colours:
    White  = TP  (predicted change, GT change)
    Black  = TN  (predicted no-change, GT no-change)
    Red    = FP  (predicted change, GT no-change)
    Green  = FN  (predicted no-change, GT change)

Boundary overlay: GT boundary in yellow, predicted boundary in cyan,
overlaid on the mean of I1 and I2.

Samples are chosen to show interesting boundary behaviour.
Output: MambaRefine_CD/figures/qualitative_boundary_examples.pdf
"""
from __future__ import annotations
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
# Remove competing repo src paths
sys.path = [p for p in sys.path if "MambaVision_experiments" not in p and "MambaFCS" not in p]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

import random
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.config import load_config
from data.dataset_builder import build_test_loader
from models.mambarefinecd import build_model

# Inline helpers to avoid training.* import conflict with other workspace repos
def _load_ckpt(ckpt_path, model, device, use_ema=True):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = dict(ckpt.get("model", {}))
    ema_used = False
    if use_ema:
        ema = ckpt.get("ema")
        shadow = ema.get("shadow") if isinstance(ema, dict) else None
        if shadow:
            state.update(shadow); ema_used = True
    model.load_state_dict(state, strict=False)
    return {"ema_used": ema_used, "best_threshold": ckpt.get("best_threshold")}

def _get_logits(output):
    if isinstance(output, dict):
        return output.get("change_logits") or output["binary_change_logits"]
    if isinstance(output, (list, tuple)):
        return output[0]
    return output

REPO = _REPO
FIG_DIR = REPO / "MambaRefine_CD" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def denorm(t: torch.Tensor) -> np.ndarray:
    """Convert normalised image tensor (C,H,W) -> uint8 HWC."""
    img = t.cpu().float().numpy().transpose(1, 2, 0)  # HWC
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def _boundary_np(mask: np.ndarray, bw: int = 1) -> np.ndarray:
    """Extract 1-pixel boundary via dilation - erosion (numpy)."""
    t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    k = 2 * bw + 1
    dilated = F.max_pool2d(t, k, stride=1, padding=bw)
    eroded  = -F.max_pool2d(-t, k, stride=1, padding=bw)
    bnd = (dilated - eroded).clamp(0, 1).squeeze().numpy()
    return (bnd > 0.5).astype(np.uint8)


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Build RGB error map: TP=white, TN=black, FP=red, FN=green."""
    H, W = gt.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    tp = (pred == 1) & (gt == 1)
    tn = (pred == 0) & (gt == 0)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    out[tp] = [255, 255, 255]  # white
    out[tn] = [0,   0,   0  ]  # black
    out[fp] = [220, 50,  50 ]  # red
    out[fn] = [50,  180, 80 ]  # green
    return out


def boundary_overlay(img_a: np.ndarray, img_b: np.ndarray,
                     pred_bnd: np.ndarray, gt_bnd: np.ndarray) -> np.ndarray:
    """Mean(I1,I2) with GT boundary yellow and predicted boundary cyan."""
    base = ((img_a.astype(np.float32) + img_b.astype(np.float32)) / 2).astype(np.uint8).copy()
    base[gt_bnd > 0] = [255, 220, 0]    # yellow
    base[pred_bnd > 0] = [0, 220, 220]  # cyan
    return base


def collect_samples(loader, model, threshold, device, n_samples=4,
                    seed=42) -> list:
    """Run inference, collect n_samples with decent boundary content."""
    rng = random.Random(seed)
    candidates = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ia = batch["image_a"].to(device)
            ib = batch["image_b"].to(device)
            gt_t = batch["mask"].to(device)

            with torch.cuda.amp.autocast(enabled=True):
                output = model(ia, ib)
            logits = _get_logits(output)
            probs = torch.sigmoid(logits)

            B = probs.shape[0]
            for b in range(B):
                prob_b = probs[b, 0].cpu()
                gt_b   = gt_t[b, 0].cpu().numpy().astype(np.uint8)
                pred_b = (prob_b.numpy() > threshold).astype(np.uint8)

                # only keep samples with some change content
                if gt_b.sum() < 100:
                    continue

                img_a_np = denorm(ia[b].cpu())
                img_b_np = denorm(ib[b].cpu())
                candidates.append({
                    "img_a": img_a_np,
                    "img_b": img_b_np,
                    "gt":   gt_b,
                    "pred": pred_b,
                })
            if len(candidates) >= max(n_samples * 5, 20):
                break

    # Sort by GT boundary pixel count to pick samples with clear edges
    def gt_bnd_pixels(c):
        return _boundary_np(c["gt"]).sum()

    candidates.sort(key=gt_bnd_pixels, reverse=True)
    # pick from the top candidates with some spread
    step = max(1, len(candidates) // n_samples)
    chosen = [candidates[i * step] for i in range(min(n_samples, len(candidates)))]
    return chosen


def make_qualitative_figure(samples: list, title: str = "MambaRefine-CD",
                             out_path: Path = None) -> None:
    n = len(samples)
    cols = 6
    fig, axes = plt.subplots(n, cols, figsize=(cols * 1.8, n * 1.8))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["$I_1$", "$I_2$", "GT", "Prediction", "Error Map", "Boundary Overlay"]
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=8, fontweight="bold", pad=3)

    for i, s in enumerate(samples):
        gt   = s["gt"]
        pred = s["pred"]
        pred_bnd = _boundary_np(pred)
        gt_bnd   = _boundary_np(gt)
        err  = error_map(pred, gt)
        bnd_ov = boundary_overlay(s["img_a"], s["img_b"], pred_bnd, gt_bnd)

        imgs = [s["img_a"], s["img_b"],
                (gt * 255).astype(np.uint8),
                (pred * 255).astype(np.uint8),
                err, bnd_ov]

        for j, im in enumerate(imgs):
            ax = axes[i, j]
            if im.ndim == 2:
                ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            else:
                ax.imshow(im, interpolation="nearest")
            ax.axis("off")

    # Legend for error map
    legend_handles = [
        mpatches.Patch(color="#ffffff", ec="#aaaaaa", label="TP"),
        mpatches.Patch(color="#000000", label="TN"),
        mpatches.Patch(color="#dc3232", label="FP"),
        mpatches.Patch(color="#32b450", label="FN"),
        mpatches.Patch(color="#ffdc00", label="GT bnd"),
        mpatches.Patch(color="#00dcdc", label="Pred bnd"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=6,
               fontsize=7, frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(title, fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.3)
    if out_path is None:
        out_path = FIG_DIR / "qualitative_boundary_examples.pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = str(REPO / "configs/ablations/dsifn/a6_full.yaml")
    ckpt_path   = str(REPO / "outputs/dsifn/a6_full"
                      "/run_dsifn_a6_full_seed42_20260501_095853"
                      "/best_model_final.pth")
    threshold   = 0.6

    print(f"Loading config: {config_path}")
    cfg = load_config(config_path)
    cfg.setdefault("evaluation", {})["threshold"] = threshold
    cfg.setdefault("eval", {})["threshold"] = threshold

    loader = build_test_loader(cfg)
    model  = build_model(cfg).to(device)
    load_info = _load_ckpt(ckpt_path, model, device, use_ema=True)
    print(f"EMA used: {load_info.get('ema_used')}")

    print("Collecting samples...")
    samples = collect_samples(loader, model, threshold, device, n_samples=4)
    print(f"Collected {len(samples)} samples.")

    make_qualitative_figure(
        samples,
        title="MambaRefine-CD: Qualitative Boundary Results on DSIFN-CD",
        out_path=FIG_DIR / "qualitative_boundary_examples.pdf",
    )


if __name__ == "__main__":
    main()
