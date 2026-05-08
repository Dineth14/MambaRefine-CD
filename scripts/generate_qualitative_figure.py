"""Generate qualitative_boundary_examples.pdf for MambaRefine-CD paper.

Phase 1: Scan ALL test samples from DSIFN-CD and WHU-CD, rank by GT change
         pixel count, run A6 inference on the top-10 from each dataset,
         save individual panels to outputs/qualitative_candidates/{dsifn,whu}/.

Phase 2: Pick the N_FINAL best from each dataset (by F1 quality score
         = 1 - |error_pixels / total_pixels|) and compose the final figure.

Output layout per row:
    I1 | I2 | GT | Prediction | Error Map | Boundary Overlay
"""
from __future__ import annotations
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC  = _REPO / "src"
_rs, _ss = str(_REPO), str(_SRC)
sys.path[:] = [_rs, _ss] + [p for p in sys.path
    if p not in (_rs, _ss)
    and "MambaVision_experiments" not in p
    and "MambaFCS" not in p]

import json
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

# ── inline helpers ───────────────────────────────────────────────────────────
def _load_ckpt(ckpt_path, model, device, use_ema=True):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
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

# ── image helpers ─────────────────────────────────────────────────────────────
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def denorm(t: torch.Tensor) -> np.ndarray:
    img = t.cpu().float().numpy().transpose(1, 2, 0)
    return np.clip(img * STD + MEAN, 0, 1)  # float [0,1]

def _boundary_t(mask_np: np.ndarray, bw: int = 1) -> np.ndarray:
    t = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    k, p = 2*bw+1, bw
    dil = F.max_pool2d(t, k, stride=1, padding=p)
    ero = -F.max_pool2d(-t, k, stride=1, padding=p)
    return ((dil - ero).clamp(0,1).squeeze().numpy() > 0.5).astype(np.uint8)

def error_map_rgb(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    H, W = gt.shape
    out = np.zeros((H, W, 3), dtype=np.float32)
    out[(pred==1)&(gt==1)] = [1.0, 1.0, 1.0]   # TP white
    out[(pred==0)&(gt==0)] = [0.0, 0.0, 0.0]   # TN black
    out[(pred==1)&(gt==0)] = [0.86, 0.20, 0.20] # FP red
    out[(pred==0)&(gt==1)] = [0.20, 0.71, 0.31] # FN green
    return out

def bnd_overlay(a: np.ndarray, b: np.ndarray,
                pred_bnd: np.ndarray, gt_bnd: np.ndarray) -> np.ndarray:
    """Float [0,1] image: mean(a,b) with yellow GT bnd and cyan pred bnd."""
    base = ((a + b) / 2.0).copy()
    base[gt_bnd > 0]   = [1.0, 0.86, 0.0]   # yellow
    base[pred_bnd > 0] = [0.0, 0.86, 0.86]  # cyan
    return base.clip(0, 1)

def sample_f1(pred, gt):
    tp = ((pred==1)&(gt==1)).sum()
    fp = ((pred==1)&(gt==0)).sum()
    fn = ((pred==0)&(gt==1)).sum()
    return (2*tp / (2*tp + fp + fn + 1e-7))


# ── Phase 1: scan all test tiles, keep top-10 by change pixels ────────────────
def scan_and_collect(cfg_path, ckpt_path, threshold, dataset_label,
                     top_k=10, device=None, out_dir=None):
    print(f"\n{'='*60}")
    print(f"Scanning {dataset_label}")
    device = device or torch.device("cpu")
    cfg = load_config(cfg_path)
    cfg.setdefault("evaluation", {})["threshold"] = threshold
    cfg.setdefault("eval",       {})["threshold"] = threshold

    loader = build_test_loader(cfg)
    model  = build_model(cfg).to(device)
    info   = _load_ckpt(ckpt_path, model, device)
    print(f"  EMA used: {info['ema_used']}")
    model.eval()

    # collect ALL samples with any change content
    pool = []
    with torch.no_grad():
        for batch in loader:
            ia = batch["image_a"].to(device)
            ib = batch["image_b"].to(device)
            gt_t = batch["mask"].to(device)

            with torch.cuda.amp.autocast(enabled=True):
                out = model(ia, ib)
            probs = torch.sigmoid(_get_logits(out))  # (B,1,H,W)

            B = probs.shape[0]
            for b in range(B):
                pb  = probs[b, 0].cpu()
                gtb = (gt_t[b, 0] if gt_t.dim()==4 else gt_t[b]).cpu().numpy().astype(np.uint8)
                n_change = int(gtb.sum())
                if n_change < 200:   # skip nearly-empty tiles
                    continue
                pred_b = (pb.numpy() > threshold).astype(np.uint8)
                f1     = sample_f1(pred_b, gtb)
                pool.append({
                    "img_a":    denorm(ia[b].cpu()),
                    "img_b":    denorm(ib[b].cpu()),
                    "gt":       gtb,
                    "pred":     pred_b,
                    "n_change": n_change,
                    "f1":       f1,
                    "label":    dataset_label,
                })
    print(f"  Total tiles with change: {len(pool)}")

    # rank by number of change pixels, pick top_k
    pool.sort(key=lambda x: x["n_change"], reverse=True)
    top = pool[:top_k]
    print(f"  Kept top {len(top)} (by change pixel count)")
    for i, s in enumerate(top):
        print(f"    [{i+1:02d}] n_change={s['n_change']:6d}  F1={s['f1']:.3f}")

    # save individual panel images
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_panels(top, out_dir, threshold)

    return top


def _save_panels(samples, out_dir, threshold):
    """Save each sample as a 6-panel PNG for manual inspection."""
    for idx, s in enumerate(samples):
        pred_bnd = _boundary_t(s["pred"])
        gt_bnd   = _boundary_t(s["gt"])
        err  = error_map_rgb(s["pred"], s["gt"])
        bov  = bnd_overlay(s["img_a"], s["img_b"], pred_bnd, gt_bnd)
        gt_disp   = np.stack([s["gt"]]*3, axis=-1).astype(np.float32)
        pred_disp = np.stack([s["pred"]]*3, axis=-1).astype(np.float32)

        panels = [s["img_a"], s["img_b"], gt_disp, pred_disp, err, bov]
        titles = ["I1", "I2", "GT", "Pred", "Error", "Boundary"]

        fig, axes = plt.subplots(1, 6, figsize=(14, 2.4))
        for ax, im, t in zip(axes, panels, titles):
            ax.imshow(im.clip(0,1), interpolation="nearest")
            ax.set_title(t, fontsize=8)
            ax.axis("off")
        fig.suptitle(f"{s['label']}  [{idx+1:02d}]  n_change={s['n_change']}  F1={s['f1']:.3f}",
                     fontsize=9, fontweight="bold")
        plt.tight_layout(pad=0.2)
        save_path = out_dir / f"{idx+1:02d}_nchg{s['n_change']}_f1{int(s['f1']*100)}.png"
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved {len(samples)} panel images to {out_dir}")


# ── Phase 2: pick best by F1, compose final paper figure ─────────────────────
def compose_figure(dsifn_pool, whu_pool, n_dsifn=3, n_whu=3,
                   out_path=None):
    """
    Pick n_dsifn best-F1 from dsifn_pool and n_whu best-F1 from whu_pool.
    Compose a single figure with (n_dsifn + n_whu) rows x 6 columns.
    """
    def pick_best(pool, n):
        ranked = sorted(pool, key=lambda x: x["f1"], reverse=True)
        return ranked[:n]

    dsifn_chosen = pick_best(dsifn_pool, n_dsifn)
    whu_chosen   = pick_best(whu_pool,   n_whu)
