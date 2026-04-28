"""Compare one sample through validation and test input pipelines."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from utils.config import load_config
from data.dataset_builder import build_dataset


def _find_index(ds, sample_id: str | None) -> int:
    if not sample_id:
        return 0
    if hasattr(ds, "index"):
        for i, entry in enumerate(ds.index):
            if sample_id in Path(entry["image_a_path"]).name:
                return i
    if hasattr(ds, "names"):
        for i, name in enumerate(ds.names):
            if sample_id in str(name):
                return i
    return 0


def _stats(t: torch.Tensor) -> dict:
    x = t.detach().float()
    return {
        "shape": tuple(x.shape),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std()),
    }


def _save_prob(path: Path, logits: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prob = torch.sigmoid(logits.detach().cpu())[0, 0].clamp(0, 1).numpy()
    Image.fromarray((prob * 255).astype("uint8")).save(path)


def _describe(label: str, item: dict) -> None:
    mask = item.get("label", item.get("mask"))
    print(f"{label} id: {item.get('id', item.get('name'))}")
    print(f"{label} image_a: {_stats(item['image_a'])}")
    print(f"{label} image_b: {_stats(item['image_b'])}")
    print(f"{label} mask unique: {sorted(float(v) for v in torch.unique(mask).tolist())}")
    print(f"{label} mask shape: {tuple(mask.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare validation and test preprocessing for one sample.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--out_dir", default="outputs/debug_levir_eval/pipeline_compare")
    parser.add_argument("--use_ema", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dc = cfg.get("dataset", {})
    seed = int(cfg.get("experiment", {}).get("seed", 42))
    val_ds = build_dataset(dc, "val", augment=False, seed=seed)
    test_ds = build_dataset(dc, "test", augment=False, seed=seed)
    val_item = val_ds[_find_index(val_ds, args.sample_id)]
    test_item = test_ds[_find_index(test_ds, args.sample_id)]

    _describe("val", val_item)
    _describe("test", test_item)
    for key in ("image_a", "image_b"):
        va = _stats(val_item[key])
        te = _stats(test_item[key])
        if abs(va["mean"] - te["mean"]) > 0.5 or abs(va["std"] - te["std"]) > 0.5:
            print(f"WARNING: {key} statistics differ strongly between selected val/test samples.")

    if args.ckpt:
        from models.mambarefinecd import build_model
        from training.checkpoint import load_for_eval
        from training.model_outputs import normalize_model_output

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg).to(device)
        info = load_for_eval(args.ckpt, model, map_location=device, use_ema=args.use_ema)
        print(f"Loaded checkpoint iter={info['iteration']} ema_found={info['ema_found']} ema_used={info['ema_used']}")
        model.eval()
        out_dir = Path(args.out_dir)
        with torch.no_grad():
            for label, item in [("val", val_item), ("test", test_item)]:
                ia = item["image_a"].unsqueeze(0).to(device)
                ib = item["image_b"].unsqueeze(0).to(device)
                logits = normalize_model_output(model(ia, ib))["change_logits"]
                _save_prob(out_dir / f"{label}_prob.png", logits)
                print(f"{label} logits: {_stats(logits.cpu())}")


if __name__ == "__main__":
    main()
