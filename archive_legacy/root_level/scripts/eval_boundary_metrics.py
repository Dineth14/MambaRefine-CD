"""Boundary metric evaluation for MambaRefine-CD ablation variants.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_boundary_metrics.py
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parents[1]
_SRC  = _REPO / "src"
_rs, _ss = str(_REPO), str(_SRC)
sys.path[:] = [_rs, _ss] + [p for p in sys.path
    if p not in (_rs, _ss) and "MambaVision_experiments" not in p and "MambaFCS" not in p]

import torch
import torch.nn.functional as F
from utils.config import load_config
from data.dataset_builder import build_test_loader
from models.mambarefinecd import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
EPS = 1e-7

# ── Inline checkpoint loader (avoids training.* import conflict) ─────────────
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
    return {"iter": ckpt.get("iteration", 0), "ema_used": ema_used}

# ── Inline output normalizer ─────────────────────────────────────────────────
def _get_logits(output):
    if isinstance(output, dict):
        return output.get("change_logits") or output["binary_change_logits"]
    if isinstance(output, (list, tuple)):
        return output[0]
    return output

# ── Morphological helpers ────────────────────────────────────────────────────
def _dilate(m, r):
    if r == 0: return m
    k = 2*r+1
    return (F.max_pool2d(m, k, stride=1, padding=r) > 0).float()

def _boundary(m, bw=1):
    k, p = 2*bw+1, bw
    return (F.max_pool2d(m,k,stride=1,padding=p) - (-F.max_pool2d(-m,k,stride=1,padding=p))).clamp(0,1)

# ── Streaming boundary metrics ───────────────────────────────────────────────
class BndMetrics:
    def __init__(self, tol=3): self.tol=tol; self.reset()
    def reset(self):
        self.bp=self.bd=self.rp=self.rd=self.ii=self.uu=self.tp=self.fp=self.fn=0.0
    def update(self, prob, gt, thr):
        with torch.no_grad():
            pb = (prob>thr).float().unsqueeze(0).unsqueeze(0)
            gb =  gt.float().unsqueeze(0).unsqueeze(0)
            pbnd=_boundary(pb); gbnd=_boundary(gb)
            gd=_dilate(gbnd,self.tol); pd=_dilate(pbnd,self.tol)
            self.bp+=(pbnd*gd).sum().item(); self.bd+=pbnd.sum().item()
            self.rp+=(gbnd*pd).sum().item(); self.rd+=gbnd.sum().item()
            pband=_dilate(pbnd,self.tol); gband=_dilate(gbnd,self.tol)
            self.ii+=(pband*gband).sum().item()
            self.uu+=((pband+gband).clamp(0,1)).sum().item()
            ins=gband.bool()
            pi=pb[ins].float(); gi=gb[ins].float()
            self.tp+=(pi*gi).sum().item(); self.fp+=(pi*(1-gi)).sum().item(); self.fn+=((1-pi)*gi).sum().item()
    def compute(self):
        p=self.bp/(self.bd+EPS); r=self.rp/(self.rd+EPS)
        bf1=2*p*r/(p+r+EPS)
        biou=self.ii/(self.uu+EPS)
        tp,fp,fn=self.tp,self.fp,self.fn
        tri_p=tp/(tp+fp+EPS); tri_r=tp/(tp+fn+EPS)
        tri=2*tri_p*tri_r/(tri_p+tri_r+EPS)
        return {"BF1":round(bf1*100,2),"BIoU":round(biou*100,2),"Trimap_F1_3px":round(tri*100,2)}

# ── Evaluation function ───────────────────────────────────────────────────────
def eval_variant(cfg_path, ckpt_path, thr, label, device):
    logger.info("="*55+f"\n  {label}\n  ckpt={ckpt_path}\n  thr={thr}")
    cfg = load_config(cfg_path)
    cfg.setdefault("evaluation",{})["threshold"]=thr
    cfg.setdefault("eval",{})["threshold"]=thr
    loader=build_test_loader(cfg)
    model=build_model(cfg).to(device)
    info=_load_ckpt(ckpt_path, model, device)
    logger.info(f"  EMA used: {info['ema_used']}")
    model.eval(); bm=BndMetrics(3)
    with torch.no_grad():
        for batch in loader:
            i1=batch["image_a"].to(device); i2=batch["image_b"].to(device); gt=batch["mask"].to(device)
            with torch.cuda.amp.autocast(enabled=True):
                out=model(i1,i2)
            logits=_get_logits(out); probs=torch.sigmoid(logits)
            for b in range(probs.shape[0]):
                pb=probs[b,0].cpu()
                gtb=(gt[b,0] if gt.dim()==4 else gt[b]).cpu().float()
                if gtb.shape!=pb.shape:
                    gtb=F.interpolate(gtb.unsqueeze(0).unsqueeze(0).float(),size=pb.shape,mode="nearest").squeeze()
                bm.update(pb, gtb, thr)
    m=bm.compute()
    logger.info(f"  BF1={m['BF1']}  BIoU={m['BIoU']}  Tri={m['Trimap_F1_3px']}")
    return m

# ── LaTeX table ───────────────────────────────────────────────────────────────
def latex_table(rows):
    hdr=[r"\begin{table}[t]",r"\centering",
         r"\caption{Boundary-specific metrics (3-pixel tolerance). BF1\,=\,boundary F1;",
         r"  BIoU\,=\,boundary band IoU; Trimap~F1\,=\,changed-class F1 in the 3-px GT boundary band.}",
         r"\label{tab:boundary_metrics}",
         r"\begin{tabular}{lccccc}",r"\toprule",
         r"Variant & F1 & IoU & BF1 & BIoU & Trimap F1-3px \\",r"\midrule"]
    body=[f"{r['label']} & {r['f1']:.2f} & {r['iou']:.2f} & {r['BF1']:.2f} & {r['BIoU']:.2f} & {r['Trimap_F1_3px']:.2f} \\\\" for r in rows]
    return "\n".join(hdr+body+[r"\bottomrule",r"\end{tabular}",r"\end{table}"])

# ── Variant list ─────────────────────────────────────────────────────────────
VARIANTS=[
  {"label":"A1 MambaVision-FPN","config":"configs/ablations/dsifn/a1_mambavision_fpn.yaml",
   "ckpt":"outputs/dsifn/a1_mambavision_fpn/run_dsifn_a1_mambavision_fpn_seed42_20260501_004250/best_model_final.pth",
   "thr":0.5,"f1":93.21,"iou":87.28},
  {"label":"A4 + ARF-FPN","config":"configs/ablations/dsifn/a4_mambavision_drbi_arf.yaml",
   "ckpt":"outputs/dsifn/a4_mambavision_drbi_arf/run_dsifn_a4_mambavision_drbi_arf_seed42_20260501_010826/best_model_final.pth",
   "thr":0.6,"f1":93.71,"iou":88.16},
  {"label":"A5 + Boundary Residual","config":"configs/ablations/dsifn/a5_mambavision_drbi_arf_boundary.yaml",
   "ckpt":"outputs/dsifn/a5_mambavision_drbi_arf_boundary/run_dsifn_a5_mambavision_drbi_arf_boundary_seed42_20260501_012826/best_model_final.pth",
   "thr":0.6,"f1":93.59,"iou":87.94},
  {"label":"A6 Full Model","config":"configs/ablations/dsifn/a6_full.yaml",
   "ckpt":"outputs/dsifn/a6_full/run_dsifn_a6_full_seed42_20260501_095853/best_model_final.pth",
   "thr":0.6,"f1":94.58,"iou":89.72},
  {"label":"WHU Full Model","config":"configs/experiments/whu_full.yaml",
   "ckpt":"outputs/whu/full/run_whu_whu_full_seed42_20260430_114506/best_model_final.pth",
   "thr":0.55,"f1":95.15,"iou":90.76},
]

def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    out=_REPO/"outputs"/"boundary_eval"; out.mkdir(parents=True, exist_ok=True)
    results=[]
    for v in VARIANTS:
        cp=str(_REPO/v["config"]); kp=str(_REPO/v["ckpt"])
        if not Path(kp).exists(): logger.warning(f"No ckpt: {kp}"); continue
        if not Path(cp).exists(): logger.warning(f"No cfg: {cp}"); continue
        try:
            m=eval_variant(cp,kp,v["thr"],v["label"],device)
            results.append({"label":v["label"],"f1":v["f1"],"iou":v["iou"],**m})
        except Exception as e:
            import traceback; logger.error(f"Failed {v['label']}: {e}"); traceback.print_exc()
    (out/"boundary_metrics_summary.json").write_text(json.dumps(results,indent=2))
    dsifn=[r for r in results if "WHU" not in r["label"]]
    whu=[r for r in results if "WHU" in r["label"]]
    tex=latex_table(dsifn+whu)
    (_REPO/"MambaRefine_CD"/"tables"/"table_boundary_metrics.tex").write_text(tex)
    logger.info("\n"+"="*62)
    logger.info(f"{'Variant':<34}{'F1':>6}{'IoU':>6}{'BF1':>6}{'BIoU':>6}{'Tri':>6}")
    logger.info("-"*62)
    for r in results:
        logger.info(f"{r['label']:<34}{r['f1']:>6.2f}{r['iou']:>6.2f}{r['BF1']:>6.2f}{r['BIoU']:>6.2f}{r['Trimap_F1_3px']:>6.2f}")

if __name__=="__main__":
    main()
