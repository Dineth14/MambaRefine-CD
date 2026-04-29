# External Baseline Weights

This repository does not invent or mirror external checkpoint URLs. Only verified public weights should be downloaded.

| Model | Repo Link | Available Pretrained Weights | Datasets Supported | Download Command | Note |
|---|---|---|---|---|---|
| ChangeFormer | https://github.com/wgcban/ChangeFormer | TODO: verify official README checkpoint URLs | LEVIR-CD, DSIFN-CD if official links are accessible | `python tools/download_baseline_weights.py --changeformer-levir-url <VERIFIED_URL>` | Do not use unverified mirrors. |
| SNUNet | TODO official repo link | None verified | TODO | N/A | No official pretrained weights found; train from scratch. |
| IFNet | TODO official repo link | None verified | TODO | N/A | No reliable official pretrained weights found; train from scratch. |
| CDMamba | TODO official repo link | None verified for LEVIR/WHU/DSIFN | TODO | N/A | Train from scratch unless the user provides a checkpoint. |
| M-CD | TODO official repo link | None verified | TODO | N/A | No verified pretrained weights found; train from scratch. |

Use:

```bash
python tools/download_baseline_weights.py
```

The command creates `external_weights/`, prints the status for each baseline, and never silently ignores failed downloads.
