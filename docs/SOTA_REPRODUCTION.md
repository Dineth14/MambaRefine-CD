# SOTA Reproduction

This pipeline evaluates external change-detection models under the local MambaRefine-CD dataset loaders and metric definitions.

## Supported models

- ChangeFormer
- BIT
- SNUNet
- STANet
- Mamba-CD
- MambaRefine-CD (re-evaluated under the same pipeline for comparability)

## Workspace

- Repositories: `external/`
- Weights: `external_weights/`
- Reproduced evaluation outputs: `outputs/sota_reproduced_eval/`
- Generated website data: `website/assets/data/reproduced_sota_results.json`

## Manual checkpoints

If an automatic weight download is not available, place the checkpoint under:

- `external_weights/<Model>/<Dataset>/`

Then run:

```bash
python scripts/discover_sota_checkpoints.py
python scripts/evaluate_sota_models.py
python scripts/write_sota_tables.py
```

## Full pipeline

```bash
python scripts/run_sota_reproduction_pipeline.py
```

Or run step-by-step:

```bash
python scripts/clone_sota_repos.py
python scripts/download_sota_weights.py
python scripts/discover_sota_checkpoints.py
python scripts/evaluate_sota_models.py
python scripts/write_sota_tables.py
python scripts/collect_website_qualitative.py
python scripts/validate_website.py
```

## Status files

- Repo clone status: `outputs/sota_reproduced_eval/reports/repo_clone_status.json`
- Weight download status: `outputs/sota_reproduced_eval/reports/weight_download_status.json`
- Checkpoint discovery: `outputs/sota_reproduced_eval/reports/checkpoint_discovery.json`
- Master pipeline status: `outputs/sota_reproduced_eval/reports/master_status.json`

Every evaluation output directory also contains `status.json`. No metric row should be trusted unless `status.json` reports `status = OK`.

## Metrics

Binary comparisons use the local metric implementation:

- `mF1`
- `F1_1`
- `F1_0`
- `mIoU`
- `IoU_1`
- `IoU_0`
- `Precision_1`
- `Recall_1`
- `OA`
- `Boundary F1`
- `Edge IoU`

Do not compare `mF1` against `F1_1`. Literature tables often report one or the other, but reproduced tables here stay within one consistent evaluation pipeline.

## Why reproduced metrics matter

Literature tables often differ in:

- split definitions
- threshold policy
- post-processing
- metric naming
- whether `mF1` or `F1_1` is reported

Reproduced evaluation under one code path is therefore more reliable than copying published summary rows without protocol alignment.
