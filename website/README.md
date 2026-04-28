# Website README

## Preview locally

```bash
cd website
python -m http.server 8000
```

Then open `http://localhost:8000`.

## Extract verified results

```bash
python scripts/extract_all_results_for_website.py
python scripts/extract_mambacd_protocol_results.py
```

Outputs:

- `website/assets/data/ours_results.json`
- `website/assets/data/ours_results.csv`
- `website/assets/data/ours_results_all_candidates.json`
- `website/assets/data/mambacd_protocol_ours.json`
- `website/assets/data/mambacd_protocol_ours.csv`
- `website/assets/data/mambacd_protocol_ours_all_candidates.json`
- `website/assets/data/mambacd_paper_comparison.json`
- `website/assets/data/mambacd_paper_comparison.csv`

## Reproduce SOTA evaluations

```bash
python scripts/run_sota_reproduction_pipeline.py
```

Outputs:

- `outputs/sota_reproduced_eval/reports/master_status.json`
- `outputs/sota_reproduced_eval/tables/`
- `website/assets/data/reproduced_sota_results.json`

## Extract external SOTA results from local sources only

```bash
python scripts/extract_external_sota_results.py
```

Outputs:

- `website/assets/data/external_sota_results.json`
- `website/assets/data/external_sota_results.csv`
- `website/assets/data/external_sota_sources.json`

## Profile model efficiency

```bash
python scripts/model_efficiency.py
```

Outputs:

- `website/assets/data/ours_efficiency.json`
- `outputs/model_efficiency/latest_efficiency.json`

## Collect qualitative images

```bash
python scripts/collect_website_qualitative.py
```

Outputs:

- `website/assets/qualitative/manifest.json`
- copied qualitative images under `website/assets/qualitative/`

## Validate website

```bash
python scripts/validate_mambacd_protocol_tables.py
python scripts/validate_website.py
```

Outputs:

- `outputs/website_validation/mambacd_protocol_validation.json`
- `outputs/website_validation/mambacd_protocol_validation.md`
- `outputs/website_validation/website_validation_report.json`
- `outputs/website_validation/website_validation_report.md`

## Deploy on GitHub Pages

1. Push the repository.
2. Open the repository settings on GitHub.
3. Go to **Pages**.
4. Choose **Deploy from a branch**.
5. Set the published folder to `/website`.

## Where to edit values manually

- Verified local results: `website/assets/data/ours_results.json`
- Mamba-CD protocol local results: `website/assets/data/mambacd_protocol_ours.json`
- Mamba-CD paper comparison rows: `website/assets/data/mambacd_paper_comparison.json`
- Reproduced comparison rows: `website/assets/data/reproduced_sota_results.json`
- External local-source results: `website/assets/data/external_sota_results.json`
- Efficiency values: `website/assets/data/ours_efficiency.json`
- Qualitative manifest: `website/assets/qualitative/manifest.json`
- Static content and section text: `website/index.html`

## Mamba-CD protocol note

- The Mamba-CD protocol table has been added to the website.
- Paper F1 corresponds to change-class `F1_1`, not `mF1`.
- MambaRefine-CD maps paper metrics to `precision_1`, `recall_1`, `F1_1`, `IoU_1`, and `OA`.
- Literature values are copied from the Mamba-CD paper tables.
- Our values are extracted from local evaluation logs and result files.

## SECOND semantic outputs

True semantic SECOND evaluations are emitted under the normal eval directory as:

- `second_metrics.json` / `second_metrics.csv`
- `second_semantic_metrics.json` / `second_semantic_metrics.csv`

The website should treat `SeK` as valid only when those semantic outputs come from `model.output_mode: semantic_change` with timestamp-wise semantic predictions enabled.
