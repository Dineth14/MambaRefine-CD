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
```

Outputs:

- `website/assets/data/ours_results.json`
- `website/assets/data/ours_results.csv`
- `website/assets/data/ours_results_all_candidates.json`

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
python scripts/validate_website.py
```

Outputs:

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
- Reproduced comparison rows: `website/assets/data/reproduced_sota_results.json`
- External local-source results: `website/assets/data/external_sota_results.json`
- Efficiency values: `website/assets/data/ours_efficiency.json`
- Qualitative manifest: `website/assets/qualitative/manifest.json`
- Static content and section text: `website/index.html`
