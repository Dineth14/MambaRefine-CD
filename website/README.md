# MambaRefine-CD Website

Research project website for **MambaRefine-CD**. Built with plain HTML/CSS/JS and MathJax v3.

## Local preview

```bash
cd website
python -m http.server 8000
# Open http://localhost:8000 in your browser
```

> **Important:** Open via a local server, not directly as a `file://` URL, so that relative paths to SVG assets resolve correctly.

## GitHub Pages deployment

1. Push the repo to GitHub.
2. Go to **Settings → Pages**.
3. Source: **Deploy from a branch** → branch `main`, folder `/website`.
4. The site will be published at `https://<username>.github.io/<repo>/`.

## Editing results

Results tables are in `index.html`. Each table is marked with:

```html
<!-- EDIT RESULTS HERE: tab-levir -->
<!-- EDIT RESULTS HERE: tab-whu -->
<!-- EDIT RESULTS HERE: tab-dsifn -->
<!-- EDIT RESULTS HERE: tab-ablation -->
```

Replace `&mdash;` placeholder cells in the **MambaRefine-CD (ours)** row with actual numbers.

## Adding qualitative images

Place images in `assets/qualitative/` following this naming convention:

```
assets/qualitative/
  example_01_A.png      # pre-change image crop
  example_01_B.png      # post-change image crop
  example_01_gt.png     # ground truth mask
  example_01_pred.png   # model prediction
  example_01_err.png    # error map (TP=green, FP=red, FN=blue)
```

Then update the `<img>` tags in the `#qualitative` section of `index.html`:

```html
<img src="assets/qualitative/example_01_A.png" alt="..."/>
```

The placeholder `<div class="img-placeholder">` divs are already in place — replace each with an `<img>` tag.

## Replacing diagrams

All SVG diagrams are in `assets/diagrams/`. They are plain SVG files with embedded Unicode text — no external fonts required.

If you want to regenerate them with ChatGPT, use `DIAGRAM_GENERATION_GUIDE.md` in this folder. It explains the architecture, the shared visual style, and gives a copy-paste prompt for each SVG.

| File | Content |
|---|---|
| `01_problem_naive_difference.svg` | Naive diff failure modes vs D-RBI |
| `02_mambavision_encoder.svg` | Shared encoder + 4-scale feature pyramids |
| `03_drbi_module.svg` | D-RBI module detail |
| `04_region_boundary_gates.svg` | Region and boundary gate parallel paths |
| `05_adaptive_rf_decoder.svg` | ARF-FPN + boundary residual decoder |
| `06_full_architecture.svg` | Complete end-to-end architecture |
| `07_experiment_timeline.svg` | Design evolution 4-stage timeline |
| `08_metric_explanation.svg` | TP/FP/FN confusion matrix + metric formulas |

## MathJax configuration

The site uses MathJax v3 with `tex-svg.js`. Inline math uses `\( ... \)` and display math uses `\[ ... \]`. Do **not** use `$` or `$$` delimiters — they are not configured and will render as plain text.

## File structure

```
website/
  index.html              ← main page (793 lines, 16 sections)
  styles.css              ← all styles (Inter font, academic palette)
  script.js               ← tabs, mobile nav, scroll progress, fade-in
  README.md               ← this file
  assets/
    diagrams/             ← 8 SVG diagrams
    results/              ← (placeholder) result figures
    qualitative/          ← (placeholder) qualitative image grids
    datasets/             ← (placeholder) dataset preview images
```
