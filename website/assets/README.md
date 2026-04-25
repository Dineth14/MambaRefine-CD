# website/assets — Image Asset Reference

This directory holds all images used by the project website (`website/index.html`).

---

## Architecture Diagram

| Filename | Used in | Description |
|----------|---------|-------------|
| `architecture_placeholder.png` | Method section | Full model architecture diagram. Replace with a vector-export (PNG or SVG) of the architecture figure. Recommended size: ≥ 1400 × 600 px. |

---

## Dataset Thumbnails

| Filename | Dataset | Description |
|----------|---------|-------------|
| `levir_placeholder.png` | LEVIR-CD card | Representative image pair + change mask from LEVIR-CD. Recommended size: 800 × 300 px (wide crop). |
| `whu_placeholder.png` | WHU-CD card | Representative image pair + change mask from WHU-CD. |
| `dsifn_placeholder.png` | DSIFN-CD card | Representative image pair + change mask from DSIFN-CD. |

---

## Qualitative Results

Name qualitative result images following this pattern:

```
qual_ex<N>_<panel>.png
```

where `<N>` is the example number (1, 2, 3, …) and `<panel>` is one of:

| Panel suffix | Content |
|-------------|---------|
| `a`         | Pre-change image $I_1$ |
| `b`         | Post-change image $I_2$ |
| `gt`        | Ground truth change mask (binary, white = change) |
| `pred`      | Model prediction (binary or probability map) |
| `err`       | Error map: green = correct change, red = false positive, blue = false negative |

Example set for 3 examples:
```
qual_ex1_a.png
qual_ex1_b.png
qual_ex1_gt.png
qual_ex1_pred.png
qual_ex1_err.png
qual_ex2_a.png
...
```

After saving images here, update the `qual-row` divs in `website/index.html`:

```html
<!-- Replace the placeholder-cell divs in each .qual-row with: -->
<div class="qual-cell">
  <img src="assets/qual_ex1_a.png" alt="Image A example 1"/>
</div>
```

---

## Recommended Image Formats

- **PNG** for masks and error maps (lossless, important for binary visualisation).
- **JPEG** at quality ≥ 90 for natural images (smaller file, acceptable for photos).
- **SVG** or high-res **PNG** for diagrams (avoid JPEG compression artefacts).

---

## After Adding Images

1. Remove any `placeholder-img` class or `onerror` handler from the `<img>` tag in `index.html` for that image.
2. Commit and push to trigger GitHub Pages rebuild.
