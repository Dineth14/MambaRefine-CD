"""Regenerate qualitative result figures with a larger legend.

This script uses the qualitative result images already stored in this
MambaRefine-CD repository. It removes the old compact bottom legend, draws a
larger centered legend, and refreshes both the top-level figures and the Beamer
slide copies.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
FONT_REGULAR = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

LEGEND_ITEMS = [
    ("TP", (255, 255, 255), (120, 120, 120)),
    ("TN", (0, 0, 0), (120, 120, 120)),
    ("FP", (219, 51, 51), None),
    ("FN", (51, 181, 79), None),
    ("GT boundary", (255, 219, 0), None),
    ("Pred boundary", (0, 219, 219), None),
]

FIGURES = [
    {
        "src": REPO / "figures" / "qualitative_dsifn_final.png",
        "outs": [
            REPO / "figures" / "qualitative_dsifn_final.png",
            REPO / "analysis" / "Metropolis_Beamer_Theme" / "figures" / "results" / "qualitative_dsifn_final.png",
        ],
        "pdf": REPO / "figures" / "qualitative_dsifn_final.pdf",
    },
    {
        "src": REPO / "figures" / "qualitative_whu_refined.png",
        "outs": [
            REPO / "figures" / "qualitative_whu_refined.png",
            REPO / "analysis" / "Metropolis_Beamer_Theme" / "figures" / "results" / "qualitative_whucd.png",
        ],
        "pdf": REPO / "figures" / "qualitative_whu_refined.pdf",
    },
]


def _content_row_runs(img: Image.Image, threshold: int = 245, min_fraction: float = 0.01) -> list[tuple[int, int]]:
    """Return vertical runs with enough non-white pixels."""
    rgb = img.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    active: list[bool] = []
    min_count = int(width * min_fraction)
    for y in range(height):
        count = 0
        for x in range(width):
            r, g, b = pixels[x, y]
            if min(r, g, b) < threshold:
                count += 1
        active.append(count > min_count)

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for y, is_active in enumerate(active):
        if is_active and start is None:
            start = y
        elif not is_active and start is not None:
            runs.append((start, y - 1))
            start = None
    if start is not None:
        runs.append((start, height - 1))
    return runs


def _crop_without_old_legend(img: Image.Image) -> Image.Image:
    """Keep the header and result panels, dropping the old legend strip."""
    width, height = img.size
    runs = _content_row_runs(img)
    panel_runs = [run for run in runs if (run[1] - run[0] + 1) > height * 0.08]
    if not panel_runs:
        return img.copy()
    grid_bottom = panel_runs[-1][1]
    crop_bottom = min(height, grid_bottom + max(36, int(height * 0.012)))
    return img.crop((0, 0, width, crop_bottom))


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _draw_large_legend(width: int) -> Image.Image:
    legend_h = max(150, int(width * 0.052))
    font_size = max(30, int(width * 0.0125))
    patch = max(34, int(font_size * 1.15))
    gap_after_patch = max(12, int(font_size * 0.45))
    item_gap = max(26, int(font_size * 0.95))
    pad_x = max(18, int(font_size * 0.70))
    pad_y = max(12, int(font_size * 0.45))

    font = ImageFont.truetype(str(FONT_REGULAR), font_size)
    legend = Image.new("RGB", (width, legend_h), "white")
    draw = ImageDraw.Draw(legend)

    item_widths: list[int] = []
    max_text_h = 0
    for label, _, _ in LEGEND_ITEMS:
        text_w, text_h = _text_size(draw, label, font)
        max_text_h = max(max_text_h, text_h)
        item_widths.append(patch + gap_after_patch + text_w)

    box_w = sum(item_widths) + item_gap * (len(item_widths) - 1) + 2 * pad_x
    box_h = max(patch, max_text_h) + 2 * pad_y
    x0 = max(8, (width - box_w) // 2)
    y0 = max(8, (legend_h - box_h) // 2)
    x1 = min(width - 8, x0 + box_w)
    y1 = min(legend_h - 8, y0 + box_h)

    draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=(255, 255, 255), outline=(198, 198, 198), width=5)

    x = x0 + pad_x
    center_y = (y0 + y1) // 2
    for (label, fill, edge), item_w in zip(LEGEND_ITEMS, item_widths):
        py0 = center_y - patch // 2
        px1 = x + patch
        py1 = py0 + patch
        draw.rectangle((x, py0, px1, py1), fill=fill, outline=edge or fill, width=max(2, font_size // 16))
        text_w, text_h = _text_size(draw, label, font)
        draw.text((px1 + gap_after_patch, center_y - text_h // 2 - 2), label, fill=(20, 20, 20), font=font)
        x += item_w + item_gap

    return legend


def regenerate_one(src: Path, outs: list[Path], pdf: Path | None) -> None:
    img = Image.open(src).convert("RGB")
    body = _crop_without_old_legend(img)
    legend = _draw_large_legend(body.size[0])
    out = Image.new("RGB", (body.size[0], body.size[1] + legend.size[1]), "white")
    out.paste(body, (0, 0))
    out.paste(legend, (0, body.size[1]))

    for path in outs:
        path.parent.mkdir(parents=True, exist_ok=True)
        out.save(path, quality=95)
        print(f"saved {path}")

    if pdf is not None:
        pdf.parent.mkdir(parents=True, exist_ok=True)
        out.save(pdf, "PDF", resolution=300.0)
        print(f"saved {pdf}")


def main() -> None:
    for figure in FIGURES:
        regenerate_one(figure["src"], figure["outs"], figure["pdf"])


if __name__ == "__main__":
    main()
