#!/usr/bin/env python3
"""Validate website assets and save a report."""
from __future__ import annotations

import json
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBSITE = ROOT / "website"
OUTPUT_DIR = ROOT / "outputs" / "website_validation"


class _Parser(HTMLParser):
    def error(self, message: str) -> None:  # pragma: no cover
        raise RuntimeError(message)


def _local_refs(html: str) -> list[str]:
    return re.findall(r'(?:src|href)="([^"]+)"', html)


def _strip_math_and_scripts(html: str) -> str:
    text = re.sub(r"<script.*?</script>", "", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.S | re.I)
    text = re.sub(r"\\\(.+?\\\)", "", text, flags=re.S)
    text = re.sub(r"\\\[.+?\\\]", "", text, flags=re.S)
    return text


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    index = WEBSITE / "index.html"
    styles = WEBSITE / "styles.css"
    script = WEBSITE / "script.js"

    report = {
      "generated_at": datetime.now().isoformat(timespec="seconds"),
      "checks": {},
      "missing_assets": [],
      "warnings": [],
    }

    report["checks"]["index_exists"] = index.exists()
    report["checks"]["styles_exists"] = styles.exists()
    report["checks"]["script_exists"] = script.exists()

    html = index.read_text(encoding="utf-8") if index.exists() else ""
    if html:
        _Parser().feed(html)

    refs = _local_refs(html)
    missing_assets = []
    for ref in refs:
        if ref.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = (WEBSITE / ref).resolve()
        if not path.exists():
            missing_assets.append(ref)
    report["missing_assets"] = missing_assets

    report["checks"]["mathjax_config_present"] = "window.MathJax" in html and "tex-svg.js" in html
    report["checks"]["qualitative_manifest_exists"] = (WEBSITE / "assets" / "qualitative" / "manifest.json").exists()
    report["checks"]["ours_results_exists"] = (WEBSITE / "assets" / "data" / "ours_results.json").exists()
    report["checks"]["external_sota_exists"] = (WEBSITE / "assets" / "data" / "external_sota_results.json").exists()
    report["checks"]["reproduced_sota_exists"] = (WEBSITE / "assets" / "data" / "reproduced_sota_results.json").exists()

    required_svgs = [
        WEBSITE / "assets" / "diagrams" / "full_architecture.svg",
        WEBSITE / "assets" / "diagrams" / "decoder_code_faithful.svg",
        WEBSITE / "assets" / "diagrams" / "drbi_module.svg",
        WEBSITE / "assets" / "diagrams" / "metrics_explanation.svg",
    ]
    report["checks"]["required_diagrams_exist"] = all(path.exists() for path in required_svgs)

    cleaned = _strip_math_and_scripts(html)
    raw_latex_hits = re.findall(r"(\\text\{|\\mathbb\{|\\frac\{|\\sigma|\\odot|\\nabla|\\tanh|\\Pr\()", cleaned)
    if raw_latex_hits:
        report["warnings"].append("Potential raw LaTeX detected outside MathJax delimiters.")

    report["checks"]["balanced_display_math"] = html.count("\\[") == html.count("\\]")
    report["checks"]["balanced_inline_math"] = html.count("\\(") == html.count("\\)")

    report_json = OUTPUT_DIR / "website_validation_report.json"
    report_md = OUTPUT_DIR / "website_validation_report.md"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Website Validation Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Checks",
    ]
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Missing Assets")
    if missing_assets:
        lines.extend([f"- `{item}`" for item in missing_assets])
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Warnings")
    if report["warnings"]:
        lines.extend([f"- {warning}" for warning in report["warnings"]])
    else:
        lines.append("- None")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {report_json}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
