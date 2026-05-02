"""Generate architecture_overview.pdf and drbi_module.pdf for the MambaRefine-CD paper.

Creates clean vector figures using matplotlib patches/arrows.
Output:
    MambaRefine_CD/figures/architecture_overview.pdf
    MambaRefine_CD/figures/drbi_module.pdf
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "MambaRefine_CD" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
C_INPUT   = "#EFF3FF"   # light blue  – input images
C_ENC     = "#BDD7EE"   # medium blue – encoder
C_DRBI    = "#F4CCCC"   # soft red    – D-RBI
C_REGION  = "#D9EAD3"   # soft green  – region stream
C_BOUND   = "#FFE599"   # soft yellow – boundary stream
C_DEC     = "#D0E0E3"   # soft teal   – decoder / ARF-FPN
C_OUT     = "#434343"   # dark grey   – output
C_EDGE    = "#666666"

FS       = 7.5          # font size
FS_SM    = 6.5
ARROW_KW = dict(arrowstyle="->", color=C_EDGE, lw=0.9,
                connectionstyle="arc3,rad=0.0")

def box(ax, x, y, w, h, label, sublabel=None, fc=C_ENC, ec=C_EDGE, fs=FS, r=0.04):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=f"round,pad={r}",
                           facecolor=fc, edgecolor=ec, linewidth=0.8, zorder=2)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + 0.01, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=3)
        ax.text(x, y - 0.10, sublabel, ha="center", va="center",
                fontsize=FS_SM, color="#444444", zorder=3)
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fs, fontweight="bold", zorder=3)

def arrow(ax, x0, y0, x1, y1, label=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=ARROW_KW, zorder=4)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx+0.02, my, label, fontsize=FS_SM-0.5, color="#555555",
                ha="left", va="center", zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Architecture Overview
# ─────────────────────────────────────────────────────────────────────────────
def make_architecture_overview():
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # ── Row 1: inputs ────────────────────────────────────────────────────────
    box(ax, 0.18, 0.88, 0.22, 0.13, r"$I_1$", "Time 1", fc=C_INPUT)
    box(ax, 0.50, 0.88, 0.22, 0.13, r"$I_2$", "Time 2", fc=C_INPUT)

    # ── Row 2: shared encoder ─────────────────────────────────────────────
    box(ax, 0.34, 0.70, 0.55, 0.12, "Shared MambaVision-S Encoder",
        r"$[96, 192, 384, 768]$  ×  4 stages", fc=C_ENC)
    arrow(ax, 0.18, 0.815, 0.18, 0.74)
    arrow(ax, 0.50, 0.815, 0.50, 0.74)
    # note shared weights
    ax.text(0.34, 0.725, "shared weights", ha="center", va="top",
            fontsize=FS_SM-1, color="#888888", style="italic", zorder=5)

    # ── Row 3: D-RBI (4 blocks) ──────────────────────────────────────────
    box(ax, 0.34, 0.535, 0.55, 0.11, "D-RBI  ×4",
        r"Region gate  $\oplus$  Boundary gate", fc=C_DRBI)
    arrow(ax, 0.34, 0.644, 0.34, 0.59, r"$\{F_1^s, F_2^s\}$")

    # ── Row 4: two streams ───────────────────────────────────────────────
    # Region stream (left)
    box(ax, 0.20, 0.38, 0.26, 0.10, "CRAMLite",
        r"$F_{\rm out} = F(1 + \alpha A)$", fc=C_REGION)
    box(ax, 0.20, 0.245, 0.26, 0.10, "ARF-FPN Decoder",
        r"dilated $\{1,2,4,8\}$", fc=C_DEC)
    arrow(ax, 0.20, 0.48, 0.20, 0.43, r"$R^s$")
    arrow(ax, 0.20, 0.33, 0.20, 0.30)

    # Boundary stream (right)
    box(ax, 0.73, 0.38, 0.30, 0.10, "Boundary Residual Head",
        r"$P_f = P_c + 0.1\tanh(\Delta)$", fc=C_BOUND)
    arrow(ax, 0.56, 0.48, 0.73, 0.43, r"$B^0$")

    # Coarse head → boundary head
    arrow(ax, 0.20, 0.195, 0.20, 0.14)  # to coarse logit
    box(ax, 0.20, 0.105, 0.26, 0.09, "Coarse logit $P_c$", fc=C_DEC)
    arrow(ax, 0.33, 0.105, 0.57, 0.38)  # Pc to boundary head

    # Boundary head → final output
    arrow(ax, 0.73, 0.33, 0.73, 0.14)
    box(ax, 0.73, 0.08, 0.30, 0.09, "Binary Change Map  $P_f$", fc=C_OUT,
        fs=FS, ec="#222222")
    ax.texts[-1].set_color("white")

    # labels on streams
    ax.annotate("", xy=(0.20, 0.48), xytext=(0.20, 0.59),
                arrowprops=ARROW_KW, zorder=4)

    # title
    ax.set_title("MambaRefine-CD: Architecture Overview",
                 fontsize=9, fontweight="bold", pad=4)

    out = FIG_DIR / "architecture_overview.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – D-RBI Module
# ─────────────────────────────────────────────────────────────────────────────
def make_drbi_module():
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    def bx(x, y, w, h, lbl, sub=None, fc=C_ENC):
        box(ax, x, y, w, h, lbl, sub, fc=fc)

    # Inputs
    bx(0.20, 0.92, 0.22, 0.10, r"$F_1^s$", fc=C_INPUT)
    bx(0.62, 0.92, 0.22, 0.10, r"$F_2^s$", fc=C_INPUT)

    # GroupNorm
    bx(0.41, 0.78, 0.60, 0.09, "GroupNorm", fc=C_ENC)
    arrow(ax, 0.20, 0.87, 0.25, 0.825)
    arrow(ax, 0.62, 0.87, 0.57, 0.825)

    # Four-stream concat
    bx(0.41, 0.645, 0.60, 0.09,
       r"Concat $[F_{1n}, F_{2n}, |F_{2n}-F_{1n}|, F_{2n}-F_{1n}]$", fc=C_ENC)
    arrow(ax, 0.41, 0.735, 0.41, 0.69)

    # 1×1 proj + DW3×3 + PW1×1
    bx(0.41, 0.515, 0.55, 0.085, "1×1 proj → DW 3×3 → PW 1×1", fc=C_ENC)
    arrow(ax, 0.41, 0.60, 0.41, 0.56, r"$T^s$")

    # Temporal descriptor
    bx(0.41, 0.405, 0.30, 0.075, r"$D^s$", fc="#F3F3F3")
    arrow(ax, 0.41, 0.472, 0.41, 0.443)

    # Split to two branches
    # Region gate (left)
    bx(0.18, 0.275, 0.28, 0.085, "Region Gate", r"DW3×3 → PW1×1 → sigmoid", fc=C_REGION)
    bx(0.18, 0.155, 0.28, 0.085, r"$G_r^s \in [0.2,\, 0.8]$", fc=C_REGION)
    bx(0.18, 0.050, 0.25, 0.075, r"$R^s = G_r^s \odot D^s$", fc=C_REGION)
    arrow(ax, 0.26, 0.405, 0.18, 0.317)
    arrow(ax, 0.18, 0.232, 0.18, 0.198)
    arrow(ax, 0.18, 0.112, 0.18, 0.088)

    # Boundary gate (right)
    bx(0.70, 0.275, 0.32, 0.085, "Boundary Gate",
       r"Sobel $S^s$ × DW3×3 → sigmoid", fc=C_BOUND)
    bx(0.70, 0.155, 0.28, 0.085, r"$G_b^s \in [0.0,\, 0.4]$", fc=C_BOUND)
    bx(0.70, 0.050, 0.25, 0.075, r"$B^s = G_b^s \odot D^s$", fc=C_BOUND)
    arrow(ax, 0.56, 0.405, 0.70, 0.317)
    arrow(ax, 0.70, 0.232, 0.70, 0.198)
    arrow(ax, 0.70, 0.112, 0.70, 0.088)

    # legend note
    ax.text(0.41, 0.005, r"$D^s$ also passed unchanged to decoder",
            ha="center", va="bottom", fontsize=FS_SM-0.5, color="#666666",
            style="italic")

    ax.set_title("D-RBI: Differential Region–Boundary Interaction",
                 fontsize=9, fontweight="bold", pad=4)

    out = FIG_DIR / "drbi_module.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    make_architecture_overview()
    make_drbi_module()
    print("Done.")
