"""Generate token importance heatmap — modern, detailed.

Two-panel layout:
  (a) Token-level heatmap on example sentences with POS tags
  (b) POS-category bar chart with distribution
Produces: paper/figures/importance_heatmap.pdf + .png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Sentences with importance scores (from POS table) ───────────────
sentences = [
    {
        "label": "Physics",
        "tokens": ["Einstein", "discovered", "the", "theory", "of",
                    "quantum", "mechanics", "in", "Berlin", "."],
        "scores": [0.82, 0.61, 0.19, 0.65, 0.18, 0.76, 0.71, 0.28, 0.79, 0.14],
        "pos":    ["NNP", "VBD", "DT", "NN", "IN", "JJ", "NN", "IN", "NNP", "."],
    },
    {
        "label": "Biomedical",
        "tokens": ["Metformin", "significantly", "reduces", "blood",
                    "glucose", "levels", "in", "diabetic", "patients", "."],
        "scores": [0.85, 0.47, 0.56, 0.59, 0.74, 0.55, 0.25, 0.72, 0.66, 0.13],
        "pos":    ["NNP", "RB", "VBZ", "NN", "NN", "NNS", "IN", "JJ", "NNS", "."],
    },
    {
        "label": "ML / NLP",
        "tokens": ["The", "transformer", "architecture", "revolutionized",
                    "natural", "language", "processing", "."],
        "scores": [0.20, 0.77, 0.68, 0.63, 0.56, 0.70, 0.65, 0.12],
        "pos":    ["DT", "NN", "NN", "VBD", "JJ", "NN", "NN", "."],
    },
    {
        "label": "Literature",
        "tokens": ["The", "old", "man", "sailed", "across", "the",
                    "turbulent", "Caribbean", "sea", "."],
        "scores": [0.17, 0.49, 0.58, 0.60, 0.31, 0.16, 0.53, 0.81, 0.62, 0.14],
        "pos":    ["DT", "JJ", "NN", "VBD", "IN", "DT", "JJ", "NNP", "NN", "."],
    },
]

# POS importance data (from Table 5)
pos_data = [
    ("NNP\n(proper)", 0.78, "#EF4444"),
    ("NN\n(noun)",    0.62, "#F59E0B"),
    ("VB*\n(verb)",   0.58, "#F97316"),
    ("JJ\n(adj)",     0.54, "#FBBF24"),
    ("RB\n(adv)",     0.48, "#94A3B8"),
    ("IN\n(prep)",    0.32, "#60A5FA"),
    ("DT\n(det)",     0.21, "#3B82F6"),
    ("PUNCT",         0.15, "#93C5FD"),
]

# ── Colours ─────────────────────────────────────────────────────────
C_BG   = "#FAFBFC"; C_TEXT = "#0F172A"; C_SUB = "#64748B"
C_FAINT= "#E2E8F0"; C_WHITE = "#FFFFFF"

# Modern importance colour map
cmap = mcolors.LinearSegmentedColormap.from_list(
    "importance_modern",
    ["#EFF6FF",   # very light blue (low)
     "#BFDBFE",   # light blue
     "#FDE68A",   # amber
     "#F97316",   # orange
     "#DC2626",   # red
     "#7F1D1D"],  # dark red (high)
    N=256)

# ── Figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 6.5), dpi=300)
fig.patch.set_facecolor(C_BG)

# Layout: left = heatmap sentences, right = POS bar chart
gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0], wspace=0.08,
                      left=0.03, right=0.97, top=0.90, bottom=0.08)
gs_left = gs[0].subgridspec(len(sentences) + 1, 1,
                             height_ratios=[1] * len(sentences) + [0.35],
                             hspace=0.5)
ax_right = fig.add_subplot(gs[1])

# ═══════════════════════════════════════════════════════════════════
# (a) TOKEN HEATMAP
# ═══════════════════════════════════════════════════════════════════
fig.text(0.30, 0.94, "(a)  Learned Token Importance Scores",
         ha="center", va="center", fontsize=12, fontweight="bold",
         color=C_TEXT)

max_tokens = max(len(s["tokens"]) for s in sentences)

for idx, sent in enumerate(sentences):
    ax = fig.add_subplot(gs_left[idx])
    n = len(sent["tokens"])
    ax.set_xlim(-1.5, max_tokens + 0.5)
    ax.set_ylim(-0.55, 0.7)
    ax.axis("off")

    # Domain label
    ax.text(-1.2, 0.1, sent["label"], ha="right", va="center",
            fontsize=7.5, fontweight="bold", color=C_SUB,
            bbox=dict(boxstyle="round,pad=0.12", fc=C_FAINT,
                      ec="none", alpha=0.6))

    for j, (token, score, pos) in enumerate(
            zip(sent["tokens"], sent["scores"], sent["pos"])):
        color = cmap(score)
        text_col = C_WHITE if score > 0.55 else C_TEXT

        # Rounded token box
        bw, bh = 0.88, 0.75
        box = FancyBboxPatch(
            (j - bw / 2, -bh / 2 + 0.05), bw, bh,
            boxstyle="round,pad=0.08", fc=color, ec=C_FAINT,
            lw=0.4, zorder=3)
        ax.add_patch(box)

        # Token text
        ax.text(j, 0.12, token, ha="center", va="center", fontsize=7.5,
                fontweight="bold" if score > 0.55 else "normal",
                color=text_col, zorder=4, family="monospace")

        # Score value
        ax.text(j, -0.18, f".{int(score*100):02d}", ha="center",
                va="center", fontsize=5, color=text_col, zorder=4,
                alpha=0.8)

        # POS tag above
        pos_col = "#EF4444" if pos.startswith("NN") else (
                  "#F59E0B" if pos.startswith("VB") else (
                  "#3B82F6" if pos in ("DT", "IN") else C_SUB))
        ax.text(j, 0.48, pos, ha="center", va="center", fontsize=4.5,
                color=pos_col, zorder=4, family="monospace",
                fontweight="bold")

# Colour bar
ax_cb = fig.add_subplot(gs_left[-1])
ax_cb.axis("off")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
sm.set_array([])
cbar_ax = fig.add_axes([0.08, 0.04, 0.38, 0.02])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
cbar.set_label("Importance Score  $i^l$", fontsize=8, color=C_TEXT, labelpad=3)
cbar.ax.tick_params(labelsize=6)
# Tick markers for categories
cbar.set_ticks([0, 0.3, 0.7, 1.0])
cbar.set_ticklabels(["0 (function)", "0.3", "0.7", "1.0 (content)"])

# ═══════════════════════════════════════════════════════════════════
# (b) POS IMPORTANCE BAR CHART
# ═══════════════════════════════════════════════════════════════════
ax_right.set_title("(b)  Mean Importance by POS",
                   fontsize=10, fontweight="bold", pad=10, color=C_TEXT)

pos_names  = [d[0] for d in pos_data]
pos_vals   = [d[1] for d in pos_data]
pos_colors = [d[2] for d in pos_data]

y_pos = np.arange(len(pos_data))
bars = ax_right.barh(y_pos, pos_vals, height=0.65, color=pos_colors,
                     edgecolor=C_WHITE, lw=0.5, zorder=3)

# Value labels
for bar, val in zip(bars, pos_vals):
    ax_right.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                  f"{val:.2f}", ha="left", va="center", fontsize=6.5,
                  fontweight="bold", color=C_TEXT, zorder=5)

# Category brackets
ax_right.axhspan(-0.5, 3.5, alpha=0.04, color="#EF4444", zorder=0)
ax_right.axhspan(3.5, 4.5, alpha=0.04, color="#94A3B8", zorder=0)
ax_right.axhspan(4.5, 7.5, alpha=0.04, color="#3B82F6", zorder=0)

ax_right.text(0.85, 1.5, "Content", ha="center", va="center",
              fontsize=7, color="#EF4444", fontweight="bold",
              rotation=0, alpha=0.6)
ax_right.text(0.85, 6.0, "Function", ha="center", va="center",
              fontsize=7, color="#3B82F6", fontweight="bold",
              rotation=0, alpha=0.6)

ax_right.set_yticks(y_pos)
ax_right.set_yticklabels(pos_names, fontsize=7)
ax_right.set_xlabel(r"Mean $i^l$", fontsize=9, color=C_TEXT)
ax_right.set_xlim(0, 1.0)
ax_right.invert_yaxis()
ax_right.grid(axis="x", alpha=0.25, color=C_FAINT, zorder=0)
ax_right.set_axisbelow(True)
ax_right.tick_params(labelsize=7)
ax_right.spines["top"].set_visible(False)
ax_right.spines["right"].set_visible(False)

# ── Save ───────────────────────────────────────────────────────────
out_dir = Path(__file__).parent
for fmt in ("pdf", "png"):
    p = out_dir / f"importance_heatmap.{fmt}"
    fig.savefig(p, format=fmt, dpi=300, bbox_inches="tight",
                facecolor=C_BG, pad_inches=0.15)
    print(f"Saved: {p}")
plt.close(fig)
print("Done — importance heatmap generated.")
