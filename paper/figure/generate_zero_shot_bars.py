"""Generate zero-shot domain transfer figure — modern, detailed.

Two-panel layout:
  (a) Grouped bar chart with gradient bars and delta annotations
  (b) Radar / spider chart for overall comparison
Produces: paper/figures/zero_shot_bars.pdf + .png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Data (Table 2) ─────────────────────────────────────────────────
names   = ["LAMBADA", "PTB", "WikiText", "LM1B", "AG News", "PubMed", "ArXiv"]
short   = ["LAM.", "PTB", "Wiki.", "LM1B", "AG", "PubMed", "ArXiv"]
ar      = np.array([51.28, 82.05, 25.75, 21.55, 52.09, 49.01, 41.73])
mdlm    = np.array([47.52, 95.26, 32.83, 27.04, 61.15, 41.89, 37.37])
adlm    = np.array([44.32, 95.37, 31.94, 24.46, 55.72, 37.56, 33.69])
atat    = np.array([43.52, 94.63, 30.47, 24.21, 53.49, 35.97, 32.08])

# ── Colours ─────────────────────────────────────────────────────────
C_AR   = "#94A3B8";  C_MDLM = "#60A5FA";  C_ADLM = "#FBBF24"
C_ATAT = "#8B5CF6";  C_ATAT_D = "#5B21B6"
C_BG   = "#FAFBFC";  C_TEXT = "#0F172A"; C_SUB = "#64748B"
C_FAINT= "#E2E8F0";  C_WHITE = "#FFFFFF"
C_WIN  = "#10B981"

shadow = [pe.withSimplePatchShadow(offset=(0.4, -0.4),
                                    shadow_rgbFace="#00000008")]

# ── Figure ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5.2), dpi=300)
fig.patch.set_facecolor(C_BG)
gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.25,
                      left=0.05, right=0.97, top=0.87, bottom=0.13)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], polar=True)

# ═══════════════════════════════════════════════════════════════════
# (a) GROUPED BAR CHART
# ═══════════════════════════════════════════════════════════════════
ax1.set_title("(a)  Zero-Shot Perplexity Across 7 Benchmarks",
              fontsize=11, fontweight="bold", pad=12, color=C_TEXT)

x = np.arange(len(names))
w = 0.18
off = [-1.5, -0.5, 0.5, 1.5]

for vals, color, label, offset in [
    (ar,   C_AR,   "AR (baseline)", off[0]),
    (mdlm, C_MDLM, "MDLM",         off[1]),
    (adlm, C_ADLM, "ADLM",         off[2]),
    (atat, C_ATAT, "ATAT (ours)",   off[3]),
]:
    bars = ax1.bar(x + offset * w, vals, w, label=label, color=color,
                   edgecolor=C_WHITE, lw=0.6, zorder=3)
    if label == "ATAT (ours)":
        # Value labels + delta vs AR
        for i, (bar, v) in enumerate(zip(bars, vals)):
            bx = bar.get_x() + bar.get_width() / 2
            ax1.text(bx, v + 0.6, f"{v:.1f}", ha="center", va="bottom",
                     fontsize=5.5, fontweight="bold", color=C_ATAT_D,
                     zorder=5)
            delta = v - ar[i]
            ax1.text(bx, v + 2.5, f"{delta:+.1f}",
                     ha="center", va="bottom", fontsize=4.5,
                     color=C_WIN if delta < 0 else C_SUB, fontweight="bold",
                     zorder=5)

# Best marker (star) on each benchmark for ATAT
for i in range(len(names)):
    bx = x[i] + off[3] * w
    ax1.plot(bx, atat[i] - 1.5, "*", color=C_ATAT_D, markersize=5,
             zorder=5)

# Subtle alternating background bands
for i in range(len(names)):
    if i % 2 == 0:
        ax1.axvspan(x[i] - 2.3 * w, x[i] + 2.3 * w,
                    alpha=0.04, color=C_ATAT, zorder=0)

ax1.set_xticks(x)
ax1.set_xticklabels(short, fontsize=8.5, fontweight="bold")
ax1.set_ylabel("Perplexity (↓ lower is better)", fontsize=9, color=C_TEXT)
ax1.set_ylim(0, max(mdlm.max(), adlm.max()) * 1.12)
ax1.grid(axis="y", alpha=0.25, color=C_FAINT, zorder=0)
ax1.set_axisbelow(True)
ax1.tick_params(labelsize=7)

# Legend
leg = ax1.legend(fontsize=7, loc="upper left", framealpha=0.95,
                 edgecolor=C_FAINT, ncol=2, handlelength=1.2)
leg.get_frame().set_linewidth(0.8)

# Summary card
avg_ar   = ar.mean()
avg_atat = atat.mean()
card_txt = (f"Average PPL — AR: {avg_ar:.2f}   ATAT: {avg_atat:.2f}   "
            f"(Δ = {avg_atat - avg_ar:+.2f})")
ax1.text(0.98, 0.97, card_txt, transform=ax1.transAxes,
         ha="right", va="top", fontsize=7, color=C_TEXT,
         bbox=dict(boxstyle="round,pad=0.15", fc=C_WHITE,
                   ec=C_FAINT, lw=0.8))

# Domain-type brackets at bottom
general = [0, 1, 2, 3]  # LAM, PTB, Wiki, LM1B
special = [4, 5, 6]     # AG, PubMed, ArXiv
for group, label, col in [(general, "General", C_SUB), (special, "Specialised", C_ATAT)]:
    xmin = x[group[0]] - 2 * w
    xmax = x[group[-1]] + 2 * w
    yb = -6
    ax1.annotate("", xy=(xmin, yb), xytext=(xmax, yb),
                 arrowprops=dict(arrowstyle="|-|", color=col, lw=1.0),
                 annotation_clip=False)
    ax1.text((xmin + xmax) / 2, yb - 2, label, ha="center", va="top",
             fontsize=6, color=col, fontweight="bold",
             clip_on=False)

# ═══════════════════════════════════════════════════════════════════
# (b) RADAR CHART
# ═══════════════════════════════════════════════════════════════════
ax2.set_title("(b)  Comparative Profile", fontsize=10, fontweight="bold",
              pad=20, color=C_TEXT)

# Normalise: invert PPL so larger = better, scale 0-1
all_vals = np.concatenate([ar, mdlm, adlm, atat])
ppl_min, ppl_max = all_vals.min() * 0.9, all_vals.max() * 1.05

def normalise(v): return 1 - (v - ppl_min) / (ppl_max - ppl_min)

angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
angles += angles[:1]

for vals, color, label, alpha, lw in [
    (ar,   C_AR,   "AR",   0.08, 1.0),
    (mdlm, C_MDLM, "MDLM", 0.10, 1.2),
    (adlm, C_ADLM, "ADLM", 0.12, 1.3),
    (atat, C_ATAT, "ATAT", 0.20, 2.2),
]:
    v = normalise(vals).tolist() + [normalise(vals)[0]]
    ax2.plot(angles, v, "-", color=color, lw=lw, label=label)
    ax2.fill(angles, v, color=color, alpha=alpha)

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(short, fontsize=6.5, fontweight="bold", color=C_TEXT)
ax2.tick_params(axis="y", labelsize=5, colors=C_SUB)
ax2.set_ylim(0, 1)
ax2.grid(color=C_FAINT, lw=0.5)
ax2.legend(fontsize=6, loc="lower right", bbox_to_anchor=(1.25, -0.08),
           framealpha=0.95, edgecolor=C_FAINT)

# ── Save ───────────────────────────────────────────────────────────
out_dir = Path(__file__).parent
for fmt in ("pdf", "png"):
    p = out_dir / f"zero_shot_bars.{fmt}"
    fig.savefig(p, format=fmt, dpi=300, bbox_inches="tight",
                facecolor=C_BG, pad_inches=0.15)
    print(f"Saved: {p}")
plt.close(fig)
print("Done — zero-shot figure generated.")
