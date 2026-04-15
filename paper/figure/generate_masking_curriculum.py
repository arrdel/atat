"""Generate balanced masking curriculum figure — modern, detailed.

Produces: paper/figures/masking_curriculum.pdf + .png
Three-panel layout:
  (a) Masking schedule surface (heatmap of g_bal over i × t)
  (b) Line plots at key timesteps
  (c) Curriculum phase diagram with token examples
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
import numpy as np

# ── Modern palette ──────────────────────────────────────────────────
C_PROP  = "#EF4444"; C_INV   = "#3B82F6"; C_BAL   = "#8B5CF6"
C_EASY  = "#10B981"; C_MED   = "#F59E0B"; C_HARD  = "#EF4444"
C_BG    = "#FAFBFC"; C_TEXT  = "#0F172A"; C_SUB   = "#64748B"
C_FAINT = "#E2E8F0"; C_WHITE = "#FFFFFF"

ETA = 0.3

def g_prop(i): return ETA + (1 - 2 * ETA) * i
def g_inv(i):  return ETA + (1 - 2 * ETA) * (1 - i)
def g_bal(i, t): return t * g_inv(i) + (1 - t) * g_prop(i)

# ── Figure  (3 panels) ─────────────────────────────────────────────
fig = plt.figure(figsize=(15, 4.8), dpi=300)
fig.patch.set_facecolor(C_BG)

gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0], wspace=0.32,
                      left=0.05, right=0.97, top=0.88, bottom=0.14)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

shadow = [pe.withSimplePatchShadow(offset=(0.5, -0.5),
                                    shadow_rgbFace="#00000010")]
i_vals = np.linspace(0, 1, 300)
t_vals = np.linspace(0, 1, 300)
I, T = np.meshgrid(i_vals, t_vals)
G = g_bal(I, T)

# ═══════════════════════════════════════════════════════════════════
# (a) HEATMAP — g_bal surface over (i, t)
# ═══════════════════════════════════════════════════════════════════
cmap_heat = LinearSegmentedColormap.from_list(
    "mask_heat",
    ["#DBEAFE", "#60A5FA", "#8B5CF6", "#F59E0B", "#EF4444"], N=256)
im = ax1.pcolormesh(I, T, G, cmap=cmap_heat, shading="gouraud",
                     vmin=0.25, vmax=0.75, rasterized=True)
ax1.set_xlabel("Token Importance  $i^l$", fontsize=9, color=C_TEXT)
ax1.set_ylabel("Diffusion Timestep  $t$", fontsize=9, color=C_TEXT)
ax1.set_title("(a)  Masking Probability Surface", fontsize=10,
              fontweight="bold", pad=10, color=C_TEXT)
# Contour lines
cs = ax1.contour(I, T, G, levels=[0.3, 0.4, 0.5, 0.6, 0.7],
                 colors="white", linewidths=0.6, alpha=0.7)
ax1.clabel(cs, inline=True, fontsize=5.5, fmt="%.1f", colors="white")

# Annotations
ax1.text(0.15, 0.92, "preserve\nanchors", ha="center", va="center",
         fontsize=6, color=C_WHITE, fontweight="bold", zorder=5,
         bbox=dict(boxstyle="round,pad=0.1", fc=C_INV, ec="none", alpha=0.8))
ax1.text(0.85, 0.08, "focus on\nhard tokens", ha="center", va="center",
         fontsize=6, color=C_WHITE, fontweight="bold", zorder=5,
         bbox=dict(boxstyle="round,pad=0.1", fc=C_PROP, ec="none", alpha=0.8))

cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cb.set_label(r"$g_{\mathrm{bal}}(i, t)$", fontsize=8, color=C_SUB)
cb.ax.tick_params(labelsize=6)

ax1.tick_params(labelsize=7)
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

# ═══════════════════════════════════════════════════════════════════
# (b) LINE PLOTS — curves at selected timesteps
# ═══════════════════════════════════════════════════════════════════
ax2.set_title("(b)  Schedule at Key Timesteps", fontsize=10,
              fontweight="bold", pad=10, color=C_TEXT)

# Base strategies
ax2.fill_between(i_vals, g_prop(i_vals), g_inv(i_vals),
                 alpha=0.06, color=C_BAL)
ax2.plot(i_vals, g_prop(i_vals), "--", color=C_PROP, lw=1.3, alpha=0.5,
         label=r"$g_{\mathrm{prop}}$ (proportional)")
ax2.plot(i_vals, g_inv(i_vals), "--", color=C_INV, lw=1.3, alpha=0.5,
         label=r"$g_{\mathrm{inv}}$ (inverse)")

# Balanced curves
cmap_t = plt.cm.cool
timesteps = [0.0, 0.25, 0.5, 0.75, 1.0]
for t in timesteps:
    c = cmap_t(t)
    lw = 2.5 if t in (0.0, 1.0) else 1.8
    ax2.plot(i_vals, g_bal(i_vals, t), "-", color=c, lw=lw,
             alpha=0.9, label=f"$t = {t:.2f}$")

# Eta floor
ax2.axhline(y=ETA, color=C_FAINT, ls=":", lw=1.0, zorder=0)
ax2.text(1.02, ETA, r"$\eta$", fontsize=7, color=C_SUB, va="center",
         ha="left", transform=ax2.get_yaxis_transform())

# Formula callout
ax2.text(0.5, 0.03,
         r"$g_{\mathrm{bal}}(i, t) = t \cdot g_{\mathrm{inv}}(i) "
         r"+ (1{-}t) \cdot g_{\mathrm{prop}}(i)$",
         ha="center", va="bottom", fontsize=7.5, color=C_BAL,
         transform=ax2.transAxes,
         bbox=dict(boxstyle="round,pad=0.12", fc=C_WHITE, ec=C_BAL,
                   lw=0.8, alpha=0.95))

ax2.set_xlabel("Token Importance  $i^l$", fontsize=9, color=C_TEXT)
ax2.set_ylabel("Masking Probability", fontsize=9, color=C_TEXT)
ax2.set_xlim(0, 1); ax2.set_ylim(0.18, 0.82)
ax2.legend(fontsize=5.8, loc="upper right", framealpha=0.95,
           edgecolor=C_FAINT, ncol=1, handlelength=1.5)
ax2.grid(True, alpha=0.25, color=C_FAINT)
ax2.tick_params(labelsize=7)

# ═══════════════════════════════════════════════════════════════════
# (c) CURRICULUM PHASES — stylised timeline
# ═══════════════════════════════════════════════════════════════════
ax3.set_title("(c)  Training Curriculum Phases", fontsize=10,
              fontweight="bold", pad=10, color=C_TEXT)

progress = np.linspace(0, 100, 400)
importance = np.linspace(0, 1, 300)
P, I2 = np.meshgrid(progress, importance)

# Build phase map (cumulative)
phase_map = np.full_like(P, 0.0)
phase_map[(P <= 20) & (I2 <= 0.3)] = 1
phase_map[(P > 20) & (P <= 60) & (I2 <= 0.7)] = 2
phase_map[(P > 60)] = 3

from matplotlib.colors import ListedColormap
cmap_ph = ListedColormap([C_FAINT, "#A7F3D0", "#FDE68A", "#FECACA"])
ax3.pcolormesh(P, I2, phase_map, cmap=cmap_ph, vmin=0, vmax=3,
               shading="auto", alpha=0.65, rasterized=True)

# Phase boundaries
for xv in [20, 60]:
    ax3.axvline(x=xv, color=C_TEXT, ls="--", lw=0.8, alpha=0.4)
for yv in [0.3, 0.7]:
    ax3.axhline(y=yv, color=C_TEXT, ls=":", lw=0.6, alpha=0.35)

# Phase labels
phase_info = [
    (10,  0.15, "Easy\n20 %", C_EASY,  r"$i \in [0,\, 0.3]$",
     "det · punct · prep"),
    (40,  0.50, "Medium\n40 %", C_MED,  r"$i \in [0.3,\, 0.7]$",
     "adj · adv · common NN"),
    (80,  0.50, "Hard\n40 %", C_HARD,   r"$i \in [0,\, 1.0]$",
     "NNP · key VB · jargon"),
]
for px, py, plabel, pcol, prange, ptokens in phase_info:
    ax3.text(px, py, plabel, ha="center", va="center", fontsize=8,
             fontweight="bold", color=C_WHITE, zorder=6,
             bbox=dict(boxstyle="round,pad=0.15", fc=pcol, ec="none",
                       alpha=0.92))
    ax3.text(px, py - 0.18, prange, ha="center", va="top", fontsize=5.5,
             color=pcol, zorder=6)
    ax3.text(px, 0.95, ptokens, ha="center", va="top", fontsize=5,
             color=C_SUB, style="italic", zorder=6)

# Difficulty arrow along bottom
ax3.annotate("", xy=(95, -0.12), xytext=(5, -0.12),
             arrowprops=dict(arrowstyle="-|>", color=C_TEXT, lw=1.2),
             annotation_clip=False)
ax3.text(50, -0.17, "Increasing difficulty  →", ha="center", va="top",
         fontsize=6.5, color=C_SUB, fontweight="bold",
         clip_on=False)

ax3.set_xlabel("Training Progress (%)", fontsize=9, color=C_TEXT)
ax3.set_ylabel("Token Importance  $i^l$", fontsize=9, color=C_TEXT)
ax3.set_xlim(0, 100); ax3.set_ylim(0, 1)
ax3.tick_params(labelsize=7)

# ── Save ───────────────────────────────────────────────────────────
out_dir = Path(__file__).parent
for fmt in ("pdf", "png"):
    p = out_dir / f"masking_curriculum.{fmt}"
    fig.savefig(p, format=fmt, dpi=300, bbox_inches="tight",
                facecolor=C_BG, pad_inches=0.15)
    print(f"Saved: {p}")
plt.close(fig)
print("Done — masking curriculum figure generated.")
