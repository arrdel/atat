"""Generate ATAT architecture diagram — modern, detailed, publication-quality.

Produces: paper/figures/atat_architecture.pdf + .png
Run:  python paper/figures/generate_architecture.py

Design: Multi-level architecture with internal component breakdowns,
gradient fills, soft shadows, data-flow annotations, and formula callouts.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# MODERN COLOUR PALETTE  (soft gradients, high contrast)
# ═══════════════════════════════════════════════════════════════════
C_ANCHOR     = "#3B82F6";  C_ANCHOR_L  = "#DBEAFE"; C_ANCHOR_D = "#1E40AF"
C_ESTIM      = "#10B981";  C_ESTIM_L   = "#D1FAE5"; C_ESTIM_D  = "#047857"
C_MASK       = "#F59E0B";  C_MASK_L    = "#FEF3C7"; C_MASK_D   = "#B45309"
C_DENOISE    = "#8B5CF6";  C_DENOISE_L = "#EDE9FE"; C_DENOISE_D= "#5B21B6"
C_SCORE      = "#EF4444";  C_SCORE_L   = "#FEE2E2"
C_INPUT      = "#334155";  C_INPUT_L   = "#F1F5F9"
C_BG         = "#FAFBFC"
C_TEXT       = "#0F172A";  C_SUB = "#64748B"; C_FAINT = "#CBD5E1"
C_WHITE      = "#FFFFFF"
C_LOSS_BG    = "#FFF7ED";  C_LOSS_BD = "#F97316"

# ═══════════════════════════════════════════════════════════════════
# CANVAS
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 9), dpi=300)
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_xlim(-0.5, 16.5)
ax.set_ylim(-2.8, 9.5)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(C_BG)

shadow = [pe.withSimplePatchShadow(offset=(0.06, -0.06),
                                    shadow_rgbFace="#00000015")]
text_glow = [pe.withStroke(linewidth=2, foreground=C_WHITE)]


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════
def module_box(x, y, w, h, fill, edge, title, subtitle=None,
               fontsize=11, badge=None, badge_color=None):
    body = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.18",
                          fc=fill, ec=edge, lw=1.6, zorder=5)
    body.set_path_effects(shadow)
    ax.add_patch(body)
    hdr_h = 0.38
    hdr = FancyBboxPatch((x + 0.05, y + h - hdr_h - 0.05), w - 0.10, hdr_h,
                         boxstyle="round,pad=0.08", fc=edge, ec="none",
                         alpha=0.88, zorder=6)
    ax.add_patch(hdr)
    ax.text(x + w / 2, y + h - hdr_h / 2 - 0.05, title, ha="center",
            va="center", fontsize=fontsize, fontweight="bold", color=C_WHITE,
            zorder=7)
    if subtitle:
        ax.text(x + w / 2, y + h - hdr_h - 0.28, subtitle, ha="center",
                va="top", fontsize=7, color=C_SUB, style="italic", zorder=7)
    if badge:
        ax.text(x + 0.18, y + h - 0.12, badge, ha="left", va="top",
                fontsize=6, fontweight="bold", color=C_WHITE, zorder=8,
                bbox=dict(boxstyle="round,pad=0.12",
                          fc=badge_color or edge, ec="none", alpha=0.95))


def sub_block(x, y, w, h, fill, edge, text, fs=6.5, bold=False):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                         fc=fill, ec=edge, lw=0.8, zorder=7)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=C_TEXT, fontweight="bold" if bold else "normal",
            zorder=8)


def arr(x1, y1, x2, y2, color=C_TEXT, lw=1.8, ls="-", rad=0, mut=15):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        connectionstyle=f"arc3,rad={rad}",
                        color=color, lw=lw, ls=ls, mutation_scale=mut,
                        zorder=4)
    ax.add_patch(a)


def lbl(x, y, text, color=C_SUB, fs=6.5, bg=C_WHITE, rot=0):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color,
            rotation=rot, zorder=9,
            bbox=dict(boxstyle="round,pad=0.1", fc=bg, ec="none", alpha=0.92))


def formula(x, y, tex, color, fs=7):
    ax.text(x, y, tex, ha="center", va="center", fontsize=fs, color=color,
            zorder=9, bbox=dict(boxstyle="round,pad=0.15", fc=C_WHITE,
                                ec=color, lw=0.8, alpha=0.95))


# ═══════════════════════════════════════════════════════════════════
# 1. INPUT TOKENS
# ═══════════════════════════════════════════════════════════════════
ix, iy, iw, ih = 0.0, 3.6, 1.8, 2.4
module_box(ix, iy, iw, ih, C_INPUT_L, C_INPUT, "Input", "Token sequence")

toks = ["The", "cat", "sat", "on", "…", "mat"]
for k, tok in enumerate(toks):
    ty = iy + ih - 0.85 - k * 0.28
    ax.text(ix + iw / 2, ty, tok, ha="center", va="center", fontsize=5.8,
            color=C_INPUT, family="monospace", zorder=8,
            bbox=dict(boxstyle="round,pad=0.06", fc=C_WHITE,
                      ec=C_FAINT, lw=0.5))

ax.text(ix + iw / 2, iy + 0.15,
        r"$\mathbf{x} = (x^1, x^2, \ldots, x^n)$",
        ha="center", va="center", fontsize=6.5, color=C_INPUT, zorder=8)

# ═══════════════════════════════════════════════════════════════════
# 2. FROZEN GPT-2 ANCHOR  (layer stack inside)
# ═══════════════════════════════════════════════════════════════════
gx, gy, gw, gh = 2.6, 2.2, 3.0, 5.2
module_box(gx, gy, gw, gh, C_ANCHOR_L, C_ANCHOR,
           "GPT-2 Anchor", "124M params · Frozen",
           badge="❄ NO GRAD", badge_color="#60A5FA")

layers = ["Layer 11", "Layer 10", "⋮", "Layer 1", "Layer 0"]
layer_sub = ["Self-Attn + FFN", "Self-Attn + FFN", "",
             "Self-Attn + FFN", "Self-Attn + FFN"]
for k, (ll, sl) in enumerate(zip(layers, layer_sub)):
    ly = gy + gh - 1.1 - k * 0.72
    if ll == "⋮":
        ax.text(gx + gw / 2, ly + 0.15, "⋮", ha="center", va="center",
                fontsize=12, color=C_ANCHOR, zorder=8)
    else:
        sub_block(gx + 0.2, ly, gw - 0.4, 0.55, "#EFF6FF", C_ANCHOR, ll,
                  fs=6.5, bold=True)
        ax.text(gx + gw / 2, ly + 0.08, sl, ha="center", va="center",
                fontsize=5, color=C_SUB, zorder=8)

sub_block(gx + 0.2, gy + 0.25, gw - 0.4, 0.5,
          "#BFDBFE", C_ANCHOR, "Token Embed + Pos Enc", fs=6, bold=True)
ax.text(gx + gw / 2, gy + 0.05, "12 layers · 768 dim · 12 heads",
        ha="center", va="center", fontsize=5.5, color=C_ANCHOR_D, zorder=8)

# ═══════════════════════════════════════════════════════════════════
# 3. IMPORTANCE ESTIMATOR  (MLP detail + frequency branch)
# ═══════════════════════════════════════════════════════════════════
ex, ey, ew, eh = 6.5, 5.0, 3.2, 4.2
module_box(ex, ey, ew, eh, C_ESTIM_L, C_ESTIM,
           "Importance Estimator", "200K params · Trainable")

mlp = [("LayerNorm",          "#D1FAE5"),
       ("Linear 768→256",     "#A7F3D0"),
       ("GELU",               "#6EE7B7"),
       ("Linear 256→1",       "#A7F3D0"),
       (r"Sigmoid → $i^l_{\mathrm{ctx}}$", "#34D399")]
for k, (name, col) in enumerate(mlp):
    ly = ey + eh - 1.0 - k * 0.52
    sub_block(ex + 0.18, ly, ew - 0.36, 0.40, col, C_ESTIM, name,
              fs=6.5, bold=(k == len(mlp) - 1))
    if k < len(mlp) - 1:
        ax.annotate("", xy=(ex + ew / 2, ly - 0.02),
                    xytext=(ex + ew / 2, ly + 0.40 + 0.02),
                    arrowprops=dict(arrowstyle="-|>", color=C_ESTIM,
                                    lw=0.7, mutation_scale=8), zorder=8)

sub_block(ex + 0.18, ey + 0.22, ew - 0.36, 0.40,
          "#FEF3C7", C_MASK, r"Freq Prior  $i^l_{\mathrm{freq}}$", fs=6)
sub_block(ex + 0.18, ey + 0.72, ew - 0.36, 0.45,
          "#ECFDF5", C_ESTIM_D, "", fs=6)
ax.text(ex + ew / 2, ey + 0.945,
        r"$i^l = 0.7\, i^l_{\mathrm{ctx}} + 0.3\, i^l_{\mathrm{freq}}$",
        ha="center", va="center", fontsize=6.5, color=C_ESTIM_D,
        fontweight="bold", zorder=9)

# ═══════════════════════════════════════════════════════════════════
# 4. IMPORTANCE SCORES HUB (central circle)
# ═══════════════════════════════════════════════════════════════════
sx, sy, sr = 10.3, 7.0, 0.55
circ = plt.Circle((sx, sy), sr, fc=C_SCORE_L, ec=C_SCORE, lw=2.0, zorder=7)
circ.set_path_effects(shadow)
ax.add_patch(circ)
ax.text(sx, sy + 0.08, r"$\mathbf{i}^l$", ha="center", va="center",
        fontsize=13, fontweight="bold", color=C_SCORE, zorder=8)
ax.text(sx, sy - 0.22, "scores", ha="center", va="center",
        fontsize=6, color=C_SCORE, zorder=8)

# ═══════════════════════════════════════════════════════════════════
# 5. BALANCED MASKING CURRICULUM (phase detail)
# ═══════════════════════════════════════════════════════════════════
mx, my, mw, mh = 8.8, 2.2, 3.0, 4.0
module_box(mx, my, mw, mh, C_MASK_L, C_MASK,
           "Masking Curriculum", "Balanced schedule")

phases = [("Hard  40 %", C_SCORE, "#FEE2E2", r"$i \in [0,\, 1.0]$"),
          ("Medium  40 %", C_MASK, "#FEF3C7", r"$i \in [0.3,\, 0.7]$"),
          ("Easy  20 %",  C_ESTIM, "#D1FAE5", r"$i \in [0,\, 0.3]$")]
for k, (pl, pc, pf, pr) in enumerate(phases):
    py = my + mh - 1.1 - k * 0.75
    sub_block(mx + 0.15, py, mw - 0.30, 0.55, pf, pc, "", fs=6)
    ax.text(mx + 0.35, py + 0.33, pl, ha="left", va="center", fontsize=6,
            fontweight="bold", color=pc, zorder=9)
    ax.text(mx + mw - 0.35, py + 0.15, pr, ha="right", va="center",
            fontsize=5.5, color=C_SUB, zorder=9)

ax.text(mx + mw / 2, my + 0.5,
        r"$g_{\mathrm{prop}} = \eta + (1{-}2\eta)\, i$",
        ha="center", va="center", fontsize=5.5, color=C_MASK_D, zorder=9)
ax.text(mx + mw / 2, my + 0.22,
        r"$g_{\mathrm{inv}} = \eta + (1{-}2\eta)\,(1{-}i)$",
        ha="center", va="center", fontsize=5.5, color=C_MASK_D, zorder=9)

# ═══════════════════════════════════════════════════════════════════
# 6. DENOISER TRANSFORMER  (layer stack + augmentation detail)
# ═══════════════════════════════════════════════════════════════════
dx, dy, dw, dh = 13.0, 2.2, 3.0, 5.2
module_box(dx, dy, dw, dh, C_DENOISE_L, C_DENOISE,
           "Denoiser", "48M params · Trainable")

dl = [("Denoiser Layer 5", "adaLN + RoPE"),
      ("Denoiser Layer 4", "Self-Attn + FFN"),
      ("⋮", ""),
      ("Denoiser Layer 1", "Self-Attn + FFN"),
      ("Denoiser Layer 0", "adaLN + RoPE")]
for k, (ll, sl) in enumerate(dl):
    ly = dy + dh - 1.1 - k * 0.68
    if ll == "⋮":
        ax.text(dx + dw / 2, ly + 0.15, "⋮", ha="center", va="center",
                fontsize=12, color=C_DENOISE, zorder=8)
    else:
        sub_block(dx + 0.15, ly, dw - 0.30, 0.50, "#F5F3FF", C_DENOISE,
                  ll, fs=6, bold=True)
        ax.text(dx + dw / 2, ly + 0.07, sl, ha="center", va="center",
                fontsize=4.8, color=C_SUB, zorder=8)

sub_block(dx + 0.15, dy + 0.75, dw - 0.30, 0.50, "#DDD6FE", C_DENOISE_D,
          r"$\oplus$ Importance Augment", fs=5.5, bold=True)
sub_block(dx + 0.15, dy + 0.18, dw - 0.30, 0.45, "#C4B5FD", C_DENOISE_D,
          r"Linear → Softmax → $\hat{x}^l$", fs=6, bold=True)
ax.text(dx + dw / 2, dy + 0.03, "6 layers · 768 dim · 12 heads · RoPE",
        ha="center", va="center", fontsize=5, color=C_DENOISE_D, zorder=8)

# ═══════════════════════════════════════════════════════════════════
# 7. OUTPUT
# ═══════════════════════════════════════════════════════════════════
ox, oy, ow, oh = 13.3, 0.0, 2.4, 1.6
module_box(ox, oy, ow, oh, C_INPUT_L, C_INPUT,
           "Output", r"$\hat{x} = (\hat{x}^1, \ldots, \hat{x}^n)$",
           fontsize=10)
ax.text(ox + ow / 2, oy + 0.2, "Token predictions",
        ha="center", va="center", fontsize=6, color=C_SUB, zorder=8)

# ═══════════════════════════════════════════════════════════════════
# ARROWS — DATA FLOW
# ═══════════════════════════════════════════════════════════════════
# Input → GPT-2
arr(ix + iw, iy + ih / 2, gx, gy + gh / 2, C_INPUT, lw=2.2)
lbl((ix + iw + gx) / 2, iy + ih / 2 + 0.25, "tokenize", C_SUB, 5.5)

# GPT-2 → Estimator  (hidden states)
arr(gx + gw, gy + gh * 0.75, ex, ey + eh * 0.65, C_ANCHOR, lw=2.2)
lbl((gx + gw + ex) / 2 + 0.1, gy + gh * 0.75 + 0.35,
    r"$h^l_{\mathrm{GPT{-}2}}$", C_ANCHOR, 7.5, C_ANCHOR_L)

# GPT-2 → Denoiser  (skip-connection for augmentation)
arr(gx + gw, gy + gh * 0.35, dx, dy + dh * 0.25,
    C_ANCHOR, lw=1.5, ls=(0, (5, 3)), rad=-0.12)
lbl(8.2, 2.85, r"$h^l$ skip", C_ANCHOR, 5.5)

# Estimator → Importance hub
arr(ex + ew, ey + eh * 0.55, sx - sr, sy, C_ESTIM, lw=2.2)
lbl((ex + ew + sx - sr) / 2, (ey + eh * 0.55 + sy) / 2 + 0.3,
    r"hybrid $i^l \in [0,1]$", C_ESTIM, 6)

# Hub → Masking
arr(sx, sy - sr, mx + mw / 2, my + mh, C_SCORE, lw=2.0)
lbl(sx - 0.65, (sy - sr + my + mh) / 2 + 0.15, "masking\nprobs", C_SCORE, 5.5)

# Hub → Denoiser  (importance projection)
arr(sx + sr * 0.7, sy - sr * 0.7, dx, dy + dh * 0.55,
    C_SCORE, lw=2.0, ls=(0, (4, 2)))
lbl((sx + sr + dx) / 2 + 0.35, (sy + dy + dh * 0.55) / 2 + 0.5,
    r"$[i^l;\; 1{-}i^l]$", C_SCORE, 7, C_SCORE_L)

# Masking → Denoiser
arr(mx + mw, my + mh * 0.5, dx, dy + dh * 0.4, C_MASK, lw=2.0)
lbl((mx + mw + dx) / 2, my + mh * 0.5 - 0.3,
    r"masked tokens $z_t$", C_MASK_D, 6)

# Denoiser → Output
arr(dx + dw / 2, dy, ox + ow / 2, oy + oh, C_DENOISE, lw=2.0)

# ═══════════════════════════════════════════════════════════════════
# ORACLE SUPERVISION (top arc)
# ═══════════════════════════════════════════════════════════════════
oracle_x = gx + gw * 0.5
ax.text(oracle_x, gy + gh + 0.65,
        r"Oracle:  $i^{l,\ast} = \min(1,\; "
        r"\frac{-\log p(x^l|x^{<l})}{\tau})$"
        r"$\;\; \tau{=}10$",
        ha="center", va="center", fontsize=6.5, color=C_ANCHOR_D, zorder=9,
        bbox=dict(boxstyle="round,pad=0.15", fc="#EFF6FF", ec=C_ANCHOR,
                  lw=0.8, alpha=0.95))
arr(oracle_x + 2.2, gy + gh + 0.5, ex + ew / 2, ey + eh,
    C_ANCHOR, lw=1.0, ls=":", rad=-0.2, mut=10)
lbl(oracle_x + 3.0, gy + gh + 0.9, "supervision", C_ANCHOR, 5)

# ═══════════════════════════════════════════════════════════════════
# TIMESTEP  t  → Denoiser  (adaLN conditioning)
# ═══════════════════════════════════════════════════════════════════
tx, ty = dx + dw + 0.15, dy + dh * 0.75
ax.text(tx, ty, r"$t$", ha="left", va="center", fontsize=11,
        fontweight="bold", color=C_DENOISE_D, zorder=8,
        bbox=dict(boxstyle="circle,pad=0.15", fc=C_DENOISE_L,
                  ec=C_DENOISE, lw=1.2))
arr(tx, ty, dx + dw, dy + dh * 0.65, C_DENOISE, lw=1.2, ls="--",
    rad=-0.15, mut=10)
ax.text(tx + 0.05, ty - 0.45, "timestep\n(adaLN)", ha="center", va="top",
        fontsize=5, color=C_SUB, zorder=8)

# ═══════════════════════════════════════════════════════════════════
# FORMULA CALLOUTS
# ═══════════════════════════════════════════════════════════════════
formula(dx + dw / 2, dy - 0.55,
        r"$\tilde{h}^l = \mathrm{LN}\left(h^l + W_i [i^l;\; 1{-}i^l]\right)$"
        r"$\;\; W_i \in \mathbb{R}^{768 \times 2}$",
        C_DENOISE, fs=6.5)

formula(mx + mw / 2, my - 0.55,
        r"$g_{\mathrm{bal}}(i,t) = t\, g_{\mathrm{inv}} "
        r"+ (1{-}t)\, g_{\mathrm{prop}}$"
        r"$\;\; \eta{=}0.3$",
        C_MASK, fs=6.5)

# ═══════════════════════════════════════════════════════════════════
# TRAINING OBJECTIVE (bottom banner)
# ═══════════════════════════════════════════════════════════════════
ly = -2.2
lb = FancyBboxPatch((2.0, ly), 12.5, 0.9, boxstyle="round,pad=0.15",
                     fc=C_LOSS_BG, ec=C_LOSS_BD, lw=1.5, zorder=5)
lb.set_path_effects(shadow)
ax.add_patch(lb)

ax.text(2.4, ly + 0.45, "Training\nObjective", ha="left", va="center",
        fontsize=8, fontweight="bold", color=C_LOSS_BD, zorder=6)
ax.text(8.25, ly + 0.55,
        r"$\mathcal{L}_{\mathrm{ATAT}} = "
        r"\mathcal{L}_{\mathrm{denoise}} "
        r"+ \;\gamma \, \mathcal{L}_{\mathrm{importance}}$",
        ha="center", va="center", fontsize=9, color=C_TEXT, zorder=6)
ax.text(8.25, ly + 0.18,
        r"$= \sum_l w(i^l)\, \ell_{\mathrm{CE}}(x^l, \hat{x}^l) "
        r"+ \gamma \sum_l (i^l - i^{l,\ast})^2$",
        ha="center", va="center", fontsize=7, color=C_SUB, zorder=6)
ax.text(8.25, ly - 0.08,
        r"$w(i^l) = 1 + \beta \cdot i^l \;\; (\beta{=}1.0, \;\gamma{=}0.003)$",
        ha="center", va="center", fontsize=6.5, color=C_SUB, zorder=6)

# ═══════════════════════════════════════════════════════════════════
# INFO CARD (top-right)
# ═══════════════════════════════════════════════════════════════════
info_x, info_y = 13.0, 8.7
ib = FancyBboxPatch((info_x, info_y), 3.3, 0.65, boxstyle="round,pad=0.12",
                     fc=C_WHITE, ec=C_FAINT, lw=1.0, zorder=5)
ax.add_patch(ib)
for k, (nm, ct, st, co) in enumerate([
    ("Anchor:", "124M", "frozen", C_ANCHOR),
    ("Estimator:", "200K", "trainable", C_ESTIM),
    ("Denoiser:", "48M", "trainable", C_DENOISE),
]):
    px = info_x + 0.15 + k * 1.1
    ax.text(px, info_y + 0.42, nm, ha="left", va="center", fontsize=5.5,
            color=C_TEXT, fontweight="bold", zorder=8)
    ax.text(px, info_y + 0.2, f"{ct} ({st})", ha="left", va="center",
            fontsize=5, color=co, zorder=8)
ax.text(info_x + 3.15, info_y + 0.32, "Total: 173M\nTrainable: 49M",
        ha="right", va="center", fontsize=6, fontweight="bold",
        color=C_TEXT, zorder=8)

# ═══════════════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════════════
ax.text(8.25, 9.2, "ATAT: Adaptive Token Attention for Text Diffusion",
        ha="center", va="center", fontsize=15, fontweight="bold",
        color=C_TEXT, zorder=10, path_effects=text_glow)
ax.text(8.25, 8.85, "Architecture Overview",
        ha="center", va="center", fontsize=10, color=C_SUB, zorder=10)

# ═══════════════════════════════════════════════════════════════════
# LEGEND
# ═══════════════════════════════════════════════════════════════════
legend_items = [
    mpatches.Patch(fc=C_ANCHOR_L, ec=C_ANCHOR, lw=1.5,
                   label="Frozen GPT-2 Anchor"),
    mpatches.Patch(fc=C_ESTIM_L, ec=C_ESTIM, lw=1.5,
                   label="Importance Estimator"),
    mpatches.Patch(fc=C_MASK_L, ec=C_MASK, lw=1.5,
                   label="Masking Curriculum"),
    mpatches.Patch(fc=C_DENOISE_L, ec=C_DENOISE, lw=1.5,
                   label="Denoiser Transformer"),
    mpatches.Patch(fc=C_SCORE_L, ec=C_SCORE, lw=1.5,
                   label="Importance Scores"),
]
leg = ax.legend(handles=legend_items, loc="lower left", fontsize=6,
                framealpha=0.95, edgecolor=C_FAINT, ncol=3,
                bbox_to_anchor=(-0.02, -0.05), handlelength=1.5,
                handleheight=1.0, columnspacing=1.0)
leg.get_frame().set_linewidth(0.8)

# ═══════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════
out_dir = Path(__file__).parent
for fmt in ("pdf", "png"):
    p = out_dir / f"atat_architecture.{fmt}"
    fig.savefig(p, format=fmt, dpi=300, bbox_inches="tight",
                facecolor=C_BG, pad_inches=0.2)
    print(f"Saved: {p}")
plt.close(fig)
print("Done — architecture diagram generated.")
