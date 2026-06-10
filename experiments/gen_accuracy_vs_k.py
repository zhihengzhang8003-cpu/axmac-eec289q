"""Generate "Accuracy vs. K (round / trunc / stochastic)" figure.

Usage:
    py -3.14 experiments/gen_accuracy_vs_k.py

Output:
    experiments/results/figures/fig_accuracy_vs_k.png
"""
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
CSV  = REPO / "experiments" / "results" / "accuracy_vs_K.csv"
OUT  = REPO / "experiments" / "results" / "figures" / "fig_accuracy_vs_k.png"

# ── read ─────────────────────────────────────────────────────────────────────
Ks, trunc_acc, round_acc, drum_acc = [], [], [], []
with CSV.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        Ks.append(int(row["K"]))
        trunc_acc.append(float(row["trunc_acc"]) * 100)
        round_acc.append(float(row["round_acc"]) * 100)
        drum_acc.append(float(row["stochastic_acc"]) * 100)

C_TRUNC = "#C0392B"
C_ROUND = "#1565C0"
C_DRUM  = "#1E8449"

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.0, 4.8), facecolor="white", dpi=150)
fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.13)

ax.set_facecolor("white")
ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.6, zorder=0)
ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_color("black"); spine.set_linewidth(0.8)
ax.tick_params(colors="black", labelsize=11)

ax.plot(Ks, trunc_acc, "o--", color=C_TRUNC, linewidth=2.0, markersize=6,
        markerfacecolor=C_TRUNC, markeredgecolor=C_TRUNC, label="trunc", zorder=4)
ax.plot(Ks, round_acc, "o-",  color=C_ROUND, linewidth=2.5, markersize=6,
        markerfacecolor=C_ROUND, markeredgecolor=C_ROUND, label="round", zorder=5)
ax.plot(Ks, drum_acc,  "o",   color=C_DRUM,  linewidth=2.0, markersize=6,
        markerfacecolor=C_DRUM,  markeredgecolor=C_DRUM,  label="stochastic",
        linestyle=(0, (4, 2)), zorder=4)

# ── annotate K=8 values ───────────────────────────────────────────────────────
for val, color, va in [
    (trunc_acc[Ks.index(8)], C_TRUNC, "top"),
    (round_acc[Ks.index(8)], C_ROUND, "bottom"),
    (drum_acc[Ks.index(8)],  C_DRUM,  "bottom"),
]:
    ax.text(8.08, val, f"{val:.1f}%", color=color,
            fontsize=9, fontweight="bold", va=va, ha="left")

# ── reference line ────────────────────────────────────────────────────────────
ax.axhline(10, color="#AAAAAA", linewidth=0.8, linestyle=":", zorder=2)
ax.text(7.9, 11.5, "chance (10%)", color="#888888", fontsize=8.5, ha="right")

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(-0.3, 8.9)
ax.set_ylim(50, 102)
ax.set_xticks(Ks)
ax.set_xticklabels([str(k) for k in Ks], color="black", fontsize=11)
ax.set_yticks(range(50, 105, 5))
ax.set_yticklabels([f"{v}%" for v in range(50, 105, 5)], color="black", fontsize=10)
ax.set_xlabel("Truncation depth  K  (bits removed per partial product)",
              color="black", fontsize=11, labelpad=7)
ax.set_ylabel("Top-1 accuracy  (MNIST, 10,000 test images)",
              color="black", fontsize=11, labelpad=7)
ax.set_title("Accuracy vs. K  ·  INT8 PTQ  ·  784→256→128→64→10 MLP  ·  MNIST",
             color="black", fontsize=12, fontweight="bold", pad=10)

ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.02),
          facecolor="white", edgecolor="#AAAAAA",
          labelcolor="black", fontsize=10.5, framealpha=0.95)

# ── inset: zoom K=6..8 ───────────────────────────────────────────────────────
axins = ax.inset_axes([0.32, 0.08, 0.38, 0.42])  # x0, y0, w, h in axes coords
zoom_ks = [6, 7, 8]
zi = [Ks.index(k) for k in zoom_ks]
axins.plot(zoom_ks, [trunc_acc[i] for i in zi], "o--", color=C_TRUNC,
           linewidth=1.8, markersize=6, markerfacecolor=C_TRUNC)
axins.plot(zoom_ks, [round_acc[i] for i in zi], "o-",  color=C_ROUND,
           linewidth=2.0, markersize=6, markerfacecolor=C_ROUND)
axins.plot(zoom_ks, [drum_acc[i]  for i in zi], "o",   color=C_DRUM,
           linewidth=1.8, markersize=6, markerfacecolor=C_DRUM,
           linestyle=(0, (4, 2)))
axins.set_facecolor("white")
axins.set_xticks(zoom_ks)
axins.set_xticklabels(["K=6", "K=7", "K=8"], fontsize=8.5, color="black")
axins.set_ylim(68, 100)
axins.set_yticks([70, 80, 90, 95, 97, 99])
axins.set_yticklabels(["70%", "80%", "90%", "95%", "97%", "99%"], fontsize=8, color="black")
axins.yaxis.grid(True, color="#EEEEEE", linewidth=0.5)
for sp in axins.spines.values():
    sp.set_color("#AAAAAA"); sp.set_linewidth(0.8)
axins.set_title("K = 6–8 (detail)", fontsize=8.5, color="black", pad=4)
# annotate exact values at K=8
for val, color, dy in [
    (trunc_acc[Ks.index(8)], C_TRUNC, -4),
    (round_acc[Ks.index(8)], C_ROUND,  1),
    (drum_acc[Ks.index(8)],  C_DRUM,  -2),
]:
    axins.text(8.05, val + dy, f"{val:.1f}%", color=color,
               fontsize=7.5, fontweight="bold", va="center")
ax.indicate_inset_zoom(axins, edgecolor="#AAAAAA", linewidth=0.8)

fig.savefig(OUT, dpi=150, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("Saved.")
