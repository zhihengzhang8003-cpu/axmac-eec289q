"""Generate "Bias at N=4096 (INT8, K=6)" bar chart.

Usage:
    py -3.14 "E:/AIDev/EEC 289Q 002 SQ 2026：Deep Learning Hardware/project/experiments/gen_bias_bar.py"

Output:
    experiments/results/figures/fig_bias_bar_n4096.png
"""
import csv, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = Path(__file__).resolve().parent.parent
CSV  = REPO / "experiments" / "results" / "bias_accumulation.csv"
OUT  = REPO / "experiments" / "results" / "figures" / "fig_bias_bar_n4096.png"

# ── read N=4096 row ──────────────────────────────────────────────────────────
with CSV.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if int(row["N"]) == 4096:
            trunc_val = abs(float(row["trunc_mean"]))
            round_val = abs(float(row["round_mean"]))
            stoch_val = abs(float(row["stochastic_mean"]))
            break

modes  = ["trunc", "round", "stochastic"]
values = [trunc_val, round_val, stoch_val]
colors = ["#C0392B", "#2471A3", "#1E8449"]   # red, blue, green

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.0, 4.2), facecolor="white", dpi=150)
fig.subplots_adjust(left=0.14, right=0.96, top=0.85, bottom=0.14)

bars = ax.bar(modes, values, color=colors, width=0.50,
              edgecolor="black", linewidth=0.7, zorder=3)

# ── log y-axis with clean grid ───────────────────────────────────────────────
ax.set_yscale("log")
ax.set_ylim(1, 200_000)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{int(x):,}" if x >= 1 else f"{x:.1f}"
))
ax.yaxis.grid(True, which="both", color="#DDDDDD", linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for spine in ax.spines.values():
    spine.set_color("black")
    spine.set_linewidth(0.8)
ax.tick_params(axis="both", colors="black", labelsize=11)
ax.set_xticks(range(len(modes)))
ax.set_xticklabels(modes, fontsize=12, color="black")

# ── value labels above each bar ──────────────────────────────────────────────
for bar, val in zip(bars, values):
    label = f"{val:,.0f}"
    ax.text(bar.get_x() + bar.get_width() / 2,
            val * 1.6,
            label,
            ha="center", va="bottom",
            fontsize=12, fontweight="bold",
            color=bar.get_facecolor())

# ── ratio annotations ─────────────────────────────────────────────────────────
ratio_ts = trunc_val / stoch_val
ratio_tr = trunc_val / round_val
ax.annotate(
    f"trunc / stochastic\n≈ {ratio_ts:.0f}×",
    xy=(2, stoch_val), xytext=(1.65, 800),
    fontsize=9.5, color="#555555",
    arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0),
    bbox=dict(facecolor="white", edgecolor="#CCCCCC",
              boxstyle="round,pad=0.3", linewidth=0.7),
)

# ── axes labels & title ───────────────────────────────────────────────────────
ax.set_ylabel("|Accumulated bias|  (log scale)", fontsize=11, color="black", labelpad=6)
ax.set_xlabel("Rounding mode", fontsize=11, color="black", labelpad=6)
ax.set_title("Bias at N = 4096  (INT8, K = 6)",
             fontsize=13, fontweight="bold", color="black", pad=10)

fig.savefig(OUT, dpi=150, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("Saved.")
