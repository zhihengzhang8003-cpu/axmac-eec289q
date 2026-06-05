"""
generate_figures.py -- generate all report figures from existing data + new experiments.

Figures produced (saved to experiments/results/figures/):
  fig1_bias_accumulation.png  -- trunc/round/stoch error vs depth N
  fig2_three_way_rounding.png -- bias comparison bar chart, K=1..6, INT8
  fig3_ppa_hardware.png       -- PPA bar chart from Quartus synthesis data
  fig4_accuracy_vs_K.png      -- Top-1 agreement vs K for trunc/round/stoch (1000 samples)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS = os.path.join(os.path.dirname(__file__), "results")
FIGDIR  = os.path.join(RESULTS, "figures")
os.makedirs(FIGDIR, exist_ok=True)

GOLDEN  = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "rtl", "golden", "mlp_toy")

# ── shared style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 150,
    "lines.linewidth": 1.8, "lines.markersize": 5,
})
C_TRUNC = "#d62728"
C_ROUND = "#1f77b4"
C_STOCH = "#2ca02c"
C_EXACT = "#7f7f7f"

# ============================================================
# FIG 1 -- bias accumulation vs depth N
# ============================================================
print("Fig 1: bias accumulation...")

def _read_csv(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            out = {}
            for k, v in r.items():
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
            rows.append(out)
    return rows

bias_rows = _read_csv(os.path.join(RESULTS, "bias_accumulation.csv"))
Ns      = [r["N"] for r in bias_rows]
t_bias  = [r["trunc_mean"]       for r in bias_rows]
r_bias  = [r["round_mean"]       for r in bias_rows]
s_bias  = [r["stochastic_mean"]  for r in bias_rows]

fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.plot(Ns, [abs(v) for v in t_bias], color=C_TRUNC, label="trunc  |bias|")
ax.plot(Ns, [abs(v) for v in r_bias], color=C_ROUND, label="round  |bias|")
ax.plot(Ns, [abs(v) for v in s_bias], color=C_STOCH, label="stochastic |bias|")

# reference lines
N_arr = np.array(Ns)
ax.plot(N_arr, N_arr * (abs(t_bias[-1]) / Ns[-1]),
        "r--", lw=1, alpha=0.4, label="∝ N  (linear)")
ax.plot(N_arr, np.sqrt(N_arr) * (abs(s_bias[-1]) / np.sqrt(Ns[-1])),
        "g--", lw=1, alpha=0.4, label="∝ √N")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Accumulation depth N"); ax.set_ylabel("|Accumulated bias|")
ax.set_title("Fig. 1: Truncation bias grows linearly with N\n(INT8, K=6)")
ax.legend(framealpha=0.9, loc="upper left")
ax.grid(True, which="both", ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig1_bias_accumulation.png"))
plt.close(fig)
print("  saved fig1_bias_accumulation.png")

# ============================================================
# FIG 2 -- three-way rounding bias bar chart
# ============================================================
print("Fig 2: three-way rounding...")

rnd_rows = _read_csv(os.path.join(RESULTS, "three_way_rounding.csv"))
Ks_rnd = sorted(set(int(r["K"]) for r in rnd_rows
                if int(r["K"]) > 0 and r.get("fmt","INT8") == "INT8"))

def get_bias_vals(mode):
    return [next((r["bias"] for r in rnd_rows
                  if int(r["K"]) == k
                  and r.get("fmt","INT8") == "INT8"
                  and r.get("rounding","trunc") == mode),
                 0.0)
            for k in Ks_rnd]

t_vals = get_bias_vals("trunc")
r_vals = get_bias_vals("round")
s_vals = get_bias_vals("stochastic")

x = np.arange(len(Ks_rnd))
w = 0.25
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.bar(x - w, t_vals, w, label="trunc",       color=C_TRUNC, alpha=0.85)
ax.bar(x,     r_vals, w, label="round",       color=C_ROUND, alpha=0.85)
ax.bar(x + w, s_vals, w, label="stochastic",  color=C_STOCH, alpha=0.85)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([f"K={k}" for k in Ks_rnd])
ax.set_xlabel("Truncation depth K"); ax.set_ylabel("Per-MAC accumulated bias")
ax.set_title("Fig. 2: Rounding mode comparison (INT8)\n"
             "round approaches stochastic at near-zero hardware cost")
ax.legend(framealpha=0.9)
ax.grid(True, axis="y", ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig2_three_way_rounding.png"))
plt.close(fig)
print("  saved fig2_three_way_rounding.png")

# ============================================================
# FIG 3 -- PPA hardware bar chart
# ============================================================
print("Fig 3: PPA hardware...")

ppa_data = {
    "Exact\n(K=0)":     (1769, 10.45),
    "K=6\ntrunc":       (1685,  8.67),
    "K=6\nround":       (1731,  9.20),
    "K=6\nstochastic":  (1817,  9.39),
}
labels = list(ppa_data.keys())
les    = [ppa_data[l][0] for l in labels]
pows   = [ppa_data[l][1] for l in labels]
colors = [C_EXACT, C_TRUNC, C_ROUND, C_STOCH]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.5))
x = np.arange(len(labels))
ax1.bar(x, les,  color=colors, alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8.5)
ax1.set_ylabel("Logic Elements (LE)")
ax1.set_title("Area (EP4CE10)")
ax1.set_ylim(1550, 1900)
for i, v in enumerate(les):
    ax1.text(i, v + 5, str(v), ha="center", va="bottom", fontsize=8)
ax1.grid(True, axis="y", ls=":", alpha=0.4)

ax2.bar(x, pows, color=colors, alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8.5)
ax2.set_ylabel("Core Dynamic Power (mW)")
ax2.set_title("Power (EP4CE10)")
ax2.set_ylim(7, 11.5)
for i, v in enumerate(pows):
    ax2.text(i, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
ax2.grid(True, axis="y", ls=":", alpha=0.4)

fig.suptitle("Fig. 3: Hardware synthesis results (Quartus Prime Lite 18.1)\n"
             "round: only +46 LE vs trunc; stochastic LFSR exceeds exact baseline",
             fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig3_ppa_hardware.png"))
plt.close(fig)
print("  saved fig3_ppa_hardware.png")

# ============================================================
# FIG 4 -- Top-1 accuracy vs K (1000 synthetic samples)
# ============================================================
print("Fig 4: Top-1 accuracy vs K (running 1000-sample inference)...")

from axmac.dnn_inference import tiny_mlp_forward
from axmac.exact_mac import IntFormat

def load_mem_as_int(path, dtype=np.int8):
    vals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("//"):
                vals.append(int(line, 16))
    return np.array(vals, dtype=np.int64)

def load_csv_flat(path):
    vals = []
    with open(path) as f:
        for line in f:
            for v in line.split(","):
                v = v.strip()
                if v:
                    vals.append(int(v))
    return np.array(vals, dtype=np.int64)

# load weights from CSV (signed int, row-major)
w0 = load_csv_flat(os.path.join(GOLDEN, "w0.csv")).reshape(64, 16)
b0 = load_csv_flat(os.path.join(GOLDEN, "b0.csv"))
w1 = load_csv_flat(os.path.join(GOLDEN, "w1.csv")).reshape(16, 10)
b1 = load_csv_flat(os.path.join(GOLDEN, "b1.csv"))
layers = [(w0, b0), (w1, b1)]
fmt = IntFormat("INT8", 8)

# one reference sample to confirm golden result
x_ref = load_csv_flat(os.path.join(GOLDEN, "x.csv")).reshape(1, 64)
logits_ref = tiny_mlp_forward(x_ref, layers, fmt=fmt, K=0)
assert np.argmax(logits_ref) == 1, f"Reference argmax mismatch: {np.argmax(logits_ref)}"

# generate 1000 synthetic test samples from the same distribution as x.csv
rng = np.random.default_rng(42)
N_SAMPLES = 1000
x_ref_mean = float(load_csv_flat(os.path.join(GOLDEN, "x.csv")).mean())
x_ref_std  = float(load_csv_flat(os.path.join(GOLDEN, "x.csv")).std())
X_test = rng.normal(x_ref_mean, max(x_ref_std, 8.0),
                    size=(N_SAMPLES, 64)).astype(np.int64)
X_test = np.clip(X_test, -128, 127)

# exact predictions as ground truth
y_exact = tiny_mlp_forward(X_test, layers, fmt=fmt, K=0)
labels_exact = np.argmax(y_exact, axis=1)

K_range = list(range(0, 9))
modes   = ["trunc", "round", "stochastic"]
colors_mode = {"trunc": C_TRUNC, "round": C_ROUND, "stochastic": C_STOCH}
results_acc = {m: [] for m in modes}

for K in K_range:
    for mode in modes:
        seed = rng if mode == "stochastic" else None
        y_approx = tiny_mlp_forward(X_test, layers, fmt=fmt, K=K, rounding=mode,
                                    rng=seed)
        preds = np.argmax(y_approx, axis=1)
        acc   = float(np.mean(preds == labels_exact))
        results_acc[mode].append(acc)
    print(f"  K={K}: trunc={results_acc['trunc'][-1]:.3f}  "
          f"round={results_acc['round'][-1]:.3f}  "
          f"stoch={results_acc['stochastic'][-1]:.3f}")

# mark the hardware-observed misclassification point
hw_K6_trunc_correct = False  # K=6 trunc -> argmax=3 (wrong) on the board

fig, ax = plt.subplots(figsize=(5.5, 3.8))
for mode in modes:
    ax.plot(K_range, [v * 100 for v in results_acc[mode]],
            "o-", color=colors_mode[mode], label=mode)

# annotate hardware validation points
ax.annotate("Board: correct\n(K=0..4 trunc, K=4 round)",
            xy=(4, results_acc["trunc"][4] * 100),
            xytext=(2.5, 55), fontsize=8,
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            color="gray")
ax.annotate("Board: argmax=3 [FAIL]\n(K=6 trunc)",
            xy=(6, results_acc["trunc"][6] * 100),
            xytext=(5.5, 25), fontsize=8,
            arrowprops=dict(arrowstyle="->", color=C_TRUNC, lw=0.8),
            color=C_TRUNC)

ax.set_xlabel("Truncation depth K")
ax.set_ylabel("Top-1 Agreement with Exact (%)")
ax.set_title(f"Fig. 4: Top-1 accuracy vs K (N={N_SAMPLES} synthetic INT8 samples)\n"
             "Hardware board results annotated")
ax.set_xticks(K_range)
ax.set_ylim(0, 105)
ax.axhline(100, color=C_EXACT, ls="--", lw=1, alpha=0.5, label="exact (K=0)")
ax.legend(framealpha=0.9)
ax.grid(True, ls=":", alpha=0.4)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig4_accuracy_vs_K.png"))
plt.close(fig)
print("  saved fig4_accuracy_vs_K.png")

# save accuracy CSV
acc_csv = os.path.join(RESULTS, "accuracy_vs_K.csv")
with open(acc_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["K", "trunc_acc", "round_acc", "stoch_acc"])
    for i, K in enumerate(K_range):
        w.writerow([K,
                    round(results_acc["trunc"][i], 6),
                    round(results_acc["round"][i], 6),
                    round(results_acc["stochastic"][i], 6)])
print(f"  saved accuracy_vs_K.csv")

# ============================================================
# FIG 5 -- CIFAR-10 INT8 accuracy vs K (trunc / round / DRUM)
# ============================================================
print("Fig 5: CIFAR-10 accuracy vs K...")

cifar_rows = _read_csv(os.path.join(RESULTS, "cifar10_accuracy.csv"))

# Parse rows: exact baseline, trunc/round per K, DRUM points
exact_acc = next(r["accuracy"] for r in cifar_rows if r["mode"] == "exact")

K_vals = sorted(set(int(r["K"]) for r in cifar_rows
                    if r["mode"] in ("trunc", "round")))

def _cifar_acc(mode):
    return [next(r["accuracy"] for r in cifar_rows
                 if r["mode"] == mode and int(r["K"]) == k)
            for k in K_vals]

trunc_acc = _cifar_acc("trunc")
round_acc = _cifar_acc("round")

# DRUM single-point data
drum4_acc = next((r["accuracy"] for r in cifar_rows if r["K"] == "drum-4"), None)
drum6_acc = next((r["accuracy"] for r in cifar_rows if r["K"] == "drum-6"), None)

C_DRUM4 = "#ff7f0e"   # orange for DRUM-4
C_DRUM6 = "#9467bd"   # purple for DRUM-6
FLOAT32_BASE = 0.8325  # float32 training accuracy

fig, ax = plt.subplots(figsize=(6.0, 4.0))

# Prepend K=0 exact baseline to both lines
Kx     = [0] + K_vals
t_plot = [exact_acc] + trunc_acc
r_plot = [exact_acc] + round_acc

ax.plot(Kx, [v * 100 for v in t_plot], "o-",
        color=C_TRUNC, label="trunc", lw=2)
ax.plot(Kx, [v * 100 for v in r_plot], "s-",
        color=C_ROUND, label="round", lw=2)

# DRUM single-point markers (x position = comparable aggressiveness)
if drum4_acc is not None:
    ax.plot(4, drum4_acc * 100, "D", color=C_DRUM4, ms=8, zorder=5,
            label=f"DRUM-4  ({drum4_acc*100:.1f}%)")
if drum6_acc is not None:
    ax.plot(2, drum6_acc * 100, "D", color=C_DRUM6, ms=8, zorder=5,
            label=f"DRUM-6  ({drum6_acc*100:.1f}%)")

# Float32 reference line
ax.axhline(FLOAT32_BASE * 100, color=C_EXACT, ls="--", lw=1.2,
           alpha=0.7, label=f"Float32 baseline ({FLOAT32_BASE*100:.2f}%)")

# Annotate the key comparison points
ax.annotate(f"trunc K=6\n{trunc_acc[-1]*100:.1f}% (collapse)",
            xy=(6, trunc_acc[-1] * 100),
            xytext=(4.2, 22), fontsize=8,
            arrowprops=dict(arrowstyle="->", color=C_TRUNC, lw=1.0),
            color=C_TRUNC, fontweight="bold")
ax.annotate(f"round K=6\n{round_acc[-1]*100:.1f}% (+0.0pp)",
            xy=(6, round_acc[-1] * 100),
            xytext=(4.5, 77), fontsize=8,
            arrowprops=dict(arrowstyle="->", color=C_ROUND, lw=1.0),
            color=C_ROUND, fontweight="bold")

ax.set_xlabel("Truncation depth K")
ax.set_ylabel("Top-1 Accuracy (%)")
ax.set_title(
    "Fig. 5: CIFAR-10 INT8 accuracy vs truncation depth K\n"
    "SimpleCNN  |  INT8 PTQ  |  10,000 test images",
    fontsize=10)
ax.set_xticks(range(0, 8))
ax.set_ylim(0, 95)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.legend(framealpha=0.92, loc="lower left")
ax.grid(True, ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "fig5_cifar10_accuracy.png"), dpi=200)
plt.close(fig)
print("  saved fig5_cifar10_accuracy.png")

print("\nAll figures saved to:", FIGDIR)
