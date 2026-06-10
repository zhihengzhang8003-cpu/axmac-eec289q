"""Fashion-MNIST MLP with AxMAC injection — same architecture as MNIST experiment.

Dataset  : Fashion-MNIST (torchvision auto-download, ~30 MB)
           10 classes: T-shirt, Trouser, Pullover, Dress, Coat,
                       Sandal, Shirt, Sneaker, Bag, Ankle boot
Input    : 28×28 grayscale → flattened 784
Model    : 784 → 256 (ReLU) → 128 (ReLU) → 64 (ReLU) → 10
PTQ      : symmetric INT8, per-layer activation calibration
Outputs  : experiments/results/fashion_mlp.pth
           experiments/results/fashion_accuracy_vs_K.csv
           experiments/results/figures/fig_fashion_accuracy_vs_k.png

Usage:
    py -3.14 experiments/train_fashion_mlp.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from axmac.dnn_inference import quantize_to_int, int_linear_approx
from axmac.exact_mac import INT8

EPS = 1e-8

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
RES_DIR  = _REPO_ROOT / "experiments" / "results"
FIG_DIR  = RES_DIR / "figures"
CACHE    = RES_DIR / "fashion_data"
CKPT     = RES_DIR / "fashion_mlp.pth"
CSV_OUT  = RES_DIR / "fashion_accuracy_vs_K.csv"
FIG_OUT  = FIG_DIR / "fig_fashion_accuracy_vs_k.png"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

# ── hypers ────────────────────────────────────────────────────────────────────
DIMS   = [(784, 256), (256, 128), (128, 64), (64, 10)]
EPOCHS = 15
LR     = 1e-3
BATCH  = 256
SEED   = 42

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal",  "Shirt",   "Sneaker",  "Bag",   "Ankle boot"]

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── dataset ───────────────────────────────────────────────────────────────────

def _load_fashion():
    tfm = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    tr_ds = torchvision.datasets.FashionMNIST(
        root=str(CACHE), train=True,  download=True, transform=tfm)
    te_ds = torchvision.datasets.FashionMNIST(
        root=str(CACHE), train=False, download=True, transform=tfm)
    print(f"FashionMNIST: train={len(tr_ds)}  test={len(te_ds)}")
    return tr_ds, te_ds


# ── model ─────────────────────────────────────────────────────────────────────

def _build_model():
    layers = []
    for i, (d_in, d_out) in enumerate(DIMS):
        layers.append(nn.Linear(d_in, d_out, bias=True))
        if i < len(DIMS) - 1:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def train(model, tr_ds, te_ds):
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=512,  shuffle=False, num_workers=0)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training 784→256→128→64→10 MLP for {EPOCHS} epochs …")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in tr_loader:
            x = x.view(x.size(0), -1)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        sched.step()
        if epoch % 5 == 0 or epoch == EPOCHS:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in te_loader:
                    x = x.view(x.size(0), -1)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total   += len(y)
            print(f"  epoch {epoch:2d}/{EPOCHS}  test_acc={correct/total*100:.2f}%")

    torch.save(model.state_dict(), CKPT)
    print(f"Checkpoint → {CKPT.name}")


# ── PTQ ───────────────────────────────────────────────────────────────────────

def _calibrate(model, te_ds):
    """Per-layer input activation scale (max-abs / 127) over 2000 images."""
    loader  = DataLoader(te_ds, batch_size=2000, shuffle=False, num_workers=0)
    x_b, _  = next(iter(loader))
    h       = x_b.view(x_b.size(0), -1)
    linears = [m for m in model if isinstance(m, nn.Linear)]
    scales  = {}
    with torch.no_grad():
        for i, lin in enumerate(linears):
            scales[f"layer{i}_in"] = max(float(h.abs().max()), EPS) / 127.0
            h = torch.relu(lin(h)) if i < len(linears) - 1 else lin(h)
    return scales


def _quantize_model(model, scales):
    """Returns list of (w_int, b_int, sw, sb) with per-layer scale info."""
    linears = [m for m in model if isinstance(m, nn.Linear)]
    result  = []
    for i, lin in enumerate(linears):
        sx = scales[f"layer{i}_in"]
        sw = max(float(lin.weight.data.abs().max()), EPS) / 127.0
        sb = sx * sw
        w_int = np.clip(np.rint(lin.weight.data.numpy().T / sw),
                        -127, 127).astype(np.int64)   # (in, out)
        b_int = np.rint(lin.bias.data.numpy() / sb).astype(np.int64)
        result.append((w_int, b_int, sw, sb))
    return result


# ── INT8 forward with dequant → ReLU → requant between layers ─────────────────

def _int8_forward(x_np, qw_list, scales, K, mode):
    """Proper PTQ forward: dequantises after each layer before re-quantising."""
    n_layers = len(qw_list)
    sx0 = scales["layer0_in"]
    h   = np.clip(np.rint(x_np / sx0), -127, 127).astype(np.int64)

    for i, (w_int, b_int, sw, sb) in enumerate(qw_list):
        h = int_linear_approx(h, w_int, fmt=INT8, K=K, bias=b_int, rounding=mode)
        if i < n_layers - 1:
            sx_out = scales[f"layer{i+1}_in"]
            h_f    = np.maximum(h.astype(np.float64) * sb, 0.0)
            h      = np.clip(np.rint(h_f / sx_out), -127, 127).astype(np.int64)
    return h   # raw int64 logits


# ── accuracy sweep ────────────────────────────────────────────────────────────

def eval_int8_accuracy(X_np, y_np, scales, qw_list, K, mode):
    logits = _int8_forward(X_np, qw_list, scales, K, mode)
    return (logits.argmax(axis=1) == y_np).mean()


# ── figure ────────────────────────────────────────────────────────────────────

def _make_figure(Ks, trunc_acc, round_acc, float_acc):
    C_TRUNC = "#C0392B"
    C_ROUND = "#1565C0"

    fig, ax = plt.subplots(figsize=(7.0, 4.6), facecolor="white", dpi=150)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.13)
    ax.set_facecolor("white")
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(0.8)
    ax.tick_params(colors="black", labelsize=11)

    ax.plot(Ks, [v*100 for v in trunc_acc], "o--", color=C_TRUNC,
            linewidth=2.0, markersize=6, label="trunc", zorder=4)
    ax.plot(Ks, [v*100 for v in round_acc], "o-",  color=C_ROUND,
            linewidth=2.5, markersize=6, label="round", zorder=5)
    ax.axhline(float_acc*100, color="#888888", linewidth=1.0,
               linestyle="--", zorder=2, label=f"float32 ({float_acc*100:.1f}%)")
    ax.axhline(10, color="#BBBBBB", linewidth=0.8, linestyle=":", zorder=1)
    ax.text(max(Ks)-0.1, 11.5, "chance (10%)", color="#AAAAAA",
            fontsize=8.5, ha="right")

    k8 = Ks.index(8)
    for val, color, va in [(trunc_acc[k8]*100, C_TRUNC, "top"),
                            (round_acc[k8]*100, C_ROUND, "bottom")]:
        ax.text(8.08, val, f"{val:.1f}%", color=color,
                fontsize=9, fontweight="bold", va=va, ha="left")

    ax.set_xlim(-0.3, 8.9)
    ax.set_ylim(5, 100)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks], color="black", fontsize=11)
    ax.set_yticks(range(10, 105, 10))
    ax.set_yticklabels([f"{v}%" for v in range(10, 105, 10)],
                       color="black", fontsize=10)
    ax.set_xlabel("Truncation depth  K  (bits removed per partial product)",
                  color="black", fontsize=11, labelpad=7)
    ax.set_ylabel("Top-1 accuracy  (Fashion-MNIST, 10 classes)",
                  color="black", fontsize=11, labelpad=7)
    ax.set_title("Accuracy vs. K  ·  INT8 PTQ  ·  Fashion-MNIST  ·  784→256→128→64→10 MLP",
                 color="black", fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="lower left", facecolor="white", edgecolor="#AAAAAA",
              labelcolor="black", fontsize=10.5, framealpha=0.95)

    # inset zoom K=6-8
    axins = ax.inset_axes([0.32, 0.08, 0.38, 0.42])
    zi    = [Ks.index(k) for k in [6, 7, 8]]
    axins.plot([6,7,8], [trunc_acc[i]*100 for i in zi], "o--",
               color=C_TRUNC, linewidth=1.8, markersize=6)
    axins.plot([6,7,8], [round_acc[i]*100 for i in zi], "o-",
               color=C_ROUND, linewidth=2.0, markersize=6)
    axins.set_facecolor("white")
    axins.set_xticks([6, 7, 8])
    axins.set_xticklabels(["K=6", "K=7", "K=8"], fontsize=8.5, color="black")
    axins.yaxis.grid(True, color="#EEEEEE", linewidth=0.5)
    for sp in axins.spines.values():
        sp.set_color("#AAAAAA"); sp.set_linewidth(0.8)
    axins.set_title("K = 6–8 (detail)", fontsize=8.5, color="black", pad=4)
    for val, color, dy in [(trunc_acc[zi[-1]]*100, C_TRUNC, -4),
                            (round_acc[zi[-1]]*100, C_ROUND,  1)]:
        axins.text(8.05, val+dy, f"{val:.1f}%", color=color,
                   fontsize=7.5, fontweight="bold", va="center")
    ax.indicate_inset_zoom(axins, edgecolor="#AAAAAA", linewidth=0.8)

    fig.savefig(FIG_OUT, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {FIG_OUT.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    tr_ds, te_ds = _load_fashion()

    model = _build_model()
    if CKPT.exists():
        print(f"Loading checkpoint {CKPT.name}")
        model.load_state_dict(torch.load(CKPT, map_location="cpu",
                                         weights_only=True))
    else:
        train(model, tr_ds, te_ds)
    model.eval()

    # float test accuracy
    te_loader = DataLoader(te_ds, batch_size=512, shuffle=False, num_workers=0)
    correct = total = 0
    with torch.no_grad():
        for x, y in te_loader:
            x = x.view(x.size(0), -1)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += len(y)
    float_acc = correct / total
    print(f"\nFloat32 test accuracy: {float_acc*100:.2f}%")

    # PTQ
    scales     = _calibrate(model, te_ds)
    qw_list    = _quantize_model(model, scales)
    print(f"Activation scales: {[f'{scales[k]:.5f}' for k in sorted(scales)]}")

    # test set arrays (all 10 000)
    te_loader2 = DataLoader(te_ds, batch_size=10000, shuffle=False, num_workers=0)
    x_b, y_b   = next(iter(te_loader2))
    X_np = x_b.view(x_b.size(0), -1).numpy()
    y_np = y_b.numpy()

    # K sweep
    Ks        = list(range(9))
    trunc_acc = []
    round_acc = []

    print("\n--- Accuracy vs K sweep ---")
    print(f"  {'K':>3}  {'trunc':>8}  {'round':>8}")
    print("  " + "-"*25)
    for K in Ks:
        t0 = time.time()
        ta = eval_int8_accuracy(X_np, y_np, scales, qw_list, K, "trunc")
        ra = eval_int8_accuracy(X_np, y_np, scales, qw_list, K, "round")
        trunc_acc.append(ta)
        round_acc.append(ra)
        print(f"  {K:>3}  {ta*100:>7.2f}%  {ra*100:>7.2f}%  ({time.time()-t0:.1f}s)")

    with CSV_OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "trunc_acc", "round_acc"])
        for K, ta, ra in zip(Ks, trunc_acc, round_acc):
            w.writerow([K, f"{ta:.4f}", f"{ra:.4f}"])
    print(f"\nSaved → {CSV_OUT.name}")

    _make_figure(Ks, trunc_acc, round_acc, float_acc)
    print("All done.")


if __name__ == "__main__":
    main()
