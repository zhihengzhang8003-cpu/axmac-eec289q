"""Binary cat-vs-dog MLP on Oxford-IIIT Pet with AxMAC injection.

Dataset  : Oxford-IIIT Pet (torchvision auto-download, ~800 MB first run)
           37 breeds → binary-category: cat=0, dog=1
Input    : 32×32 grayscale, flattened → 1024 features
Model    : 1024 → 256 (ReLU) → 64 (ReLU) → 2
PTQ      : symmetric INT8, per-layer calibration (same as train_mnist_mlp.py)
Outputs  : experiments/results/catdog_mlp.pth
           experiments/results/catdog_accuracy_vs_K.csv
           experiments/results/figures/fig_catdog_accuracy_vs_k.png

Usage:
    py -3.14 experiments/train_catdog_mlp.py
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

from axmac.dnn_inference import quantize_to_int, tiny_mlp_forward
from axmac.exact_mac import INT8

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
RES_DIR  = _REPO_ROOT / "experiments" / "results"
FIG_DIR  = RES_DIR / "figures"
CACHE    = RES_DIR / "pet_data"
CKPT     = RES_DIR / "catdog_mlp.pth"
CSV_OUT  = RES_DIR / "catdog_accuracy_vs_K.csv"
FIG_OUT  = FIG_DIR / "fig_catdog_accuracy_vs_k.png"
RES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

# ── hypers ────────────────────────────────────────────────────────────────────
IMG_SIZE  = 32
IN_DIMS   = IMG_SIZE * IMG_SIZE          # 1024  (grayscale)
DIMS      = [(IN_DIMS, 256), (256, 64), (64, 2)]
EPOCHS    = 30
LR        = 3e-4
BATCH     = 128
SEED      = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── dataset ───────────────────────────────────────────────────────────────────

def _load_pet():
    tfm = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.Grayscale(),
        T.ToTensor(),                    # → [0, 1]
        T.Normalize((0.5,), (0.5,)),     # → [-1, 1]
    ])
    print("Loading Oxford-IIIT Pet (downloading if needed) …")
    full = torchvision.datasets.OxfordIIITPet(
        root=str(CACHE), split="trainval",
        target_types="binary-category",
        download=True, transform=tfm,
    )
    test_ds = torchvision.datasets.OxfordIIITPet(
        root=str(CACHE), split="test",
        target_types="binary-category",
        download=True, transform=tfm,
    )
    n_val = max(200, len(full) // 10)
    n_tr  = len(full) - n_val
    tr_ds, val_ds = random_split(full, [n_tr, n_val],
                                  generator=torch.Generator().manual_seed(SEED))
    print(f"  train={n_tr}  val={n_val}  test={len(test_ds)}")
    return tr_ds, val_ds, test_ds


# ── model ─────────────────────────────────────────────────────────────────────

def _build_model():
    layers = []
    for i, (d_in, d_out) in enumerate(DIMS):
        layers.append(nn.Linear(d_in, d_out, bias=True))
        if i < len(DIMS) - 1:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _flatten(batch):
    x, y = batch
    return x.view(x.size(0), -1), y


def train(model, tr_ds, val_ds):
    tr_loader  = DataLoader(tr_ds,  batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256,  shuffle=False, num_workers=0)
    opt     = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training 1024→256→64→2 MLP for {EPOCHS} epochs …")
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in tr_loader:
            x, y = _flatten(batch)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        sched.step()
        if epoch % 5 == 0 or epoch == EPOCHS:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for batch in val_loader:
                    x, y = _flatten(batch)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total   += len(y)
            acc = correct / total
            best_acc = max(best_acc, acc)
            print(f"  epoch {epoch:2d}/{EPOCHS}  val_acc={acc*100:.1f}%")

    torch.save(model.state_dict(), CKPT)
    print(f"Checkpoint → {CKPT.name}  (best_val={best_acc*100:.1f}%)")
    return model


# ── PTQ calibration ───────────────────────────────────────────────────────────

def _calibrate(model, val_ds):
    """Per-layer input activation scale: max |x| / 127 across 512 val samples."""
    loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)
    x_batch, _ = _flatten(next(iter(loader)))

    scales = []
    h = x_batch
    linears = [m for m in model if isinstance(m, nn.Linear)]
    relus   = [m for m in model if isinstance(m, nn.ReLU)]
    for i, lin in enumerate(linears):
        scale = float(h.abs().max()) / 127.0
        scales.append(scale)
        with torch.no_grad():
            h = lin(h)
            if i < len(relus):
                h = torch.relu(h)
    return scales


def _quantize_model(model, act_scales):
    linears = [m for m in model if isinstance(m, nn.Linear)]
    layers_int = []
    for i, lin in enumerate(linears):
        sw = float(lin.weight.data.abs().max()) / 127.0
        sb = act_scales[i] * sw
        w_int = np.clip(np.rint(lin.weight.data.numpy().T / sw), -127, 127).astype(np.int64)
        b_int = np.clip(np.rint(lin.bias.data.numpy()   / sb), -127, 127).astype(np.int64)
        layers_int.append((w_int, b_int))
    return layers_int


# ── INT8 accuracy sweep ───────────────────────────────────────────────────────

def _to_int8(x_float, act_scale):
    return quantize_to_int(x_float / act_scale, INT8)


def eval_int8_accuracy(X_np, y_np, act_scale0, layers_int, K, mode):
    X_int = _to_int8(X_np, act_scale0)
    correct = 0
    for i in range(len(X_int)):
        out = tiny_mlp_forward(X_int[i:i+1], layers_int, fmt=INT8,
                               K=K, rounding=mode)
        if int(out.argmax()) == int(y_np[i]):
            correct += 1
    return correct / len(y_np)


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
    ax.axhline(50, color="#BBBBBB", linewidth=0.8, linestyle=":", zorder=1)
    ax.text(max(Ks)-0.1, 51.5, "chance (50%)", color="#AAAAAA",
            fontsize=8.5, ha="right")

    # annotate K=8
    k8 = Ks.index(8)
    for val, color, va in [(trunc_acc[k8]*100, C_TRUNC, "top"),
                            (round_acc[k8]*100, C_ROUND, "bottom")]:
        ax.text(8.08, val, f"{val:.1f}%", color=color,
                fontsize=9, fontweight="bold", va=va, ha="left")

    ax.set_xlim(-0.3, 8.9)
    ax.set_ylim(45, 100)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks], color="black", fontsize=11)
    ax.set_yticks(range(45, 105, 5))
    ax.set_yticklabels([f"{v}%" for v in range(45, 105, 5)],
                       color="black", fontsize=10)
    ax.set_xlabel("Truncation depth  K  (bits removed per partial product)",
                  color="black", fontsize=11, labelpad=7)
    ax.set_ylabel("Binary accuracy  (cat vs dog)",
                  color="black", fontsize=11, labelpad=7)
    ax.set_title("Accuracy vs. K  ·  INT8 PTQ  ·  Oxford-IIIT Pet  ·  1024→256→64→2 MLP",
                 color="black", fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="lower left", facecolor="white", edgecolor="#AAAAAA",
              labelcolor="black", fontsize=10.5, framealpha=0.95)

    fig.savefig(FIG_OUT, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure → {FIG_OUT.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    tr_ds, val_ds, test_ds = _load_pet()

    # ── float training ────────────────────────────────────────────────────────
    model = _build_model()
    if CKPT.exists():
        print(f"Loading checkpoint {CKPT.name}")
        model.load_state_dict(torch.load(CKPT, map_location="cpu",
                                         weights_only=True))
    else:
        train(model, tr_ds, val_ds)
    model.eval()

    # float test accuracy
    te_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    correct = total = 0
    with torch.no_grad():
        for batch in te_loader:
            x, y = _flatten(batch)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += len(y)
    float_acc = correct / total
    print(f"\nFloat32 test accuracy: {float_acc*100:.2f}%")

    # ── PTQ ───────────────────────────────────────────────────────────────────
    act_scales  = _calibrate(model, val_ds)
    layers_int  = _quantize_model(model, act_scales)
    act_scale0  = act_scales[0]
    print(f"Activation scales (layer inputs): "
          f"{[f'{s:.5f}' for s in act_scales]}")

    # build test arrays
    te_loader2 = DataLoader(test_ds, batch_size=2000, shuffle=False, num_workers=0)
    x_b, y_b   = _flatten(next(iter(te_loader2)))
    X_np = x_b.numpy()
    y_np = y_b.numpy()

    # ── K sweep ───────────────────────────────────────────────────────────────
    Ks         = list(range(9))       # K = 0..8
    trunc_acc  = []
    round_acc  = []

    print("\n--- Accuracy vs K sweep ---")
    print(f"  {'K':>3}  {'trunc':>8}  {'round':>8}")
    print("  " + "-" * 25)
    for K in Ks:
        t0 = time.time()
        ta = eval_int8_accuracy(X_np, y_np, act_scale0, layers_int, K, "trunc")
        ra = eval_int8_accuracy(X_np, y_np, act_scale0, layers_int, K, "round")
        trunc_acc.append(ta)
        round_acc.append(ra)
        print(f"  {K:>3}  {ta*100:>7.2f}%  {ra*100:>7.2f}%  ({time.time()-t0:.1f}s)")

    # ── save CSV ──────────────────────────────────────────────────────────────
    with CSV_OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "trunc_acc", "round_acc"])
        for K, ta, ra in zip(Ks, trunc_acc, round_acc):
            w.writerow([K, f"{ta:.4f}", f"{ra:.4f}"])
    print(f"\nSaved → {CSV_OUT.name}")

    # ── figure ────────────────────────────────────────────────────────────────
    _make_figure(Ks, trunc_acc, round_acc, float_acc)
    print("All done.")


if __name__ == "__main__":
    main()
