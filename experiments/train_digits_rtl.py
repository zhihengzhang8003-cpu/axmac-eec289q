"""Train 64→16→10 MLP on sklearn.datasets.load_digits() and export RTL golden vectors.

The trained model's INTEGER representation replaces the random-weight golden
vectors in rtl/golden/mlp_toy/, enabling real-digit RTL validation against
a model that actually classifies handwritten digit images.

Usage:
    py -3.14 experiments/train_digits_rtl.py

Outputs:
    experiments/results/digits_mlp.pth
    experiments/results/digits_mlp_weights.npz
    rtl/golden/mlp_toy/x.csv / x.mem           -- one real test image (INT8)
    rtl/golden/mlp_toy/w0.csv, b0.csv, ...     -- layer 0/1 INT8 weights + biases
    rtl/golden/mlp_toy/w0.mem, b0.mem, ...     -- $readmemh hex for Verilog
    rtl/golden/mlp_toy/y_K0_trunc.csv  (and K2, K4, each in trunc + round)
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

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── paths ─────────────────────────────────────────────────────────────────────
RES_DIR    = _REPO_ROOT / "experiments" / "results"
GOLDEN_DIR = _REPO_ROOT / "rtl" / "golden" / "mlp_toy"
RES_DIR.mkdir(parents=True, exist_ok=True)
GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

CKPT = RES_DIR / "digits_mlp.pth"

# ── hypers ────────────────────────────────────────────────────────────────────
DIMS   = [(64, 16), (16, 10)]   # must match RTL mlp_top.v
EPOCHS = 200
LR     = 1e-3
SEED   = 42

# Pixel values in sklearn digits are integers 0–16.
# Map to INT8 range [0, 127] for the RTL.
X_SCALE = 127.0 / 16.0         # 7.9375


# ── helpers ───────────────────────────────────────────────────────────────────

def _quant_weights(w_float: np.ndarray) -> np.ndarray:
    """Symmetric INT8 quantization: scale to fill [-127, 127]."""
    peak = np.abs(w_float).max()
    if peak == 0:
        return np.zeros_like(w_float, dtype=np.int64)
    scale = peak / 127.0
    return np.clip(np.rint(w_float / scale).astype(np.int64), -127, 127)


def _quant_bias(b_float: np.ndarray) -> np.ndarray:
    """Symmetric INT8 bias quantization (independent axis from weights)."""
    peak = np.abs(b_float).max()
    if peak == 0:
        return np.zeros_like(b_float, dtype=np.int64)
    scale = peak / 63.0         # use half range for numerical headroom
    return np.clip(np.rint(b_float / scale).astype(np.int64), -127, 127)


def _write_csv(path: Path, m: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        for row in np.atleast_2d(m):
            w.writerow([int(v) for v in row])


def _write_mem(path: Path, m: np.ndarray) -> None:
    """Flat row-major two's-complement hex for $readmemh."""
    with path.open("w") as f:
        for v in np.atleast_2d(m).flatten():
            f.write(f"{int(v) & 0xFF:02x}\n")


# ── PyTorch model ─────────────────────────────────────────────────────────────

def _build_model():
    """64→16 (ReLU) → 10  — matches RTL mlp_top topology."""
    layers = []
    for i, (d_in, d_out) in enumerate(DIMS):
        layers.append(nn.Linear(d_in, d_out, bias=True))
        if i < len(DIMS) - 1:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def train(model, X_tr_t, y_tr_t, X_te_t, y_te_t):
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training 64→16→10 MLP on sklearn digits for {EPOCHS} epochs …")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()
        sched.step()
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(X_te_t).argmax(1) == y_te_t).float().mean().item()
            print(f"  epoch {epoch:3d}/{EPOCHS}  test_acc={acc*100:.2f}%")

    model.eval()
    with torch.no_grad():
        acc = (model(X_te_t).argmax(1) == y_te_t).float().mean().item()
    print(f"Float32 accuracy: {acc*100:.2f}%")
    return acc


# ── INT8 inference + accuracy sweep ──────────────────────────────────────────

def _int8_model_from_torch(model):
    """Return (x_int_test, layers_int) from trained PyTorch model."""
    w0 = model[0].weight.data.numpy().T  # (64, 16)
    b0 = model[0].bias.data.numpy()      # (16,)
    w1 = model[2].weight.data.numpy().T  # (16, 10)
    b1 = model[2].bias.data.numpy()      # (10,)
    return (
        _quant_weights(w0), _quant_bias(b0),
        _quant_weights(w1), _quant_bias(b1),
    )


def eval_int8_accuracy(X_int, y, w0i, b0i, w1i, b1i, K=0, mode="trunc"):
    layers = [(w0i, b0i), (w1i, b1i)]
    correct = 0
    for i in range(len(X_int)):
        x = X_int[i:i+1]
        out = tiny_mlp_forward(x, layers, fmt=INT8, K=K, rounding=mode)
        if int(out.argmax()) == int(y[i]):
            correct += 1
    return correct / len(y)


# ── golden file export ────────────────────────────────────────────────────────

def export_golden(x_sample: np.ndarray, true_label: int,
                  w0i, b0i, w1i, b1i) -> None:
    """Write all mlp_toy/ golden files from one INT8 input + trained weights."""
    layers = [(w0i, b0i), (w1i, b1i)]

    # ── weight / bias files ────────────────────────────────────────────────
    _write_csv(GOLDEN_DIR / "w0.csv", w0i)
    _write_csv(GOLDEN_DIR / "b0.csv", b0i.reshape(1, -1))
    _write_csv(GOLDEN_DIR / "w1.csv", w1i)
    _write_csv(GOLDEN_DIR / "b1.csv", b1i.reshape(1, -1))

    _write_mem(GOLDEN_DIR / "w0.mem", w0i)
    _write_mem(GOLDEN_DIR / "b0.mem", b0i)
    _write_mem(GOLDEN_DIR / "w1.mem", w1i)
    _write_mem(GOLDEN_DIR / "b1.mem", b1i)

    # ── input vector ───────────────────────────────────────────────────────
    _write_csv(GOLDEN_DIR / "x.csv", x_sample)
    _write_mem(GOLDEN_DIR / "x.mem", x_sample)

    # ── reference outputs for each (K, mode) ──────────────────────────────
    for K in (0, 2, 4):
        for mode in ("trunc", "round"):
            y = tiny_mlp_forward(x_sample, layers, fmt=INT8, K=K, rounding=mode)
            _write_csv(GOLDEN_DIR / f"y_K{K}_{mode}.csv", y)
            pred = int(y.argmax())
            print(f"    K={K} {mode:5s}: pred={pred}  logits={y.flatten().tolist()}")

    print(f"\n  True label: {true_label}")
    print(f"  Wrote {sum(1 for _ in GOLDEN_DIR.iterdir())} files → {GOLDEN_DIR}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not HAS_SKLEARN:
        sys.exit("sklearn not found — run: pip install scikit-learn")
    if not HAS_TORCH:
        sys.exit("PyTorch not found — run: pip install torch")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── load dataset ──────────────────────────────────────────────────────
    digits  = load_digits()
    X_raw   = digits.data.astype(np.float32)    # (1797, 64), values 0-16
    y_all   = digits.target                       # (1797,)

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y_all, test_size=0.2, random_state=SEED
    )

    # Scale to INT8 range for training input (so weights are INT8-compatible)
    X_tr_f = torch.tensor(X_tr_raw * X_SCALE / 127.0)   # back to [0,1] for stable training
    X_te_f = torch.tensor(X_te_raw * X_SCALE / 127.0)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    y_te_t = torch.tensor(y_te, dtype=torch.long)

    # ── train ─────────────────────────────────────────────────────────────
    model = _build_model()
    if CKPT.exists():
        print(f"Loading checkpoint from {CKPT.name}")
        model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            acc = (model(X_te_f).argmax(1) == y_te_t).float().mean().item()
        print(f"Float32 accuracy: {acc*100:.2f}%")
    else:
        train(model, X_tr_f, y_tr_t, X_te_f, y_te_t)
        torch.save(model.state_dict(), CKPT)
        print(f"Checkpoint → {CKPT.name}")

    # ── quantize weights ──────────────────────────────────────────────────
    w0i, b0i, w1i, b1i = _int8_model_from_torch(model)

    # INT8 input: x_int = round(pixel * X_SCALE), clipped to [0, 127]
    X_te_int = np.clip(
        np.rint(X_te_raw * X_SCALE).astype(np.int64), 0, INT8.max_val
    )

    print("\n--- INT8 accuracy sweep (test set) ---")
    for K in (0, 2, 4):
        for mode in ("trunc", "round"):
            t0  = time.time()
            acc = eval_int8_accuracy(X_te_int, y_te, w0i, b0i, w1i, b1i,
                                     K=K, mode=mode)
            print(f"  K={K} {mode:5s}: {acc*100:.2f}%  ({time.time()-t0:.1f}s)")

    # Save weights array
    np.savez(RES_DIR / "digits_mlp_weights.npz",
             w0=w0i, b0=b0i, w1=w1i, b1=b1i,
             x_scale=np.float64(X_SCALE))
    print(f"\nWeights → digits_mlp_weights.npz")

    # ── pick golden sample ────────────────────────────────────────────────
    # Choose first test image correctly classified by INT8 K=0 trunc (argmax match)
    layers = [(w0i, b0i), (w1i, b1i)]
    golden_idx = None
    for i in range(len(X_te_int)):
        x = X_te_int[i:i+1]
        out = tiny_mlp_forward(x, layers, fmt=INT8, K=0, rounding="trunc")
        if int(out.argmax()) == int(y_te[i]):
            golden_idx = i
            break

    if golden_idx is None:
        print("WARNING: no correctly classified sample at K=0 trunc; using sample 0")
        golden_idx = 0

    x_sample   = X_te_int[golden_idx:golden_idx+1]   # shape (1, 64)
    true_label = int(y_te[golden_idx])
    print(f"\n--- Exporting golden vectors (test sample #{golden_idx}, label={true_label}) ---")
    print(f"  Pixel values (INT8): {x_sample.flatten().tolist()}")
    export_golden(x_sample, true_label, w0i, b0i, w1i, b1i)


if __name__ == "__main__":
    main()
