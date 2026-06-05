"""CIFAR-10 experiment: train a SimpleCNN, then evaluate INT8 approximate-MAC
inference across K ∈ {0..6} and rounding modes {trunc, round}.

The script measures the top-1 accuracy drop caused by K-bit per-MAC truncation
(AxMAC) relative to the K=0 exact INT8 baseline. Results are saved to
experiments/results/cifar10_accuracy.csv.

Usage:
    py -3.14 experiments/cifar10_experiment.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError for arrows etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ---------------------------------------------------------------------------
# Path setup – make the project root importable so `axmac` can be found.
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from axmac.dnn_inference import (
    int_conv2d_approx, int_conv2d_drum,
    int_linear_approx, int_linear_drum,
    quantize_to_int,
)
from axmac.exact_mac import INT8

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXPERIMENTS_DIR = project_root / "experiments"
DATA_DIR        = EXPERIMENTS_DIR / "data" / "cifar10"
RESULTS_DIR     = EXPERIMENTS_DIR / "results"
CHECKPOINT_PATH = RESULTS_DIR / "cifar10_cnn.pth"
CSV_PATH        = RESULTS_DIR / "cifar10_accuracy.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
BATCH_TRAIN  = 128
BATCH_EVAL   = 64
EPOCHS       = 30
LR           = 1e-3
CALIB_BATCHES = 8   # number of test batches used for per-layer scale calibration
K_LIST       = [0, 1, 2, 3, 4, 5, 6]
MODE_LIST    = ["trunc", "round"]
DRUM_K_LIST  = [4, 6]   # DRUM-k: k MSBs per operand; k=4 ~ trunc K=4, k=6 ~ trunc K=2
EPS          = 1e-8  # guard against zero scale


# ============================================================
# 1. Model definition (no BatchNorm)
# ============================================================

class SimpleCNN(nn.Module):
    """3-layer conv + 2-layer FC, no BatchNorm."""

    def __init__(self) -> None:
        super().__init__()
        # Conv1: 3→32, 3×3, pad=1 → 32×32×32 → pool → 32×16×16
        self.conv1 = nn.Conv2d(3,   32,  3, padding=1, bias=True)
        # Conv2: 32→64, 3×3, pad=1 → 64×16×16 → pool → 64×8×8
        self.conv2 = nn.Conv2d(32,  64,  3, padding=1, bias=True)
        # Conv3: 64→128, 3×3, pad=1 → 128×8×8 → pool → 128×4×4
        self.conv3 = nn.Conv2d(64,  128, 3, padding=1, bias=True)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU(inplace=True)
        # FC1: 128×4×4=2048 → 256
        self.fc1   = nn.Linear(2048, 256, bias=True)
        # FC2: 256 → 10
        self.fc2   = nn.Linear(256,  10,  bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))  # 32×16×16
        x = self.pool(self.relu(self.conv2(x)))  # 64×8×8
        x = self.pool(self.relu(self.conv3(x)))  # 128×4×4
        x = x.view(x.size(0), -1)               # 2048
        x = self.relu(self.fc1(x))              # 256
        x = self.fc2(x)                          # 10
        return x


# ============================================================
# 2. Data loaders
# ============================================================

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)


def get_loaders():
    """Return (train_loader, test_loader)."""
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=True,  download=True, transform=train_tf)
    test_set  = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=BATCH_TRAIN, shuffle=True,
        num_workers=2, pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(
        test_set,  batch_size=BATCH_EVAL,  shuffle=False,
        num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ============================================================
# 3. Training
# ============================================================

def train(model: SimpleCNN, train_loader, test_loader, device: torch.device) -> None:
    """Train model for EPOCHS epochs; saves checkpoint afterwards."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n=== Training SimpleCNN on CIFAR-10 for {EPOCHS} epochs ===")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        scheduler.step()

        if epoch % 10 == 0:
            acc = evaluate_float(model, test_loader, device)
            avg_loss = running_loss / len(train_loader.dataset)
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  test_acc={acc*100:.2f}%")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Checkpoint saved → {CHECKPOINT_PATH}\n")


def evaluate_float(model: SimpleCNN, loader, device: torch.device) -> float:
    """Top-1 accuracy with float32 exact inference."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)
    return correct / total


# ============================================================
# 4. INT8 PTQ helpers
# ============================================================

def extract_weights(model: SimpleCNN):
    """Return raw float32 numpy arrays for all conv/linear weight+bias."""
    def _np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    return {
        "conv1_w": _np(model.conv1.weight),
        "conv1_b": _np(model.conv1.bias),
        "conv2_w": _np(model.conv2.weight),
        "conv2_b": _np(model.conv2.bias),
        "conv3_w": _np(model.conv3.weight),
        "conv3_b": _np(model.conv3.bias),
        "fc1_w":   _np(model.fc1.weight),
        "fc1_b":   _np(model.fc1.bias),
        "fc2_w":   _np(model.fc2.weight),
        "fc2_b":   _np(model.fc2.bias),
    }


def calibrate(model: SimpleCNN, test_loader, device: torch.device) -> dict[str, float]:
    """Run CALIB_BATCHES of float32 inference and record per-layer input scales.

    Returns a dict with keys: 'input', 'conv1_out', 'conv2_out', 'conv3_out', 'fc1_out'.
    Each value is max(|activation|) / 127 (scale for INT8 quantization).
    """
    model.eval()
    acts: dict[str, list[float]] = {
        "input":    [],
        "conv1_out": [],
        "conv2_out": [],
        "conv3_out": [],
        "fc1_out":  [],
    }

    with torch.no_grad():
        for i, (xb, _) in enumerate(test_loader):
            if i >= CALIB_BATCHES:
                break
            xb = xb.to(device)

            # Record input scale
            acts["input"].append(xb.abs().max().item())

            # Layer-by-layer forward to capture intermediate activations
            out1 = model.pool(model.relu(model.conv1(xb)))  # 32×16×16
            acts["conv1_out"].append(out1.abs().max().item())

            out2 = model.pool(model.relu(model.conv2(out1)))  # 64×8×8
            acts["conv2_out"].append(out2.abs().max().item())

            out3 = model.pool(model.relu(model.conv3(out2)))  # 128×4×4
            acts["conv3_out"].append(out3.abs().max().item())

            flat = out3.view(out3.size(0), -1)
            fc1_out = model.relu(model.fc1(flat))
            acts["fc1_out"].append(fc1_out.abs().max().item())

    # Convert max-abs to per-tensor scale (x / scale ≈ INT8)
    scales: dict[str, float] = {}
    for k, vals in acts.items():
        max_abs = float(np.max(vals)) if vals else 0.0
        scales[k] = max(max_abs / 127.0, EPS)
    return scales


def quantize_weights(float_weights: dict[str, np.ndarray],
                     scales: dict[str, float]) -> dict[str, np.ndarray]:
    """Quantize each layer's weight and bias to INT8 / INT32 respectively.

    Weight scale: scale_w = max(|w|) / 127
    Bias  scale:  scale_b = scale_x * scale_w  (shared across the layer)
    Bias is quantized to int64 so it can be added directly to the INT32 accumulator.

    Returns a dict with keys like 'conv1_w_q', 'conv1_b_q', 'conv1_sw', etc.
    """
    result: dict[str, np.ndarray | float] = {}

    layer_info = [
        # (layer_name, input_scale_key)
        ("conv1", "input"),
        ("conv2", "conv1_out"),
        ("conv3", "conv2_out"),
        ("fc1",   "conv3_out"),
        ("fc2",   "fc1_out"),
    ]

    for lname, sx_key in layer_info:
        w = float_weights[f"{lname}_w"]
        b = float_weights[f"{lname}_b"]
        sx = scales[sx_key]

        max_abs_w = float(np.max(np.abs(w)))
        sw = max(max_abs_w / 127.0, EPS)

        # Quantize weight: round + clip to [-127, 127], then cast to int64
        w_q = np.clip(np.round(w / sw), -127, 127).astype(np.int64)

        # Bias scale = scale_x * scale_w; bias quantized to int64
        sb = sx * sw
        b_q = np.round(b / sb).astype(np.int64)

        result[f"{lname}_w_q"] = w_q
        result[f"{lname}_b_q"] = b_q
        result[f"{lname}_sw"]  = sw
        result[f"{lname}_sb"]  = sb

    return result


# ============================================================
# 5. NumPy maxpool2d (2×2, stride 2)
# ============================================================

def maxpool2x2(x: np.ndarray) -> np.ndarray:
    """2×2 max-pool with stride 2. Input: (N, C, H, W), H and W must be even."""
    n, c, h, w = x.shape
    assert h % 2 == 0 and w % 2 == 0, f"maxpool2x2: H/W must be even, got {h}×{w}"
    x_r = x.reshape(n, c, h // 2, 2, w // 2, 2)
    return x_r.max(axis=(3, 5))


# ============================================================
# 6. INT8 approximate inference (pure NumPy)
# ============================================================

def int8_approx_forward(
    x_float: np.ndarray,
    qw: dict,
    scales: dict[str, float],
    K: int,
    mode: str,
    drum_k: int | None = None,
) -> np.ndarray:
    """Run a full INT8 approximate-MAC forward pass on a batch.

    Parameters
    ----------
    x_float : (N, 3, 32, 32) float32 numpy array
    qw      : quantized weights dict from quantize_weights()
    scales  : activation scales dict from calibrate()
    K       : number of LSBs to drop per MAC product (trunc/round path)
    mode    : rounding mode ('trunc' or 'round'); ignored when drum_k is set
    drum_k  : if not None, use DRUM-k approximation instead of K/mode

    Returns
    -------
    logits_int : (N, 10) int64 array (raw accumulator output of FC2)
    """
    use_drum = drum_k is not None

    def _conv(x, w_key, b_key, sx_in_scale, sw_key):
        if use_drum:
            return int_conv2d_drum(
                x, qw[w_key], fmt=INT8, k=drum_k, bias=qw[b_key], stride=1, padding=1,
            ), sx_in_scale * qw[sw_key]
        else:
            return int_conv2d_approx(
                x, qw[w_key], fmt=INT8, K=K, bias=qw[b_key],
                stride=1, padding=1, rounding=mode,
            ), sx_in_scale * qw[sw_key]

    def _linear(x, w_key, b_key, sx_in_scale, sw_key):
        if use_drum:
            return int_linear_drum(
                x, qw[w_key].T, fmt=INT8, k=drum_k, bias=qw[b_key],
            ), sx_in_scale * qw[sw_key]
        else:
            return int_linear_approx(
                x, qw[w_key].T, fmt=INT8, K=K, bias=qw[b_key], rounding=mode,
            ), sx_in_scale * qw[sw_key]

    # -- Input quantization --
    sx_in = scales["input"]
    x_q = np.clip(np.round(x_float / sx_in), -127, 127).astype(np.int64)

    # ---- Conv1 ----
    out1_int, dq1 = _conv(x_q, "conv1_w_q", "conv1_b_q", sx_in, "conv1_sw")
    out1_f = np.maximum(out1_int.astype(np.float32) * dq1, 0.0)
    out1_f = maxpool2x2(out1_f)
    sx1_out = scales["conv1_out"]
    out1_q = np.clip(np.round(out1_f / sx1_out), -127, 127).astype(np.int64)

    # ---- Conv2 ----
    out2_int, dq2 = _conv(out1_q, "conv2_w_q", "conv2_b_q", sx1_out, "conv2_sw")
    out2_f = np.maximum(out2_int.astype(np.float32) * dq2, 0.0)
    out2_f = maxpool2x2(out2_f)
    sx2_out = scales["conv2_out"]
    out2_q = np.clip(np.round(out2_f / sx2_out), -127, 127).astype(np.int64)

    # ---- Conv3 ----
    out3_int, dq3 = _conv(out2_q, "conv3_w_q", "conv3_b_q", sx2_out, "conv3_sw")
    out3_f = np.maximum(out3_int.astype(np.float32) * dq3, 0.0)
    out3_f = maxpool2x2(out3_f)
    sx3_out = scales["conv3_out"]
    out3_q = np.clip(np.round(out3_f / sx3_out), -127, 127).astype(np.int64)

    # ---- Flatten ----
    flat_q = out3_q.reshape(out3_q.shape[0], -1)  # (N, 2048)

    # ---- FC1 ----
    fc1_int, dq4 = _linear(flat_q, "fc1_w_q", "fc1_b_q", sx3_out, "fc1_sw")
    fc1_f = np.maximum(fc1_int.astype(np.float32) * dq4, 0.0)
    sx4_out = scales["fc1_out"]
    fc1_q = np.clip(np.round(fc1_f / sx4_out), -127, 127).astype(np.int64)

    # ---- FC2 ----
    logits_int, _ = _linear(fc1_q, "fc2_w_q", "fc2_b_q", sx4_out, "fc2_sw")
    return logits_int


# ============================================================
# 7. Evaluation loop
# ============================================================

def evaluate_int8(
    test_loader,
    qw: dict,
    scales: dict[str, float],
    K: int,
    mode: str,
) -> float:
    """Evaluate INT8 approx inference on the full test set. Returns top-1 accuracy."""
    correct = total = 0
    for xb_t, yb_t in test_loader:
        x_np = xb_t.numpy()   # (N, 3, 32, 32) float32
        y_np = yb_t.numpy()   # (N,)

        logits = int8_approx_forward(x_np, qw, scales, K, mode)
        preds  = logits.argmax(axis=1)
        correct += int((preds == y_np).sum())
        total   += y_np.shape[0]

    return correct / total


# ============================================================
# 8. Main
# ============================================================

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_loaders()

    # ---- Build / load model ----
    model = SimpleCNN().to(device)

    if CHECKPOINT_PATH.exists():
        print(f"Checkpoint found at {CHECKPOINT_PATH} — skipping training.")
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    else:
        train(model, train_loader, test_loader, device)

    # Float32 baseline accuracy
    float_acc = evaluate_float(model, test_loader, device)
    print(f"\nFloat32 test accuracy: {float_acc*100:.2f}%")

    # ---- Calibration ----
    print("\nCalibrating per-layer activation scales …")
    scales = calibrate(model, test_loader, device)
    for k, v in scales.items():
        print(f"  scale[{k}] = {v:.6f}")

    # ---- Weight quantization ----
    float_weights = extract_weights(model)
    qw = quantize_weights(float_weights, scales)

    # Use CPU data loader (numpy) for INT8 eval
    test_loader_cpu = torch.utils.data.DataLoader(
        test_loader.dataset, batch_size=BATCH_EVAL, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # ---- K=0 exact INT8 baseline (single evaluation, mode irrelevant) ----
    print("\nEvaluating K=0 (exact INT8 baseline) …")
    t0 = time.time()
    exact_acc = evaluate_int8(test_loader_cpu, qw, scales, K=0, mode="trunc")
    print(f"  K=0 exact  accuracy = {exact_acc*100:.2f}%  ({time.time()-t0:.1f}s)")

    # ---- Sweep K and mode ----
    rows: list[dict] = []
    rows.append({"K": 0, "mode": "exact", "accuracy": exact_acc,
                 "delta_from_exact": 0.0})

    # Summary table data: {K: {mode: acc}}
    summary: dict[int, dict[str, float]] = {K: {} for K in K_LIST}
    summary[0]["trunc"] = exact_acc
    summary[0]["round"] = exact_acc

    print(f"\n{'K':>3}  {'mode':<8}  {'accuracy':>8}  {'delta':>8}")
    print("-" * 38)
    print(f"  0  {'exact':<8}  {exact_acc*100:7.2f}%  {'0.00pp':>8}")

    for K in K_LIST:
        if K == 0:
            continue  # already computed
        for mode in MODE_LIST:
            t0 = time.time()
            acc = evaluate_int8(test_loader_cpu, qw, scales, K=K, mode=mode)
            delta = (acc - exact_acc) * 100
            elapsed = time.time() - t0
            print(f"  {K}  {mode:<10}  {acc*100:7.2f}%  {delta:+.2f}pp  ({elapsed:.1f}s)")
            rows.append({"K": K, "mode": mode, "accuracy": acc,
                         "delta_from_exact": delta})
            summary[K][mode] = acc

    # ---- DRUM sweep (fixed k values, no K parameter) ----
    print("\n-- DRUM-k evaluation --")
    drum_summary: dict[int, float] = {}
    for dk in DRUM_K_LIST:
        t0 = time.time()
        # evaluate_int8 passes K and mode; for DRUM we add drum_k via the loader
        correct = total = 0
        for xb_t, yb_t in test_loader_cpu:
            x_np, y_np = xb_t.numpy(), yb_t.numpy()
            logits = int8_approx_forward(x_np, qw, scales, K=0, mode="trunc",
                                          drum_k=dk)
            correct += int((logits.argmax(axis=1) == y_np).sum())
            total += y_np.shape[0]
        acc = correct / total
        delta = (acc - exact_acc) * 100
        elapsed = time.time() - t0
        label = f"drum-{dk}"
        print(f"  -  {label:<10}  {acc*100:7.2f}%  {delta:+.2f}pp  ({elapsed:.1f}s)")
        rows.append({"K": f"drum-{dk}", "mode": "drum", "accuracy": acc,
                     "delta_from_exact": delta})
        drum_summary[dk] = acc

    # ---- Write CSV ----
    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["K", "mode", "accuracy", "delta_from_exact"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved → {CSV_PATH}")

    # ---- Summary table ----
    drum_cols = "".join(f" | {'drum-'+str(dk):>8}" for dk in DRUM_K_LIST)
    header = f"{'K':<4} | {'trunc':>7} | {'round':>7} | {'round-trunc':>11}" + drum_cols
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for K in K_LIST:
        t_acc = summary[K].get("trunc", exact_acc if K == 0 else float("nan"))
        r_acc = summary[K].get("round", exact_acc if K == 0 else float("nan"))
        delta_str = f"{(r_acc - t_acc)*100:+.1f}pp"
        drum_vals = "".join(
            f" | {drum_summary.get(dk, float('nan'))*100:7.1f}%" for dk in DRUM_K_LIST
        ) if K == 0 else "".join(f" | {'  -':>8}" for _ in DRUM_K_LIST)
        print(f"{K:<4} | {t_acc*100:6.1f}% | {r_acc*100:6.1f}% | {delta_str:>11}" + drum_vals)
    # Print DRUM rows separately
    for dk in DRUM_K_LIST:
        d_acc = drum_summary.get(dk, float("nan"))
        d_delta = f"{(d_acc - exact_acc)*100:+.1f}pp"
        print(f"drum-{dk:<2} {'':>8} {'':>8} {d_delta:>11}" +
              "".join(f" | {drum_summary.get(dk2, float('nan'))*100:7.1f}%" if dk2 == dk
                      else f" | {'':>8}" for dk2 in DRUM_K_LIST))
    print("=" * len(header))


if __name__ == "__main__":
    main()
