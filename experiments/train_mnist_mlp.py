"""Train a 784→256→128→64→10 MLP on MNIST, post-training quantize to INT8,
and regenerate truncation-injection experiment CSVs.

Regenerates:
    experiments/results/mnist_mlp_weights.npz   -- INT8 quantized weights
    experiments/results/accuracy_vs_K.csv        -- MNIST Top-1 accuracy vs K
    experiments/results/bias_accumulation.csv    -- bias stats with real MNIST inputs
    experiments/results/layer_allocation.csv     -- sensitivity allocation with real inputs

Usage:
    py -3.14 experiments/train_mnist_mlp.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from axmac.dnn_inference import int_linear_approx, tiny_mlp_forward
from axmac.exact_mac import INT8
from axmac.sensitivity import (
    allocate_K, layer_mac_counts, layer_sensitivity,
    output_divergence, uniform_K,
)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = _REPO / "experiments" / "data" / "mnist"
RESULTS_DIR = _REPO / "experiments" / "results"
CKPT        = RESULTS_DIR / "mnist_mlp.pth"
WEIGHTS_NPZ = RESULTS_DIR / "mnist_mlp_weights.npz"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── hyper-params ──────────────────────────────────────────────────────────────
EPOCHS       = 15
LR           = 1e-3
BATCH        = 256
CALIB_N      = 2000   # calibration images for activation scale
K_MAX        = 8
EPS          = 1e-8

# ── topology ──────────────────────────────────────────────────────────────────
DIMS = [(784, 256), (256, 128), (128, 64), (64, 10)]


# ════════════════════════════════════════════════════════════════════════════
# 1.  PyTorch model
# ════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers = []
        for i, (d_in, d_out) in enumerate(DIMS):
            layers.append(nn.Linear(d_in, d_out, bias=True))
            if i < len(DIMS) - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


# ════════════════════════════════════════════════════════════════════════════
# 2.  Data
# ════════════════════════════════════════════════════════════════════════════

def get_loaders():
    tf = transforms.Compose([transforms.ToTensor()])
    train_set = torchvision.datasets.MNIST(str(DATA_DIR), train=True,  download=True, transform=tf)
    test_set  = torchvision.datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH, shuffle=True,  num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=BATCH, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ════════════════════════════════════════════════════════════════════════════
# 3.  Training
# ════════════════════════════════════════════════════════════════════════════

def train(model: MLP, train_loader, test_loader, device) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Training 784→256→128→64→10 MLP on MNIST for {EPOCHS} epochs ...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            nn.CrossEntropyLoss()(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()
        if epoch % 5 == 0 or epoch == EPOCHS:
            acc = _float_acc(model, test_loader, device)
            print(f"  epoch {epoch:2d}/{EPOCHS}  test_acc={acc*100:.2f}%")

    torch.save(model.state_dict(), CKPT)
    print(f"Checkpoint saved → {CKPT}")


def _float_acc(model: MLP, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(1)
            correct += (pred == yb.to(device)).sum().item()
            total   += yb.size(0)
    return correct / total


# ════════════════════════════════════════════════════════════════════════════
# 4.  Calibration + PTQ
# ════════════════════════════════════════════════════════════════════════════

def calibrate(model: MLP, test_loader, device) -> dict[str, float]:
    """Record per-layer input-activation max-abs over CALIB_N images."""
    model.eval()
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    act_max = [[] for _ in linears]

    n_seen = 0
    with torch.no_grad():
        for xb, _ in test_loader:
            if n_seen >= CALIB_N:
                break
            xb = xb.to(device)
            h = xb.view(xb.size(0), -1)
            for i, lin in enumerate(linears):
                act_max[i].append(h.abs().max().item())
                h = torch.relu(lin(h)) if i < len(linears) - 1 else lin(h)
            n_seen += xb.size(0)

    scales = {}
    for i, vals in enumerate(act_max):
        scales[f"layer{i}_in"] = max(float(np.max(vals)), EPS) / 127.0
    return scales


def quantize_weights(model: MLP, scales: dict[str, float]):
    """Per-layer symmetric INT8 quantization.

    Returns list of (w_int, b_int, sw, sb) tuples, one per layer.
    """
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    result = []
    for i, lin in enumerate(linears):
        w = lin.weight.detach().cpu().numpy()   # (out, in)
        b = lin.bias.detach().cpu().numpy()     # (out,)
        sx = scales[f"layer{i}_in"]
        sw = max(float(np.max(np.abs(w))), EPS) / 127.0
        sb = sx * sw
        w_int = np.clip(np.round(w / sw), -127, 127).astype(np.int64).T  # (in, out)
        b_int = np.round(b / sb).astype(np.int64)
        result.append((w_int, b_int, sw, sb))
    return result


# ════════════════════════════════════════════════════════════════════════════
# 5.  INT8 approximate forward (layer-by-layer with dequant)
# ════════════════════════════════════════════════════════════════════════════

def _int8_forward(x_np: np.ndarray, qw_list, scales: dict,
                  K_vec: list[int], mode: str) -> np.ndarray:
    """Full forward pass with per-layer K truncation injection.

    x_np: (N, 784) float32 in [0, 1]
    Returns: (N, 10) int64 raw logits.
    """
    n_layers = len(qw_list)
    # quantize input
    sx_in = scales["layer0_in"]
    h = np.clip(np.round(x_np / sx_in), -127, 127).astype(np.int64)

    for i, (w_int, b_int, sw, sb) in enumerate(qw_list):
        # approximate matmul
        h = int_linear_approx(h, w_int, fmt=INT8, K=K_vec[i],
                               bias=b_int, rounding=mode)
        if i < n_layers - 1:
            # dequant → relu → requant
            sx_out = scales[f"layer{i+1}_in"]
            h_float = np.maximum(h.astype(np.float64) * sb, 0.0)
            h = np.clip(np.round(h_float / sx_out), -127, 127).astype(np.int64)
    return h   # raw int64 logits


def evaluate_int8(test_loader, qw_list, scales, K_vec, mode) -> float:
    correct = total = 0
    for xb, yb in test_loader:
        x_np = xb.numpy().reshape(xb.size(0), -1)
        logits = _int8_forward(x_np, qw_list, scales, K_vec, mode)
        preds = logits.argmax(axis=1)
        correct += int((preds == yb.numpy()).sum())
        total   += yb.size(0)
    return correct / total


# ════════════════════════════════════════════════════════════════════════════
# 6.  Accuracy vs K sweep
# ════════════════════════════════════════════════════════════════════════════

def sweep_accuracy(test_loader, qw_list, scales) -> None:
    print("\n--- Accuracy vs K sweep ---")
    rows = []
    print(f"{'K':>3}  {'trunc':>7}  {'round':>7}  {'stochastic':>10}")
    print("-" * 38)
    for K in range(0, K_MAX + 1):
        row = {"K": K}
        for mode in ("trunc", "round", "stochastic"):
            t0 = time.time()
            K_vec = [K] * len(qw_list)
            acc = evaluate_int8(test_loader, qw_list, scales, K_vec, mode)
            row[f"{mode}_acc"] = round(acc, 4)
            print(f"  {K}  {mode:<12}  {acc*100:.2f}%  ({time.time()-t0:.1f}s)")
        rows.append(row)

    out = RESULTS_DIR / "accuracy_vs_K.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["K", "trunc_acc", "round_acc", "stochastic_acc"])
        w.writeheader(); w.writerows(rows)
    print(f"Saved → {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# 7.  Bias accumulation with real MNIST inputs
# ════════════════════════════════════════════════════════════════════════════

def sweep_bias(test_loader, qw_list, scales) -> None:
    """Replicate bias_accumulation.csv using real MNIST inputs.

    We measure accumulated bias of the first linear layer only,
    varying N = batch size.  Matches the spirit of redesign_experiments.py.
    """
    print("\n--- Bias accumulation (real MNIST inputs, layer 0) ---")
    w_int, b_int, sw, sb = qw_list[0]
    sx_in = scales["layer0_in"]

    # Collect a large pool of quantized inputs
    all_x = []
    for xb, _ in test_loader:
        h = np.clip(np.round(xb.numpy().reshape(xb.size(0), -1) / sx_in),
                    -127, 127).astype(np.int64)
        all_x.append(h)
        if sum(a.shape[0] for a in all_x) >= 4096:
            break
    pool = np.concatenate(all_x, axis=0)[:4096]

    Ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    K = 6
    rows = []
    for N in Ns:
        x_n = pool[:N]
        exact = int_linear_approx(x_n, w_int, fmt=INT8, K=0,
                                   bias=None, rounding="trunc").astype(np.float64)
        stats = {}
        for mode in ("trunc", "round", "stochastic"):
            approx = int_linear_approx(x_n, w_int, fmt=INT8, K=K,
                                        bias=None, rounding=mode).astype(np.float64)
            diff = (approx - exact).flatten()
            stats[f"{mode}_mean"] = float(diff.mean())
            stats[f"{mode}_rms"]  = float(np.sqrt(np.mean(diff**2)))
        rows.append({"N": N, **stats})
        print(f"  N={N:5d}  trunc_mean={stats['trunc_mean']:+.1f}  "
              f"round_mean={stats['round_mean']:+.1f}  "
              f"stoch_mean={stats['stochastic_mean']:+.1f}")

    out = RESULTS_DIR / "bias_accumulation.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "N", "trunc_mean", "trunc_rms",
            "round_mean", "round_rms",
            "stochastic_mean", "stochastic_rms"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Saved → {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# 8.  Layer allocation with real MNIST inputs
# ════════════════════════════════════════════════════════════════════════════

def sweep_allocation(test_loader, qw_list, scales) -> None:
    """Regenerate layer_allocation.csv using trained weights + MNIST inputs."""
    print("\n--- Layer allocation sweep ---")

    # Build integer layers for tiny_mlp_forward
    # Use simple max-abs scaling (no dequant between layers — consistent with sensitivity.py)
    layers_simple = []
    for w_int, b_int, sw, sb in qw_list:
        layers_simple.append((w_int, b_int))

    # Collect calibration batch
    x_pool = []
    for xb, _ in test_loader:
        sx_in = scales["layer0_in"]
        h = np.clip(np.round(xb.numpy().reshape(xb.size(0), -1) / sx_in),
                    -127, 127).astype(np.int64)
        x_pool.append(h)
        if sum(a.shape[0] for a in x_pool) >= 64:
            break
    x_calib = np.concatenate(x_pool, axis=0)[:64]

    # Probe sensitivity
    sens = layer_sensitivity(x_calib, layers_simple, fmt=INT8, K_probe=4)
    macs = layer_mac_counts(layers_simple)
    print(f"  Sensitivities: {[round(s, 4) for s in sens]}")
    print(f"  MAC counts:    {macs}")

    rows = []
    rng = np.random.default_rng(42)
    for budget in range(0, 25):
        for policy in ("uniform", "allocated"):
            if policy == "uniform":
                K_vec = uniform_K(len(layers_simple), budget)
            else:
                K_vec = allocate_K(sens, macs, total_budget=budget, K_max=8)
            # measure NRMSE
            y_ref  = tiny_mlp_forward(x_calib, layers_simple, fmt=INT8, K=0)
            y_approx = tiny_mlp_forward(x_calib, layers_simple, fmt=INT8, K=K_vec)
            nrmse = output_divergence(y_approx, y_ref, metric="logit_nrmse")
            energy = sum(
                macs[i] * (1.0 - K_vec[i] / 16.0) * 0.284375
                for i in range(len(layers_simple))
            )
            rows.append({
                "budget": budget,
                "policy": policy,
                "K_vector": "|".join(map(str, K_vec)),
                "logit_nrmse": round(nrmse, 6),
                "energy_per_inference_pJ": round(energy, 3),
            })
        print(f"  B={budget:2d}  uniform NRMSE={rows[-2]['logit_nrmse']:.4f}"
              f"  alloc NRMSE={rows[-1]['logit_nrmse']:.4f}")

    out = RESULTS_DIR / "layer_allocation.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "budget", "policy", "K_vector",
            "logit_nrmse", "energy_per_inference_pJ"])
        writer.writeheader(); writer.writerows(rows)
    print(f"Saved → {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# 9.  Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader = get_loaders()
    model = MLP().to(device)

    if CKPT.exists():
        print(f"Checkpoint found — skipping training.")
        model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    else:
        train(model, train_loader, test_loader, device)

    float_acc = _float_acc(model, test_loader, device)
    print(f"\nFloat32 accuracy: {float_acc*100:.2f}%")

    print("Calibrating activation scales ...")
    scales = calibrate(model, test_loader, device)
    for k, v in scales.items():
        print(f"  {k}: scale={v:.5f}")

    qw_list = quantize_weights(model, scales)

    # Save quantized weights for reuse
    save_dict = {}
    for i, (w_int, b_int, sw, sb) in enumerate(qw_list):
        save_dict[f"w{i}"] = w_int
        save_dict[f"b{i}"] = b_int
        save_dict[f"sw{i}"] = np.array(sw)
        save_dict[f"sb{i}"] = np.array(sb)
    np.savez(WEIGHTS_NPZ, **save_dict)
    print(f"Weights saved → {WEIGHTS_NPZ.name}")

    sweep_accuracy(test_loader, qw_list, scales)
    sweep_bias(test_loader, qw_list, scales)
    sweep_allocation(test_loader, qw_list, scales)

    print("\nAll done.")


if __name__ == "__main__":
    main()
