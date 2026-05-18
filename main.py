"""End-to-end driver: runs the full (precision, K, ACA-window) sweep,
extracts Pareto fronts, runs a tiny MLP inference sanity experiment, and
dumps results to ``experiments/results/`` as CSV for downstream plotting.

Usage (from any working directory; the fullwidth-colon project path
breaks Python's startup if used as cwd, see README):

    python -m main

Outputs:
    experiments/results/int_sweep.csv     — every INT (fmt, K, W) config
    experiments/results/fp_sweep.csv      — every FP (fmt, K) config
    experiments/results/int_pareto.csv    — non-dominated subset of INT
    experiments/results/fp_pareto.csv     — non-dominated subset of FP
    experiments/results/mlp_inference.csv — per-K MLP output divergence
    experiments/results/run_summary.txt   — human-readable summary
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from axmac.dnn_inference import tiny_mlp_forward
from axmac.exact_mac import BF16, FP16, FP32, INT4, INT8, INT16
from axmac.pareto import (
    DesignPoint,
    pareto_front,
    sort_front_by_energy,
    sweep_fp_designs,
    sweep_int_designs,
)
from axmac.power_model import mac_int_energy


# ============================================================
# Output paths
# ============================================================

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def _write_designpoint_csv(path: Path, points: list[DesignPoint]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "fmt", "is_fp", "K", "aca_window",
            "energy_pJ", "mult_pJ", "add_pJ",
            "med", "rmse", "nmed", "max_abs",
        ])
        for p in points:
            w.writerow([
                p.fmt_name, p.is_fp, p.K,
                "" if p.aca_window is None else p.aca_window,
                f"{p.energy_pJ:.6f}",
                f"{p.energy_breakdown.multiplier_pJ:.6f}",
                f"{p.energy_breakdown.adder_pJ:.6f}",
                f"{p.error_med:.6e}",
                f"{p.error_rmse:.6e}",
                f"{p.error_nmed:.6e}",
                f"{p.error_max_abs:.6e}",
            ])


def _print_front(label: str, front: list[DesignPoint]) -> str:
    lines = [f"--- {label} Pareto front ({len(front)} points) ---"]
    for p in sort_front_by_energy(front):
        w = "—" if p.aca_window is None else f"W={p.aca_window}"
        lines.append(
            f"  {p.fmt_name:6s} K={p.K} {w:5s} "
            f"E={p.energy_pJ:7.4f} pJ  "
            f"MED={p.error_med:10.4g}  "
            f"NMED={p.error_nmed:10.4e}  "
            f"RMSE={p.error_rmse:10.4g}"
        )
    text = "\n".join(lines)
    print(text)
    return text


# ============================================================
# A) INT design-space sweep
# ============================================================

def section_int_sweep() -> tuple[list[DesignPoint], list[DesignPoint]]:
    print("\n=== A) INT design-space sweep ===")
    fmts = [INT4, INT8, INT16]
    Ks = [0, 1, 2, 3, 4, 5, 6]
    Ws = [None, 32, 16, 8, 4]
    print(f"Configs: {len(fmts)} fmts × {len(Ks)} K × {len(Ws)} W = "
          f"{len(fmts) * len(Ks) * len(Ws)}")
    points = sweep_int_designs(fmts, Ks, Ws, n_samples=1000)
    _write_designpoint_csv(RESULTS_DIR / "int_sweep.csv", points)
    print(f"wrote {RESULTS_DIR / 'int_sweep.csv'}  ({len(points)} rows)")

    front_nmed = pareto_front(points, x_key="energy_pJ", y_key="error_nmed")
    _write_designpoint_csv(RESULTS_DIR / "int_pareto.csv", front_nmed)
    return points, front_nmed


# ============================================================
# B) FP design-space sweep
# ============================================================

def section_fp_sweep() -> tuple[list[DesignPoint], list[DesignPoint]]:
    print("\n=== B) FP design-space sweep ===")
    fmts = [FP16, BF16, FP32]
    Ks = [0, 1, 2, 3, 4, 5, 6]
    print(f"Configs: {len(fmts)} fmts × {len(Ks)} K = {len(fmts) * len(Ks)}")
    points = sweep_fp_designs(fmts, Ks, n_samples=1000)
    _write_designpoint_csv(RESULTS_DIR / "fp_sweep.csv", points)
    print(f"wrote {RESULTS_DIR / 'fp_sweep.csv'}  ({len(points)} rows)")

    front = pareto_front(points, x_key="energy_pJ", y_key="error_nmed")
    _write_designpoint_csv(RESULTS_DIR / "fp_pareto.csv", front)
    return points, front


# ============================================================
# C) Tiny MLP inference: K sweep on synthetic MNIST-shaped data
# ============================================================

def section_mlp_inference() -> list[dict]:
    print("\n=== C) Tiny MLP inference K-sweep (INT8, 784→128→32→10) ===")
    rng = np.random.default_rng(0xCAFE)
    batch = 64
    fmt = INT8

    # Synthetic MNIST-shaped inputs in INT8 range, 50% sparse (post-ReLU-ish).
    x = rng.integers(0, 64, size=(batch, 784))
    x = np.where(rng.random((batch, 784)) < 0.5, 0, x)

    # Random INT8-quantized weights for a 2-hidden-layer MLP.
    w1 = rng.integers(-32, 32, size=(784, 128))
    b1 = rng.integers(-200, 200, size=(128,))
    w2 = rng.integers(-32, 32, size=(128, 32))
    b2 = rng.integers(-200, 200, size=(32,))
    w3 = rng.integers(-32, 32, size=(32, 10))
    b3 = rng.integers(-200, 200, size=(10,))
    layers = [(w1, b1), (w2, b2), (w3, b3)]

    # Per-layer MAC counts (used to scale per-MAC energy to per-inference).
    macs_per_inference = sum(
        w.shape[0] * w.shape[1] for (w, _) in layers
    )
    print(f"batch={batch}, MACs/inference={macs_per_inference:,}")

    # Reference: K=0 output.
    y_exact = tiny_mlp_forward(x, layers, fmt=fmt, K=0)
    pred_exact = y_exact.argmax(axis=1)
    print(f"K=0 baseline argmax distribution: "
          f"{np.bincount(pred_exact, minlength=10).tolist()}")

    rows: list[dict] = []
    for K in [0, 1, 2, 3, 4, 5, 6]:
        y = tiny_mlp_forward(x, layers, fmt=fmt, K=K)
        pred = y.argmax(axis=1)
        agreement = float(np.mean(pred == pred_exact))
        # Output-vector divergence (NRMSE in logit space).
        diff = (y - y_exact).astype(np.float64)
        denom = np.sqrt(np.mean(y_exact.astype(np.float64) ** 2)) + 1e-12
        nrmse = float(np.sqrt(np.mean(diff ** 2)) / denom)
        e_per_mac = mac_int_energy(fmt, K=K).total_pJ
        e_per_inference_pJ = e_per_mac * macs_per_inference

        rows.append(dict(
            K=K, agreement=agreement, logit_nrmse=nrmse,
            energy_per_mac_pJ=e_per_mac,
            energy_per_inference_pJ=e_per_inference_pJ,
        ))
        print(
            f"  K={K}:  argmax agreement = {agreement * 100:5.1f}%   "
            f"logit NRMSE = {nrmse:8.4f}   "
            f"E/MAC = {e_per_mac:.4f} pJ   "
            f"E/inference = {e_per_inference_pJ / 1e3:.3f} nJ"
        )

    csv_path = RESULTS_DIR / "mlp_inference.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {csv_path}  ({len(rows)} rows)")
    return rows


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    print("=" * 60)
    print("AxMAC end-to-end experiment driver")
    print("=" * 60)

    int_points, int_front = section_int_sweep()
    int_front_txt = _print_front("INT (energy, NMED)", int_front)

    fp_points, fp_front = section_fp_sweep()
    fp_front_txt = _print_front("FP (energy, NMED)", fp_front)

    mlp_rows = section_mlp_inference()

    # Human-readable summary log.
    summary_lines = [
        "AxMAC end-to-end experiment summary",
        "=" * 60,
        f"INT sweep:  {len(int_points)} configs, front size {len(int_front)}",
        f"FP sweep:   {len(fp_points)} configs, front size {len(fp_front)}",
        f"MLP K-sweep: {len(mlp_rows)} K values",
        "",
        int_front_txt,
        "",
        fp_front_txt,
        "",
        "--- MLP inference (INT8, 784→128→32→10) ---",
    ]
    for r in mlp_rows:
        summary_lines.append(
            f"  K={r['K']}  agreement={r['agreement'] * 100:5.1f}%  "
            f"NRMSE={r['logit_nrmse']:.4f}  "
            f"E/MAC={r['energy_per_mac_pJ']:.4f} pJ  "
            f"E/inference={r['energy_per_inference_pJ'] / 1e3:.3f} nJ"
        )
    summary_text = "\n".join(summary_lines)
    (RESULTS_DIR / "run_summary.txt").write_text(summary_text, encoding="utf-8")
    print(f"\nwrote {RESULTS_DIR / 'run_summary.txt'}")
    print("=" * 60)
    print("done.")


if __name__ == "__main__":
    main()
