"""Export Python-side golden test vectors for the RTL testbench.

Three families of CSVs are produced; the cocotb / SystemVerilog testbenches in
``rtl/tb/`` consume them as the bit-exact reference for the corresponding DUT.

  1. mac_int8.csv       — INT8 truncated/rounded multiplier (DUT: mac_unit.v).
                          Covers K = 0..6 under trunc and round modes. These
                          two modes are deterministic, so the RTL must match
                          ``product_rounded`` bit-for-bit.
  2. mac_int8_stoch.csv — Stochastic-rounding samples for distributional check
                          (mean / RMSE). Bit-exact match is NOT expected: the
                          Python golden uses ``random.Random``, the RTL will
                          use an LFSR. We compare statistics, not bits.
  3. aca.csv            — Approximate-Carry-Adder reference at several window
                          widths (DUT: aca_adder.v).
  4. mlp_toy/           — Bundle for end-to-end MLP test (DUT: mlp_top.v on
                          the toy 64->16->10 model that fits EP4CE10).

Run from the project root:

  py -3.14 rtl/golden/export_golden.py

Outputs land in this directory (rtl/golden/).
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from axmac.approx_mac import _apply_rounding, aca_add  # noqa: E402
from axmac.dnn_inference import quantize_to_int, tiny_mlp_forward  # noqa: E402
from axmac.exact_mac import INT8, booth_radix4_pps  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


# ============================================================
# 1) MAC unit — deterministic rounding modes (trunc, round)
# ============================================================

def _int8_test_pairs(rng: random.Random, n_random: int = 200) -> list[tuple[int, int]]:
    """A reusable (a, b) test set: edges + grid + random samples."""
    edges = [INT8.min_val, -1, 0, 1, INT8.max_val]
    pairs = [(a, b) for a in edges for b in edges]
    grid_step = 32
    for a in range(INT8.min_val, INT8.max_val + 1, grid_step):
        for b in range(INT8.min_val, INT8.max_val + 1, grid_step):
            pairs.append((a, b))
    for _ in range(n_random):
        pairs.append(
            (rng.randint(INT8.min_val, INT8.max_val),
             rng.randint(INT8.min_val, INT8.max_val))
        )
    return pairs


MODE_CODE = {"trunc": 0, "round": 1, "stochastic": 2}


def export_mac_int8_deterministic() -> Path:
    """Numeric ``mode`` column (0=trunc 1=round 2=stoch) so SV $sscanf with
    plain ``%d`` parses every field -- ModelSim-Altera 10.5b lacks the
    ``%[^,]`` C-scanf extension."""
    rng = random.Random(0xA11CE)
    pairs = _int8_test_pairs(rng)
    path = OUT_DIR / "mac_int8.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "K", "mode", "product_full", "product_rounded"])
        for a, b in pairs:
            product_full = sum(booth_radix4_pps(a, b, INT8.bits))
            assert product_full == a * b, "Booth PPs must sum to exact product"
            for K in range(7):
                for mode in ("trunc", "round"):
                    rounded = _apply_rounding(product_full, K, mode, None)
                    w.writerow([a, b, K, MODE_CODE[mode], product_full, rounded])
    return path


def export_mac_int8_stochastic(n_samples_per_K: int = 256) -> Path:
    """Stochastic samples for distributional checks; not bit-exact vs RTL LFSR."""
    rng = random.Random(0xBEE5)
    path = OUT_DIR / "mac_int8_stoch.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "K", "seed_call_idx", "product_full", "product_rounded"])
        call_idx = 0
        for K in range(1, 7):
            for _ in range(n_samples_per_K):
                a = rng.randint(INT8.min_val, INT8.max_val)
                b = rng.randint(INT8.min_val, INT8.max_val)
                product_full = a * b
                rounded = _apply_rounding(product_full, K, "stochastic", rng)
                w.writerow([a, b, K, call_idx, product_full, rounded])
                call_idx += 1
    return path


# ============================================================
# 2) ACA adder — various window widths
# ============================================================

def export_aca() -> Path:
    rng = random.Random(0xC0DE)
    path = OUT_DIR / "aca.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "bits", "window", "sum"])
        bits = 32
        for window in (4, 8, 16, 32):
            for _ in range(256):
                a = rng.randrange(-(1 << (bits - 1)), 1 << (bits - 1))
                b = rng.randrange(-(1 << (bits - 1)), 1 << (bits - 1))
                s = aca_add(a, b, bits, window)
                w.writerow([a, b, bits, window, s])
    return path


# ============================================================
# 3) Toy MLP end-to-end (64 -> 16 -> 10), fits EP4CE10
# ============================================================

def _write_int_matrix(path: Path, m: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        for row in np.atleast_2d(m):
            w.writerow([int(v) for v in row])


def _write_int8_hex(path: Path, m: np.ndarray) -> None:
    """Flat row-major INT8 dump in two's-complement hex, one byte per line.

    Format consumable by Verilog ``$readmemh``. Negative values land as
    ``80..ff``. A weight matrix W[in, out] is flattened so that
    ``mem[in * OUT + out]`` is W[in, out] (row-major).
    """
    with path.open("w") as f:
        for v in np.atleast_2d(m).flatten():
            f.write(f"{int(v) & 0xFF:02x}\n")


def export_toy_mlp() -> Path:
    """Bundle: input vector + per-layer weights/biases + reference outputs."""
    rng_np = np.random.default_rng(0xD0D0)
    bundle_dir = OUT_DIR / "mlp_toy"
    bundle_dir.mkdir(exist_ok=True)

    # Toy 64 -> 16 -> 10 MLP, INT8 weights/activations.
    x_float = rng_np.uniform(0.0, 1.0, size=(1, 64))
    w0_f = rng_np.uniform(-0.25, 0.25, size=(64, 16))
    b0_f = rng_np.uniform(-1.0, 1.0, size=(16,))
    w1_f = rng_np.uniform(-0.25, 0.25, size=(16, 10))
    b1_f = rng_np.uniform(-1.0, 1.0, size=(10,))

    x_scale = INT8.max_val // 2
    w_scale = INT8.max_val
    b_scale = INT8.max_val // 2
    x = quantize_to_int(x_float * x_scale, INT8)
    w0 = quantize_to_int(w0_f * w_scale, INT8)
    b0 = quantize_to_int(b0_f * b_scale, INT8)
    w1 = quantize_to_int(w1_f * w_scale, INT8)
    b1 = quantize_to_int(b1_f * b_scale, INT8)

    # CSV form for human inspection / SV testbenches.
    _write_int_matrix(bundle_dir / "x.csv", x)
    _write_int_matrix(bundle_dir / "w0.csv", w0)
    _write_int_matrix(bundle_dir / "b0.csv", b0.reshape(1, -1))
    _write_int_matrix(bundle_dir / "w1.csv", w1)
    _write_int_matrix(bundle_dir / "b1.csv", b1.reshape(1, -1))

    # Hex memory image form for $readmemh in mlp_top.v / bram_wrapper.v.
    # The hex dumps are flat row-major byte streams (in * OUT + out indexing).
    _write_int8_hex(bundle_dir / "x.mem", x)
    _write_int8_hex(bundle_dir / "w0.mem", w0)
    _write_int8_hex(bundle_dir / "b0.mem", b0)
    _write_int8_hex(bundle_dir / "w1.mem", w1)
    _write_int8_hex(bundle_dir / "b1.mem", b1)

    layers = [(w0, b0), (w1, b1)]
    for K in (0, 2, 4):
        for mode in ("trunc", "round"):
            y = tiny_mlp_forward(x, layers, fmt=INT8, K=K, rounding=mode)
            _write_int_matrix(bundle_dir / f"y_K{K}_{mode}.csv", y)

    return bundle_dir


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    print(f"writing golden vectors under {OUT_DIR}")
    paths = [
        export_mac_int8_deterministic(),
        export_mac_int8_stochastic(),
        export_aca(),
        export_toy_mlp(),
    ]
    for p in paths:
        if p.is_dir():
            n_files = sum(1 for _ in p.iterdir())
            print(f"  {p.relative_to(OUT_DIR)}/  ({n_files} files)")
        else:
            n_rows = sum(1 for _ in p.open()) - 1
            print(f"  {p.relative_to(OUT_DIR)}  ({n_rows} rows)")


if __name__ == "__main__":
    main()
