"""Switching-activity power model, calibrated to 45 nm reference data.

Two layers:

1. **Analytical datasheet** — :func:`mac_int_energy`, :func:`mac_fp_energy`
   return per-MAC energy as a function of format and approximation knobs
   (K, aca_window). Base numbers come from Horowitz, *Computing's Energy
   Problem*, ISSCC 2014 keynote (45 nm, 1 V, 1 GHz).

2. **Data-dependent switching** — :func:`pp_switching_activity` and
   :func:`mantissa_switching_activity` compute Hamming distances over
   consecutive operand cycles. :func:`energy_from_activity` scales the
   base multiplier energy by the observed activity factor, so a sequence
   of correlated activations (typical in CNNs) reports lower energy than
   the random-input default.

Approximation savings model:

* Truncated multiplier (drop bottom K bits of the product): a triangular
  region of the N×N partial-product matrix is removed. Fractional savings
  ``K(2N - K) / (2N²)`` up to a cap of 0.95.
* ACA adder with window W in an acc_bits-wide chain: shortening the carry
  chain saves roughly half the adder's dynamic energy on the segments
  that no longer ripple-propagate. Fractional savings
  ``(acc_bits - W) / acc_bits * 0.5``.

These are simple-but-defensible analytical models; the data-dependent
hooks let Week-5 trace experiments tighten them with real activations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .exact_mac import BF16, FP16, FP32, INT4, INT8, INT16, FPFormat, IntFormat


# ============================================================
# Horowitz ISSCC 2014, 45 nm, 1 V — base per-op energy (pJ)
# ============================================================

_BASE_PJ: dict[tuple[str, str], float] = {
    # INT — INT8 anchored to Horowitz (0.2 pJ mult, 0.03 pJ add).
    # INT4/INT16 scaled by mantissa-product area (≈ N²) and ripple length (≈ N).
    ("INT4", "mult"): 0.050,
    ("INT4", "add"):  0.015,
    ("INT8", "mult"): 0.200,
    ("INT8", "add"):  0.030,
    ("INT16", "mult"): 0.800,
    ("INT16", "add"):  0.060,
    # FP — Horowitz exact for FP16 (1.1/0.4) and FP32 (3.7/0.9).
    # BF16: same exponent as FP32 but 7-bit mantissa → slightly cheaper than FP16.
    ("FP16", "mult"): 1.100,
    ("FP16", "add"):  0.400,
    ("BF16", "mult"): 0.950,
    ("BF16", "add"):  0.380,
    ("FP32", "mult"): 3.700,
    ("FP32", "add"):  0.900,
}

# 32-bit accumulator add cost (used for the INT MAC's accumulator stage).
ACC32_ADD_PJ = 0.100


def base_mult_pJ(fmt: IntFormat | FPFormat) -> float:
    return _BASE_PJ[(fmt.name, "mult")]


def base_add_pJ(fmt: IntFormat | FPFormat) -> float:
    return _BASE_PJ[(fmt.name, "add")]


# ============================================================
# Approximation savings
# ============================================================

def truncation_savings(n_bits: int, K: int) -> float:
    """Fractional multiplier energy saved by dropping the bottom K bits of the product.

    Models the lost area of the N×N partial-product matrix as a triangle of
    side K. Returns a value in [0, 0.95].
    """
    if K <= 0:
        return 0.0
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    raw = K * (2 * n_bits - K) / (2 * n_bits * n_bits)
    return max(0.0, min(0.95, raw))


def aca_savings(acc_bits: int, window: int | None) -> float:
    """Fractional adder energy saved by ACA with carry-isolated window ``window``."""
    if window is None or window >= acc_bits:
        return 0.0
    if window <= 0:
        raise ValueError("aca window must be >= 1")
    return (acc_bits - window) / acc_bits * 0.5


# ============================================================
# Energy reports
# ============================================================

@dataclass(frozen=True)
class Energy:
    multiplier_pJ: float
    adder_pJ: float

    @property
    def total_pJ(self) -> float:
        return self.multiplier_pJ + self.adder_pJ

    def __repr__(self) -> str:  # nicer print in pytest failures
        return (
            f"Energy(mult={self.multiplier_pJ:.4f} pJ, "
            f"add={self.adder_pJ:.4f} pJ, total={self.total_pJ:.4f} pJ)"
        )


def mac_int_energy(
    fmt: IntFormat,
    *,
    K: int = 0,
    aca_window: int | None = None,
    acc_bits: int = 32,
) -> Energy:
    """Per-MAC energy for the INT path with the given approximation knobs.

    K=0 + aca_window=None returns the calibrated 45 nm baseline.
    """
    if K < 0:
        raise ValueError("K must be >= 0")
    mult = base_mult_pJ(fmt) * (1.0 - truncation_savings(fmt.bits, K))
    add = ACC32_ADD_PJ * (acc_bits / 32) * (1.0 - aca_savings(acc_bits, aca_window))
    return Energy(mult, add)


def mac_fp_energy(fmt: FPFormat, *, K: int = 0) -> Energy:
    """Per-MAC energy for the FP path with mantissa truncation K."""
    if K < 0:
        raise ValueError("K must be >= 0")
    # Mantissa product width is (mant_bits + 1) per operand (with hidden bit).
    mant_width = fmt.mant_bits + 1
    mult = base_mult_pJ(fmt) * (1.0 - truncation_savings(mant_width, K))
    add = base_add_pJ(fmt)  # FP add not approximated in W3
    return Energy(mult, add)


# ============================================================
# Data-dependent switching activity
# ============================================================

def hamming_distance(a: int, b: int, n_bits: int) -> int:
    """Number of differing bits between ``a`` and ``b`` over the low ``n_bits`` bits.

    Works for signed inputs (Python ints) by first masking to the bit window.
    """
    if n_bits <= 0:
        return 0
    mask = (1 << n_bits) - 1
    return ((a & mask) ^ (b & mask)).bit_count()


def pp_switching_activity(
    prev_pps: Sequence[int],
    cur_pps: Sequence[int],
    bits_per_pp: int,
) -> int:
    """Total bit-toggles across the partial-product matrix between two cycles."""
    if len(prev_pps) != len(cur_pps):
        raise ValueError("prev_pps and cur_pps must have the same length")
    return sum(hamming_distance(p, c, bits_per_pp) for p, c in zip(prev_pps, cur_pps))


def mantissa_switching_activity(prev: int, cur: int, mant_bits: int) -> int:
    """Hamming distance between two mantissa products of the same FP format."""
    # Mantissa product is up to 2*(mant_bits+1) bits wide.
    return hamming_distance(prev, cur, 2 * (mant_bits + 1))


def energy_from_activity(
    fmt: IntFormat | FPFormat,
    activity_bits: int,
    *,
    baseline_alpha: float = 0.5,
) -> float:
    """Scale base multiplier energy by observed switching activity.

    ``activity_bits`` is the Hamming distance between consecutive PP / mantissa
    products (use :func:`pp_switching_activity` or
    :func:`mantissa_switching_activity` to compute it). ``baseline_alpha`` is
    the assumed switching factor behind the Horowitz datasheet number
    (0.5 = uncorrelated random inputs).

    Returns multiplier energy in pJ.
    """
    if not (0.0 < baseline_alpha <= 1.0):
        raise ValueError("baseline_alpha must be in (0, 1]")
    if activity_bits < 0:
        raise ValueError("activity_bits must be non-negative")
    if isinstance(fmt, IntFormat):
        n_bits = fmt.bits
        # PP matrix area scales like N * (N/2) Booth-radix-4 partial products
        # of width up to 2N → roughly N² bits.
        total_bits = n_bits * n_bits
    else:
        mant_width = fmt.mant_bits + 1
        total_bits = 2 * mant_width * mant_width
    if total_bits == 0:
        return 0.0
    observed_alpha = activity_bits / total_bits
    return base_mult_pJ(fmt) * (observed_alpha / baseline_alpha)


# ============================================================
# Convenience: precomputed datasheet snapshot for sanity prints
# ============================================================

def datasheet_snapshot() -> dict[str, Energy]:
    """Return calibrated baseline energies for every supported format."""
    return {
        "INT4":  mac_int_energy(INT4),
        "INT8":  mac_int_energy(INT8),
        "INT16": mac_int_energy(INT16),
        "FP16":  mac_fp_energy(FP16),
        "BF16":  mac_fp_energy(BF16),
        "FP32":  mac_fp_energy(FP32),
    }
