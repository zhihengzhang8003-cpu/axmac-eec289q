"""Arithmetic accuracy evaluation: RMSE, MED, max error, histograms.

Stage 4 deliverable. Compares :mod:`axmac.approx_mac` against the exact MAC
of :mod:`axmac.exact_mac` over operand samples drawn from a chosen
distribution, then aggregates the per-MAC errors into a small set of
metrics that feed the Week-6 Pareto analysis.

Metrics (computed in the decimal domain so they are comparable across
formats):

* **MED**  — mean(|exact − approx|)
* **RMSE** — sqrt(mean((exact − approx)²))
* **max_abs_err** — max(|exact − approx|)
* **NMED** — MED divided by the dynamic range of the format
              (max representable magnitude), so error scales are
              comparable across INT4 ↔ FP32.
* **bias** — mean(exact − approx); positive bias on integer truncation
             is expected because zeroing low bits always rounds toward
             −∞ for non-negative products.

Sample generators:

* :func:`int_samples`   — uniform / normal / relu-like over an :class:`IntFormat`
* :func:`fp_samples`    — uniform / normal / relu-like over a  :class:`FPFormat`
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal, Sequence

from .approx_mac import approx_mac_fp, approx_mac_int
from .exact_mac import (
    FPFormat,
    IntFormat,
    decode_fp,
    encode_fp,
    mac_fp,
    mac_int,
)


Distribution = Literal["uniform", "normal", "relu"]


# ============================================================
# Result container
# ============================================================

@dataclass(frozen=True)
class ErrorStats:
    n_samples: int
    med: float
    rmse: float
    max_abs_err: float
    nmed: float
    bias: float

    def __repr__(self) -> str:
        return (
            f"ErrorStats(n={self.n_samples}, med={self.med:.4g}, "
            f"rmse={self.rmse:.4g}, max={self.max_abs_err:.4g}, "
            f"nmed={self.nmed:.4g}, bias={self.bias:+.4g})"
        )


def _aggregate(errors: Sequence[float], dynamic_range: float) -> ErrorStats:
    if len(errors) == 0:
        raise ValueError("cannot aggregate an empty error list")
    if dynamic_range <= 0:
        raise ValueError("dynamic_range must be positive")
    n = len(errors)
    abs_errs = [abs(e) for e in errors]
    med = sum(abs_errs) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    return ErrorStats(
        n_samples=n,
        med=med,
        rmse=rmse,
        max_abs_err=max(abs_errs),
        nmed=med / dynamic_range,
        bias=sum(errors) / n,
    )


# ============================================================
# Sample generators
# ============================================================

def int_samples(
    fmt: IntFormat,
    n: int,
    *,
    distribution: Distribution = "uniform",
    sigma_frac: float = 0.25,
    seed: int | None = None,
) -> list[int]:
    """Draw ``n`` samples from ``fmt`` under the requested distribution.

    ``sigma_frac`` is sigma as a fraction of ``fmt.max_val`` for normal/relu.
    All samples are clipped to ``[fmt.min_val, fmt.max_val]``.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    rng = random.Random(seed)
    sigma = max(1.0, sigma_frac * fmt.max_val)
    out: list[int] = []
    for _ in range(n):
        if distribution == "uniform":
            x = rng.randint(fmt.min_val, fmt.max_val)
        elif distribution == "normal":
            x = round(rng.gauss(0.0, sigma))
        elif distribution == "relu":
            x = round(max(0.0, rng.gauss(0.0, sigma)))
        else:
            raise ValueError(f"unknown distribution {distribution!r}")
        out.append(max(fmt.min_val, min(fmt.max_val, x)))
    return out


def fp_samples(
    fmt: FPFormat,
    n: int,
    *,
    distribution: Distribution = "uniform",
    scale: float = 1.0,
    seed: int | None = None,
) -> list[int]:
    """Draw ``n`` FP bit-pattern samples under the requested distribution.

    ``scale`` controls the magnitude of the underlying float draws; samples
    are encoded into ``fmt`` so subnormals / overflow round into format.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    rng = random.Random(seed)
    out: list[int] = []
    for _ in range(n):
        if distribution == "uniform":
            x = rng.uniform(-scale, scale)
        elif distribution == "normal":
            x = rng.gauss(0.0, scale)
        elif distribution == "relu":
            x = max(0.0, rng.gauss(0.0, scale))
        else:
            raise ValueError(f"unknown distribution {distribution!r}")
        out.append(encode_fp(x, fmt))
    return out


# ============================================================
# Per-config error stats
# ============================================================

def _int_dynamic_range(fmt: IntFormat) -> float:
    # Product range fits in 2*fmt.bits; use it for NMED so that
    # K=fmt.bits truncation gives nmed close to 1.
    return float(fmt.max_val) * float(fmt.max_val)


def error_stats_int(
    fmt: IntFormat,
    a_samples: Sequence[int],
    b_samples: Sequence[int],
    acc_samples: Sequence[int],
    *,
    K: int = 0,
    aca_window: int | None = None,
    acc_bits: int = 32,
) -> ErrorStats:
    """Run paired exact / approx INT MAC over the sample tuples; aggregate.

    All three sample lists must have equal length and be ``len > 0``.
    """
    n = len(a_samples)
    if n == 0:
        raise ValueError("sample lists must be non-empty")
    if len(b_samples) != n or len(acc_samples) != n:
        raise ValueError("a/b/acc samples must have matching length")
    errors: list[float] = []
    for a, b, acc in zip(a_samples, b_samples, acc_samples):
        exact = mac_int(a, b, acc, fmt, acc_bits=acc_bits)
        approx = approx_mac_int(
            a, b, acc, fmt, K=K, aca_window=aca_window, acc_bits=acc_bits
        )
        errors.append(float(exact - approx))
    return _aggregate(errors, _int_dynamic_range(fmt))


def _fp_dynamic_range(fmt: FPFormat) -> float:
    # Largest representable finite magnitude (ignoring overflow into Inf).
    max_exp = fmt.exp_all_ones - 1 - fmt.bias
    mant_max = (1 << fmt.mant_bits) - 1
    mantissa = 1.0 + mant_max / (1 << fmt.mant_bits)
    return mantissa * (2.0 ** max_exp)


def error_stats_fp(
    fmt: FPFormat,
    a_samples: Sequence[int],
    b_samples: Sequence[int],
    acc_samples: Sequence[int],
    *,
    K: int = 0,
) -> ErrorStats:
    """Run paired exact / approx FP MAC over the sample tuples; aggregate.

    Non-finite reference results (Inf, NaN) are skipped — they indicate the
    chosen input distribution overflows the format and the error is
    undefined.
    """
    n = len(a_samples)
    if n == 0:
        raise ValueError("sample lists must be non-empty")
    if len(b_samples) != n or len(acc_samples) != n:
        raise ValueError("a/b/acc samples must have matching length")
    errors: list[float] = []
    for a, b, acc in zip(a_samples, b_samples, acc_samples):
        exact_bits = mac_fp(a, b, acc, fmt)
        approx_bits = approx_mac_fp(a, b, acc, fmt, K=K)
        exact = decode_fp(exact_bits, fmt)
        approx = decode_fp(approx_bits, fmt)
        if not (math.isfinite(exact) and math.isfinite(approx)):
            continue
        errors.append(exact - approx)
    if not errors:
        raise ValueError("all samples produced non-finite results; cannot aggregate")
    return _aggregate(errors, _fp_dynamic_range(fmt))


# ============================================================
# Sweeps
# ============================================================

def sweep_int(
    fmt: IntFormat,
    K_values: Sequence[int],
    aca_windows: Sequence[int | None],
    *,
    n_samples: int = 2000,
    distribution: Distribution = "uniform",
    acc_bits: int = 32,
    seed: int = 0xA11CE,
) -> dict[tuple[int, int | None], ErrorStats]:
    """Sweep (K, aca_window) for an INT format and return per-config error stats.

    Uses one draw of samples per format so the configs are evaluated on the
    same inputs (paired comparison).
    """
    a = int_samples(fmt, n_samples, distribution=distribution, seed=seed)
    b = int_samples(fmt, n_samples, distribution=distribution, seed=seed + 1)
    acc_max = (1 << (acc_bits - 1)) - 1
    acc_min = -(1 << (acc_bits - 1))
    rng = random.Random(seed + 2)
    # Keep accumulator small enough that we don't bias toward wrap-around.
    headroom = min(acc_max, 1 << max(0, acc_bits - 8))
    acc = [rng.randint(-headroom, headroom) for _ in range(n_samples)]
    out: dict[tuple[int, int | None], ErrorStats] = {}
    for K in K_values:
        for W in aca_windows:
            out[(K, W)] = error_stats_int(
                fmt, a, b, acc, K=K, aca_window=W, acc_bits=acc_bits
            )
    return out


def sweep_fp(
    fmt: FPFormat,
    K_values: Sequence[int],
    *,
    n_samples: int = 2000,
    distribution: Distribution = "uniform",
    scale: float = 1.0,
    seed: int = 0xB0B,
) -> dict[int, ErrorStats]:
    """Sweep K for an FP format and return per-K error stats."""
    a = fp_samples(fmt, n_samples, distribution=distribution, scale=scale, seed=seed)
    b = fp_samples(fmt, n_samples, distribution=distribution, scale=scale, seed=seed + 1)
    acc = fp_samples(fmt, n_samples, distribution=distribution, scale=scale, seed=seed + 2)
    return {K: error_stats_fp(fmt, a, b, acc, K=K) for K in K_values}


# ============================================================
# Histogram
# ============================================================

def error_histogram(
    errors: Sequence[float],
    *,
    n_bins: int = 50,
    rng_pad: float = 1e-12,
) -> tuple[list[float], list[int]]:
    """Compute an equal-width histogram over ``errors``.

    Returns ``(edges, counts)`` where ``len(edges) == n_bins + 1`` and
    ``len(counts) == n_bins``. The right edge is inclusive on the last bin.
    ``rng_pad`` ensures non-zero width when all errors are equal.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be >= 1")
    if not errors:
        raise ValueError("cannot histogram an empty sequence")
    lo = min(errors)
    hi = max(errors)
    if hi == lo:
        hi = lo + rng_pad
    width = (hi - lo) / n_bins
    edges = [lo + i * width for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for e in errors:
        idx = int((e - lo) / width)
        if idx >= n_bins:
            idx = n_bins - 1
        counts[idx] += 1
    return edges, counts
