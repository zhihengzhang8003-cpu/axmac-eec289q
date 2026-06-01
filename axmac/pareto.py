"""Full (precision, K, ACA-window) design-space sweep and Pareto analysis.

Stage 6 deliverable — redesigned harness (v2).

Design principles
-----------------
Common reference.
    Every format is scored against the same float64 dot-product.  Quantization
    error, truncation error, and ACA carry-isolation error all land on one
    ``error_nrmse`` axis, so INT4 / INT8 / INT16 / FP8 / FP16 / BF16 / FP32
    can share a single Pareto plane.

Equivalent-W deduplication.
    For a ``acc_bits``-wide accumulator, any ACA window W >= acc_bits is
    mathematically identical to W = None (exact carry chain).  The original
    code kept both ``None`` and ``32`` in the sweep, producing duplicate
    DesignPoints with identical energy and error that polluted the Pareto
    front.  ``_canonicalise_windows`` collapses all W >= acc_bits to ``None``
    and deduplicates before the sweep runs.

ACA noise-floor filter.
    When the ACA window is very narrow relative to acc_bits the carry-isolation
    error explodes (e.g. INT4 K=4 W=4 -> NRMSE >> 1).  Such points are not
    interesting engineering trade-offs; they are artefacts of an untailored
    grid.  ``_filter_usable`` removes any point whose ``error_nrmse`` exceeds
    ``MAX_NRMSE_FACTOR`` times the baseline NRMSE of that format at K=0 /
    W=None (pure quantisation error).  The factor is generous (10x) so
    borderline noisy-but-interesting configs survive.

Extended sweep ranges.
    INT16 has 16 mantissa bits, so K values up to ~12 are meaningful.
    FP8 formats (E4M3 / E5M2) have 3-4 mantissa bits; K > 4 exceeds the full
    mantissa width and is capped accordingly.  The default ``K_values`` dicts
    in ``DEFAULT_K_RANGES`` encode these per-format limits.
    ACA windows are chosen relative to ``acc_bits`` for each format so the
    grid is always meaningful.

Pareto structure
----------------
``per_format_front(points, fmt_name)``
    Non-dominated (K, W) configs within one precision.  The primary tool for
    understanding each format's own efficiency curve.

``per_format_fronts(points)``
    Run ``per_format_front`` for every format present; returns a dict.

``global_front(points)``
    Cross-format non-dominated set on (energy_pJ, error_nrmse).  Interleaves
    formats by energy; cheap end = small INT + large K/W, accurate end =
    FP32 K=0.  This is the headline deliverable.

Both functions accept the full point list from ``sweep_all_designs``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np

from .accuracy_eval import error_stats_fp, error_stats_int, fp_samples, int_samples
from .approx_mac import RoundingMode, approx_mac_fp, approx_mac_int
from .exact_mac import FPFormat, IntFormat, decode_fp, encode_fp
from .power_model import Energy, mac_fp_energy, mac_int_energy


# ============================================================
# Data point
# ============================================================

@dataclass(frozen=True)
class DesignPoint:
    fmt_name: str
    is_fp: bool
    K: int
    aca_window: int | None          # None = exact carry chain
    energy_pJ: float
    energy_breakdown: Energy
    error_med: float                # mean |err|, absolute
    error_rmse: float               # absolute RMS error
    error_nmed: float               # MED / format dynamic range (within-format only)
    error_max_abs: float
    error_nrmse: float = 0.0        # relative RMSE vs float64 ref — primary axis
    error_rel_bias: float = 0.0     # mean(err)/rms(ref); tracks Contribution-A bias

    def __repr__(self) -> str:
        w = "—" if self.aca_window is None else str(self.aca_window)
        return (
            f"DesignPoint({self.fmt_name}, K={self.K}, W={w}, "
            f"E={self.energy_pJ:.3f} pJ, "
            f"NRMSE={self.error_nrmse:.3g}, bias={self.error_rel_bias:+.2g})"
        )


# ============================================================
# Equivalent-W deduplication
# ============================================================

def _canonicalise_windows(
    aca_windows: Sequence[int | None],
    acc_bits: int,
) -> list[int | None]:
    """Collapse any W >= acc_bits to None, then deduplicate, preserving order.

    For a signed ``acc_bits``-wide accumulator, ``aca_add`` with window >=
    acc_bits takes the exact carry path.  Keeping both ``None`` and 32 (when
    acc_bits=32) produces DesignPoints with identical (energy, error) that
    pollute the Pareto front with false duplicates.
    """
    seen: set[int | None] = set()
    out: list[int | None] = []
    for w in aca_windows:
        canonical = None if (w is None or w >= acc_bits) else w
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


# ============================================================
# Default per-format K ranges
# ============================================================

# Maximum meaningful K per format:
#   INT:  product has 2*bits width; truncating > bits is pointless.
#   FP:   mantissa product is 2*(mant_bits+1) wide.
# We cap at slightly below the full product width so at least a few bits remain.

DEFAULT_K_RANGES: dict[str, range] = {
    "INT4":  range(0, 5),    # 4-bit operands -> 8-bit product; K > 4 kills it
    "INT8":  range(0, 8),    # 8-bit operands -> 16-bit product
    "INT16": range(0, 13),   # 16-bit operands -> 32-bit product; K up to 12
    "E4M3":  range(0, 5),    # 3-bit mantissa + hidden -> product 8 bits
    "E5M2":  range(0, 5),    # 2-bit mantissa + hidden -> product 6 bits
    "FP16":  range(0, 8),    # 10-bit mantissa + hidden -> product 22 bits
    "BF16":  range(0, 6),    # 7-bit mantissa + hidden -> product 16 bits
    "FP32":  range(0, 12),   # 23-bit mantissa + hidden -> product 48 bits
}


def _k_range_for(fmt: IntFormat | FPFormat) -> range:
    return DEFAULT_K_RANGES.get(fmt.name, range(0, 7))


# ============================================================
# Pareto front  (metric-agnostic, smaller-is-better on both axes)
# ============================================================

_VALID_KEYS = {
    "energy_pJ", "error_med", "error_rmse", "error_nmed",
    "error_max_abs", "error_nrmse",
}

# Metrics NOT comparable across formats.
_CROSS_FORMAT_BLOCKED = {"error_nmed"}


def dominates(
    a: DesignPoint,
    b: DesignPoint,
    *,
    x_key: str = "energy_pJ",
    y_key: str = "error_nrmse",
) -> bool:
    """``a`` dominates ``b`` iff a <= b on both axes and strictly < on one."""
    if x_key not in _VALID_KEYS or y_key not in _VALID_KEYS:
        raise ValueError(f"keys must be in {_VALID_KEYS}")
    ax, ay = getattr(a, x_key), getattr(a, y_key)
    bx, by = getattr(b, x_key), getattr(b, y_key)
    return ax <= bx and ay <= by and (ax < bx or ay < by)


def pareto_front(
    points: Sequence[DesignPoint],
    *,
    x_key: str = "energy_pJ",
    y_key: str = "error_nrmse",
) -> list[DesignPoint]:
    """Non-dominated subset of ``points``. O(n^2), fine for <= a few hundred points."""
    if x_key not in _VALID_KEYS or y_key not in _VALID_KEYS:
        raise ValueError(f"keys must be in {_VALID_KEYS}")
    pts = list(points)
    front = []
    for i, p in enumerate(pts):
        if not any(
            dominates(q, p, x_key=x_key, y_key=y_key)
            for j, q in enumerate(pts) if j != i
        ):
            front.append(p)
    return front


def sort_front_by_energy(points: Sequence[DesignPoint]) -> list[DesignPoint]:
    return sorted(points, key=lambda p: p.energy_pJ)


def per_format_front(
    points: Sequence[DesignPoint],
    fmt_name: str,
    *,
    y_key: str = "error_nrmse",
) -> list[DesignPoint]:
    """Pareto-optimal (K, W) configs within one format, sorted by energy.

    Use this to understand each format's own K+W efficiency curve before
    combining formats in ``global_front``.
    """
    sub = [p for p in points if p.fmt_name == fmt_name]
    return sort_front_by_energy(pareto_front(sub, y_key=y_key))


def per_format_fronts(
    points: Sequence[DesignPoint],
    *,
    y_key: str = "error_nrmse",
) -> dict[str, list[DesignPoint]]:
    """Run ``per_format_front`` for every format present in ``points``.

    Returns ``{fmt_name: sorted_front}``.  Convenient for plotting all
    per-format efficiency curves in one call.
    """
    fmt_names = dict.fromkeys(p.fmt_name for p in points)
    return {
        name: per_format_front(points, name, y_key=y_key)
        for name in fmt_names
    }


def global_front(
    points: Sequence[DesignPoint],
    *,
    y_key: str = "error_nrmse",
) -> list[DesignPoint]:
    """Cross-format Pareto front on (energy_pJ, error_nrmse), sorted by energy.

    ``error_nrmse`` (relative RMSE vs float64) is the only metric safe for
    cross-format comparison; ``error_nmed`` is blocked because its denominator
    (``max_val**2``) differs by orders of magnitude between INT4 and INT16.
    """
    if y_key in _CROSS_FORMAT_BLOCKED:
        raise ValueError(
            f"y_key={y_key!r} is not comparable across formats. "
            f"Use 'error_nrmse' for cross-format Pareto."
        )
    return sort_front_by_energy(pareto_front(points, y_key=y_key))


# ============================================================
# ACA noise-floor filter
# ============================================================

MAX_NRMSE_FACTOR = 10.0


def _filter_usable(points: list[DesignPoint]) -> list[DesignPoint]:
    """Remove configs whose NRMSE exceeds MAX_NRMSE_FACTOR * format baseline.

    Baseline for each format is the K=0, W=None point (pure quantisation
    error, no approximation).  If no such baseline point exists in the list,
    the format's points pass through unfiltered (conservative default).

    This eliminates ACA configurations where the window is so narrow that
    carry-isolation error dominates and the config is useless in practice
    (e.g. INT4 K=4 W=4 -> NRMSE >> 1).
    """
    baselines: dict[str, float] = {}
    for p in points:
        if p.K == 0 and p.aca_window is None:
            baselines[p.fmt_name] = p.error_nrmse

    usable = []
    for p in points:
        base = baselines.get(p.fmt_name)
        if base is None:
            usable.append(p)
            continue
        # Small floor prevents INT16 (near-zero quant error) from filtering everything.
        threshold = max(base * MAX_NRMSE_FACTOR, 1e-6)
        if p.error_nrmse <= threshold:
            usable.append(p)
    return usable


# ============================================================
# Shared float64 signal
# ============================================================

def make_operands(
    n_dot: int,
    L: int,
    *,
    weight_std: float = 1.0,
    act: str = "relu",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """``n_dot`` pairs of length-``L`` float64 vectors (activations x weights).

    ``act="relu"`` clips activations to non-negative (half-normal), matching
    typical post-ReLU distributions in DNN inference.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, weight_std, size=(n_dot, L))
    b = rng.normal(0.0, weight_std, size=(n_dot, L))
    if act == "relu":
        a = np.maximum(a, 0.0)
    elif act != "normal":
        raise ValueError(f"unknown act {act!r}; use 'relu' or 'normal'")
    return a, b


def float_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """float64 dot products — the single ground-truth reference."""
    return np.einsum("nl,nl->n", a.astype(np.float64), b.astype(np.float64))


def acc_bits_for(fmt: IntFormat, L: int, *, guard: int = 4) -> int:
    """Accumulator width (bits) that won't overflow a length-``L`` INT dot product."""
    return 2 * fmt.bits + max(1, math.ceil(math.log2(max(1, L)))) + guard


# ============================================================
# Per-config evaluation
# ============================================================

class _Err(NamedTuple):
    nrmse: float
    rel_bias: float
    med: float
    rmse: float
    max_abs: float


def _stats(out: np.ndarray, ref: np.ndarray) -> _Err:
    err = out - ref
    denom = math.sqrt(float(np.mean(ref ** 2))) + 1e-12
    return _Err(
        nrmse=math.sqrt(float(np.mean(err ** 2))) / denom,
        rel_bias=float(np.mean(err)) / denom,
        med=float(np.mean(np.abs(err))),
        rmse=math.sqrt(float(np.mean(err ** 2))),
        max_abs=float(np.max(np.abs(err))),
    )


def _int_scale(x: np.ndarray, fmt: IntFormat) -> float:
    m = float(np.max(np.abs(x)))
    return m / fmt.max_val if m > 0 else 1.0


def eval_int_config(
    a: np.ndarray,
    b: np.ndarray,
    ref: np.ndarray,
    fmt: IntFormat,
    *,
    K: int,
    aca_window: int | None,
    acc_bits: int,
    rounding: RoundingMode = "trunc",
    seed: int = 0,
) -> _Err:
    """Quantise the shared signal into ``fmt`` and score one (K, W) config.

    Exact accumulation (W None or >= acc_bits) is vectorised for speed.
    Windowed ACA uses the scalar approx_mac_int loop.
    """
    s_a, s_b = _int_scale(a, fmt), _int_scale(b, fmt)
    qa = np.clip(np.rint(a / s_a), fmt.min_val, fmt.max_val).astype(np.int64)
    qb = np.clip(np.rint(b / s_b), fmt.min_val, fmt.max_val).astype(np.int64)
    exact_acc = (aca_window is None) or (aca_window >= acc_bits)

    if exact_acc:
        prod = qa * qb
        if K > 0:
            if rounding == "round":
                prod = prod + (1 << (K - 1))
            elif rounding == "stochastic":
                gen = np.random.default_rng(seed)
                prod = prod + gen.integers(0, 1 << K, size=prod.shape, dtype=np.int64)
            prod = prod & ~((1 << K) - 1)
        acc = prod.sum(axis=1)
        mask = (1 << acc_bits) - 1
        half = 1 << (acc_bits - 1)
        acc = acc & mask
        acc = np.where(acc >= half, acc - (1 << acc_bits), acc)
        out = s_a * s_b * acc.astype(np.float64)
    else:
        import random as _random
        rng_sr = _random.Random(seed) if rounding == "stochastic" else None
        out = np.empty(len(ref), dtype=np.float64)
        for n in range(len(ref)):
            acc = 0
            row_a, row_b = qa[n], qb[n]
            for i in range(qa.shape[1]):
                acc = approx_mac_int(
                    int(row_a[i]), int(row_b[i]), acc, fmt,
                    K=K, aca_window=aca_window, acc_bits=acc_bits,
                    rounding=rounding, rng=rng_sr,
                )
            out[n] = s_a * s_b * acc

    return _stats(out, ref)


def eval_fp_config(
    a: np.ndarray,
    b: np.ndarray,
    ref: np.ndarray,
    fmt: FPFormat,
    *,
    K: int,
    rounding: RoundingMode = "trunc",
    seed: int = 0,
) -> _Err:
    """Encode the shared signal into ``fmt`` and score one K config."""
    import random as _random
    rng_sr = _random.Random(seed) if rounding == "stochastic" else None
    za = np.vectorize(lambda v: encode_fp(float(v), fmt))(a).astype(np.int64)
    zb = np.vectorize(lambda v: encode_fp(float(v), fmt))(b).astype(np.int64)
    zero = encode_fp(0.0, fmt)
    out = np.empty(len(ref), dtype=np.float64)
    for n in range(len(ref)):
        acc = zero
        for i in range(za.shape[1]):
            acc = approx_mac_fp(
                int(za[n, i]), int(zb[n, i]), acc, fmt,
                K=K, rounding=rounding, rng=rng_sr,
            )
        out[n] = decode_fp(acc, fmt)
    return _stats(out, ref)


# ============================================================
# Sweeps
# ============================================================

def sweep_int_designs(
    fmts: Sequence[IntFormat],
    K_values: Sequence[int] | None,
    aca_windows: Sequence[int | None],
    *,
    n_dot: int = 150,
    L: int = 128,
    act: str = "relu",
    rounding: RoundingMode = "trunc",
    seed: int = 0,
    filter_usable: bool = True,
) -> list[DesignPoint]:
    """Sweep the INT design space on the shared float64 reference.

    Parameters
    ----------
    K_values:
        Pass ``None`` to use ``DEFAULT_K_RANGES`` per format (recommended).
    aca_windows:
        W >= acc_bits entries are collapsed to ``None`` before sweeping,
        eliminating duplicate DesignPoints.
    filter_usable:
        Drop configs whose NRMSE > MAX_NRMSE_FACTOR * baseline (default True).
    """
    a, b = make_operands(n_dot, L, act=act, seed=seed)
    ref = float_reference(a, b)
    points: list[DesignPoint] = []
    for fmt in fmts:
        acc_bits = acc_bits_for(fmt, L)
        k_vals = K_values if K_values is not None else _k_range_for(fmt)
        windows = _canonicalise_windows(aca_windows, acc_bits)
        for K in k_vals:
            for W in windows:
                e = eval_int_config(
                    a, b, ref, fmt, K=K, aca_window=W,
                    acc_bits=acc_bits, rounding=rounding, seed=seed + 7,
                )
                en = mac_int_energy(fmt, K=K, aca_window=W, acc_bits=acc_bits, rounding=rounding)
                points.append(DesignPoint(
                    fmt_name=fmt.name, is_fp=False, K=K, aca_window=W,
                    energy_pJ=en.total_pJ, energy_breakdown=en,
                    error_med=e.med, error_rmse=e.rmse,
                    error_nmed=e.nrmse,
                    error_max_abs=e.max_abs,
                    error_nrmse=e.nrmse,
                    error_rel_bias=e.rel_bias,
                ))
    if filter_usable:
        points = _filter_usable(points)
    return points


def sweep_fp_designs(
    fmts: Sequence[FPFormat],
    K_values: Sequence[int] | None,
    *,
    n_dot: int = 150,
    L: int = 128,
    act: str = "relu",
    rounding: RoundingMode = "trunc",
    seed: int = 0,
) -> list[DesignPoint]:
    """Sweep the FP design space on the shared float64 reference.

    ``K_values=None`` uses ``DEFAULT_K_RANGES`` per format (e.g. E4M3 caps at K=4).
    """
    a, b = make_operands(n_dot, L, act=act, seed=seed)
    ref = float_reference(a, b)
    points: list[DesignPoint] = []
    for fmt in fmts:
        k_vals = K_values if K_values is not None else _k_range_for(fmt)
        for K in k_vals:
            e = eval_fp_config(a, b, ref, fmt, K=K, rounding=rounding, seed=seed + 7)
            en = mac_fp_energy(fmt, K=K, rounding=rounding)
            points.append(DesignPoint(
                fmt_name=fmt.name, is_fp=True, K=K, aca_window=None,
                energy_pJ=en.total_pJ, energy_breakdown=en,
                error_med=e.med, error_rmse=e.rmse,
                error_nmed=e.nrmse,
                error_max_abs=e.max_abs,
                error_nrmse=e.nrmse,
                error_rel_bias=e.rel_bias,
            ))
    return points


def sweep_all_designs(
    int_fmts: Sequence[IntFormat],
    fp_fmts: Sequence[FPFormat],
    int_K_values: Sequence[int] | None = None,
    int_aca_windows: Sequence[int | None] = (4, 8, 16, None),
    fp_K_values: Sequence[int] | None = None,
    *,
    n_dot: int = 150,
    L: int = 128,
    act: str = "relu",
    rounding: RoundingMode = "trunc",
    seed: int = 0,
    filter_usable: bool = True,
) -> list[DesignPoint]:
    """Run INT and FP sweeps on one shared signal and concatenate.

    Recommended entry point.  Pass the result to ``global_front`` or
    ``per_format_fronts``.

    Defaults
    --------
    ``int_K_values=None``       Use DEFAULT_K_RANGES per format.
    ``int_aca_windows``         [4, 8, 16, None] — narrow to exact carry chain.
    ``fp_K_values=None``        Use DEFAULT_K_RANGES per format.
    ``filter_usable=True``      Drop ACA configs too noisy to be useful.
    """
    return [
        *sweep_int_designs(
            int_fmts, int_K_values, int_aca_windows,
            n_dot=n_dot, L=L, act=act, rounding=rounding,
            seed=seed, filter_usable=filter_usable,
        ),
        *sweep_fp_designs(
            fp_fmts, fp_K_values,
            n_dot=n_dot, L=L, act=act, rounding=rounding, seed=seed,
        ),
    ]


# ============================================================
# Legacy sweeps (per-MAC, in-format reference, kept for accuracy_eval tests)
# ============================================================
# DO NOT use for cross-format Pareto.

def sweep_int_designs_legacy(
    fmts: Sequence[IntFormat],
    K_values: Sequence[int],
    aca_windows: Sequence[int | None],
    *,
    n_samples: int = 1000,
    distribution: str = "uniform",
    acc_bits: int = 32,
    seed: int = 0xA11CE,
) -> list[DesignPoint]:
    """Legacy per-MAC sweep (zero accumulator, in-format NMED)."""
    points: list[DesignPoint] = []
    for fmt in fmts:
        a = int_samples(fmt, n_samples, distribution=distribution, seed=seed)  # type: ignore[arg-type]
        b = int_samples(fmt, n_samples, distribution=distribution, seed=seed + 1)  # type: ignore[arg-type]
        acc = [0] * n_samples
        for K in K_values:
            for W in aca_windows:
                energy = mac_int_energy(fmt, K=K, aca_window=W, acc_bits=acc_bits)
                stats = error_stats_int(fmt, a, b, acc, K=K, aca_window=W, acc_bits=acc_bits)
                points.append(DesignPoint(
                    fmt_name=fmt.name, is_fp=False, K=K, aca_window=W,
                    energy_pJ=energy.total_pJ, energy_breakdown=energy,
                    error_med=stats.med, error_rmse=stats.rmse,
                    error_nmed=stats.nmed, error_max_abs=stats.max_abs_err,
                    error_nrmse=stats.nmed, error_rel_bias=0.0,
                ))
    return points


def sweep_fp_designs_legacy(
    fmts: Sequence[FPFormat],
    K_values: Sequence[int],
    *,
    n_samples: int = 1000,
    distribution: str = "uniform",
    scale: float = 1.0,
    seed: int = 0xB0B,
) -> list[DesignPoint]:
    """Legacy per-MAC FP sweep (in-format NMED)."""
    points: list[DesignPoint] = []
    for fmt in fmts:
        a = fp_samples(fmt, n_samples, distribution=distribution, scale=scale, seed=seed)  # type: ignore[arg-type]
        b = fp_samples(fmt, n_samples, distribution=distribution, scale=scale, seed=seed + 1)  # type: ignore[arg-type]
        acc = fp_samples(fmt, n_samples, distribution=distribution, scale=scale, seed=seed + 2)  # type: ignore[arg-type]
        for K in K_values:
            energy = mac_fp_energy(fmt, K=K)
            stats = error_stats_fp(fmt, a, b, acc, K=K)
            points.append(DesignPoint(
                fmt_name=fmt.name, is_fp=True, K=K, aca_window=None,
                energy_pJ=energy.total_pJ, energy_breakdown=energy,
                error_med=stats.med, error_rmse=stats.rmse,
                error_nmed=stats.nmed, error_max_abs=stats.max_abs_err,
                error_nrmse=stats.nmed, error_rel_bias=0.0,
            ))
    return points


# ============================================================
# Smoke test
# ============================================================

if __name__ == "__main__":
    from .exact_mac import BF16, FP8_E4M3, FP8_E5M2, FP16, FP32, INT4, INT8, INT16

    pts = sweep_all_designs(
        int_fmts=[INT4, INT8, INT16],
        fp_fmts=[FP8_E4M3, FP8_E5M2, FP16, BF16, FP32],
        # K_values=None -> use DEFAULT_K_RANGES per format
        int_aca_windows=[4, 8, 16, None],
        n_dot=80, L=64, act="relu",
    )

    print(f"{len(pts)} configs after usability filter")

    print("\n--- per-format Pareto fronts ---")
    for fmt_name, front in per_format_fronts(pts).items():
        print(f"  {fmt_name} ({len(front)} points):")
        for p in front:
            print(f"    K={p.K} W={str(p.aca_window):>4}  "
                  f"E={p.energy_pJ:.4f} pJ  NRMSE={p.error_nrmse:.4e}")

    print("\n--- global (energy, NRMSE) front ---")
    for p in global_front(pts):
        print(f"  {p.fmt_name:5s} K={p.K} W={str(p.aca_window):>4}  "
              f"E={p.energy_pJ:.4f} pJ  NRMSE={p.error_nrmse:.4e}  "
              f"bias={p.error_rel_bias:+.2e}")