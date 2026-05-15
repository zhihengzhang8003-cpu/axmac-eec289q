"""Full (precision, K, ACA-window) design-space sweep and Pareto analysis.

Stage 6 deliverable. Joins :mod:`axmac.power_model` energy estimates with
:mod:`axmac.accuracy_eval` error metrics to give per-config (energy, error)
points, then extracts the Pareto-optimal front — the set of configs that
no other config dominates in both dimensions simultaneously.

The :func:`pareto_front` routine is metric-agnostic: pass any two
attribute names of :class:`DesignPoint` (e.g. ``energy_pJ`` and
``error_nmed``, or ``energy_pJ`` and ``error_rmse``) and it returns the
non-dominated subset under "smaller is better" on both axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .accuracy_eval import error_stats_fp, error_stats_int, fp_samples, int_samples
from .exact_mac import FPFormat, IntFormat
from .power_model import Energy, mac_fp_energy, mac_int_energy


# ============================================================
# Data point
# ============================================================

@dataclass(frozen=True)
class DesignPoint:
    fmt_name: str
    is_fp: bool
    K: int
    aca_window: int | None
    energy_pJ: float
    energy_breakdown: Energy
    error_med: float
    error_rmse: float
    error_nmed: float
    error_max_abs: float

    def __repr__(self) -> str:
        w = "—" if self.aca_window is None else str(self.aca_window)
        return (
            f"DesignPoint({self.fmt_name}, K={self.K}, W={w}, "
            f"E={self.energy_pJ:.3f} pJ, "
            f"MED={self.error_med:.3g}, NMED={self.error_nmed:.3g}, "
            f"RMSE={self.error_rmse:.3g})"
        )


# ============================================================
# Sweep
# ============================================================

def sweep_int_designs(
    fmts: Sequence[IntFormat],
    K_values: Sequence[int],
    aca_windows: Sequence[int | None],
    *,
    n_samples: int = 1000,
    distribution: str = "uniform",
    acc_bits: int = 32,
    seed: int = 0xA11CE,
) -> list[DesignPoint]:
    """Sweep the INT design space, returning one DesignPoint per (fmt, K, W)."""
    points: list[DesignPoint] = []
    for fmt in fmts:
        a = int_samples(fmt, n_samples, distribution=distribution, seed=seed)  # type: ignore[arg-type]
        b = int_samples(fmt, n_samples, distribution=distribution, seed=seed + 1)  # type: ignore[arg-type]
        acc = [0] * n_samples  # zero acc isolates per-MAC behavior
        for K in K_values:
            for W in aca_windows:
                energy = mac_int_energy(fmt, K=K, aca_window=W, acc_bits=acc_bits)
                stats = error_stats_int(fmt, a, b, acc, K=K, aca_window=W, acc_bits=acc_bits)
                points.append(DesignPoint(
                    fmt_name=fmt.name,
                    is_fp=False,
                    K=K,
                    aca_window=W,
                    energy_pJ=energy.total_pJ,
                    energy_breakdown=energy,
                    error_med=stats.med,
                    error_rmse=stats.rmse,
                    error_nmed=stats.nmed,
                    error_max_abs=stats.max_abs_err,
                ))
    return points


def sweep_fp_designs(
    fmts: Sequence[FPFormat],
    K_values: Sequence[int],
    *,
    n_samples: int = 1000,
    distribution: str = "uniform",
    scale: float = 1.0,
    seed: int = 0xB0B,
) -> list[DesignPoint]:
    """Sweep the FP design space, returning one DesignPoint per (fmt, K)."""
    points: list[DesignPoint] = []
    for fmt in fmts:
        a = fp_samples(fmt, n_samples, distribution=distribution, scale=scale, seed=seed)  # type: ignore[arg-type]
        b = fp_samples(fmt, n_samples, distribution=distribution, scale=scale, seed=seed + 1)  # type: ignore[arg-type]
        acc = fp_samples(fmt, n_samples, distribution=distribution, scale=scale, seed=seed + 2)  # type: ignore[arg-type]
        for K in K_values:
            energy = mac_fp_energy(fmt, K=K)
            stats = error_stats_fp(fmt, a, b, acc, K=K)
            points.append(DesignPoint(
                fmt_name=fmt.name,
                is_fp=True,
                K=K,
                aca_window=None,
                energy_pJ=energy.total_pJ,
                energy_breakdown=energy,
                error_med=stats.med,
                error_rmse=stats.rmse,
                error_nmed=stats.nmed,
                error_max_abs=stats.max_abs_err,
            ))
    return points


# ============================================================
# Pareto front
# ============================================================

_VALID_KEYS = {"energy_pJ", "error_med", "error_rmse", "error_nmed", "error_max_abs"}


def dominates(a: DesignPoint, b: DesignPoint, *, x_key: str, y_key: str) -> bool:
    """``a`` dominates ``b`` iff a ≤ b on both axes and a < b on at least one.

    Both axes are interpreted as "smaller is better" (energy in pJ, any
    error metric).
    """
    if x_key not in _VALID_KEYS or y_key not in _VALID_KEYS:
        raise ValueError(f"keys must be in {_VALID_KEYS}")
    ax, ay = getattr(a, x_key), getattr(a, y_key)
    bx, by = getattr(b, x_key), getattr(b, y_key)
    return ax <= bx and ay <= by and (ax < bx or ay < by)


def pareto_front(
    points: Sequence[DesignPoint],
    *,
    x_key: str = "energy_pJ",
    y_key: str = "error_nmed",
) -> list[DesignPoint]:
    """Return the non-dominated subset of ``points``.

    Time complexity O(n²) — fine for the dozens-to-hundreds of points
    produced by the design-space sweep.
    """
    if x_key not in _VALID_KEYS or y_key not in _VALID_KEYS:
        raise ValueError(f"keys must be in {_VALID_KEYS}")
    front: list[DesignPoint] = []
    for i, p in enumerate(points):
        dominated_by_other = False
        for j, q in enumerate(points):
            if i == j:
                continue
            if dominates(q, p, x_key=x_key, y_key=y_key):
                dominated_by_other = True
                break
        if not dominated_by_other:
            front.append(p)
    return front


def sort_front_by_energy(points: Sequence[DesignPoint]) -> list[DesignPoint]:
    """Return ``points`` sorted by ascending energy — convenient for plotting."""
    return sorted(points, key=lambda p: p.energy_pJ)


# ============================================================
# Convenience: full INT+FP sweep
# ============================================================

def sweep_all_designs(
    int_fmts: Sequence[IntFormat],
    fp_fmts: Sequence[FPFormat],
    int_K_values: Sequence[int],
    int_aca_windows: Sequence[int | None],
    fp_K_values: Sequence[int],
    *,
    n_samples: int = 1000,
) -> list[DesignPoint]:
    """Convenience wrapper running both INT and FP sweeps and concatenating."""
    return [
        *sweep_int_designs(int_fmts, int_K_values, int_aca_windows, n_samples=n_samples),
        *sweep_fp_designs(fp_fmts, fp_K_values, n_samples=n_samples),
    ]
