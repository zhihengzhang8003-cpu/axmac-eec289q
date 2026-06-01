"""W6 tests for axmac.pareto.

Contracts:
  1. dominates() implements weak Pareto dominance (≤ both, < on at least one).
  2. pareto_front returns only non-dominated points.
  3. The exact config (K=0, W=None) is always on the front for its format
     under any error metric (it has the lowest error → cannot be dominated).
  4. The smallest-format / most-truncated config is always on the front
     under energy-vs-error (it has the lowest energy → cannot be dominated).
  5. Sweep returns one DesignPoint per (fmt, K, W) Cartesian product.
"""

from __future__ import annotations

import pytest

from axmac.exact_mac import BF16, FP16, FP32, INT4, INT8, INT16
from axmac.pareto import (
    DesignPoint,
    dominates,
    pareto_front,
    sort_front_by_energy,
    sweep_all_designs,
    sweep_fp_designs,
    sweep_int_designs,
)
from axmac.power_model import Energy


def _mk_point(*, energy, err, fmt="INT8", K=0, W=None) -> DesignPoint:
    return DesignPoint(
        fmt_name=fmt,
        is_fp=False,
        K=K,
        aca_window=W,
        energy_pJ=energy,
        energy_breakdown=Energy(energy / 2, energy / 2),
        error_med=err,
        error_rmse=err,
        error_nmed=err,
        error_max_abs=err,
    )


# ============================================================
# dominates()
# ============================================================

def test_dominates_strict_better_on_both():
    a = _mk_point(energy=1.0, err=0.1)
    b = _mk_point(energy=2.0, err=0.2)
    assert dominates(a, b, x_key="energy_pJ", y_key="error_med")
    assert not dominates(b, a, x_key="energy_pJ", y_key="error_med")


def test_dominates_equal_on_one_strict_on_other():
    a = _mk_point(energy=1.0, err=0.1)
    b = _mk_point(energy=1.0, err=0.2)
    assert dominates(a, b, x_key="energy_pJ", y_key="error_med")
    assert not dominates(b, a, x_key="energy_pJ", y_key="error_med")


def test_dominates_equal_on_both_is_false():
    a = _mk_point(energy=1.0, err=0.1)
    b = _mk_point(energy=1.0, err=0.1)
    assert not dominates(a, b, x_key="energy_pJ", y_key="error_med")
    assert not dominates(b, a, x_key="energy_pJ", y_key="error_med")


def test_dominates_tradeoff_neither_dominates():
    a = _mk_point(energy=1.0, err=0.5)  # cheap but high error
    b = _mk_point(energy=5.0, err=0.1)  # expensive but accurate
    assert not dominates(a, b, x_key="energy_pJ", y_key="error_med")
    assert not dominates(b, a, x_key="energy_pJ", y_key="error_med")


def test_dominates_rejects_invalid_key():
    a = _mk_point(energy=1.0, err=0.1)
    b = _mk_point(energy=2.0, err=0.2)
    with pytest.raises(ValueError):
        dominates(a, b, x_key="energy", y_key="error_med")  # type: ignore[arg-type]


# ============================================================
# pareto_front
# ============================================================

def test_pareto_front_empty():
    assert pareto_front([]) == []


def test_pareto_front_single_point():
    p = _mk_point(energy=1.0, err=0.1)
    assert pareto_front([p]) == [p]


def test_pareto_front_only_returns_non_dominated():
    points = [
        _mk_point(energy=1.0, err=0.5),   # extreme cheap
        _mk_point(energy=3.0, err=0.1),   # extreme accurate
        _mk_point(energy=2.0, err=0.3),   # tradeoff: between the two
        _mk_point(energy=5.0, err=0.6),   # dominated by everything
    ]
    front = pareto_front(points, x_key="energy_pJ", y_key="error_med")
    front_set = {(p.energy_pJ, p.error_med) for p in front}
    assert (1.0, 0.5) in front_set
    assert (3.0, 0.1) in front_set
    assert (2.0, 0.3) in front_set
    assert (5.0, 0.6) not in front_set


def test_pareto_front_duplicates_kept():
    """Two identical points: neither dominates the other (strict inequality fails)."""
    p1 = _mk_point(energy=1.0, err=0.1)
    p2 = _mk_point(energy=1.0, err=0.1, K=2)  # different K so not the same object
    front = pareto_front([p1, p2])
    assert len(front) == 2


def test_pareto_front_with_alt_metric():
    """Front under error_rmse should match front under error_med when both
    metrics agree (we use same value in _mk_point)."""
    points = [_mk_point(energy=e, err=err) for e, err in [(1, 0.5), (2, 0.1), (3, 0.3)]]
    f1 = pareto_front(points, x_key="energy_pJ", y_key="error_med")
    f2 = pareto_front(points, x_key="energy_pJ", y_key="error_rmse")
    assert {(p.energy_pJ, p.error_med) for p in f1} == {(p.energy_pJ, p.error_rmse) for p in f2}


def test_pareto_front_rejects_invalid_key():
    points = [_mk_point(energy=1.0, err=0.1)]
    with pytest.raises(ValueError):
        pareto_front(points, x_key="bogus", y_key="error_med")


# ============================================================
# sort_front_by_energy
# ============================================================

def test_sort_front_by_energy():
    points = [_mk_point(energy=e, err=0.1) for e in [3.0, 1.0, 2.0]]
    sorted_pts = sort_front_by_energy(points)
    assert [p.energy_pJ for p in sorted_pts] == [1.0, 2.0, 3.0]


# ============================================================
# Sweeps
# ============================================================

def test_sweep_int_grid_size():
    fmts = [INT4, INT8]
    Ks = [0, 2, 4]
    Ws = [None]  # single canonical window avoids deduplication ambiguity
    points = sweep_int_designs(fmts, Ks, Ws, n_dot=20, L=16, filter_usable=False)
    assert len(points) == len(fmts) * len(Ks)
    assert all(not p.is_fp for p in points)


def test_sweep_fp_grid_size():
    fmts = [FP16, BF16]
    Ks = [0, 1, 2, 4]
    points = sweep_fp_designs(fmts, Ks, n_dot=20, L=16)
    assert len(points) == len(fmts) * len(Ks)
    assert all(p.is_fp for p in points)


def test_sweep_baseline_zero_error():
    """K=0 / W=None yields the smallest error: pure quantisation noise vs float64."""
    points = sweep_int_designs([INT8], [0], [None], n_dot=50, L=32)
    assert len(points) == 1
    # Not exactly zero (quantisation noise vs float64 reference), but small.
    assert points[0].error_nrmse < 0.05


def test_sweep_all_designs_concatenates():
    pts = sweep_all_designs(
        int_fmts=[INT8],
        fp_fmts=[FP16],
        int_K_values=[0, 2],
        int_aca_windows=[None],
        fp_K_values=[0, 2],
        n_dot=20, L=16,
        filter_usable=False,
    )
    assert len(pts) == 4  # 2 INT + 2 FP
    assert sum(not p.is_fp for p in pts) == 2
    assert sum(p.is_fp for p in pts) == 2


# ============================================================
# Cross-format Pareto: smallest format dominates energy axis;
# largest format K=0 dominates accuracy axis.
# ============================================================

def test_within_format_pareto_shows_tradeoffs():
    """Inside a single format, the (energy, NRMSE) front is non-trivial:
    K=0 is the high-energy/low-error corner, K=max is the cheapest/most-
    erroneous corner, and at least one intermediate K can be on the front.
    filter_usable=False keeps all K values regardless of noise level.
    """
    points = sweep_int_designs([INT8], [0, 1, 2, 4, 6], [None], n_dot=50, L=32, filter_usable=False)
    front = pareto_front(points, x_key="energy_pJ", y_key="error_nrmse")
    assert any(p.K == 0 for p in front), "K=0 (corner of error axis) must be on front"
    assert any(p.K == 6 for p in front), "K=6 (corner of energy axis) must be on front"
    assert len(front) >= 2


def test_cross_format_pareto_spans_formats():
    """Global Pareto front (NRMSE vs energy) spans from cheap/imprecise to accurate.

    With NRMSE vs float64 as the cross-format error metric, INT4 K=0 carries
    larger quantisation noise than INT16 K=0, so the front is multi-format:
    INT4 occupies the low-energy end, INT16 the high-accuracy end.
    """
    points = sweep_int_designs([INT4, INT8, INT16], [0, 2, 4], [None], n_dot=50, L=32)
    front = pareto_front(points, x_key="energy_pJ", y_key="error_nrmse")
    fmts_on_front = {p.fmt_name for p in front}
    assert "INT4" in fmts_on_front, "INT4 must be on front (lowest energy)"
    assert "INT16" in fmts_on_front, "INT16 must be on front (lowest NRMSE)"
    assert len(fmts_on_front) > 1, "front must span multiple formats"
    # The lowest-energy point is always INT4.
    min_e_pt = min(front, key=lambda p: p.energy_pJ)
    assert min_e_pt.fmt_name == "INT4"


def test_pareto_front_lowest_energy_is_on_front():
    """The point with strictly lowest energy is always on the front."""
    points = sweep_int_designs([INT4, INT8, INT16], [0, 2, 4], [None], n_dot=30, L=16)
    front = pareto_front(points, x_key="energy_pJ", y_key="error_med")
    min_energy = min(p.energy_pJ for p in points)
    assert any(p.energy_pJ == min_energy for p in front)


def test_design_point_repr_contains_metrics():
    p = _mk_point(energy=1.5, err=0.02, fmt="INT8", K=3, W=8)
    r = repr(p)
    for tag in ["INT8", "K=3", "W=8", "E=1.500", "NRMSE=", "bias="]:
        assert tag in r
