"""W3 tests for axmac.power_model.

Three contracts:
  1. K=0 + aca_window=None returns the calibrated Horowitz 45 nm baseline
     (locks the datasheet so later refactors can't silently drift).
  2. Monotonicity: more truncation -> less mult energy; smaller ACA window
     -> less adder energy.
  3. Switching-activity helpers obey their definitions (Hamming distance,
     identical inputs ⇒ 0 activity, all-flipped ⇒ N bits).
"""

from __future__ import annotations

import random

import pytest

from axmac.exact_mac import BF16, FP16, FP32, INT4, INT8, INT16, booth_radix4_pps
from axmac.power_model import (
    ACC32_ADD_PJ,
    Energy,
    aca_savings,
    base_add_pJ,
    base_mult_pJ,
    datasheet_snapshot,
    energy_from_activity,
    hamming_distance,
    mac_fp_energy,
    mac_int_energy,
    mantissa_switching_activity,
    pp_switching_activity,
    truncation_savings,
)


# ============================================================
# Datasheet baselines
# ============================================================

@pytest.mark.parametrize(
    "fmt, mult_pJ, add_pJ",
    [
        (INT4, 0.050, 0.015),
        (INT8, 0.200, 0.030),
        (INT16, 0.800, 0.060),
    ],
)
def test_int_datasheet(fmt, mult_pJ, add_pJ):
    assert base_mult_pJ(fmt) == pytest.approx(mult_pJ)
    assert base_add_pJ(fmt) == pytest.approx(add_pJ)


@pytest.mark.parametrize(
    "fmt, mult_pJ, add_pJ",
    [
        (FP16, 1.100, 0.400),
        (BF16, 0.950, 0.380),
        (FP32, 3.700, 0.900),
    ],
)
def test_fp_datasheet(fmt, mult_pJ, add_pJ):
    assert base_mult_pJ(fmt) == pytest.approx(mult_pJ)
    assert base_add_pJ(fmt) == pytest.approx(add_pJ)


def test_int_baseline_matches_horowitz():
    """K=0 + default acc returns the 45 nm baseline; ACC32_ADD_PJ is the adder cost."""
    e = mac_int_energy(INT8)
    assert e.multiplier_pJ == pytest.approx(0.200)
    assert e.adder_pJ == pytest.approx(ACC32_ADD_PJ)


def test_fp_baseline_matches_horowitz():
    e = mac_fp_energy(FP16)
    assert e.multiplier_pJ == pytest.approx(1.100)
    assert e.adder_pJ == pytest.approx(0.400)


def test_datasheet_snapshot_covers_all_formats():
    snap = datasheet_snapshot()
    assert set(snap.keys()) == {"INT4", "INT8", "INT16", "FP16", "BF16", "FP32"}
    for label, energy in snap.items():
        assert isinstance(energy, Energy)
        assert energy.total_pJ > 0


# ============================================================
# Approximation savings models
# ============================================================

def test_truncation_savings_zero_at_k_zero():
    for n in [4, 8, 16, 24]:
        assert truncation_savings(n, 0) == 0.0


def test_truncation_savings_increasing_in_k():
    for n in [8, 16]:
        prev = -1.0
        for K in range(0, n):
            s = truncation_savings(n, K)
            assert s >= prev - 1e-12, (n, K, s, prev)
            prev = s


def test_truncation_savings_capped():
    """Extreme K (full truncation) should not exceed the cap."""
    assert truncation_savings(8, 16) <= 0.95
    assert truncation_savings(8, 100) <= 0.95


def test_aca_savings_none_or_full_window_is_zero():
    assert aca_savings(32, None) == 0.0
    assert aca_savings(32, 32) == 0.0
    assert aca_savings(32, 64) == 0.0


def test_aca_savings_increasing_as_window_shrinks():
    prev = -1.0
    for W in [32, 24, 16, 8, 4, 2, 1]:
        s = aca_savings(32, W)
        assert s >= prev - 1e-12, (W, s, prev)
        prev = s
    assert aca_savings(32, 1) == pytest.approx(31 / 32 * 0.5)


# ============================================================
# Composite mac_*_energy monotonicity
# ============================================================

@pytest.mark.parametrize("fmt", [INT4, INT8, INT16])
def test_int_mac_energy_monotone_in_K(fmt):
    base = mac_int_energy(fmt, K=0).multiplier_pJ
    prev = base + 1e-9
    for K in range(0, fmt.bits + 1):
        m = mac_int_energy(fmt, K=K).multiplier_pJ
        assert m <= prev + 1e-12, (K, m, prev)
        prev = m


@pytest.mark.parametrize("fmt", [INT8, INT16])
def test_int_mac_energy_monotone_in_aca(fmt):
    prev = mac_int_energy(fmt, aca_window=32).adder_pJ + 1e-9
    for W in [32, 24, 16, 8, 4, 2, 1]:
        a = mac_int_energy(fmt, aca_window=W).adder_pJ
        assert a <= prev + 1e-12, (W, a, prev)
        prev = a


@pytest.mark.parametrize("fmt", [FP16, BF16, FP32])
def test_fp_mac_energy_monotone_in_K(fmt):
    prev = mac_fp_energy(fmt, K=0).multiplier_pJ + 1e-9
    for K in range(0, fmt.mant_bits + 2):
        m = mac_fp_energy(fmt, K=K).multiplier_pJ
        assert m <= prev + 1e-12, (K, m, prev)
        prev = m


def test_int_mac_energy_acc_bits_scaling():
    """Adder cost scales linearly with acc_bits (no ACA)."""
    e32 = mac_int_energy(INT8, acc_bits=32).adder_pJ
    e16 = mac_int_energy(INT8, acc_bits=16).adder_pJ
    e64 = mac_int_energy(INT8, acc_bits=64).adder_pJ
    assert e16 == pytest.approx(e32 / 2)
    assert e64 == pytest.approx(e32 * 2)


# ============================================================
# Validation against approx_mac semantics
# ============================================================

def test_negative_K_rejected():
    with pytest.raises(ValueError):
        mac_int_energy(INT8, K=-1)
    with pytest.raises(ValueError):
        mac_fp_energy(FP16, K=-1)


def test_aca_window_zero_rejected():
    with pytest.raises(ValueError):
        mac_int_energy(INT8, aca_window=0)


# ============================================================
# Hamming distance / switching activity
# ============================================================

def test_hamming_distance_basic():
    assert hamming_distance(0, 0, 8) == 0
    assert hamming_distance(0xFF, 0x00, 8) == 8
    assert hamming_distance(0b1010_1010, 0b0101_0101, 8) == 8
    assert hamming_distance(0b1111_0000, 0b1100_0011, 8) == 4


def test_hamming_distance_masks_to_window():
    # Bit 9 is outside the 8-bit window and must be ignored.
    assert hamming_distance(0, 1 << 9, 8) == 0
    assert hamming_distance(0xFF, 0xFF | (1 << 9), 8) == 0


def test_hamming_distance_handles_negative():
    # -1 in two's complement is all-ones within any window.
    assert hamming_distance(-1, 0, 8) == 8
    assert hamming_distance(-1, -1, 16) == 0


def test_pp_switching_activity_zero_when_identical():
    pps = booth_radix4_pps(13, -7, 8)
    assert pp_switching_activity(pps, pps, 2 * 8) == 0


def test_pp_switching_activity_random_consecutive():
    """Random uncorrelated inputs should average roughly half the PP bits."""
    rng = random.Random(0xACAC)
    bits_per_pp = 16
    total_dist = 0
    samples = 0
    for _ in range(500):
        a1, b1 = rng.randint(-128, 127), rng.randint(-128, 127)
        a2, b2 = rng.randint(-128, 127), rng.randint(-128, 127)
        prev = booth_radix4_pps(a1, b1, 8)
        cur = booth_radix4_pps(a2, b2, 8)
        total_dist += pp_switching_activity(prev, cur, bits_per_pp)
        samples += len(prev) * bits_per_pp
    avg_alpha = total_dist / samples
    assert 0.35 <= avg_alpha <= 0.65, avg_alpha


def test_pp_switching_length_mismatch_rejected():
    with pytest.raises(ValueError):
        pp_switching_activity([1, 2, 3], [1, 2], 8)


def test_mantissa_switching_activity_zero_when_identical():
    assert mantissa_switching_activity(0xDEAD, 0xDEAD, 10) == 0
    assert mantissa_switching_activity(0, 0, 23) == 0


def test_mantissa_switching_activity_full_flip():
    # 2*(mant_bits+1) bit window. mant_bits=10 ⇒ 22-bit window.
    mask = (1 << 22) - 1
    assert mantissa_switching_activity(mask, 0, 10) == 22


# ============================================================
# energy_from_activity
# ============================================================

def test_energy_from_activity_zero_when_no_switching():
    assert energy_from_activity(INT8, 0) == 0.0
    assert energy_from_activity(FP16, 0) == 0.0


def test_energy_from_activity_scales_linearly_in_activity():
    e1 = energy_from_activity(INT8, 8)
    e2 = energy_from_activity(INT8, 16)
    e3 = energy_from_activity(INT8, 32)
    assert e2 == pytest.approx(2 * e1)
    assert e3 == pytest.approx(4 * e1)


def test_energy_from_activity_matches_base_at_baseline_alpha():
    """If observed alpha == baseline_alpha, the data-dependent energy
    should equal the datasheet number."""
    # INT8: total_bits = 64, baseline_alpha = 0.5 ⇒ activity_bits = 32.
    e = energy_from_activity(INT8, 32, baseline_alpha=0.5)
    assert e == pytest.approx(base_mult_pJ(INT8))


def test_energy_from_activity_rejects_bad_alpha():
    with pytest.raises(ValueError):
        energy_from_activity(INT8, 1, baseline_alpha=0.0)
    with pytest.raises(ValueError):
        energy_from_activity(INT8, 1, baseline_alpha=1.5)


def test_energy_from_activity_rejects_negative_activity():
    with pytest.raises(ValueError):
        energy_from_activity(INT8, -1)
