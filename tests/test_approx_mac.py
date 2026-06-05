"""W2 regression + bounded-error tests for axmac.approx_mac.

Two contracts:
  1. K=0 + aca_window=None must match axmac.exact_mac byte-for-byte across
     INT4/8/16 and FP16/BF16/FP32 — this is the baseline guarantee.
  2. K>0 / aca_window<acc_bits introduce bounded, predictable error:
     INT truncation drops at most 2^K - 1; ACA sum differs from exact sum
     in the segments where a carry would have crossed a boundary.
"""

from __future__ import annotations

import math
import random

import pytest

import numpy as np

from axmac.approx_mac import aca_add, approx_mac_fp, approx_mac_int
from axmac.approx_mac import drum_quantize_operand, drum_multiply, approx_mac_int_drum
from axmac.dnn_inference import _drum_quantize_arr, int_matmul_drum, int_conv2d_drum
from axmac.exact_mac import (
    BF16,
    FP16,
    FP32,
    INT4,
    INT8,
    INT16,
    encode_fp,
    mac_fp,
    mac_int,
)


# ============================================================
# K=0 / window=None equivalence to exact MAC
# ============================================================

@pytest.mark.parametrize("fmt", [INT4, INT8, INT16])
def test_int_k0_matches_exact(fmt):
    rng = random.Random(0xA001 + fmt.bits)
    for _ in range(2000):
        a = rng.randint(fmt.min_val, fmt.max_val)
        b = rng.randint(fmt.min_val, fmt.max_val)
        acc = rng.randint(-(1 << 31), (1 << 31) - 1)
        assert approx_mac_int(a, b, acc, fmt, K=0) == mac_int(a, b, acc, fmt)


@pytest.mark.parametrize("fmt", [INT4, INT8, INT16])
def test_int_aca_full_window_matches_exact(fmt):
    rng = random.Random(0xA002 + fmt.bits)
    for _ in range(1000):
        a = rng.randint(fmt.min_val, fmt.max_val)
        b = rng.randint(fmt.min_val, fmt.max_val)
        acc = rng.randint(-(1 << 31), (1 << 31) - 1)
        # window >= acc_bits ⇒ no carry isolation
        got = approx_mac_int(a, b, acc, fmt, K=0, aca_window=32)
        assert got == mac_int(a, b, acc, fmt)
        got2 = approx_mac_int(a, b, acc, fmt, K=0, aca_window=64)
        assert got2 == mac_int(a, b, acc, fmt)


@pytest.mark.parametrize("fmt", [FP16, BF16, FP32])
def test_fp_k0_matches_exact(fmt):
    rng = random.Random(0xB000 + fmt.total_bits)
    # Mix small and large magnitudes, signed.
    sample_floats = []
    for _ in range(500):
        sample_floats.append(rng.uniform(-10.0, 10.0))
    for _ in range(500):
        sample_floats.append(rng.uniform(-1e3, 1e3))
    for _ in range(200):
        sample_floats.append(rng.uniform(-1e-3, 1e-3))

    for _ in range(1500):
        x = rng.choice(sample_floats)
        y = rng.choice(sample_floats)
        z = rng.choice(sample_floats)
        a_bits = encode_fp(x, fmt)
        b_bits = encode_fp(y, fmt)
        c_bits = encode_fp(z, fmt)
        assert approx_mac_fp(a_bits, b_bits, c_bits, fmt, K=0) == mac_fp(a_bits, b_bits, c_bits, fmt)


def test_fp_k0_specials_match():
    fmt = FP32
    # Build a battery of special FP bit patterns.
    pos_inf = (0xFF << 23)
    neg_inf = (1 << 31) | (0xFF << 23)
    qnan = (0xFF << 23) | (1 << 22)
    pos_zero = 0
    neg_zero = 1 << 31
    one = encode_fp(1.0, fmt)
    neg_one = encode_fp(-1.0, fmt)
    seven = encode_fp(7.5, fmt)

    cases = [
        (pos_inf, one, neg_one),
        (one, pos_inf, seven),
        (pos_zero, pos_inf, seven),
        (pos_inf, neg_zero, seven),
        (neg_zero, pos_zero, seven),
        (qnan, seven, one),
        (seven, neg_one, neg_one),
    ]
    for a_bits, b_bits, c_bits in cases:
        # NaN equality: bit patterns must match because we route through the
        # same _make_qnan helper.
        assert approx_mac_fp(a_bits, b_bits, c_bits, fmt, K=0) == mac_fp(a_bits, b_bits, c_bits, fmt)


# ============================================================
# Truncation has the right magnitude
# ============================================================

@pytest.mark.parametrize("fmt", [INT8, INT16])
@pytest.mark.parametrize("K", [1, 2, 4, 8])
def test_int_truncation_drops_low_bits(fmt, K):
    """approx_mac_int(K) = mac_int(0) with bottom K bits of (a*b) zeroed."""
    rng = random.Random(0xC0DE ^ (fmt.bits << 4) ^ K)
    for _ in range(500):
        a = rng.randint(fmt.min_val, fmt.max_val)
        b = rng.randint(fmt.min_val, fmt.max_val)
        acc = 0
        got = approx_mac_int(a, b, acc, fmt, K=K)
        # Expected: (a*b) with bottom K bits zeroed, then wrapped to 32 bits.
        truncated = (a * b) & ~((1 << K) - 1)
        wrapped = truncated & 0xFFFFFFFF
        if wrapped >= (1 << 31):
            wrapped -= 1 << 32
        assert got == wrapped, (a, b, K, got, wrapped)


@pytest.mark.parametrize("fmt", [INT8, INT16])
@pytest.mark.parametrize("K", [1, 2, 4])
def test_int_truncation_error_bounded(fmt, K):
    """|exact - approx| < 2^K when no accumulator wrap is triggered."""
    rng = random.Random(0xBAD ^ (fmt.bits << 8) ^ K)
    bound = 1 << K
    for _ in range(500):
        a = rng.randint(fmt.min_val, fmt.max_val)
        b = rng.randint(fmt.min_val, fmt.max_val)
        # acc small enough that acc + a*b never crosses 32-bit boundary
        acc = rng.randint(-(1 << 24), (1 << 24) - 1)
        exact = mac_int(a, b, acc, fmt)
        approx = approx_mac_int(a, b, acc, fmt, K=K)
        assert 0 <= exact - approx < bound, (a, b, K, exact, approx)


# ============================================================
# ACA adder direct unit tests
# ============================================================

def test_aca_full_window_matches_plain_add():
    rng = random.Random(0xACA0)
    for _ in range(1000):
        a = rng.randint(-(1 << 31), (1 << 31) - 1)
        b = rng.randint(-(1 << 31), (1 << 31) - 1)
        expected = (a + b) & 0xFFFFFFFF
        if expected >= (1 << 31):
            expected -= 1 << 32
        assert aca_add(a, b, 32, 32) == expected
        assert aca_add(a, b, 32, 64) == expected


def test_aca_window_1_zero_carry():
    """window=1 means each bit-add ignores carry — equivalent to XOR."""
    rng = random.Random(0xACA1)
    for _ in range(500):
        a = rng.randint(0, (1 << 32) - 1)
        b = rng.randint(0, (1 << 32) - 1)
        xor = a ^ b
        if xor >= (1 << 31):
            xor -= 1 << 32
        assert aca_add(a, b, 32, 1) == xor


@pytest.mark.parametrize("window", [4, 8, 16])
def test_aca_segment_independence(window):
    """Each segment sums independently; flipping bits in segment j must not
    change the sum in segment i != j."""
    rng = random.Random(0xACA2 + window)
    bits = 32
    for _ in range(200):
        a = rng.randint(0, (1 << bits) - 1)
        b = rng.randint(0, (1 << bits) - 1)
        base = aca_add(a, b, bits, window)
        # Flip bit in some other segment of b.
        flip_segment = rng.randint(0, bits // window - 1)
        flip_bit = flip_segment * window + rng.randint(0, window - 1)
        b2 = b ^ (1 << flip_bit)
        flipped = aca_add(a, b2, bits, window)
        # Mask out the segment that contained the flip; everything else equal.
        other_mask = ((1 << bits) - 1) ^ (((1 << window) - 1) << (flip_segment * window))
        # Use unsigned-bit view for the comparison.
        assert (base & other_mask) == (flipped & other_mask)


# ============================================================
# FP truncation: bounded relative error
# ============================================================

@pytest.mark.parametrize("fmt", [FP16, BF16, FP32])
@pytest.mark.parametrize("K", [1, 2, 4])
def test_fp_truncation_relative_error(fmt, K):
    """Mantissa truncation by K bits yields product within 2^-mant_bits relative error
    (very loose bound — actual error is much smaller for small K)."""
    from axmac.exact_mac import decode_fp

    rng = random.Random(0xF00D ^ (fmt.total_bits << 4) ^ K)
    # acc = 0 isolates the multiply step
    zero_bits = encode_fp(0.0, fmt)
    for _ in range(300):
        x = rng.uniform(-10.0, 10.0)
        y = rng.uniform(-10.0, 10.0)
        if abs(x * y) < 1e-6:
            continue
        a_bits = encode_fp(x, fmt)
        b_bits = encode_fp(y, fmt)
        exact = decode_fp(mac_fp(a_bits, b_bits, zero_bits, fmt), fmt)
        approx = decode_fp(approx_mac_fp(a_bits, b_bits, zero_bits, fmt, K=K), fmt)
        if exact == 0.0:
            continue
        rel = abs(exact - approx) / abs(exact)
        # Truncating K bits of an (mb+1)-bit mantissa product is at most
        # 2^(K - 2*mb - 1) relative on the unrounded product, then RNE
        # rounds to the format. Use a slack bound.
        loose = 2 ** (K - fmt.mant_bits + 1)
        assert rel <= loose, (x, y, K, rel, loose, exact, approx)


# ============================================================
# Rounding modes — Contribution A (error-compensated multiplier)
# ============================================================
#
# Beyond the W2 baseline, approx_mac_int / approx_mac_fp accept a `rounding`
# knob: "trunc" (the biased baseline), "round" (add the deterministic
# correction constant 2^(K-1); Schulte & Swartzlander 1993) and "stochastic"
# (add a uniform random offset in [0, 2^K); Gupta et al. 2015). These tests
# pin the contract of the two new modes.

@pytest.mark.parametrize("fmt", [INT4, INT8, INT16])
@pytest.mark.parametrize("rounding", ["trunc", "round", "stochastic"])
def test_int_k0_matches_exact_every_rounding(fmt, rounding):
    """K=0 is the exact product whatever the rounding mode — the baseline
    guarantee must survive the Contribution-A extension."""
    rng = random.Random(0xD00D + fmt.bits)
    for _ in range(1000):
        a = rng.randint(fmt.min_val, fmt.max_val)
        b = rng.randint(fmt.min_val, fmt.max_val)
        acc = rng.randint(-(1 << 31), (1 << 31) - 1)
        got = approx_mac_int(a, b, acc, fmt, K=0, rounding=rounding,
                             rng=random.Random(1))
        assert got == mac_int(a, b, acc, fmt)


@pytest.mark.parametrize("fmt", [INT8, INT16])
@pytest.mark.parametrize("K", [1, 2, 4, 6])
def test_int_round_error_bounded_and_two_signed(fmt, K):
    """`round` keeps |exact-approx| <= 2^(K-1) — half the trunc window — and
    for K>=2 the error is two-signed, unlike the one-signed bias of plain
    truncation. (K=1 is degenerate: the dropped field is a single bit, so the
    +2^(K-1)=+1 tie-break only ever rounds up — error is {0, -1}.)"""
    rng = random.Random(0x5A1 ^ (fmt.bits << 4) ^ K)
    half = 1 << (K - 1)
    low_mask = (1 << K) - 1
    seen_neg = seen_pos = False
    for _ in range(800):
        a = rng.randint(fmt.min_val, fmt.max_val)
        b = rng.randint(fmt.min_val, fmt.max_val)
        exact = mac_int(a, b, 0, fmt)
        approx = approx_mac_int(a, b, 0, fmt, K=K, rounding="round")
        err = exact - approx
        assert -half <= err <= half, (a, b, K, err)
        assert approx & low_mask == 0          # low K bits cleared
        seen_neg |= err < 0
        seen_pos |= err > 0
    assert seen_neg                            # always some downward rounding
    if K >= 2:
        assert seen_pos                        # K>=2: genuinely two-signed


@pytest.mark.parametrize("fmt", [INT8, INT16])
@pytest.mark.parametrize("K", [1, 2, 4, 6])
def test_int_stochastic_error_bounded(fmt, K):
    """`stochastic` keeps |exact-approx| < 2^K and clears the low K bits."""
    rng = random.Random(0x57C ^ (fmt.bits << 4) ^ K)
    bound = 1 << K
    low_mask = bound - 1
    for _ in range(800):
        a = rng.randint(fmt.min_val, fmt.max_val)
        b = rng.randint(fmt.min_val, fmt.max_val)
        exact = mac_int(a, b, 0, fmt)
        approx = approx_mac_int(a, b, 0, fmt, K=K, rounding="stochastic", rng=rng)
        err = exact - approx
        assert -bound < err < bound, (a, b, K, err)
        assert approx & low_mask == 0


def test_int_stochastic_is_rng_deterministic():
    """A seeded Random makes stochastic rounding reproducible; an independent
    seed lets it differ — the randomness is real but fully controllable."""
    fmt = INT8
    pairs = [(a, b) for a in range(-50, 50, 9) for b in range(-50, 50, 9)]

    def run(seed):
        g = random.Random(seed)
        return [approx_mac_int(a, b, 0, fmt, K=5, rounding="stochastic", rng=g)
                for a, b in pairs]

    assert run(2024) == run(2024)          # same seed -> identical stream
    assert run(2024) != run(99)            # different seed -> differs somewhere


@pytest.mark.parametrize("bad", ["nearest", "rne", "", "TRUNC"])
def test_int_rejects_unknown_rounding(bad):
    with pytest.raises(ValueError, match="rounding"):
        approx_mac_int(3, 5, 0, INT8, K=2, rounding=bad)  # type: ignore[arg-type]


def test_fp_rejects_unknown_rounding():
    with pytest.raises(ValueError, match="rounding"):
        approx_mac_fp(encode_fp(1.0, FP32), encode_fp(2.0, FP32),
                      encode_fp(0.0, FP32), FP32, K=2,
                      rounding="bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize("fmt", [FP16, BF16, FP32])
@pytest.mark.parametrize("rounding", ["round", "stochastic"])
def test_fp_k0_matches_exact_every_rounding(fmt, rounding):
    """FP K=0 leaves the mantissa product untouched whatever the rounding
    mode — the exact-MAC baseline holds for Contribution A on the FP path."""
    rng = random.Random(0xBEEF + fmt.total_bits)
    for _ in range(800):
        a_bits = encode_fp(rng.uniform(-50.0, 50.0), fmt)
        b_bits = encode_fp(rng.uniform(-50.0, 50.0), fmt)
        c_bits = encode_fp(rng.uniform(-50.0, 50.0), fmt)
        got = approx_mac_fp(a_bits, b_bits, c_bits, fmt, K=0,
                            rounding=rounding, rng=random.Random(7))
        assert got == mac_fp(a_bits, b_bits, c_bits, fmt)


@pytest.mark.parametrize("fmt", [FP16, BF16])
@pytest.mark.parametrize("rounding", ["round", "stochastic"])
def test_fp_rounding_modes_stay_finite_and_bounded(fmt, rounding):
    """round / stochastic on the FP mantissa product yield finite results
    within the same loose relative bound the trunc path satisfies."""
    from axmac.exact_mac import decode_fp

    rng = random.Random(0xF1A ^ (fmt.total_bits << 3))
    zero = encode_fp(0.0, fmt)
    K = 3
    loose = 2 ** (K - fmt.mant_bits + 1)
    for _ in range(300):
        x = rng.uniform(-8.0, 8.0)
        y = rng.uniform(-8.0, 8.0)
        a_bits = encode_fp(x, fmt)
        b_bits = encode_fp(y, fmt)
        exact = decode_fp(mac_fp(a_bits, b_bits, zero, fmt), fmt)
        approx = decode_fp(
            approx_mac_fp(a_bits, b_bits, zero, fmt, K=K,
                          rounding=rounding, rng=rng),
            fmt,
        )
        assert math.isfinite(approx)
        if abs(exact) > 1e-6:
            assert abs(exact - approx) / abs(exact) <= loose, (x, y, exact, approx)


# ============================================================
# DRUM approximate multiplier tests
# ============================================================

def test_drum_quantize_zero_mean():
    """DRUM error distribution has zero mean (3-sigma statistical test)."""
    rng = random.Random(42)
    N = 50_000
    errors = []
    for _ in range(N):
        a = rng.randint(INT8.min_val, INT8.max_val)
        b = rng.randint(INT8.min_val, INT8.max_val)
        errors.append(drum_multiply(a, b, INT8, k=4) - a * b)
    mean_err = sum(errors) / N
    variance = sum((e - mean_err) ** 2 for e in errors) / N
    std_err = variance ** 0.5
    # 3-sigma confidence interval for the sample mean
    assert abs(mean_err) < 3 * std_err / (N ** 0.5), (
        f"mean_error={mean_err:.4f} is more than 3-sigma from zero "
        f"(std={std_err:.4f}, N={N})"
    )


def test_drum_quantize_operand_basic():
    """drum_quantize_operand: spot-check known values."""
    assert drum_quantize_operand(0, 4) == 0
    # 127 = 0b01111111; msb=6, shift=3 -> (127>>3)<<3 = 15<<3 = 120
    assert drum_quantize_operand(127, 4) == 120
    # -64: abs=64=2^6, msb=6, shift=3 -> (64>>3)<<3=64, sign -> -64
    assert drum_quantize_operand(-64, 4) == -64
    # 1: msb=0, shift=max(0,0-3)=0 -> unchanged
    assert drum_quantize_operand(1, 4) == 1
    # k=0 -> 0 regardless of x
    assert drum_quantize_operand(127, 0) == 0
    assert drum_quantize_operand(-42, 0) == 0


def test_drum_quantize_arr_matches_scalar():
    """_drum_quantize_arr matches scalar drum_quantize_operand element-wise."""
    rng = random.Random(7)
    vals = [rng.randint(INT8.min_val, INT8.max_val) for _ in range(200)]
    arr = np.array(vals, dtype=np.int64)
    result = _drum_quantize_arr(arr, k=4)
    expected = np.array([drum_quantize_operand(v, 4) for v in vals], dtype=np.int64)
    assert np.array_equal(result, expected), (
        f"Mismatch at indices: {np.where(result != expected)[0].tolist()}"
    )


def test_drum_multiply_full_precision_exact():
    """drum_multiply with k=8 (full INT8 precision) equals exact product."""
    rng = random.Random(99)
    for _ in range(100):
        a = rng.randint(INT8.min_val, INT8.max_val)
        b = rng.randint(INT8.min_val, INT8.max_val)
        assert drum_multiply(a, b, INT8, k=8) == a * b, (
            f"a={a}, b={b}: drum_multiply gave {drum_multiply(a, b, INT8, k=8)}, "
            f"exact={a * b}"
        )


def test_int_matmul_drum_shape():
    """int_matmul_drum returns correct shape and int64 dtype."""
    rng = np.random.default_rng(0)
    x = rng.integers(INT8.min_val, INT8.max_val + 1, size=(4, 8), dtype=np.int8)
    w = rng.integers(INT8.min_val, INT8.max_val + 1, size=(8, 16), dtype=np.int8)
    out = int_matmul_drum(x, w, fmt=INT8, k=4)
    assert out.shape == (4, 16), f"Expected shape (4, 16), got {out.shape}"
    assert out.dtype == np.int64, f"Expected int64 dtype, got {out.dtype}"


def test_int_conv2d_drum_shape():
    """int_conv2d_drum returns correct output shape for 3x3 conv with padding=1."""
    rng = np.random.default_rng(1)
    x = rng.integers(INT8.min_val, INT8.max_val + 1, size=(2, 3, 8, 8), dtype=np.int8)
    w = rng.integers(INT8.min_val, INT8.max_val + 1, size=(16, 3, 3, 3), dtype=np.int8)
    out = int_conv2d_drum(x, w, fmt=INT8, k=4, padding=1)
    assert out.shape == (2, 16, 8, 8), f"Expected shape (2, 16, 8, 8), got {out.shape}"
