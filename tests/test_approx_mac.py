"""W2 regression + bounded-error tests for axmac.approx_mac.

Two contracts:
  1. K=0 + aca_window=None must match axmac.exact_mac byte-for-byte across
     INT4/8/16 and FP16/BF16/FP32 — this is the baseline guarantee.
  2. K>0 / aca_window<acc_bits introduce bounded, predictable error:
     INT truncation drops at most 2^K - 1; ACA sum differs from exact sum
     in the segments where a carry would have crossed a boundary.
"""

from __future__ import annotations

import random

import pytest

from axmac.approx_mac import aca_add, approx_mac_fp, approx_mac_int
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
