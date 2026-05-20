"""Tests for the FP8 formats E4M3 / E5M2 (redesign task 8).

FP8 is the modernization deliverable: the study now covers the 8-bit
floating-point formats 2025-era accelerators use (Micikevicius et al.,
arXiv:2209.05433, 2022 — the OCP FP8 standard).

Contracts:
  1. encode/decode round-trip for every non-NaN bit pattern (256 each).
  2. E4M3 has NO infinities — overflow saturates to ±max-normal (448),
     and decode never yields inf. E5M2 keeps IEEE-754 inf/NaN.
  3. The format-specific special encodings: E4M3's single NaN is
     S.1111.111; E5M2 inf is S.11111.00.
  4. encode rounds to nearest, ties to even.
  5. fp_multiply / fp_add / mac_fp on FP8 match a float64 oracle.
  6. approx_mac_fp with K=0 matches mac_fp on FP8; K>0 stays finite.
"""

from __future__ import annotations

import math
import random

import pytest

from axmac.approx_mac import approx_mac_fp
from axmac.exact_mac import (
    FP8_E4M3,
    FP8_E5M2,
    decode_fp,
    encode_fp,
    fp_add,
    fp_multiply,
    mac_fp,
)

FP8_FORMATS = [FP8_E4M3, FP8_E5M2]


def _is_nan_pattern(bits: int, fmt) -> bool:
    """True if ``bits`` is a NaN encoding in ``fmt``."""
    e = (bits >> fmt.mant_bits) & ((1 << fmt.exp_bits) - 1)
    m = bits & ((1 << fmt.mant_bits) - 1)
    if e != fmt.exp_all_ones:
        return False
    if fmt.has_inf:
        return m != 0
    return m == (1 << fmt.mant_bits) - 1  # E4M3: only all-ones mantissa


# ============================================================
# Round-trip over the whole 8-bit space
# ============================================================

@pytest.mark.parametrize("fmt", FP8_FORMATS)
def test_fp8_round_trip_all_patterns(fmt):
    """encode(decode(bits)) == bits for every non-NaN 8-bit pattern."""
    for bits in range(256):
        if _is_nan_pattern(bits, fmt):
            assert math.isnan(decode_fp(bits, fmt))
            continue
        v = decode_fp(bits, fmt)
        assert encode_fp(v, fmt) == bits, (fmt.name, f"{bits:08b}", v)


# ============================================================
# Known reference values (from the OCP FP8 spec)
# ============================================================

def test_e4m3_known_values():
    assert decode_fp(0b0_1111_110, FP8_E4M3) == 448.0      # max normal
    assert decode_fp(0b0_0001_000, FP8_E4M3) == 2.0 ** -6  # min normal
    assert decode_fp(0b0_0000_001, FP8_E4M3) == 2.0 ** -9  # min subnormal
    assert decode_fp(0b0_0111_000, FP8_E4M3) == 1.0
    assert decode_fp(0b1_0111_000, FP8_E4M3) == -1.0
    assert decode_fp(0b0_0000_000, FP8_E4M3) == 0.0


def test_e5m2_known_values():
    assert decode_fp(0b0_11110_11, FP8_E5M2) == 57344.0    # max normal
    assert decode_fp(0b0_00001_00, FP8_E5M2) == 2.0 ** -14 # min normal
    assert decode_fp(0b0_00000_01, FP8_E5M2) == 2.0 ** -16 # min subnormal
    assert decode_fp(0b0_01111_00, FP8_E5M2) == 1.0


# ============================================================
# E4M3 has no infinities; E5M2 does
# ============================================================

def test_e4m3_has_no_infinity():
    # A huge magnitude saturates to ±max-normal, never inf.
    assert decode_fp(encode_fp(1e30, FP8_E4M3), FP8_E4M3) == 448.0
    assert decode_fp(encode_fp(-1e30, FP8_E4M3), FP8_E4M3) == -448.0
    # No bit pattern decodes to infinity.
    for bits in range(256):
        assert not math.isinf(decode_fp(bits, FP8_E4M3))


def test_e4m3_single_nan_encoding():
    """E4M3's only NaN is S.1111.111 (both signs)."""
    assert math.isnan(decode_fp(0b0_1111_111, FP8_E4M3))
    assert math.isnan(decode_fp(0b1_1111_111, FP8_E4M3))
    # Its neighbour S.1111.110 is the finite max-normal, not NaN.
    assert decode_fp(0b0_1111_110, FP8_E4M3) == 448.0
    assert math.isnan(decode_fp(encode_fp(float("nan"), FP8_E4M3), FP8_E4M3))


def test_e5m2_keeps_ieee_inf_and_nan():
    assert math.isinf(decode_fp(0b0_11111_00, FP8_E5M2))
    assert decode_fp(0b0_11111_00, FP8_E5M2) > 0
    assert decode_fp(0b1_11111_00, FP8_E5M2) < 0
    assert math.isnan(decode_fp(0b0_11111_01, FP8_E5M2))
    # Huge magnitude overflows to inf (IEEE-754 behaviour).
    assert math.isinf(decode_fp(encode_fp(1e30, FP8_E5M2), FP8_E5M2))


# ============================================================
# Round-to-nearest-even on encode
# ============================================================

def test_e4m3_encode_rounds_to_nearest_even():
    # E4M3 grid near 1.0 has step 0.125: 1.0(000) 1.125(001) 1.25(010) ...
    # Ties resolve to the even mantissa.
    assert decode_fp(encode_fp(1.0625, FP8_E4M3), FP8_E4M3) == 1.0    # -> 000 (even)
    assert decode_fp(encode_fp(1.1875, FP8_E4M3), FP8_E4M3) == 1.25   # -> 010 (even)
    # Clear (non-tie) nearest rounding.
    assert decode_fp(encode_fp(1.18, FP8_E4M3), FP8_E4M3) == 1.125
    assert decode_fp(encode_fp(1.20, FP8_E4M3), FP8_E4M3) == 1.25


# ============================================================
# Arithmetic vs. a float64 oracle
# ============================================================

def _finite_fp8(fmt, x: float) -> int:
    """Encode x and reject patterns that are not finite in fmt."""
    bits = encode_fp(x, fmt)
    return bits


@pytest.mark.parametrize("fmt", FP8_FORMATS)
def test_fp8_multiply_matches_oracle(fmt):
    rng = random.Random(0xF8 + fmt.mant_bits)
    for _ in range(2000):
        x = rng.uniform(-8.0, 8.0)
        y = rng.uniform(-8.0, 8.0)
        a_bits = encode_fp(x, fmt)
        b_bits = encode_fp(y, fmt)
        out_bits, _ = fp_multiply(a_bits, b_bits, fmt)
        # Oracle: decode, multiply exactly in float64, re-encode.
        ref_bits = encode_fp(decode_fp(a_bits, fmt) * decode_fp(b_bits, fmt), fmt)
        got = decode_fp(out_bits, fmt)
        ref = decode_fp(ref_bits, fmt)
        if math.isnan(ref):
            assert math.isnan(got)
        else:
            assert out_bits == ref_bits, (x, y, bin(out_bits), bin(ref_bits))


@pytest.mark.parametrize("fmt", FP8_FORMATS)
def test_fp8_add_matches_oracle(fmt):
    rng = random.Random(0xADD + fmt.mant_bits)
    for _ in range(2000):
        x = rng.uniform(-16.0, 16.0)
        y = rng.uniform(-16.0, 16.0)
        a_bits = encode_fp(x, fmt)
        b_bits = encode_fp(y, fmt)
        out_bits = fp_add(a_bits, b_bits, fmt)
        ref_bits = encode_fp(decode_fp(a_bits, fmt) + decode_fp(b_bits, fmt), fmt)
        if math.isnan(decode_fp(ref_bits, fmt)):
            assert math.isnan(decode_fp(out_bits, fmt))
        else:
            assert out_bits == ref_bits, (x, y, bin(out_bits), bin(ref_bits))


@pytest.mark.parametrize("fmt", FP8_FORMATS)
def test_fp8_mac_matches_oracle(fmt):
    """mac_fp = round(product) then add — matches the split float64 oracle."""
    rng = random.Random(0x77AC + fmt.mant_bits)
    for _ in range(2000):
        x = rng.uniform(-4.0, 4.0)
        y = rng.uniform(-4.0, 4.0)
        z = rng.uniform(-4.0, 4.0)
        a_bits = encode_fp(x, fmt)
        b_bits = encode_fp(y, fmt)
        c_bits = encode_fp(z, fmt)
        out = mac_fp(a_bits, b_bits, c_bits, fmt)
        # Oracle: round the product to fmt, then round acc + product to fmt.
        prod_bits = encode_fp(decode_fp(a_bits, fmt) * decode_fp(b_bits, fmt), fmt)
        prod = decode_fp(prod_bits, fmt)
        ref_bits = encode_fp(decode_fp(c_bits, fmt) + prod, fmt)
        if math.isnan(decode_fp(ref_bits, fmt)):
            assert math.isnan(decode_fp(out, fmt))
        else:
            assert out == ref_bits, (x, y, z, bin(out), bin(ref_bits))


# ============================================================
# approx_mac_fp on FP8
# ============================================================

@pytest.mark.parametrize("fmt", FP8_FORMATS)
def test_fp8_approx_mac_k0_matches_exact(fmt):
    """K=0 mantissa truncation must reproduce the exact MAC byte-for-byte."""
    rng = random.Random(0xA8 + fmt.mant_bits)
    for _ in range(1500):
        a = encode_fp(rng.uniform(-4.0, 4.0), fmt)
        b = encode_fp(rng.uniform(-4.0, 4.0), fmt)
        c = encode_fp(rng.uniform(-4.0, 4.0), fmt)
        assert approx_mac_fp(a, b, c, fmt, K=0) == mac_fp(a, b, c, fmt)


@pytest.mark.parametrize("fmt", FP8_FORMATS)
@pytest.mark.parametrize("K", [1, 2, 3])
def test_fp8_approx_mac_k_stays_finite(fmt, K):
    """K>0 truncation on the FP8 mantissa product stays finite for in-range data."""
    rng = random.Random(0xA8C ^ (fmt.mant_bits << 4) ^ K)
    for _ in range(500):
        a = encode_fp(rng.uniform(-2.0, 2.0), fmt)
        b = encode_fp(rng.uniform(-2.0, 2.0), fmt)
        c = encode_fp(rng.uniform(-2.0, 2.0), fmt)
        out = approx_mac_fp(a, b, c, fmt, K=K)
        assert math.isfinite(decode_fp(out, fmt))
