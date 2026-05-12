"""Bit-accuracy tests for axmac.exact_mac.

Reference oracles:
  INT  -> Python's arbitrary-precision integers (then two's-complement wrap)
  FP32 -> numpy.float32 native ops
  FP16 -> numpy.float16 native ops (split: round product, then add)
  BF16 -> numpy.float32 ops + truncation-to-BF16 (round-to-nearest-even)
"""

from __future__ import annotations

import math
import random
import struct

import numpy as np
import pytest

from axmac.exact_mac import (
    BF16,
    FP16,
    FP32,
    INT4,
    INT8,
    INT16,
    booth_radix4_pps,
    decode_fp,
    encode_fp,
    fp_add,
    fp_multiply,
    mac_fp,
    mac_int,
)


# ============================================================
# INT MAC
# ============================================================

@pytest.mark.parametrize("fmt", [INT4, INT8, INT16])
def test_booth_pps_recover_product(fmt):
    rng = random.Random(0xB007)
    for _ in range(2000):
        a = rng.randint(fmt.min_val, fmt.max_val)
        b = rng.randint(fmt.min_val, fmt.max_val)
        pps = booth_radix4_pps(a, b, fmt.bits)
        assert sum(pps) == a * b, (a, b, pps)


@pytest.mark.parametrize("fmt", [INT4, INT8, INT16])
def test_int_mac_random(fmt):
    rng = random.Random(0xC0DE + fmt.bits)
    for _ in range(2000):
        a = rng.randint(fmt.min_val, fmt.max_val)
        b = rng.randint(fmt.min_val, fmt.max_val)
        acc = rng.randint(-(1 << 31), (1 << 31) - 1)
        got = mac_int(a, b, acc, fmt)
        expected = (acc + a * b) & 0xFFFFFFFF
        if expected >= (1 << 31):
            expected -= 1 << 32
        assert got == expected, (a, b, acc, got, expected)


@pytest.mark.parametrize("fmt", [INT4, INT8, INT16])
def test_int_mac_min_times_min(fmt):
    # MIN * MIN is the classic edge case (only positive value of magnitude 2^(2n-2)).
    a = b = fmt.min_val
    got = mac_int(a, b, 0, fmt)
    expected = a * b
    assert got == expected


def test_int_mac_wraps_on_overflow():
    # Force accumulator overflow: max 32-bit + 1 wraps to min.
    a, b = INT8.max_val, 1  # product = 127
    acc = (1 << 31) - 1 - 100  # close to max
    got = mac_int(a, b, acc, INT8)
    expected = (acc + a * b) & 0xFFFFFFFF
    if expected >= (1 << 31):
        expected -= 1 << 32
    assert got == expected
    # Sanity: this case should actually wrap.
    assert got < 0


def test_int_mac_returns_pps_on_request():
    res = mac_int(7, 3, 0, INT8, return_pps=True)
    assert res.value == 21
    assert sum(res.partial_products) == 21


def test_int_mac_range_validation():
    with pytest.raises(ValueError):
        mac_int(128, 0, 0, INT8)  # a out of range
    with pytest.raises(ValueError):
        mac_int(0, -129, 0, INT8)  # b out of range


# ============================================================
# FP encode / decode
# ============================================================

def _f32_to_u32(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _f16_np_to_u16(x):
    return int(np.frombuffer(np.float16(x).tobytes(), dtype="<u2")[0])


def _bf16_oracle(x_f32):
    """Round-to-nearest-even encode an f32 to BF16 (top 16 bits)."""
    bits = _f32_to_u32(x_f32)
    e = (bits >> 23) & 0xFF
    m = bits & 0x7FFFFF
    if e == 0xFF and m != 0:
        return ((bits >> 16) | 0x40) & 0xFFFF
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    return ((bits + rounding_bias) >> 16) & 0xFFFF


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_encode_fp32_matches_struct(seed):
    rng = random.Random(seed)
    for _ in range(500):
        x = rng.uniform(-1e30, 1e30)
        assert encode_fp(x, FP32) == _f32_to_u32(x)


def test_encode_fp16_matches_numpy():
    rng = random.Random(0xA1)
    cases = [rng.uniform(-65000, 65000) for _ in range(500)]
    cases += [rng.uniform(-1e-4, 1e-4) for _ in range(500)]  # subnormal range
    cases += [0.0, -0.0, 1.0, -1.0, 65504.0, -65504.0, 1e-7, -1e-7]
    for x in cases:
        # Round x first to f32 to match what encode_fp does internally.
        x32 = float(np.float32(x))
        ours = encode_fp(x32, FP16)
        ref = _f16_np_to_u16(x32)
        # Both sides must agree, including for inf/nan/subnormal.
        if math.isnan(x32):
            continue  # NaN payload is unspecified
        assert ours == ref, (x32, hex(ours), hex(ref))


def test_encode_bf16_matches_oracle():
    rng = random.Random(0xB2)
    for _ in range(2000):
        x = rng.uniform(-1e30, 1e30)
        x32 = float(np.float32(x))
        assert encode_fp(x32, BF16) == _bf16_oracle(x32)


def test_decode_fp_round_trips():
    # encode(decode(x)) == x for non-NaN bit patterns. Use a sweep of bit
    # patterns to also catch subnormals.
    for fmt, n in [(FP16, 0x10000), (BF16, 0x10000)]:
        for bits in range(0, n, 13):  # stride to keep fast
            s, e, m = (bits >> (fmt.exp_bits + fmt.mant_bits)) & 1, \
                      (bits >> fmt.mant_bits) & ((1 << fmt.exp_bits) - 1), \
                      bits & ((1 << fmt.mant_bits) - 1)
            if e == fmt.exp_all_ones and m != 0:
                continue  # NaN
            v = decode_fp(bits, fmt)
            assert encode_fp(v, fmt) == bits, (fmt.name, hex(bits), v)


# ============================================================
# FP multiply
# ============================================================

def _np_dtype(fmt):
    return {FP32: np.float32, FP16: np.float16}[fmt]


@pytest.mark.parametrize("fmt", [FP32, FP16])
def test_fp_multiply_matches_numpy(fmt):
    rng = np.random.default_rng(42 + fmt.mant_bits)
    dtype = _np_dtype(fmt)
    n = 5000
    # Mix scales to exercise different exponents.
    scales = [1e-3, 1.0, 1e3] if fmt is FP16 else [1e-20, 1.0, 1e20]
    a_vals = rng.uniform(-1, 1, n).astype(dtype) * dtype(rng.choice(scales, n))
    b_vals = rng.uniform(-1, 1, n).astype(dtype) * dtype(rng.choice(scales, n))
    for a, b in zip(a_vals, b_vals):
        a_bits = encode_fp(float(a), fmt)
        b_bits = encode_fp(float(b), fmt)
        out_bits, _ = fp_multiply(a_bits, b_bits, fmt)
        ref = (a * b).astype(dtype)
        ref_bits = encode_fp(float(ref), fmt)
        # NaN payloads can differ; accept any NaN as a match for NaN.
        if math.isnan(float(ref)):
            assert math.isnan(decode_fp(out_bits, fmt))
        else:
            assert out_bits == ref_bits, (float(a), float(b), hex(out_bits), hex(ref_bits))


def test_bf16_multiply_matches_oracle():
    rng = np.random.default_rng(0xBF16)
    n = 5000
    scales = [1e-20, 1e-3, 1.0, 1e3, 1e20]
    a_vals = rng.uniform(-1, 1, n).astype(np.float32) * rng.choice(scales, n).astype(np.float32)
    b_vals = rng.uniform(-1, 1, n).astype(np.float32) * rng.choice(scales, n).astype(np.float32)
    for a, b in zip(a_vals, b_vals):
        # Round inputs to BF16 first.
        a_bits = encode_fp(float(a), BF16)
        b_bits = encode_fp(float(b), BF16)
        out_bits, _ = fp_multiply(a_bits, b_bits, BF16)
        # Oracle: decode -> f32 multiply -> RNE round to BF16.
        a_dec = decode_fp(a_bits, BF16)
        b_dec = decode_fp(b_bits, BF16)
        prod_f32 = float(np.float32(a_dec) * np.float32(b_dec))
        ref_bits = _bf16_oracle(prod_f32)
        if math.isnan(prod_f32):
            assert math.isnan(decode_fp(out_bits, BF16))
        else:
            assert out_bits == ref_bits, (a_dec, b_dec, hex(out_bits), hex(ref_bits))


# ============================================================
# FP add
# ============================================================

@pytest.mark.parametrize("fmt", [FP32, FP16])
def test_fp_add_matches_numpy(fmt):
    rng = np.random.default_rng(123 + fmt.mant_bits)
    dtype = _np_dtype(fmt)
    n = 5000
    a_vals = rng.uniform(-100, 100, n).astype(dtype)
    b_vals = rng.uniform(-100, 100, n).astype(dtype)
    for a, b in zip(a_vals, b_vals):
        a_bits = encode_fp(float(a), fmt)
        b_bits = encode_fp(float(b), fmt)
        out_bits = fp_add(a_bits, b_bits, fmt)
        ref = (a + b).astype(dtype)
        ref_bits = encode_fp(float(ref), fmt)
        if math.isnan(float(ref)):
            assert math.isnan(decode_fp(out_bits, fmt))
        else:
            assert out_bits == ref_bits, (float(a), float(b), hex(out_bits), hex(ref_bits))


def test_bf16_add_matches_oracle():
    rng = np.random.default_rng(0xADD16)
    n = 5000
    a_vals = rng.uniform(-1000, 1000, n).astype(np.float32)
    b_vals = rng.uniform(-1000, 1000, n).astype(np.float32)
    for a, b in zip(a_vals, b_vals):
        a_bits = encode_fp(float(a), BF16)
        b_bits = encode_fp(float(b), BF16)
        out_bits = fp_add(a_bits, b_bits, BF16)
        a_dec = decode_fp(a_bits, BF16)
        b_dec = decode_fp(b_bits, BF16)
        sum_f32 = float(np.float32(a_dec) + np.float32(b_dec))
        ref_bits = _bf16_oracle(sum_f32)
        if math.isnan(sum_f32):
            assert math.isnan(decode_fp(out_bits, BF16))
        else:
            assert out_bits == ref_bits, (a_dec, b_dec, hex(out_bits), hex(ref_bits))


# ============================================================
# FP MAC (split: round product, then add)
# ============================================================

@pytest.mark.parametrize("fmt", [FP32, FP16])
def test_fp_mac_matches_numpy(fmt):
    rng = np.random.default_rng(7 + fmt.mant_bits)
    dtype = _np_dtype(fmt)
    n = 3000
    a = rng.uniform(-10, 10, n).astype(dtype)
    b = rng.uniform(-10, 10, n).astype(dtype)
    c = rng.uniform(-10, 10, n).astype(dtype)
    for av, bv, cv in zip(a, b, c):
        out = mac_fp(
            encode_fp(float(av), fmt),
            encode_fp(float(bv), fmt),
            encode_fp(float(cv), fmt),
            fmt,
        )
        ref_prod = (av * bv).astype(dtype)
        ref = (cv + ref_prod).astype(dtype)
        ref_bits = encode_fp(float(ref), fmt)
        if math.isnan(float(ref)):
            assert math.isnan(decode_fp(out, fmt))
        else:
            assert out == ref_bits, (float(av), float(bv), float(cv), hex(out), hex(ref_bits))


def test_bf16_mac_matches_oracle():
    rng = np.random.default_rng(0xBF1AC)
    n = 3000
    a = rng.uniform(-10, 10, n).astype(np.float32)
    b = rng.uniform(-10, 10, n).astype(np.float32)
    c = rng.uniform(-10, 10, n).astype(np.float32)
    for av, bv, cv in zip(a, b, c):
        a_bits = encode_fp(float(av), BF16)
        b_bits = encode_fp(float(bv), BF16)
        c_bits = encode_fp(float(cv), BF16)
        out = mac_fp(a_bits, b_bits, c_bits, BF16)
        # Oracle: split MAC in BF16.
        a_dec = decode_fp(a_bits, BF16)
        b_dec = decode_fp(b_bits, BF16)
        c_dec = decode_fp(c_bits, BF16)
        prod_bf16 = decode_fp(_bf16_oracle(float(np.float32(a_dec) * np.float32(b_dec))), BF16)
        sum_bf16 = _bf16_oracle(float(np.float32(c_dec) + np.float32(prod_bf16)))
        if math.isnan(decode_fp(sum_bf16, BF16)):
            assert math.isnan(decode_fp(out, BF16))
        else:
            assert out == sum_bf16, (a_dec, b_dec, c_dec, hex(out), hex(sum_bf16))


# ============================================================
# Special values
# ============================================================

@pytest.mark.parametrize("fmt", [FP32, FP16, BF16])
def test_special_zero_times_anything(fmt):
    pos_zero = encode_fp(0.0, fmt)
    neg_zero = encode_fp(-0.0, fmt)
    one = encode_fp(1.0, fmt)
    minus_one = encode_fp(-1.0, fmt)
    out, _ = fp_multiply(pos_zero, one, fmt)
    assert decode_fp(out, fmt) == 0.0
    out, _ = fp_multiply(pos_zero, minus_one, fmt)
    assert decode_fp(out, fmt) == 0.0  # value-equal to -0
    assert out == neg_zero
    out, _ = fp_multiply(neg_zero, minus_one, fmt)
    assert out == pos_zero


@pytest.mark.parametrize("fmt", [FP32, FP16, BF16])
def test_special_inf_propagation(fmt):
    inf = encode_fp(float("inf"), fmt)
    neg_inf = encode_fp(float("-inf"), fmt)
    one = encode_fp(1.0, fmt)
    pos_zero = encode_fp(0.0, fmt)

    out, _ = fp_multiply(inf, one, fmt)
    assert out == inf
    out, _ = fp_multiply(inf, encode_fp(-2.0, fmt), fmt)
    assert out == neg_inf
    # inf * 0 -> NaN
    out, _ = fp_multiply(inf, pos_zero, fmt)
    assert math.isnan(decode_fp(out, fmt))


@pytest.mark.parametrize("fmt", [FP32, FP16, BF16])
def test_special_nan_propagation(fmt):
    nan_bits = encode_fp(float("nan"), fmt)
    one = encode_fp(1.0, fmt)
    out, _ = fp_multiply(nan_bits, one, fmt)
    assert math.isnan(decode_fp(out, fmt))
    out = fp_add(nan_bits, one, fmt)
    assert math.isnan(decode_fp(out, fmt))
