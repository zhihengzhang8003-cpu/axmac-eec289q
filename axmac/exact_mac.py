"""Bit-accurate exact MAC reference for INT4/8/16 and FP16/BF16/FP32.

Stage 1 deliverable. The MAC computes ``acc + a * b`` and returns the result
in the same numeric format. Two interfaces are exposed:

* ``mac_int`` — Booth radix-4 partial products + saturating/wrapping
  integer accumulator. Partial products are made available so that the
  Week-3 power model can score switching activity.
* ``mac_fp`` — sign/exponent/mantissa unpacking via ``struct``, integer
  mantissa multiply, IEEE-754 round-to-nearest-even, then a fmt-domain
  add. The mantissa product is exposed for the same reason.

Reference checks live in ``tests/test_exact_mac.py``: every result is
compared against NumPy's native ``float32``/``float16`` arithmetic, and
BF16 results are checked against a round-to-bf16(numpy.float32) oracle.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import NamedTuple


# ============================================================
# Format descriptors
# ============================================================

@dataclass(frozen=True)
class IntFormat:
    name: str
    bits: int

    @property
    def min_val(self) -> int:
        return -(1 << (self.bits - 1))

    @property
    def max_val(self) -> int:
        return (1 << (self.bits - 1)) - 1


INT4 = IntFormat("INT4", 4)
INT8 = IntFormat("INT8", 8)
INT16 = IntFormat("INT16", 16)


@dataclass(frozen=True)
class FPFormat:
    name: str
    exp_bits: int
    mant_bits: int

    @property
    def bias(self) -> int:
        return (1 << (self.exp_bits - 1)) - 1

    @property
    def total_bits(self) -> int:
        return 1 + self.exp_bits + self.mant_bits

    @property
    def exp_all_ones(self) -> int:
        return (1 << self.exp_bits) - 1


FP32 = FPFormat("FP32", 8, 23)
FP16 = FPFormat("FP16", 5, 10)
BF16 = FPFormat("BF16", 8, 7)


# ============================================================
# Result containers (so power_model can introspect)
# ============================================================

class IntMacResult(NamedTuple):
    value: int
    partial_products: list[int]


class FpMacResult(NamedTuple):
    bits: int
    mantissa_product: int  # full integer mantissa product before normalization


# ============================================================
# INT MAC: Booth radix-4 partial products
# ============================================================

_BOOTH_DECODE = (0, 1, 1, 2, -2, -1, -1, 0)


def booth_radix4_pps(a: int, b: int, n: int) -> list[int]:
    """Radix-4 modified-Booth partial products of ``a * b``.

    ``a`` and ``b`` are signed n-bit two's complement integers. Python's
    arithmetic right shift naturally sign-extends, so out-of-range bit
    reads stay correct.
    """
    num_pps = (n + 1) // 2
    pps: list[int] = []
    for i in range(num_pps):
        b_prev = 0 if i == 0 else (b >> (2 * i - 1)) & 1
        b_lo = (b >> (2 * i)) & 1
        b_hi = (b >> (2 * i + 1)) & 1
        code = (b_hi << 2) | (b_lo << 1) | b_prev
        pps.append((_BOOTH_DECODE[code] * a) << (2 * i))
    return pps


def mac_int(
    a: int,
    b: int,
    acc: int,
    fmt: IntFormat,
    *,
    acc_bits: int = 32,
    return_pps: bool = False,
) -> int | IntMacResult:
    """Compute ``acc + a * b`` with bit-accurate two's complement wrapping.

    The accumulator wraps at ``acc_bits`` (default 32, matching typical
    DNN-accelerator MAC arrays). No saturation.
    """
    if not (fmt.min_val <= a <= fmt.max_val):
        raise ValueError(f"a={a} out of range for {fmt.name}")
    if not (fmt.min_val <= b <= fmt.max_val):
        raise ValueError(f"b={b} out of range for {fmt.name}")
    acc_min = -(1 << (acc_bits - 1))
    acc_max = (1 << (acc_bits - 1)) - 1
    if not (acc_min <= acc <= acc_max):
        raise ValueError(f"acc={acc} out of range for {acc_bits}-bit accumulator")

    pps = booth_radix4_pps(a, b, fmt.bits)
    product = sum(pps)
    raw = (acc + product) & ((1 << acc_bits) - 1)
    if raw >= (1 << (acc_bits - 1)):
        raw -= 1 << acc_bits
    if return_pps:
        return IntMacResult(raw, pps)
    return raw


# ============================================================
# FP raw bit codecs
# ============================================================

def f32_to_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def bits_to_f32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def fp_unpack(bits: int, fmt: FPFormat) -> tuple[int, int, int]:
    """Return (sign, biased_exponent, mantissa)."""
    sign = (bits >> (fmt.exp_bits + fmt.mant_bits)) & 1
    exp = (bits >> fmt.mant_bits) & ((1 << fmt.exp_bits) - 1)
    mant = bits & ((1 << fmt.mant_bits) - 1)
    return sign, exp, mant


def fp_pack(sign: int, exp: int, mant: int, fmt: FPFormat) -> int:
    return (sign << (fmt.exp_bits + fmt.mant_bits)) | (exp << fmt.mant_bits) | mant


# ============================================================
# Encode / decode floats <-> fmt bits (round-to-nearest-even)
# ============================================================

def encode_fp(x: float, fmt: FPFormat) -> int:
    """Encode a Python float (IEEE-754 binary64) into ``fmt`` bits with RNE.

    Implementation note: we go float -> f32 first via ``struct``, then for
    half/bf16 perform a small RNE on the trailing bits. This keeps NaN
    payloads and signed zeros consistent across formats.
    """
    if fmt is FP32:
        return f32_to_bits(x)
    bits32 = f32_to_bits(x)
    s = (bits32 >> 31) & 1
    e32 = (bits32 >> 23) & 0xFF
    m32 = bits32 & 0x7FFFFF

    if fmt is BF16:
        if e32 == 0xFF:
            if m32 != 0:
                # Quiet NaN: ensure MSB of resulting mantissa is set.
                return ((bits32 >> 16) | 0x40) & 0xFFFF
            return ((s << 15) | (0xFF << 7)) & 0xFFFF
        rounding_bias = 0x7FFF + ((bits32 >> 16) & 1)
        return ((bits32 + rounding_bias) >> 16) & 0xFFFF

    if fmt is FP16:
        return _f32_bits_to_fp16(bits32)

    raise ValueError(f"Unknown FP format {fmt}")


def decode_fp(bits: int, fmt: FPFormat) -> float:
    """Decode ``fmt`` bits to a Python float (exact, no information loss)."""
    if fmt is FP32:
        return bits_to_f32(bits)
    if fmt is BF16:
        return bits_to_f32((bits & 0xFFFF) << 16)
    if fmt is FP16:
        return _fp16_to_f32(bits & 0xFFFF)
    raise ValueError(f"Unknown FP format {fmt}")


def _f32_bits_to_fp16(bits32: int) -> int:
    """Round f32 bit pattern to fp16 (IEEE-754 binary16), RNE."""
    s = (bits32 >> 31) & 1
    e = (bits32 >> 23) & 0xFF
    m = bits32 & 0x7FFFFF

    if e == 0xFF:
        if m != 0:
            # Quiet NaN, preserve top mantissa bits.
            qnan = 0x200 | ((m >> 13) & 0x1FF)
            if qnan == 0:
                qnan = 0x200
            return (s << 15) | (0x1F << 10) | qnan
        return (s << 15) | (0x1F << 10)

    # Unbiased exponent in f32; need to rebias to f16 (bias 15).
    if e == 0:
        # f32 zero or subnormal — both round to fp16 zero (smallest f32
        # subnormal is ~1.4e-45, far below fp16 underflow ~6e-8).
        return s << 15

    new_e = e - 127 + 15  # tentative biased fp16 exponent
    if new_e >= 0x1F:
        return (s << 15) | (0x1F << 10)  # overflow to inf

    if new_e <= 0:
        # Subnormal in fp16. Construct full mantissa with implicit 1, then
        # right-shift by (1 - new_e) bits with RNE.
        full = (1 << 23) | m  # 24-bit
        shift = 13 + (1 - new_e)  # drop 13 trailing bits + extra subnormal shift
        if shift >= 25:
            return s << 15  # underflow to zero (RNE could give smallest subnormal)
        guard = (full >> (shift - 1)) & 1
        sticky = (full & ((1 << (shift - 1)) - 1)) != 0
        keep = full >> shift
        if guard and (sticky or (keep & 1)):
            keep += 1
        # If keep overflows into bit 10, we promote to smallest normal (e=1).
        if keep >> 10:
            return (s << 15) | (1 << 10)
        return (s << 15) | keep

    # Normal range. Drop 13 mantissa bits with RNE.
    guard = (m >> 12) & 1
    sticky = (m & 0xFFF) != 0
    keep = m >> 13
    if guard and (sticky or (keep & 1)):
        keep += 1
        if keep >> 10:
            keep >>= 1
            new_e += 1
            if new_e >= 0x1F:
                return (s << 15) | (0x1F << 10)
    return (s << 15) | (new_e << 10) | (keep & 0x3FF)


def _fp16_to_f32(bits16: int) -> float:
    s = (bits16 >> 15) & 1
    e = (bits16 >> 10) & 0x1F
    m = bits16 & 0x3FF

    if e == 0x1F:
        e32 = 0xFF
        m32 = (m << 13) if m != 0 else 0
        return bits_to_f32((s << 31) | (e32 << 23) | m32)

    if e == 0:
        if m == 0:
            return bits_to_f32(s << 31)
        # Subnormal: renormalize.
        leading = m.bit_length() - 1  # in [0, 9]
        shift = 10 - leading
        m = (m << shift) & 0x3FF
        e32 = 127 - 15 - shift + 1
        return bits_to_f32((s << 31) | (e32 << 23) | (m << 13))

    e32 = e - 15 + 127
    return bits_to_f32((s << 31) | (e32 << 23) | (m << 13))


# ============================================================
# FP MAC: shared (M, E) representation
# ============================================================
#
# Internally we use the representation V = (-1)^s * M * 2^(E - mant_bits),
# where M is a non-negative integer with leading bit at position
# ``mant_bits`` for normals, or strictly below that for subnormals.

def _to_internal(bits: int, fmt: FPFormat) -> tuple[int, int, int, str]:
    """Unpack to (sign, M, E, kind).

    kind ∈ {"zero", "normal", "subnormal", "inf", "nan"}.
    """
    s, e, m = fp_unpack(bits, fmt)
    if e == fmt.exp_all_ones:
        return s, m, 0, "nan" if m != 0 else "inf"
    if e == 0:
        if m == 0:
            return s, 0, 0, "zero"
        return s, m, 1 - fmt.bias, "subnormal"
    M = (1 << fmt.mant_bits) | m
    return s, M, e - fmt.bias, "normal"


def _make_qnan(fmt: FPFormat) -> int:
    return fp_pack(0, fmt.exp_all_ones, 1 << (fmt.mant_bits - 1), fmt)


def _renormalize_and_pack(sign: int, M: int, E: int, fmt: FPFormat) -> int:
    """Round V = M * 2^(E - mant_bits) into fmt with RNE, in a single shift.

    Doing it in one shot avoids the double-rounding artifact you get from
    "first normalize, then subnormal-shift".
    """
    if M == 0:
        return fp_pack(sign, 0, 0, fmt)

    mb = fmt.mant_bits
    bias = fmt.bias
    max_norm_E = fmt.exp_all_ones - 1 - bias
    min_norm_E = 1 - bias

    L = M.bit_length() - 1
    tentative_E = E + L - mb  # exponent we'd assign in the normal range

    if tentative_E < min_norm_E:
        # Subnormal path: shift M right by (min_norm_E - E) bits with one RNE.
        shift = min_norm_E - E
        if shift > L + 1:
            return fp_pack(sign, 0, 0, fmt)  # underflow to zero
        if shift <= 0:
            keep = M  # already fits below subnormal threshold
        else:
            guard = (M >> (shift - 1)) & 1
            sticky = (M & ((1 << (shift - 1)) - 1)) != 0
            keep = M >> shift
            if guard and (sticky or (keep & 1)):
                keep += 1
        if keep >= (1 << mb):
            # Round-up promoted us to the smallest normal value.
            return fp_pack(sign, 1, keep & ((1 << mb) - 1), fmt)
        return fp_pack(sign, 0, keep, fmt)

    # Normal path: place leading bit at position mb.
    if L > mb:
        shift = L - mb
        guard = (M >> (shift - 1)) & 1
        sticky = (M & ((1 << (shift - 1)) - 1)) != 0
        keep = M >> shift
        if guard and (sticky or (keep & 1)):
            keep += 1
            if keep >> (mb + 1):
                keep >>= 1
                tentative_E += 1
        M = keep
    elif L < mb:
        M <<= mb - L

    if tentative_E > max_norm_E:
        return fp_pack(sign, fmt.exp_all_ones, 0, fmt)  # overflow -> inf
    return fp_pack(sign, tentative_E + bias, M & ((1 << mb) - 1), fmt)


def fp_multiply(a_bits: int, b_bits: int, fmt: FPFormat) -> tuple[int, int]:
    """Multiply ``a * b`` in ``fmt``. Returns (result_bits, mantissa_product).

    The mantissa product is the unrounded integer M_a * M_b, exposed for
    use by the power model and the Week-2 truncated multiplier.
    """
    sa, Ma, Ea, ka = _to_internal(a_bits, fmt)
    sb, Mb, Eb, kb = _to_internal(b_bits, fmt)
    s_out = sa ^ sb

    if ka == "nan" or kb == "nan":
        return _make_qnan(fmt), 0
    if (ka == "inf" and kb == "zero") or (kb == "inf" and ka == "zero"):
        return _make_qnan(fmt), 0
    if ka == "inf" or kb == "inf":
        return fp_pack(s_out, fmt.exp_all_ones, 0, fmt), 0
    if ka == "zero" or kb == "zero":
        return fp_pack(s_out, 0, 0, fmt), 0

    M_prod = Ma * Mb
    E_prod = Ea + Eb - fmt.mant_bits  # see header comment for the algebra
    return _renormalize_and_pack(s_out, M_prod, E_prod, fmt), M_prod


def fp_add(a_bits: int, b_bits: int, fmt: FPFormat) -> int:
    """Add ``a + b`` in ``fmt`` with RNE."""
    sa, Ma, Ea, ka = _to_internal(a_bits, fmt)
    sb, Mb, Eb, kb = _to_internal(b_bits, fmt)

    if ka == "nan" or kb == "nan":
        return _make_qnan(fmt)
    if ka == "inf" and kb == "inf":
        if sa != sb:
            return _make_qnan(fmt)
        return fp_pack(sa, fmt.exp_all_ones, 0, fmt)
    if ka == "inf":
        return fp_pack(sa, fmt.exp_all_ones, 0, fmt)
    if kb == "inf":
        return fp_pack(sb, fmt.exp_all_ones, 0, fmt)
    if ka == "zero" and kb == "zero":
        # IEEE-754: +0 + +0 = +0; -0 + -0 = -0; +0 + -0 = +0 (RNE).
        return fp_pack(sa & sb, 0, 0, fmt)
    if ka == "zero":
        return b_bits
    if kb == "zero":
        return a_bits

    # Align mantissas: shift the smaller-exponent operand right, keeping
    # extra precision bits so RNE has guard/round/sticky available.
    GUARD_BITS = 3  # G, R, S
    Ma_ext = Ma << GUARD_BITS
    Mb_ext = Mb << GUARD_BITS
    if Ea >= Eb:
        diff = Ea - Eb
        # Sticky-collapse the bits shifted off the right.
        if diff > 0:
            shifted_off = Mb_ext & ((1 << diff) - 1) if diff <= Mb_ext.bit_length() else Mb_ext
            sticky = 1 if shifted_off != 0 else 0
            Mb_ext = (Mb_ext >> diff) | sticky
        E = Ea
    else:
        diff = Eb - Ea
        if diff > 0:
            shifted_off = Ma_ext & ((1 << diff) - 1) if diff <= Ma_ext.bit_length() else Ma_ext
            sticky = 1 if shifted_off != 0 else 0
            Ma_ext = (Ma_ext >> diff) | sticky
        E = Eb

    if sa == sb:
        sign_out = sa
        M_sum = Ma_ext + Mb_ext
    else:
        if Ma_ext >= Mb_ext:
            sign_out = sa
            M_sum = Ma_ext - Mb_ext
        else:
            sign_out = sb
            M_sum = Mb_ext - Ma_ext

    if M_sum == 0:
        return fp_pack(0, 0, 0, fmt)  # +0 (RNE; -0 only if both inputs were -0, handled above)

    # Renormalize: leading bit should land at position mb + GUARD_BITS so
    # that dropping GUARD_BITS at the end gives mb-bit fraction.
    target_pos = fmt.mant_bits + GUARD_BITS
    L = M_sum.bit_length() - 1
    if L > target_pos:
        shift = L - target_pos
        # Sticky-collapse the bits we'll drop here too.
        shifted_off = M_sum & ((1 << shift) - 1)
        sticky = 1 if shifted_off != 0 else 0
        M_sum = (M_sum >> shift) | sticky
        E += shift
    elif L < target_pos:
        shift = target_pos - L
        M_sum <<= shift
        E -= shift

    # Now drop the GUARD_BITS with RNE.
    guard = (M_sum >> (GUARD_BITS - 1)) & 1
    sticky = (M_sum & ((1 << (GUARD_BITS - 1)) - 1)) != 0
    keep = M_sum >> GUARD_BITS
    if guard and (sticky or (keep & 1)):
        keep += 1

    return _renormalize_and_pack(sign_out, keep, E, fmt)


def mac_fp(
    a_bits: int,
    b_bits: int,
    acc_bits: int,
    fmt: FPFormat,
    *,
    return_intermediates: bool = False,
) -> int | FpMacResult:
    """Compute ``acc + a * b`` in ``fmt`` (split: round product, then add).

    All three inputs are bit-pattern integers in ``fmt``'s encoding. The
    result is returned in the same encoding.
    """
    prod_bits, mant_prod = fp_multiply(a_bits, b_bits, fmt)
    out = fp_add(acc_bits, prod_bits, fmt)
    if return_intermediates:
        return FpMacResult(out, mant_prod)
    return out
