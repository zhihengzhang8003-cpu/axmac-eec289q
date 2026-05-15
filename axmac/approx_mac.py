"""Approximate MAC: truncated multiplier + ACA-style approximate adder.

Stage 2 deliverable. Two knobs:

* ``K``  — drop the lowest K bits of the (integer or mantissa) product.
* ``W``  — Approximate Carry Adder window: split the accumulator addition
           into fixed-width segments with no carry across segment boundaries.

K=0 and W=None (or W >= acc_bits) reproduce :mod:`axmac.exact_mac` byte-for-byte;
this is enforced by the W2 regression tests so every later experiment can
read its baseline directly off the exact path.

INT path: drop the bottom K bits of the unrounded integer product (matches
the "truncated multiplier" model of Mahdiani et al. / Liu et al.), then add
to the accumulator through :func:`aca_add`.

FP path: zero the bottom K bits of the integer mantissa product before
:func:`_renormalize_and_pack`. The FP accumulate stage stays exact for W2
— ACA on a variable-alignment FP add is deferred (W5+ if useful).
"""

from __future__ import annotations

from .exact_mac import (
    FPFormat,
    FpMacResult,
    IntFormat,
    IntMacResult,
    _make_qnan,
    _renormalize_and_pack,
    _to_internal,
    booth_radix4_pps,
    fp_add,
    fp_pack,
)


def _wrap(x: int, bits: int) -> int:
    raw = x & ((1 << bits) - 1)
    if raw >= (1 << (bits - 1)):
        raw -= 1 << bits
    return raw


def aca_add(a: int, b: int, bits: int, window: int) -> int:
    """Signed ``bits``-wide add with carry-isolated segments of width ``window``.

    Each segment computes ``(a_seg + b_seg) mod 2^window`` independently;
    carry-out of segment *i* never reaches segment *i+1*. ``window >= bits``
    reduces to the exact two's-complement sum.
    """
    if window <= 0:
        raise ValueError("aca window must be >= 1")
    mask = (1 << bits) - 1
    a_u = a & mask
    b_u = b & mask
    if window >= bits:
        return _wrap(a_u + b_u, bits)
    out = 0
    pos = 0
    while pos < bits:
        w = min(window, bits - pos)
        seg_mask = (1 << w) - 1
        a_seg = (a_u >> pos) & seg_mask
        b_seg = (b_u >> pos) & seg_mask
        out |= ((a_seg + b_seg) & seg_mask) << pos
        pos += w
    return _wrap(out, bits)


def approx_mac_int(
    a: int,
    b: int,
    acc: int,
    fmt: IntFormat,
    *,
    K: int = 0,
    aca_window: int | None = None,
    acc_bits: int = 32,
    return_pps: bool = False,
) -> int | IntMacResult:
    """Approximate INT MAC: ``acc + trunc_K(a*b)`` with optional ACA carry chain."""
    if K < 0:
        raise ValueError("K must be >= 0")
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
    if K > 0:
        product &= ~((1 << K) - 1)  # zero K LSBs; sign-correct on Python ints

    window = aca_window if aca_window is not None else acc_bits
    out = aca_add(acc, product, acc_bits, window)
    if return_pps:
        return IntMacResult(out, pps)
    return out


def approx_mac_fp(
    a_bits: int,
    b_bits: int,
    acc_bits: int,
    fmt: FPFormat,
    *,
    K: int = 0,
    return_intermediates: bool = False,
) -> int | FpMacResult:
    """Approximate FP MAC: drop K LSBs of mantissa product, then exact RNE + add."""
    if K < 0:
        raise ValueError("K must be >= 0")
    sa, Ma, Ea, ka = _to_internal(a_bits, fmt)
    sb, Mb, Eb, kb = _to_internal(b_bits, fmt)
    s_out = sa ^ sb

    if ka == "nan" or kb == "nan":
        prod_bits, M_prod = _make_qnan(fmt), 0
    elif (ka == "inf" and kb == "zero") or (kb == "inf" and ka == "zero"):
        prod_bits, M_prod = _make_qnan(fmt), 0
    elif ka == "inf" or kb == "inf":
        prod_bits, M_prod = fp_pack(s_out, fmt.exp_all_ones, 0, fmt), 0
    elif ka == "zero" or kb == "zero":
        prod_bits, M_prod = fp_pack(s_out, 0, 0, fmt), 0
    else:
        M_full = Ma * Mb
        M_prod = M_full & ~((1 << K) - 1) if K > 0 else M_full
        E_prod = Ea + Eb - fmt.mant_bits
        prod_bits = _renormalize_and_pack(s_out, M_prod, E_prod, fmt)

    out = fp_add(acc_bits, prod_bits, fmt)
    if return_intermediates:
        return FpMacResult(out, M_prod)
    return out
