"""Approximate MAC: truncated multiplier + ACA-style approximate adder.

Stage 2 deliverable, extended for the redesigned project's error-compensation
study. Three knobs:

* ``K``         — drop the lowest K bits of the (integer or mantissa) product.
* ``W``         — Approximate Carry Adder window: split the accumulator
                  addition into fixed-width segments with no carry across
                  segment boundaries.
* ``rounding``  — how the dropped K bits are handled (see below).

K=0 and W=None (or W >= acc_bits) reproduce :mod:`axmac.exact_mac` byte-for-byte
regardless of ``rounding``; this is enforced by the W2 regression tests so
every later experiment can read its baseline directly off the exact path.

Rounding modes
--------------
Dropping the low K bits of a product is the classic *truncated multiplier*
(Mahdiani et al., IEEE TCAS-I 2010; Schulte & Swartzlander, VLSI Signal
Processing VI 1993). Plain truncation rounds every product toward -inf, so its
error always lies in [0, 2^K): a *one-signed* error with mean ~2^(K-1). Across
a DNN inference that mean accumulates *coherently* over thousands of MACs (it
grows like N, whereas a zero-mean error would only grow like sqrt(N)) — the
dominant accuracy loss this project targets.

* ``"trunc"``      — plain truncation, ``product & ~((1<<K)-1)``. Biased.
                     The project's *baseline* (prior work).
* ``"round"``      — deterministic error compensation: add the correction
                     constant 2^(K-1) (the expected value of the dropped
                     field) before masking, i.e. round-to-nearest. Error in
                     [-2^(K-1), 2^(K-1)); near-zero mean. One extra constant
                     addend, folded into the partial-product tree (Schulte &
                     Swartzlander 1993).
* ``"stochastic"`` — stochastic rounding: add a uniform random integer in
                     [0, 2^K) before masking. Also zero-mean, and the modern
                     choice in low-precision / FP8 hardware (Gupta et al.,
                     ICML 2015; Croci et al., R. Soc. Open Sci. 2022), but it
                     needs a per-MAC random-number generator. ``round`` is the
                     cheaper deterministic alternative we compare against it.

INT path: drop the bottom K bits of the unrounded integer product, then add
to the accumulator through :func:`aca_add`.

FP path: drop the bottom K bits of the integer mantissa product before
:func:`_renormalize_and_pack`. The FP accumulate stage stays exact for W2
— ACA on a variable-alignment FP add is deferred (W5+ if useful).
"""

from __future__ import annotations

import random
from typing import Literal

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


RoundingMode = Literal["trunc", "round", "stochastic"]

_VALID_ROUNDING = ("trunc", "round", "stochastic")


def _apply_rounding(
    product: int,
    K: int,
    rounding: RoundingMode,
    rng: random.Random | None,
) -> int:
    """Drop the low K bits of ``product`` under the chosen rounding mode.

    * ``"trunc"``      — plain truncation toward -inf; error in [0, 2^K),
                         biased (mean ~2^(K-1)). Project baseline.
    * ``"round"``      — add the correction constant 2^(K-1) before masking
                         (round-to-nearest); near-zero-mean error in
                         [-2^(K-1), 2^(K-1)). Schulte & Swartzlander 1993.
    * ``"stochastic"`` — add a uniform random integer in [0, 2^K); zero-mean
                         in expectation. Gupta et al. 2015.

    ``& ~mask`` rounds toward -inf on Python's arbitrary-precision ints, so
    the constant / random offset works correctly for signed products. K<=0
    is a no-op (the exact product), regardless of ``rounding``.
    """
    if K <= 0:
        return product
    if rounding == "round":
        product += 1 << (K - 1)
    elif rounding == "stochastic":
        product += rng.randrange(1 << K) if rng is not None else random.randrange(1 << K)
    return product & ~((1 << K) - 1)


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
    rounding: RoundingMode = "trunc",
    rng: random.Random | None = None,
    return_pps: bool = False,
) -> int | IntMacResult:
    """Approximate INT MAC: ``acc + round_K(a*b)`` with optional ACA carry chain.

    ``rounding`` selects how the dropped K bits are handled — ``"trunc"``
    (biased baseline), ``"round"`` (deterministic compensation) or
    ``"stochastic"`` (uses ``rng``); see the module docstring.
    """
    if K < 0:
        raise ValueError("K must be >= 0")
    if rounding not in _VALID_ROUNDING:
        raise ValueError(f"rounding must be one of {_VALID_ROUNDING}, got {rounding!r}")
    if not (fmt.min_val <= a <= fmt.max_val):
        raise ValueError(f"a={a} out of range for {fmt.name}")
    if not (fmt.min_val <= b <= fmt.max_val):
        raise ValueError(f"b={b} out of range for {fmt.name}")
    acc_min = -(1 << (acc_bits - 1))
    acc_max = (1 << (acc_bits - 1)) - 1
    if not (acc_min <= acc <= acc_max):
        raise ValueError(f"acc={acc} out of range for {acc_bits}-bit accumulator")

    pps = booth_radix4_pps(a, b, fmt.bits)
    product = _apply_rounding(sum(pps), K, rounding, rng)

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
    rounding: RoundingMode = "trunc",
    rng: random.Random | None = None,
    return_intermediates: bool = False,
) -> int | FpMacResult:
    """Approximate FP MAC: round K LSBs of the mantissa product, then exact RNE + add.

    ``rounding`` selects trunc / round / stochastic on the mantissa product;
    see the module docstring.
    """
    if K < 0:
        raise ValueError("K must be >= 0")
    if rounding not in _VALID_ROUNDING:
        raise ValueError(f"rounding must be one of {_VALID_ROUNDING}, got {rounding!r}")
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
        M_prod = _apply_rounding(M_full, K, rounding, rng)
        E_prod = Ea + Eb - fmt.mant_bits
        prod_bits = _renormalize_and_pack(s_out, M_prod, E_prod, fmt)

    out = fp_add(acc_bits, prod_bits, fmt)
    if return_intermediates:
        return FpMacResult(out, M_prod)
    return out
