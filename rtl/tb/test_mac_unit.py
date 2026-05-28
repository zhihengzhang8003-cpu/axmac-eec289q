"""cocotb test for src/mac_unit.v -- bit-exact against rtl/golden/mac_int8.csv.

Two tests:

* ``test_mac_unit_deterministic`` -- every (a, b, K, mode='trunc'|'round') row
  in mac_int8.csv must produce ``product_rounded`` exactly.
* ``test_mac_unit_stochastic_stats`` -- distributional sanity check: drive the
  DUT in stochastic mode with cocotb's RNG, then verify mean-error
  near zero (Gupta 2015 unbiasedness property).
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import cocotb
from cocotb.triggers import Timer


GOLDEN_CSV = Path(__file__).parent.parent / "golden" / "mac_int8.csv"

MODE_TRUNC = 0b00
MODE_ROUND = 0b01
MODE_STOCHASTIC = 0b10

MODE_CODE = {"trunc": MODE_TRUNC, "round": MODE_ROUND}


async def _settle(dut) -> None:
    """Combinational DUT -- one ns is plenty for signals to propagate."""
    await Timer(1, units="ns")


@cocotb.test()
async def test_mac_unit_deterministic(dut):
    """Bit-exact match on trunc and round across every row of mac_int8.csv."""
    if not GOLDEN_CSV.exists():
        raise FileNotFoundError(
            f"missing {GOLDEN_CSV}; run rtl/golden/export_golden.py first"
        )

    n_checked = 0
    n_fail = 0
    first_fail = None

    with GOLDEN_CSV.open() as f:
        for row in csv.DictReader(f):
            mode_name = row["mode"]
            if mode_name not in MODE_CODE:
                continue

            a = int(row["a"])
            b = int(row["b"])
            K = int(row["K"])
            expected = int(row["product_rounded"])

            dut.a.value = a
            dut.b.value = b
            dut.K.value = K
            dut.mode.value = MODE_CODE[mode_name]
            dut.rnd.value = 0

            await _settle(dut)

            actual = dut.product_rounded.value.signed_integer
            n_checked += 1
            if actual != expected:
                n_fail += 1
                if first_fail is None:
                    first_fail = (a, b, K, mode_name, expected, actual)

    dut._log.info(f"deterministic: {n_checked} cases checked, {n_fail} failures")
    if n_fail:
        a, b, K, mode_name, expected, actual = first_fail
        raise AssertionError(
            f"mac_unit mismatch ({n_fail}/{n_checked} failed). "
            f"First failure: a={a}, b={b}, K={K}, mode={mode_name}, "
            f"expected={expected}, got={actual}"
        )


@cocotb.test()
async def test_mac_unit_stochastic_stats(dut):
    """Stochastic-mode mean error should be near zero (Gupta 2015 property).

    Not bit-exact vs the Python golden -- RNG sources differ. We check the
    statistical property that justifies stochastic rounding in the first place:
    over many MACs at the same K, the mean rounding error tends to zero.
    """
    rng = random.Random(0xC0FFEE)
    n_per_K = 2000
    # Theoretical worst case |mean| for n samples uniform in [0, 2^K): the
    # sample mean has stddev (2^K) / sqrt(12 * n). Allow 5 sigma -> ~0 false fail.
    for K in range(1, 7):
        sigma = (1 << K) / (12 * n_per_K) ** 0.5
        tol = 5.0 * sigma

        errs = []
        for _ in range(n_per_K):
            a = rng.randint(-128, 127)
            b = rng.randint(-128, 127)
            product = a * b
            rnd = rng.getrandbits(16)

            dut.a.value = a
            dut.b.value = b
            dut.K.value = K
            dut.mode.value = MODE_STOCHASTIC
            dut.rnd.value = rnd

            await _settle(dut)

            actual = dut.product_rounded.value.signed_integer
            errs.append(actual - product)

        mean_err = sum(errs) / len(errs)
        dut._log.info(f"K={K}: mean error = {mean_err:+.3f}  (tol = {tol:.3f})")
        assert abs(mean_err) <= tol, (
            f"stochastic K={K} biased: mean_err={mean_err:.3f}, tol={tol:.3f}"
        )
