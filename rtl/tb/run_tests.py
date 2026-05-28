"""Windows-friendly test driver -- supports Icarus and ModelSim-Altera.

Pure SystemVerilog testbenches under tb/tb_*.sv read the golden CSVs in
rtl/golden/ and report PASS / FAIL via $display + $fatal. This runner just
compiles each (DUT, tb) pair with the chosen simulator and runs from the
tb/ directory so the testbench's relative "../golden/..." paths resolve.

  py run_tests.py                          # run every available test (icarus)
  py run_tests.py --sim modelsim           # use ModelSim instead
  py run_tests.py mac_unit                 # one test
  py run_tests.py mac_unit aca_adder

Backends:
  icarus    -- needs iverilog on PATH (https://bleyer.org/icarus/).
  modelsim  -- needs vlog/vsim on PATH. Ships with Quartus Prime Lite or
               standalone as ModelSim-Altera Starter Edition.

Path caveat: ModelSim-Altera Starter 10.5b refuses to chdir into paths that
contain Unicode characters such as the full-width colon U+FF1A. The project
lives at a path that hits exactly that bug. Workaround: create a junction
with an ASCII-only path, then run from inside the junction. Critically, do
NOT call Path.resolve() in this script -- resolve() follows the junction
back to the original Unicode path and defeats the workaround. .absolute()
preserves the junction prefix. Example junction creation:

    powershell -Command 'New-Item -ItemType Junction -Path E:\\axmac_rtl
                         -Target (Resolve-Path .).Path'

then run this script from inside E:\\axmac_rtl. The Icarus backend has no
such limit.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

RTL_DIR = Path(__file__).absolute().parent.parent
SRC_DIR = RTL_DIR / "src"
TB_DIR = RTL_DIR / "tb"
BUILD_DIR = RTL_DIR / "build"


# Each entry: sources (DUT + tb), top-level module of the tb.
TESTS = {
    "mac_unit": {
        "sources": [SRC_DIR / "mac_unit.v", TB_DIR / "tb_mac_unit.sv"],
        "toplevel": "tb_mac_unit",
    },
    "aca_adder": {
        "sources": [SRC_DIR / "aca_adder.v", TB_DIR / "tb_aca_adder.sv"],
        "toplevel": "tb_aca_adder",
    },
    "mac_array": {
        "sources": [
            SRC_DIR / "mac_unit.v",
            SRC_DIR / "aca_adder.v",
            SRC_DIR / "mac_array.v",
            TB_DIR / "tb_mac_array.sv",
        ],
        "toplevel": "tb_mac_array",
    },
    "mlp_top": {
        "sources": [
            SRC_DIR / "mac_unit.v",
            SRC_DIR / "aca_adder.v",
            SRC_DIR / "mac_array.v",
            SRC_DIR / "lfsr.v",
            SRC_DIR / "mlp_top.v",
            TB_DIR / "tb_mlp_top.sv",
        ],
        "toplevel": "tb_mlp_top",
    },
}


def _run_icarus(name: str, spec: dict) -> bool:
    if shutil.which("iverilog") is None:
        print("  iverilog not on PATH. Install Icarus Verilog:")
        print("    https://bleyer.org/icarus/  (pick the latest installer)")
        return False

    BUILD_DIR.mkdir(exist_ok=True)
    vvp_path = BUILD_DIR / f"{name}.vvp"

    cmd = [
        "iverilog",
        "-g2012",
        "-o", str(vvp_path),
        "-s", spec["toplevel"],
    ] + [str(p) for p in spec["sources"]]
    print("  $ " + " ".join(cmd))
    if subprocess.call(cmd) != 0:
        print("  COMPILE FAILED")
        return False

    # Run vvp from tb/ so "../golden/..." resolves.
    rc = subprocess.call(["vvp", str(vvp_path)], cwd=str(TB_DIR))
    if rc != 0:
        print(f"  SIM FAILED (rc={rc})")
        return False
    return True


def _run_modelsim(name: str, spec: dict) -> bool:
    if shutil.which("vsim") is None or shutil.which("vlog") is None:
        print("  vsim/vlog not on PATH. Add ModelSim-Altera win32aloem/ to PATH.")
        return False

    # Per-test work library so concurrent runs don't trample each other.
    work_dir = BUILD_DIR / f"work_{name}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # vlib needs to be run with cwd=work_dir parent so it creates ./work there;
    # use -work flag instead and run from tb/ so relative golden paths work.
    work_lib = work_dir / "work"

    print("  $ vlib " + str(work_lib))
    if subprocess.call(["vlib", str(work_lib)], cwd=str(TB_DIR)) != 0:
        print("  vlib FAILED")
        return False

    vlog_cmd = [
        "vlog",
        "-sv",
        "-quiet",
        "-work", str(work_lib),
    ] + [str(p) for p in spec["sources"]]
    print("  $ " + " ".join(vlog_cmd))
    if subprocess.call(vlog_cmd, cwd=str(TB_DIR)) != 0:
        print("  vlog FAILED")
        return False

    # -c    -- console mode (no GUI)
    # -do   -- finish cleanly after the testbench calls $finish/$fatal
    vsim_cmd = [
        "vsim",
        "-c",
        "-work", str(work_lib),
        "-do", "run -all; quit -f",
        spec["toplevel"],
    ]
    print("  $ " + " ".join(vsim_cmd))
    rc = subprocess.call(vsim_cmd, cwd=str(TB_DIR))
    if rc != 0:
        print(f"  vsim FAILED (rc={rc})")
        return False
    return True


def _run_one(name: str, sim: str) -> bool:
    if name not in TESTS:
        print(f"  [SKIP] {name}: not in TESTS dict")
        return False
    spec = TESTS[name]
    print(f"\n=== {name} (sim={sim}) ===")

    if sim == "icarus":
        return _run_icarus(name, spec)
    if sim == "modelsim":
        return _run_modelsim(name, spec)
    print(f"  sim={sim} not yet wired in run_tests.py")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("modules", nargs="*", help="modules to test (default: all)")
    ap.add_argument("--sim", default="icarus",
                    choices=["icarus", "verilator", "xsim", "modelsim"],
                    help="simulator backend (default: icarus)")
    args = ap.parse_args()

    targets = args.modules or list(TESTS.keys())
    ok = True
    for name in targets:
        ok = _run_one(name, args.sim) and ok
    print("\n", "OK" if ok else "FAIL", sep="")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
