# AxMAC — RTL Implementation

Hardware implementation of the AxMAC arithmetic primitives, alongside the
Python testbed in `../axmac/`. Two parallel flows share the same RTL:

| Flow                  | Vendor   | Target chip   | Toolchain               | Purpose |
|-----------------------|----------|---------------|-------------------------|---------|
| High-performance sim  | Xilinx   | xc7a100t      | Vivado ML Standard      | Paper-grade PPA: post-impl simulation + SAIF + Power Report on 28 nm Artix-7. Replaces the analytical `power_model.py` numbers in the paper's Contribution A. |
| Physical run          | Altera   | EP4CE10       | Quartus Prime Lite      | Demo on 野火征途 board: bitstream + LED/UART showing the toy MLP classifying a held-out input. PowerPlay numbers are 60 nm LP, used only for *relative* trunc/round/stochastic comparison. |

Eighty percent of the RTL is vendor-agnostic Verilog and lives in `src/`. Only
the memory-IP wrapper and the top-level pin/timing constraints are duplicated
per vendor.

## Directory layout

```
rtl/
  src/                       # vendor-agnostic Verilog (shared by both flows)
    mac_unit.v   [done]      # INT8 multiplier + K-bit trunc/round/stochastic
    aca_adder.v  [done]      # W-parameterised approximate-carry adder
    mac_array.v  [done]      # parameterisable R x C output-stationary array
    mlp_top.v    [TODO P4]   # FSM + tiling controller for the toy MLP
    bram_wrapper.v [TODO P4] # behavioural RAM interface (vendor IP plugs in here)
  tb/                        # SystemVerilog testbenches (shared)
    tb_mac_unit.sv   [done]
    tb_aca_adder.sv  [done]
    tb_mac_array.sv  [done]
    tb_mlp_top.sv    [TODO P4]
    test_mac_unit.py [cocotb backup -- needs Python 3.13, see Toolchain notes]
    run_tests.py     # Icarus driver (Windows-friendly, no make)
  golden/                    # Python-side reference vectors (see below)
    export_golden.py
    mac_int8.csv             # 4046 rows (K=0..6 x trunc/round)
    mac_int8_stoch.csv       # 1536 rows (distributional check)
    aca.csv                  # 1024 rows (W=4/8/16/32)
    mlp_toy/                 # 11 files: 64->16->10 x/w/b + 6 (K, mode) outputs
  build/                     # iverilog .vvp build artifacts (gitignored)
  vendor/
    xilinx/                  # Xilinx-only files (TODO Phase 5a)
      bram_xpm.v             # XPM_MEMORY macro wrapper
      mlp_top_a100t.xdc      # pin + timing constraints for xc7a100t
      build_vivado.tcl       # synth + impl + SAIF + Power Report
    altera/                  # Altera-only files (TODO Phase 5b)
      bram_altsync.v         # altsyncram megafunction wrapper
      mlp_top_ep4ce10.qsf    # pin assignments for 野火征途
      build_quartus.tcl      # synth + .sof + PowerPlay
```

## Status

| Phase | Work | Status |
|-------|------|--------|
| 0 | Directory skeleton, golden exporter, README | Done |
| 1 | mac_unit.v + tb_mac_unit.sv (bit-exact trunc/round + stochastic mean check) | **PASS** — 4046 deterministic cases bit-exact; stochastic mean within tolerance (ModelSim 2026-06-01) |
| 2 | aca_adder.v + tb_aca_adder.sv (W=4/8/16/32 across 32-bit) | **PASS** — 1024 cases bit-exact (ModelSim 2026-06-01) |
| 3 | mac_array.v + tb_mac_array.sv (4×4 output-stationary, inline reference) | **PASS** — 16 cases (ModelSim 2026-06-01) |
| 4 | mlp_top.v + tb_mlp_top.sv (FSM + tiling, toy 64→16→10) | **PASS** — 10/10 outputs bit-exact vs Python golden (ModelSim 2026-06-01) |
| 4b | mlp_top_demo.v + tb_mlp_top_demo.sv (board wrapper, LED display) | **PASS** — led_class=1 matches argmax of golden logits (ModelSim 2026-06-01) |
| 5a | Vivado synth + impl + SAIF + Power Report on xc7a100t | **TODO** — `vendor/xilinx/` empty; waits for Vivado install |
| 5b | Quartus synth + .sof + PowerPlay on EP4CE10 + 野火征途 board burn | **TODO** — synthesis scripts + pin constraints ready; **bitstream has NOT been programmed to board** |

## Golden CSV contract

`export_golden.py` is the single source of truth for what the RTL must
reproduce. It calls the existing `axmac/` modules and dumps reference vectors
in this directory.

| File                        | Columns                                          | What the RTL must do |
|-----------------------------|--------------------------------------------------|----------------------|
| `mac_int8.csv`              | `a,b,K,mode,product_full,product_rounded`        | For `mode ∈ {trunc,round}`: bit-exact match on `product_rounded`. Both modes are deterministic. |
| `mac_int8_stoch.csv`        | `a,b,K,seed_call_idx,product_full,product_rounded` | Distributional match only (mean / RMSE). The Python RNG and the RTL LFSR generate different sample sequences; only the statistics need to agree. |
| `aca.csv`                   | `a,b,bits,window,sum`                            | Bit-exact match on `sum` for the 32-bit ACA across W ∈ {4,8,16,32}. |
| `mlp_toy/x.csv`             | flat 1×64 input vector                           | Drive `mlp_top` once. |
| `mlp_toy/w{i}.csv,b{i}.csv` | per-layer weight matrices and bias vectors       | Preload BRAM before the run. |
| `mlp_toy/y_K{K}_{mode}.csv` | flat 1×10 expected output, one file per config   | Bit-exact match for the (K, mode) configuration the DUT is built with. |

The toy MLP topology is **64 → 16 → 10**. It is deliberately small enough to
fit in the 23 multipliers and 414 Kbit of memory on EP4CE10, and big enough
that tiling and FSM logic are non-trivial. The xc7a100t flow runs the same
RTL and the same golden, plus the larger 784 → 128 → 32 → 10 MLP from the
Python project (instantiated with a 16×16 array instead of 4×4).

Regenerate the golden whenever the Python-side reference changes:

```powershell
py -3.14 rtl/golden/export_golden.py
```

## Toolchain notes

| Tool | Why | Status |
|------|-----|--------|
| **Icarus Verilog** | Fastest local simulator for Phase 1-4 unit tests; no admin needed | **Required before any test runs.** Install from https://bleyer.org/icarus/ (Windows installer ~30 MB). |
| **Vivado ML Standard 2024.x** | Phase 5a synth/impl/power report on xc7a100t | Download from https://www.xilinx.com/support/download.html (~50 GB). Free tier covers xc7a100t. |
| **Quartus Prime Lite 22.1** | Phase 5b synth + bitstream + PowerPlay for EP4CE10 | Download from https://www.intel.com/content/www/us/en/software-kit/795188/ (~7 GB; ships with ModelSim-Altera Starter). |
| ~~cocotb~~ | Was the initial testbench plan | **Abandoned** — cocotb 2.0.1 requires Python ≤ 3.13, but the project uses 3.14. The `test_*.py` file is kept as reference; the live testbenches are pure SystemVerilog `tb_*.sv` instead. |

## How the testbenches are structured

Each `tb/tb_<module>.sv` opens its golden CSV with `$fopen`, parses rows with
`$sscanf`, drives the DUT, and reports PASS / FAIL via `$display` + `$fatal`.
This keeps the testbench portable across **Icarus, XSim, Verilator, and
ModelSim-Altera** — only `run_tests.py`'s simulator backend changes per
flow. `tb_mac_array.sv` computes its reference inline (small enough), so it
doesn't need a golden file; the others read `golden/*.csv`.

## How to run the tests (once Icarus is installed)

```powershell
py rtl/tb/run_tests.py                   # run every available test
py rtl/tb/run_tests.py mac_unit          # run a single one
py rtl/tb/run_tests.py mac_unit aca_adder
```

The driver compiles with `iverilog -g2012`, writes the .vvp to `rtl/build/`,
and runs `vvp` from `rtl/tb/` so the `"../golden/..."` paths in the
testbench resolve correctly. Each test prints a final `ALL TESTS PASSED`
line on success or `FATAL` + `$fatal` on the first mismatch.
