# AxMAC — RTL Implementation

Hardware implementation of the AxMAC arithmetic primitives, alongside the
Python testbed in `../axmac/`. Target: Altera Cyclone IV EP4CE10F17C8 (野火征途 Pro).

| Flow         | Vendor | Target chip | Toolchain          | Purpose |
|--------------|--------|-------------|--------------------|---------|
| Physical run | Altera | EP4CE10     | Quartus Prime Lite | Demo on 野火征途: bitstream + LED display showing the toy MLP classifying a held-out input. PowerPlay numbers (60 nm LP) used for relative trunc/round/stochastic comparison. |

Vendor-agnostic Verilog lives in `src/`. Board-specific pin/timing constraints are in `vendor/altera/`.

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
  build/                     # ModelSim work libs + Quartus artifacts (gitignored)
  vendor/
    altera/                  # Altera / Quartus files
      build_demo.tcl         # full flow (map+fit+asm+sta) → demo.sof
      build_quartus.tcl      # analysis+synthesis only (mac_array area estimate)
      mlp_top_ep4ce10_pins.tcl  # board pin assignments (clk/rst_n/led[3:0])
      ppa_sweep.tcl          # rounding-mode PPA sweep script
      ppa_results.csv        # PPA data (LE / registers / power by mode)
      ppa_summary.txt        # human-readable PPA summary
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
| 5 | Quartus synth + .sof + board burn on EP4CE10 (野火征途) | **TODO** — scripts + pin constraints ready; running synthesis now |

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
that tiling and FSM logic are non-trivial.

Regenerate the golden whenever the Python-side reference changes:

```powershell
py -3.14 rtl/golden/export_golden.py
```

## Toolchain notes

| Tool | Why | Status |
|------|-----|--------|
| **ModelSim-Altera 10.5b** | Simulation (all testbenches) | Installed at `E:\Quart\modelsim_ase\`; use junction `E:\axmac_rtl` to bypass Unicode-path bug |
| **Quartus Prime Lite 18.1** | Synthesis + bitstream + PowerPlay for EP4CE10 | Installed at `E:\Quart\quartus\` |
| ~~cocotb~~ | Was the initial testbench plan | **Abandoned** — requires Python ≤ 3.13; project uses 3.14 |

## How the testbenches are structured

Each `tb/tb_<module>.sv` opens its golden CSV with `$fopen`, parses rows with
`$sscanf`, drives the DUT, and reports PASS / FAIL via `$display` + `$fatal`.
This keeps the testbench portable across **Icarus, XSim, Verilator, and
ModelSim-Altera** — only `run_tests.py`'s simulator backend changes per
flow. `tb_mac_array.sv` computes its reference inline (small enough), so it
doesn't need a golden file; the others read `golden/*.csv`.

## How to run the tests

```powershell
py rtl/tb/run_tests.py                   # run every available test
py rtl/tb/run_tests.py mac_unit          # run a single one
py rtl/tb/run_tests.py mac_unit aca_adder
```

The driver compiles with `iverilog -g2012`, writes the .vvp to `rtl/build/`,
and runs `vvp` from `rtl/tb/` so the `"../golden/..."` paths in the
testbench resolve correctly. Each test prints a final `ALL TESTS PASSED`
line on success or `FATAL` + `$fatal` on the first mismatch.
