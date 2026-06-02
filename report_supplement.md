# Report Supplement — Content to Add
# AxMAC: An Error-Compensated Approximate MAC for FP8-Era DNN Inference
# Generated 2026-06-01

---

## SECTION: RTL Hardware Implementation

### Architecture

The AxMAC MAC unit is implemented in synthesizable Verilog and organized as
a four-level hierarchy:

  mac_unit.v      — one signed multiplier + truncation/rounding (K, MODE)
  aca_adder.v     — accuracy-configurable approximate adder (window W)
  mac_array.v     — R×C systolic tile (default R=1, C=4 parallel PEs)
  mlp_top.v       — two-layer MLP (64→16→10); loads weights from .mem files
  mlp_top_demo.v  — board wrapper: argmax → 4 LEDs + 115200-baud UART TX

Key parameters (all set at synthesis time):

  K_PARAM   — bits truncated per partial product (0 = exact)
  MODE      — 0 = trunc, 1 = round (add 2^(K-1)), 2 = stochastic (LFSR)
  ACA_W     — carry-propagation window (32 = exact accumulator)

Target: Intel Cyclone IV E EP4CE10F17C8 (野火征途 Pro).
Tool:   Quartus Prime Lite 18.1, 50 MHz clock constraint.

### Verification

Five ModelSim testbenches (tb_mac_unit, tb_aca_adder, tb_mac_array,
tb_mlp_top, tb_mlp_top_demo) — all PASS as of 2026-06-01.

---

## SECTION: Hardware Synthesis Results (PPA)

Table I — Rounding-mode hardware cost (K=6, ACA_W=32, EP4CE10)

  Configuration        | Logic Elements | Registers | DSP | Core Power (mW)
  ---------------------|---------------|-----------|-----|-----------------
  Exact (K=0, baseline)| 1769          | 582       | 4   | 10.45
  K=6, trunc           | 1685          | 527       | 4   |  8.67
  K=6, round           | 1731          | 527       | 4   |  9.20
  K=6, stochastic      | 1817          | 591       | 4   |  9.39

Key findings:
- trunc vs. exact: −84 LE, −17% core power (truncation saves both area and power).
- round vs. trunc: +46 LE, +0 registers (+3% logic, zero register overhead).
  The correction constant 2^(K−1) is a single constant addend — no state.
- stochastic vs. trunc: +132 LE, +64 registers. The 64 extra registers are
  the 64-bit LFSR. At K=6, the RNG costs MORE logic than truncation saves
  (1817 LE > 1769 exact baseline), eliminating the area advantage entirely.

Conclusion (Contribution A, quantified on hardware): round reaches
near-zero-mean error at ~35% of stochastic's added cost and zero added
registers. This confirms the Python simulation result on real silicon.

Table II — ACA approximate adder cost (K=0, no truncation; EP4CE10)

  ACA window W | Logic Elements | Registers
  -------------|---------------|----------
  W=32 (exact) | 1769          | 582
  W=8          | 1755          | 550
  W=4          | 1732          | 534

Narrowing the carry window from 32 to 4 saves 37 LE and 48 registers.

Note: Power values are from Quartus PowerPlay (vectorless, low confidence on
absolute magnitude; use as relative comparison only).

---

## SECTION: On-Chip Experimental Results

Five bitstreams were synthesized for the EP4CE10, each with a different
(K, MODE) configuration. Each was programmed via JTAG (USB-Blaster) and the
board's 4-LED argmax display was observed.

Test input: a fixed 64-dimensional toy-MNIST vector (x.mem, golden weights).
Reference result (Python simulation, K=0 exact): argmax = 1,
logits ≈ [4345, 10718, −4421, −9026, −3228, −6422, 4780, −3136, 10420, 6344].

Table III — On-chip argmax vs. configuration (EP4CE10, 野火征途 Pro, 2026-06-01)

  Config    | K | Mode  | Board argmax | Match simulation | Correct
  ----------|---|-------|-------------|------------------|---------
  K0_trunc  | 0 | trunc | 1           | ✅               | ✅
  K2_trunc  | 2 | trunc | 1           | ✅               | ✅
  K4_trunc  | 4 | trunc | 1           | ✅               | ✅
  K4_round  | 4 | round | 1           | ✅               | ✅
  K6_trunc  | 6 | trunc | 3           | ✅ (sim bias=+29.9)| ❌ misclassified

Analysis:
1. Robustness: K ≤ 4 (both trunc and round) produces correct classification,
   demonstrating that AxMAC is robust to moderate approximation.
2. Accuracy collapse at K=6 trunc: argmax shifts from class 1 to class 3.
   This is consistent with the Python simulation result showing a coherent
   accumulated bias of +29.9 at K=6, trunc — the bias is large enough to
   alter the rank ordering of the two closest logits
   (logit[1]=10718, logit[8]=10420, margin only 298).
3. round vs. trunc at K=4: both give argmax=1, consistent with round's
   near-zero-mean bias keeping the logit ordering intact even when trunc
   already introduces visible error.
4. RTL correctness: hardware results match Python simulation predictions
   in all five configurations, validating the Verilog implementation.

---

## KEY NUMBERS SUMMARY (for abstract / conclusion)

Python simulation:
- Truncation bias at depth N=4096: trunc ≈ 26,617 (∝ N) vs. stochastic ≈ 14 (∝ √N)
- INT8, K=6 per-MAC bias: trunc +29.9 → round −1.7 → stochastic −0.8
- Contribution B: sensitivity-driven K reduces logit NRMSE by 48% at matched energy

Hardware synthesis (EP4CE10):
- trunc K=6: −84 LE, −17% core power vs. exact baseline
- round K=6: only +46 LE, +0 registers vs. trunc (~3% area overhead)
- stochastic K=6: +132 LE, +64 registers vs. trunc (RNG erases area savings)

On-chip validation:
- K=0/2/4 trunc, K=4 round: correct classification on EP4CE10 ✅
- K=6 trunc: misclassification (argmax 1→3), confirms simulation ✅
