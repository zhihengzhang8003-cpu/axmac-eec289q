# AxMAC — Project Redesign Notes

> Status doc for the repositioned project. Created 2026-05-19.
> Tracks the research framing and roadmap; the per-paper citation index
> lives in `reference/README.md`.

## Why this redesign

TA feedback: the original project lacked a clear research purpose — it
reproduced existing approximate-MAC techniques (truncated multiplier, ACA
adder) and characterized them, but did not identify a deficiency in prior
work and propose an improvement. A paper-shaped project needs:
**baseline → deficiency → improvement → quantified gain.**

## Baseline (prior work reproduced)

- Truncated multiplier — Mahdiani et al., IEEE TCAS-I 2010.
- ACA (accuracy-configurable) approximate adder — Kahng & Kang, DAC 2012.

## Deficiencies identified

1. **Truncation is biased.** Zeroing the low K product bits rounds every
   product toward -inf; the per-MAC error is always in [0, 2^K) — a
   one-signed error with mean ~2^(K-1). Over a DNN inference this bias
   accumulates *coherently* across thousands of MACs (grows like N), while
   a zero-mean error would only grow like sqrt(N). It is the dominant
   accuracy loss, and worsens as accumulation depth grows (transformers
   accumulate over thousands of elements).
2. **Uniform K wastes inter-layer sensitivity.** One global K is applied to
   every layer, yet layers differ in error tolerance and in MAC count (the
   demo MLP's first layer is 95.8% of all MACs).
3. **per-MAC NMED is the wrong Pareto metric** — it does not capture bias
   accumulation, so the design-space sweep ranks configs by a misleading
   score.

## Contributions

### A. Error-compensated multiplier

`approx_mac.py` gains a `rounding` knob with three modes:

- `trunc` — plain truncation (biased baseline; default, byte-for-byte
  unchanged).
- `round` — deterministic error compensation: add the correction constant
  2^(K-1) before masking (round-to-nearest). Near-zero-mean error.
  Schulte & Swartzlander 1993.
- `stochastic` — stochastic rounding (Gupta et al. 2015), the technique
  modern FP8 hardware uses to remove bias — but it needs a per-MAC RNG.

**Thesis:** for *inference*, deterministic compensation (`round`) matches
stochastic rounding's unbiasedness at near-zero hardware cost (one constant
addend vs. a per-MAC RNG/LFSR). Evaluated as a three-way comparison:
trunc vs. round vs. stochastic.

### B. Layer-wise non-uniform K allocation

Sensitivity-driven per-layer K assignment (cf. ALWANN, ICCAD 2019; HAWQ,
ICCV 2019) instead of a single global K.

## Modernization (FP8 era)

- Add FP8 formats E4M3 / E5M2 (Micikevicius et al. 2022; NVIDIA
  Hopper/Blackwell) so the study covers the formats 2025 hardware uses,
  not just FP16/BF16/FP32.
- Motivation framed around FP8 inference and LLM quantization (GPTQ, AWQ,
  SmoothQuant); recent survey: Armeniakos et al., ACM CSUR 2022.

## Status & roadmap

| # | Task | Status |
|---|------|--------|
| 1 | Literature index (`reference/README.md`) | done — commit 8978308 |
| 2 | Contribution A core: `rounding` modes in `approx_mac.py` | done — commit 8978308 |
| 3 | `power_model.py`: cost of `round` (~`trunc`) vs `stochastic` (+RNG) | done — commit 9157177 |
| 4 | Contribution B: per-layer K + sensitivity allocation | done — commit 3fb5b83 |
| 5 | Experiments: bias accumulation, 3-way rounding, FP8, Pareto | done — commit 695bd2b |
| 6 | pytest tests for new code | done — commit 7ef07ca, **343 passed** |
| 7 | README + proposal `.docx` rewrite (FP8 narrative + recent refs) | done — commit 86620ea / 3cfa1d3 |
| 8 | FP8 formats E4M3/E5M2 in `exact_mac.py` | done — commit a80a680 |
| 9 | RTL Verilog implementation (mac_unit/aca_adder/mac_array/mlp_top + testbenches + Quartus scripts) | done — commit 0c2325a; ModelSim PASS; Quartus synthesis 21% LE, timing closed; **board burn ✅ 2026-06-01** |
| 10 | Unified INT4/8/16 vs FP32 task-level K-sweep (`main.py` Section D) | done — commit d1d51e4 |
| 11 | RTL board demo wrapper (mlp_top_demo, LED display, 野火征途 Pro) | done — commit 95ec92e; **board burn ✅ 2026-06-01, LED argmax=1 confirmed** |
| 12 | Pareto v2: cross-format NRMSE, FP8 sweep, per-format fronts | done — commit 931ac3a |
| 13 | UART TX output (uart_tx/uart_framer) + on-chip multi-K test | done — commit e9141b5; **K=6 trunc misclassification confirmed on hardware** |

**All tasks complete.** Python experiments (343 pytest passed), RTL simulation (ModelSim all 5 tb PASS), Quartus synthesis (EP4CE10, 21% LE, 50 MHz closed), board burn + on-chip K-sweep (5 configs, K=6 trunc argmax 1→3 confirmed).

## On-chip test results (2026-06-01, EP4CE10 野火征途 Pro)

| Config | K | Mode | Board argmax | Correct |
|--------|---|------|-------------|---------|
| K0_trunc | 0 | trunc (exact baseline) | 1 | ✅ |
| K2_trunc | 2 | trunc | 1 | ✅ |
| K4_trunc | 4 | trunc | 1 | ✅ |
| K4_round | 4 | round | 1 | ✅ |
| K6_trunc | 6 | trunc | **3** | ❌ misclassified |

K=6 trunc misclassification (argmax 1→3) matches Python sim bias +29.9 — validates RTL correctness and demonstrates accuracy collapse at aggressive truncation.

## Formal Error Analysis (P0.2)

### Theorem 1 — Bias Accumulation Bounds

**Setup.** Consider N integer MAC operations. Let ε_i denote the per-MAC
truncation error for the i-th operation when K bits of the product are
dropped (i.i.d., uniform distribution assumption over operand pairs).

**Trunc mode** (`product & ~((1<<K)-1)`):

    E[ε_i]         = (2^K - 1) / 2  ≈  2^(K-1)      [always positive]
    Var[ε_i]       = (2^K)² / 12    ≈  4^K / 12
    E[∑ε_i]        = N · 2^(K-1)                      [LINEAR in N]
    Std[∑ε_i]      = √N · 2^K / √12

**Round mode** (`(product + 2^(K-1)) & ~((1<<K)-1)`):

    E[δ_i]         = 0               (exactly)        [zero-mean]
    Var[δ_i]       = (2^K)² / 12     (same as trunc)
    E[∑δ_i]        = 0                                 [ZERO for any N]
    Std[∑δ_i]      = √N · 2^K / √12  (same as trunc)

**Proof sketch.** Trunc maps a product p to p - r where r = p mod 2^K is
uniform on [0, 2^K). Hence E[r] = (2^K-1)/2 > 0. Round maps p to
p - r + 2^(K-1) where r is still uniform on [0, 2^K), so the residual
r - 2^(K-1) is uniform on [-(2^(K-1)), 2^(K-1)) with zero mean. Both
errors have identical variance = E[r²] - (E[r])² = (2^K-1)(2^K+1)/12 ≈
4^K/12. By the law of total expectation, N accumulated errors scale as
stated. □

**Corollary (Transformer Attention).** In scaled dot-product attention,
each query-key score accumulates over d_head MACs (d_head = 64 or 128
for typical models). With INT8 and K=4:

    trunc bias per score = d_head × 2^(K-1) = 64 × 8  = 512 LSBs
    round bias per score = 0

A bias of 512 in INT8 range (±127) means scores are systematically
inflated beyond the representable range — softmax inputs are saturated,
attention patterns collapse. Round mode eliminates this at zero area cost.

**Stochastic mode** (Gupta et al. 2015): also zero-mean, identical
variance, but requires a K-bit LFSR RNG per MAC cycle; round mode
achieves the same bias cancellation deterministically.

### P0 / P1 Improvement Log

| Label | Work | Status |
|-------|------|--------|
| P0.1 | CIFAR-10 approximate MAC accuracy experiment (SimpleCNN, INT8 PTQ) | `experiments/cifar10_experiment.py` added |
| P0.2 | Formal bias accumulation theorem (Theorem 1 above) | Added to this doc |
| P1.1 | DRUM-k multiplier: Python (`drum_quantize_operand`, `drum_multiply`, `int_conv2d_drum`) + Verilog (`rtl/src/drum_multiplier.v`) | Done |
| P1.2 | Quartus synthesis comparison: DRUM vs round vs trunc (see `rtl/vendor/altera/ppa_sweep.tcl`) | TCL update pending |

## References

`reference/README.md` maps every formula/model in the project to its
source paper. PDFs are kept locally and gitignored (copyright); 7 paywalled
papers still need UC Davis library access — DOIs are listed in that file.
