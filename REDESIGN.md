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
| 2 | Contribution A core: `rounding` modes in `approx_mac.py` | done — commit 8978308, 193 tests pass |
| 3 | `power_model.py`: cost of `round` (~`trunc`) vs `stochastic` (+RNG) | todo |
| 4 | Contribution B: per-layer K + sensitivity allocation | todo |
| 5 | Experiments: bias accumulation, 3-way rounding, FP8, Pareto | todo |
| 6 | pytest tests for new code | todo |
| 7 | README / proposal rewrite (FP8 narrative + recent refs) | todo |
| 8 | FP8 formats E4M3/E5M2 in `exact_mac.py` | todo |

Suggested order: 8 → 3 → 4 → 5 → 6 → 7. The `rounding` knob still needs
threading through `accuracy_eval.py` / `dnn_inference.py` before the
experiments (task 5) can use it.

## References

`reference/README.md` maps every formula/model in the project to its
source paper. PDFs are kept locally and gitignored (copyright); 7 paywalled
papers still need UC Davis library access — DOIs are listed in that file.
