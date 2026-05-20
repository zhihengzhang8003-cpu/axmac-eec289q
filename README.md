# AxMAC: An Error-Compensated Approximate MAC for FP8-Era DNN Inference

EEC 289Q 002 SQ 2026 — Deep Learning Hardware course project.
Authors: Jiabo Zhang, Yuxuan Wang.

A Python testbed that **reproduces** a prior-work approximate MAC (truncated
multiplier + ACA adder), **identifies a concrete deficiency** in it, and
**proposes two improvements** — quantifying the gain. See `REDESIGN.md` for the
full research framing and `reference/README.md` for the per-formula citation
index.

## Research framing: baseline → deficiency → improvement

**Baseline (prior work reproduced).** A truncated multiplier (Mahdiani et al.,
TCAS-I 2010) feeding an accuracy-configurable approximate adder (Kahng & Kang,
DAC 2012), parameterized by `K` (low product bits dropped) and `W` (ACA carry
window).

**Deficiency.** Plain truncation rounds every product toward −∞: the per-MAC
error is one-signed, in `[0, 2^K)`, mean ≈ `2^(K-1)`. Across a DNN inference
this bias accumulates *coherently* over thousands of MACs — it grows like `N`,
not `√N`. Measured here: a truncated INT8 dot product at accumulation depth
`N = 4096` is off by ≈ **26,600**, whereas a genuinely zero-mean error stays
≈ 14 (it grows only like `√N`).

**Contribution A — error-compensated multiplier.** `approx_mac.py` gains a
`rounding` knob:

- `trunc` — plain truncation (biased baseline; default, bit-for-bit unchanged).
- `round` — add the deterministic correction constant `2^(K-1)` before masking,
  i.e. round-to-nearest (Schulte & Swartzlander, 1993). Near-zero-mean error;
  costs ~one extra constant addend, folded into the partial-product tree.
- `stochastic` — add a uniform random offset in `[0, 2^K)` (Gupta et al.,
  2015), the technique modern FP8 hardware uses to remove bias — but it needs
  a per-MAC RNG/LFSR.

Finding: `round` cuts the coherent bias several-fold at near-zero hardware
cost; `stochastic` is the only genuinely zero-mean mode (`√N` growth), but
pays for the per-MAC RNG.

**Contribution B — layer-wise non-uniform K.** A sensitivity-driven per-layer
`K` budget (cf. ALWANN, ICCAD 2019; HAWQ, ICCV 2019) instead of one global
`K`. `sensitivity.py` probes each layer's output divergence and steers the
budget toward error-tolerant, MAC-heavy layers; on the (energy, error) plane
it dominates uniform `K`.

**Modernization (FP8 era).** `exact_mac.py` adds the FP8 formats E4M3 / E5M2
(Micikevicius et al., 2022; NVIDIA Hopper/Blackwell) so the study covers the
formats 2025 inference hardware actually uses, not just FP16/BF16/FP32.

## Layout

```
project/
  axmac/
    exact_mac.py        # Bit-accurate INT4/8/16 + FP8/FP16/BF16/FP32 MAC
    approx_mac.py       # Truncated multiplier + ACA adder; K, W, rounding knobs
    power_model.py      # Switching-activity energy model (45 nm); rounding cost
    accuracy_eval.py    # MED/RMSE/max/NMED/bias sweeps across K and rounding
    dnn_inference.py    # Vectorized INT inference + optional torch backend
    sensitivity.py      # Contribution B: per-layer sensitivity + K allocation
    pareto.py           # (precision, K) design-space sweep + Pareto fronts
  experiments/
    redesign_experiments.py   # baseline -> deficiency -> improvement driver
    results/                  # CSV outputs + console summaries
  tests/                # pytest unit tests (343 passing, 1 skipped)
  main.py               # original Week-6 design-space driver
  REDESIGN.md           # research framing + roadmap
  reference/README.md   # per-formula -> paper citation index
```

## Supported formats

| Format   | Bits | Layout       | Source                         |
|----------|------|--------------|--------------------------------|
| INT4     | 4    | two's compl. | custom                         |
| INT8     | 8    | two's compl. | custom                         |
| INT16    | 16   | two's compl. | custom                         |
| FP8 E4M3 | 8    | s1 e4 m3     | OCP / Micikevicius et al. 2022 |
| FP8 E5M2 | 8    | s1 e5 m2     | OCP / Micikevicius et al. 2022 |
| FP16     | 16   | s1 e5 m10    | IEEE 754 binary16              |
| BF16     | 16   | s1 e8 m7     | bfloat16                       |
| FP32     | 32   | s1 e8 m23    | IEEE 754 binary32              |

## Running

From the project root:

```powershell
python -m pytest tests                         # unit tests
python experiments/redesign_experiments.py     # regenerate redesign results
```

`tests/` is a package and the experiment driver puts the repo root on
`sys.path` itself, so do **not** set `PYTHONPATH`. The driver writes CSVs plus
a console summary to `experiments/results/`.

## Headline results

(from `experiments/results/redesign_summary.txt` and `three_way_rounding.csv`)

- **Deficiency — coherent bias.** Accumulated INT8 truncation error at depth
  `N = 4096`: `trunc` ≈ 26,617 (linear in `N`) vs. `stochastic` ≈ 14
  (`√N`-bounded).
- **Contribution A — bias compensation.** Three-way per-MAC comparison (INT8,
  `K = 6`): bias drops from **+29.9** (`trunc`) to **−1.7** (`round`) to
  **−0.8** (`stochastic`) — `round` reaching it at near-zero added hardware
  cost.
- **Contribution B — non-uniform K.** At matched per-inference energy,
  sensitivity-driven `K` allocation lowers logit-NRMSE versus a uniform global
  `K` on a 256-wide MLP.

## Dependencies

- Python 3.12+ (developed on 3.14)
- numpy >= 2.0
- pytest (development / tests)
- torch (optional — only `make_approx_linear_torch` needs it; the test that
  exercises it is skipped when torch is absent)
