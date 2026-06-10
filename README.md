# AxMAC: An Error-Compensated Approximate MAC for FP8-Era DNN Inference

EEC 289Q 002 SQ 2026 — Deep Learning Hardware course project.
Authors: Jiabo Zhang, Yuxuan Wang.

A Python testbed that **reproduces** a prior-work approximate MAC (truncated
multiplier + ACA adder), **identifies a concrete deficiency** in it, and
**proposes two improvements** — quantifying the gain. See `docs/REDESIGN.md` for the
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
  axmac/                          # Python source package
    exact_mac.py                  # Bit-accurate INT4/8/16 + FP8/FP16/BF16/FP32 MAC
    approx_mac.py                 # Truncated multiplier + ACA adder; K, W, rounding knobs
    power_model.py                # Switching-activity energy model (45 nm); rounding cost
    accuracy_eval.py              # MED/RMSE/max/NMED/bias sweeps across K and rounding
    dnn_inference.py              # Vectorized INT inference + optional torch backend
    sensitivity.py                # Contribution B: per-layer sensitivity + K allocation
    pareto.py                     # (precision, K) design-space sweep + Pareto fronts
  tests/                          # pytest unit tests (350 passed)
  experiments/                    # Experiment drivers and results
    redesign_experiments.py       # baseline -> deficiency -> improvement driver
    cifar10_experiment.py         # CIFAR-10 INT8 PTQ accuracy sweep
    generate_figures.py           # Regenerate paper figures from CSVs
    results/                      # CSV outputs + console summaries
      figures/                    # 5 paper figures (fig1-fig5)
    EXPERIMENT_CONCLUSIONS.md     # Detailed conclusions from all experiments
  rtl/                            # Verilog RTL — 9 modules, 6 ModelSim testbenches
  docs/                           # Project documentation
    REDESIGN.md                   # Research framing + roadmap + task status
    CODE_WALKTHROUGH.md           # Python codebase step-by-step (Steps 1-12)
    RTL_WALKTHROUGH.md            # RTL codebase step-by-step (Steps 1-12)
    ONCHIP_TEST_RESULTS.md        # On-chip test analysis: PPA + board UART
    project_overview.html         # Visual project overview
  deliverables/                   # Final submission files
    AxMAC_Defense_Slides.pptx     # 11-slide defense presentation (~10 min)
    Bias_Aware_Approximate_MAC_IEEE_v4.docx  # IEEE-format report
    Bias_Aware_Approximate_MAC_IEEE_v3.docx  # Previous version (archived)
  scripts/                        # Utility and generation scripts
    main.py                       # Week-6 design-space driver (run: python scripts/main.py)
    make_slides.py                # Regenerate PPTX from scratch
    update_report_v4.py           # Upgrade report v3 -> v4
    read_uart.py                  # UART logit readout from board
    add_refs_slide.py             # Append references slide to PPTX
    burn/                         # Hardware programming scripts (Quartus JTAG)
      burn_K0.bat                 # Program K=0 trunc bitstream
      burn_K2_trunc.bat
      burn_K4_round.bat
      burn_K4_trunc.bat
      burn_K6_trunc.bat
      burn_capture.bat
      run_drum_tb.bat             # Run DRUM ModelSim testbench
  reference/                      # Citation index
    README.md                     # Per-formula -> paper mapping
    参考文献清单.txt               # Full reference list
  requirements.txt
  pytest.ini
  README.md                       # This file
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
python -m pytest tests                          # unit tests
python experiments/redesign_experiments.py      # regenerate redesign results
python experiments/cifar10_experiment.py        # CIFAR-10 accuracy sweep
python scripts/main.py                          # Week-6 design-space sweep
py -3.14 scripts/make_slides.py                 # regenerate defense slides
```

`tests/` is a package and the experiment driver puts the repo root on
`sys.path` itself, so do **not** set `PYTHONPATH`. The driver writes CSVs plus
a console summary to `experiments/results/`.

## Headline results

(from `experiments/results/redesign_summary.txt` and `three_way_rounding.csv`)

- **Deficiency — coherent bias.** Accumulated INT8 truncation error at depth
  `N = 4096`: `trunc` ≈ 26,617 (linear in `N`) vs. `stochastic` ≈ 14
  (`√N`-bounded).
- **Theorem 1 — formal bias bounds.** E[bias_N] = N·2^(K−1) for trunc (linear); 0 for
  round (zero for any N). Transformer Corollary: d_head=64, K=4 → 512 LSB bias per
  attention score, exceeding INT8 range.
- **Contribution A — bias compensation.** Three-way per-MAC comparison (INT8,
  `K = 6`): bias drops from **+29.9** (`trunc`) to **−1.7** (`round`) to
  **−0.8** (`stochastic`) — `round` reaching it at near-zero added hardware cost.
  **CIFAR-10 result:** round K=6 → **83.0%**; trunc K=6 → **10.8%** (72.1 pp gap).
- **Contribution B — non-uniform K.** At matched per-inference energy,
  sensitivity-driven `K` allocation lowers logit-NRMSE versus a uniform global
  `K` on a 256-wide MLP.

## Dependencies

- Python 3.12+ (developed on 3.14)
- numpy >= 2.0
- pytest (development / tests)
- torch (optional — only `make_approx_linear_torch` needs it; the test that
  exercises it is skipped when torch is absent)

## RTL & hardware results

RTL implementation targets Altera Cyclone IV EP4CE10 (野火征途 Pro, 50 MHz).

| Phase | Work | Status |
|-------|------|--------|
| 1–3 | mac_unit / aca_adder / mac_array + testbenches | ModelSim **PASS** |
| 4 | mlp_top FSM (64→16→10, 8-state, 5 tile configs) | ModelSim **PASS** — 10/10 logits bit-exact vs Python golden |
| 4b | mlp_top_demo + UART TX — LED argmax display | ModelSim **PASS**; board burn ✅ |
| 5 | Quartus synthesis + 5-config K-sweep on hardware | **PASS** — K=6 trunc argmax 1→3 misclassification confirmed on-chip |
| P1.1 | drum_multiplier.v + tb_drum_multiplier.sv | ModelSim **PASS** — 200 vectors 0 failed (2026-06-05) |

**Key on-chip finding:** K=6 truncation causes actual misclassification on hardware (argmax 1→3); K=4 round mode preserves correct argmax — strongest empirical support for Contribution A.

**Quartus PPA (EP4CE10, 50 MHz, 60 nm LP):**

| Config | Logic Elements | Registers | Core dyn. power |
|--------|---------------|-----------|----------------|
| Exact (K=0, W=32) | 1769 | 582 | 10.45 mW |
| K=6 trunc | 1685 | 527 | 8.67 mW (−17%) |
| K=6 round | 1731 | 527 | 9.20 mW (−12%) |
| K=6 stochastic | 1817 | 591 | 9.39 mW (stochastic adds exactly 64 reg = LFSR width) |
