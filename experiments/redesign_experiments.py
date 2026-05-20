"""Redesign experiment driver — baseline -> deficiency -> improvement.

This is the experimental backbone of the repositioned project (see
``REDESIGN.md``). It is kept separate from ``main.py`` (the original W6
design-space driver) so the redesign results stand on their own.

Four experiments, each tied to a deficiency / contribution:

  1. Bias accumulation vs. depth  — deficiency 1: plain truncation has a
     one-signed error, so it accumulates *coherently* (∝ N) over a dot
     product, while error-compensated rounding only grows like √N.
  2. Three-way rounding (INT8)    — contribution A: trunc vs. round vs.
     stochastic on per-MAC MED / RMSE / bias.
  3. FP8 study (E4M3 / E5M2)      — modernization: error + energy across K,
     and that deterministic compensation removes the FP8 mantissa-truncation
     bias just as it does for integers.
  4. Layer-wise K allocation     — contribution B: a sensitivity-driven
     non-uniform K budget vs. uniform global K, on a 784->128->32->10 MLP.

Run it (the full-width-colon project path breaks Python's startup if used as
the working directory, so the script puts the repo root on sys.path itself):

    py -3.14 experiments/redesign_experiments.py

Outputs land in ``experiments/results/`` as CSV plus a console summary.

Every metric / model used here has a backing reference; see
``reference/README.md``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

# Windows consoles default to cp1252; the project path (full-width colon) and
# a few report symbols fall outside it. Force UTF-8 so prints never crash.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# The repo root must be importable regardless of the working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from axmac.accuracy_eval import sweep_fp, sweep_int  # noqa: E402
from axmac.dnn_inference import tiny_mlp_forward, truncate_products  # noqa: E402
from axmac.exact_mac import FP8_E4M3, FP8_E5M2, FP16, INT8  # noqa: E402
from axmac.power_model import mac_fp_energy, mac_int_energy  # noqa: E402
from axmac.sensitivity import (  # noqa: E402
    allocate_K,
    layer_mac_counts,
    layer_sensitivity,
    output_divergence,
    uniform_K,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ROUNDINGS = ("trunc", "round", "stochastic")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote results/{path.name}  ({len(rows)} rows)")


# ============================================================
# 1) Bias accumulation vs. accumulation depth
# ============================================================

def section_bias_accumulation() -> list[dict]:
    """Deficiency 1: how the per-MAC error accumulates with depth N.

    For each depth N we form T independent INT8 dot products, truncate every
    product (K bits) under each rounding mode, and measure the accumulated
    error (exact_sum - approx_sum). The accumulator is kept exact (int64, no
    wrap) so the experiment isolates the *truncation* error.

    The finding (reported per-MAC so depths are comparable):
      * trunc       — a fixed positive per-MAC bias; the total error grows
                      linearly with N (coherent accumulation).
      * round       — a much smaller, still one-signed per-MAC bias: the
                      integer correction constant 2^(K-1) cannot cancel the
                      data-dependent skew of the product LSBs, so the total
                      error is still linear in N, just with a smaller slope.
      * stochastic  — a genuinely zero-mean per-MAC error; the total grows
                      only like sqrt(N). This is why FP8 hardware uses it.
    """
    print("\n=== 1) Error accumulation vs. depth (INT8, K=4) ===")
    K = 4
    T = 2000
    depths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    rng = np.random.default_rng(0xB1A5)

    rows: list[dict] = []
    print(f"  {'N':>6} | {'trunc/MAC':>9} {'round/MAC':>9} {'stoch/MAC':>10} "
          f"| {'stoch rms/sqrtN':>16}")
    for N in depths:
        a = rng.integers(INT8.min_val, INT8.max_val + 1, size=(T, N), dtype=np.int64)
        b = rng.integers(INT8.min_val, INT8.max_val + 1, size=(T, N), dtype=np.int64)
        prod = a * b
        exact = prod.sum(axis=1).astype(np.float64)

        row: dict = {"N": N}
        for mode in ROUNDINGS:
            sub_rng = np.random.default_rng(0x5EED) if mode == "stochastic" else None
            approx = truncate_products(prod, K, rounding=mode, rng=sub_rng)
            err = exact - approx.sum(axis=1).astype(np.float64)
            row[f"{mode}_mean"] = float(err.mean())
            row[f"{mode}_rms"] = float(np.sqrt(np.mean(err ** 2)))
        rows.append(row)
        print(f"  {N:>6} | {row['trunc_mean'] / N:>9.3f} "
              f"{row['round_mean'] / N:>9.3f} "
              f"{row['stochastic_mean'] / N:>10.4f} | "
              f"{row['stochastic_rms'] / np.sqrt(N):>16.3f}")

    trunc_pm = float(np.mean([r["trunc_mean"] / r["N"] for r in rows]))
    round_pm = float(np.mean([r["round_mean"] / r["N"] for r in rows]))
    stoch_pm = float(np.mean([r["stochastic_mean"] / r["N"] for r in rows]))
    stoch_rms_n = [r["stochastic_rms"] / np.sqrt(r["N"]) for r in rows]
    big = rows[-1]
    print(f"  -> trunc : per-MAC bias ~ {trunc_pm:+.2f}; total error grows "
          f"linearly with N (coherent).")
    print(f"  -> round : per-MAC bias ~ {round_pm:+.2f} "
          f"({abs(trunc_pm) / abs(round_pm):.1f}x smaller than trunc) -- "
          f"reduced, but still one-signed and linear in N.")
    print(f"  -> stoch : per-MAC bias ~ {stoch_pm:+.4f}; rms/sqrt(N) flat at "
          f"~{np.mean(stoch_rms_n):.2f} -- zero-mean, error grows like sqrt(N).")
    print(f"  at N={big['N']}: |error| trunc {abs(big['trunc_mean']):.0f}  "
          f"round {abs(big['round_mean']):.0f}  stoch {abs(big['stochastic_mean']):.0f}")

    _write_csv(
        RESULTS_DIR / "bias_accumulation.csv",
        ["N", "trunc_mean", "trunc_rms", "round_mean", "round_rms",
         "stochastic_mean", "stochastic_rms"],
        rows,
    )
    return rows


# ============================================================
# 2) Three-way rounding comparison on INT8
# ============================================================

def section_three_way_rounding() -> list[dict]:
    """Contribution A: trunc vs. round vs. stochastic, per-MAC error stats."""
    print("\n=== 2) Three-way rounding (INT8, per-MAC error) ===")
    rows: list[dict] = []
    print(f"  {'K':>2} {'mode':>11} | {'MED':>10} {'RMSE':>10} {'bias':>12}")
    for K in [1, 2, 3, 4, 5, 6]:
        for mode in ROUNDINGS:
            stats = sweep_int(
                INT8, [K], [None], n_samples=4000, rounding=mode,
            )[(K, None)]
            rows.append(dict(
                fmt="INT8", K=K, rounding=mode,
                med=stats.med, rmse=stats.rmse,
                nmed=stats.nmed, bias=stats.bias,
            ))
            print(f"  {K:>2} {mode:>11} | {stats.med:>10.3f} "
                  f"{stats.rmse:>10.3f} {stats.bias:>+12.3f}")

    k6 = {r["rounding"]: r for r in rows if r["K"] == 6}
    print(f"  -> at K=6: bias trunc {k6['trunc']['bias']:+.1f} -> "
          f"round {k6['round']['bias']:+.1f} -> stochastic "
          f"{k6['stochastic']['bias']:+.2f}; MED stays comparable "
          f"({k6['trunc']['med']:.1f} / {k6['round']['med']:.1f} / "
          f"{k6['stochastic']['med']:.1f}).")
    print("     trunc's error is one-signed (bias == MED); round and "
          "stochastic both make it near-zero-mean -- stochastic most so.")

    _write_csv(
        RESULTS_DIR / "three_way_rounding.csv",
        ["fmt", "K", "rounding", "med", "rmse", "nmed", "bias"],
        rows,
    )
    return rows


# ============================================================
# 3) FP8 study: E4M3 / E5M2 error + energy across K
# ============================================================

def section_fp8_study() -> list[dict]:
    """Modernization: FP8 (E4M3 / E5M2) error + energy sweep across K.

    Unlike the integer path, FP8 mantissa truncation is RMSE-dominated, not
    bias-dominated: the FP datapath's mandatory renormalization RNE re-rounds
    the result, so the strong systematic bias the integer path shows does not
    appear. The interesting trade-off here is error vs. per-MAC energy, and
    that FP8 is several times cheaper than FP16 to begin with.
    """
    print("\n=== 3) FP8 study (E4M3 / E5M2 mantissa truncation) ===")
    rows: list[dict] = []
    formats = [("E4M3", FP8_E4M3, 6), ("E5M2", FP8_E5M2, 5), ("FP16", FP16, 6)]

    print(f"  {'fmt':>6} {'K':>2} {'mode':>6} | {'NMED':>11} {'RMSE':>11} "
          f"{'bias':>12} {'E/MAC pJ':>9}")
    for name, fmt, k_hi in formats:
        for mode in ("trunc", "round"):
            stats_by_K = sweep_fp(
                fmt, list(range(0, k_hi + 1)), n_samples=4000, rounding=mode,
            )
            for K, stats in stats_by_K.items():
                energy = mac_fp_energy(fmt, K=K, rounding=mode).total_pJ
                rows.append(dict(
                    fmt=name, K=K, rounding=mode,
                    nmed=stats.nmed, rmse=stats.rmse,
                    bias=stats.bias, energy_pJ=energy,
                ))
                print(f"  {name:>6} {K:>2} {mode:>6} | {stats.nmed:>11.3e} "
                      f"{stats.rmse:>11.3e} {stats.bias:>+12.3e} {energy:>9.4f}")

    # Headline: FP8 error/energy trade-off, and the FP8-vs-FP16 energy gap.
    for name in ("E4M3", "E5M2"):
        fr = [r for r in rows if r["fmt"] == name and r["rounding"] == "trunc"]
        k_hi = max(r["K"] for r in fr)
        deep = next(r for r in fr if r["K"] == k_hi)
        base = next(r for r in fr if r["K"] == 0)
        e_save = 100 * (base["energy_pJ"] - deep["energy_pJ"]) / base["energy_pJ"]
        print(f"  -> {name}: K={k_hi} cuts MAC energy "
              f"{base['energy_pJ']:.3f}->{deep['energy_pJ']:.3f} pJ "
              f"({e_save:.0f}%) at RMSE {deep['rmse']:.3f}; "
              f"|bias| stays ~{abs(deep['bias']):.0e} (renorm RNE suppresses it).")
    fp16_0 = next(r for r in rows
                  if r["fmt"] == "FP16" and r["K"] == 0 and r["rounding"] == "trunc")
    e4m3_0 = next(r for r in rows
                  if r["fmt"] == "E4M3" and r["K"] == 0 and r["rounding"] == "trunc")
    print(f"  -> FP8 starts {fp16_0['energy_pJ'] / e4m3_0['energy_pJ']:.1f}x "
          f"cheaper per MAC than FP16 ({e4m3_0['energy_pJ']:.2f} vs "
          f"{fp16_0['energy_pJ']:.2f} pJ), before any truncation.")

    _write_csv(
        RESULTS_DIR / "fp8_study.csv",
        ["fmt", "K", "rounding", "nmed", "rmse", "bias", "energy_pJ"],
        rows,
    )
    return rows


# ============================================================
# 4) Layer-wise K allocation: uniform vs. sensitivity-driven
# ============================================================

def _build_mlp(seed: int = 0xCAFE):
    """A 256->256->256->256->10 INT8 MLP on synthetic data (random, fixed seed).

    The three equal-width hidden layers carry the *same* MAC count, but —
    sitting at different depths — they differ in how much truncation error
    they inject into the output. That is exactly the inter-layer heterogeneity
    Contribution B exploits: uniform K cannot see it, a sensitivity-driven
    allocation can. Weights are random (untrained); the experiment measures
    approximate-vs-exact INT8 fidelity, not task accuracy.
    """
    rng = np.random.default_rng(seed)
    batch = 64
    x = rng.integers(0, 48, size=(batch, 256))
    x = np.where(rng.random((batch, 256)) < 0.5, 0, x)
    layers = []
    for din, dout in [(256, 256), (256, 256), (256, 256), (256, 10)]:
        w = rng.integers(-24, 24, size=(din, dout))
        b = rng.integers(-300, 300, size=(dout,))
        layers.append((w, b))
    return x, layers, INT8


def _mlp_energy_pJ(mac_counts: list[int], K_vec: list[int], fmt) -> float:
    """Per-inference INT MAC energy for a per-layer K vector."""
    return sum(
        m * mac_int_energy(fmt, K=k).total_pJ
        for m, k in zip(mac_counts, K_vec)
    )


def _pareto_min(points: list[dict], x_key: str, y_key: str) -> list[dict]:
    """Non-dominated subset under 'smaller is better' on both keys."""
    front: list[dict] = []
    for i, p in enumerate(points):
        if not any(
            j != i
            and q[x_key] <= p[x_key] and q[y_key] <= p[y_key]
            and (q[x_key] < p[x_key] or q[y_key] < p[y_key])
            for j, q in enumerate(points)
        ):
            front.append(p)
    return front


def section_layer_allocation() -> list[dict]:
    """Contribution B: sensitivity-driven non-uniform K vs. uniform global K.

    Quality is the logit-space NRMSE of the approximate MLP output against
    the *exact* INT8 output (K=0) — a continuous, monotone measure of pure
    approximation error. (Argmax agreement on an untrained net is too noisy
    to optimise against.) The allocator should reach a given NRMSE at lower
    energy, or a lower NRMSE at given energy, than uniform K.
    """
    print("\n=== 4) Layer-wise K allocation (256->256->256->256->10 INT8 MLP) ===")
    x, layers, fmt = _build_mlp()
    macs = layer_mac_counts(layers)
    total_macs = sum(macs)
    print(f"  per-layer MACs: {macs}  "
          f"(layer 1 = {100 * macs[0] / total_macs:.1f}% of all MACs)")

    y_ref = tiny_mlp_forward(x, layers, fmt=fmt, K=0)
    sens = layer_sensitivity(x, layers, fmt=fmt, K_probe=4)
    print(f"  per-layer sensitivity (K_probe=4 logit NRMSE): "
          f"{[round(s, 4) for s in sens]}")

    K_max = 6
    n_layers = len(layers)
    rows: list[dict] = []
    print(f"  {'budget':>6} {'policy':>9} {'K vector':>12} | "
          f"{'logit NRMSE':>12} {'E/inf nJ':>9}")
    for budget in range(0, n_layers * K_max + 1):
        for policy in ("uniform", "allocated"):
            if policy == "uniform":
                K_vec = uniform_K(n_layers, budget)
            else:
                K_vec = allocate_K(sens, macs, total_budget=budget, K_max=K_max)
            if any(k > K_max for k in K_vec):
                continue
            y = tiny_mlp_forward(x, layers, fmt=fmt, K=K_vec)
            nrmse = output_divergence(y, y_ref, metric="logit_nrmse")
            energy_pJ = _mlp_energy_pJ(macs, K_vec, fmt)
            rows.append(dict(
                budget=budget, policy=policy, K_vector="|".join(map(str, K_vec)),
                logit_nrmse=nrmse, energy_per_inference_pJ=energy_pJ,
            ))
            if budget in (4, 8, 12, 16, 20):
                print(f"  {budget:>6} {policy:>9} {str(K_vec):>12} | "
                      f"{nrmse:>12.4f} {energy_pJ / 1e3:>9.3f}")

    _write_csv(
        RESULTS_DIR / "layer_allocation.csv",
        ["budget", "policy", "K_vector", "logit_nrmse",
         "energy_per_inference_pJ"],
        rows,
    )

    # Joint (energy, NRMSE) Pareto front — which policy owns it?
    front = _pareto_min(rows, "energy_per_inference_pJ", "logit_nrmse")
    front.sort(key=lambda r: r["energy_per_inference_pJ"])
    n_alloc = sum(1 for r in front if r["policy"] == "allocated")
    print(f"  -> joint (energy, NRMSE) Pareto front: {len(front)} points, "
          f"{n_alloc} allocated / {len(front) - n_alloc} uniform")

    # Matched-energy comparison: lowest-NRMSE allocated config at no more
    # energy than each uniform config.
    best_drop = 0.0
    for u in rows:
        if u["policy"] != "uniform" or u["logit_nrmse"] <= 0.0:
            continue
        cands = [r for r in rows if r["policy"] == "allocated"
                 and r["energy_per_inference_pJ"]
                 <= u["energy_per_inference_pJ"] + 1e-9]
        if not cands:
            continue
        best = min(cands, key=lambda r: r["logit_nrmse"])
        drop = (u["logit_nrmse"] - best["logit_nrmse"]) / u["logit_nrmse"]
        best_drop = max(best_drop, drop)
    print(f"  -> at matched energy, sensitivity-driven allocation cuts logit "
          f"NRMSE by up to {best_drop * 100:.0f}% vs uniform K")
    return rows


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    print("=" * 64)
    print("AxMAC redesign experiments - baseline / deficiency / improvement")
    print("=" * 64)

    bias_rows = section_bias_accumulation()
    rounding_rows = section_three_way_rounding()
    fp8_rows = section_fp8_study()
    alloc_rows = section_layer_allocation()

    big = bias_rows[-1]
    summary = [
        "AxMAC redesign experiment summary",
        "=" * 64,
        f"1) error accumulation: {len(bias_rows)} depths (N=1..{big['N']})",
        f"2) three-way rounding: {len(rounding_rows)} (K, mode) configs",
        f"3) FP8 study         : {len(fp8_rows)} (fmt, K, mode) configs",
        f"4) layer allocation  : {len(alloc_rows)} (budget, policy) configs",
        "",
        "Deficiency 1 - truncation error is coherent: at N=" + str(big["N"])
        + f" the accumulated trunc error is {big['trunc_mean']:.0f} (grows"
        " linearly with N), vs a much smaller but still linear round error of"
        f" {big['round_mean']:.0f} and a near-zero stochastic error of"
        f" {big['stochastic_mean']:+.0f}.",
        "Contribution A - stochastic rounding is the only mode with a truly"
        " zero-mean per-MAC error (sqrt(N) growth); deterministic round cuts"
        " the coherent bias several-fold at zero hardware cost but cannot"
        " remove it, because no data-independent constant cancels the"
        " product-LSB skew.",
        "Modernization - the FP8 formats E4M3 / E5M2 are swept for error and"
        " energy across K, under both trunc and round.",
        "Contribution B - a sensitivity-driven non-uniform K budget is"
        " compared against uniform global K on the (energy, error) plane.",
    ]
    (RESULTS_DIR / "redesign_summary.txt").write_text(
        "\n".join(summary), encoding="utf-8")
    print("\nwrote results/redesign_summary.txt")
    print("=" * 64)
    print("done.")


if __name__ == "__main__":
    main()
