"""Contribution B: layer-wise non-uniform K allocation.

Deficiency 2 in the project redesign: a single global truncation depth K is
applied to every layer, but layers differ both in error tolerance and in MAC
count (the demo MLP's first layer is 95.8 % of all MACs). Spending the same
K everywhere wastes that structure — a sensitive layer is needlessly
degraded, while a tolerant, MAC-heavy layer is left under-approximated.

This module measures per-layer sensitivity and distributes a fixed
approximation budget across layers so that error-tolerant, MAC-heavy layers
absorb most of the truncation. The approach follows the layer-wise
mixed-precision literature:

* HAWQ (Dong et al., ICCV 2019) — order layers by sensitivity, then assign
  precision along a Pareto frontier.
* ALWANN (Mrazek et al., ICCAD 2019) — per-layer selection of approximate
  multipliers for a DNN accelerator, without retraining.
* HAQ (Wang et al., CVPR 2019) — hardware-aware mixed-precision budget
  allocation.

Sensitivity is probed empirically here — one layer perturbed at a time —
rather than via a Hessian (HAWQ) or a learned policy (HAQ); for the truncated
multiplier the perturbation is cheap and exact, so a direct probe is the
honest choice. The budget is then handed out by a greedy marginal allocator.

Every formula below has a backing reference; see ``reference/README.md``.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from .dnn_inference import tiny_mlp_forward
from .exact_mac import IntFormat


SensitivityMetric = Literal["logit_nrmse", "argmax"]

Layers = Sequence[tuple[np.ndarray, "np.ndarray | None"]]


# ============================================================
# MAC counts
# ============================================================

def layer_mac_counts(layers: Layers) -> list[int]:
    """MAC count per layer for one input vector (batch size 1).

    A linear layer with weight matrix ``W`` of shape ``(in, out)`` performs
    ``in * out`` multiply-accumulates per input vector. The per-layer counts
    drive the energy side of the allocation: truncating a layer that runs
    more MACs saves proportionally more multiplier energy.
    """
    counts: list[int] = []
    for w, _b in layers:
        if w.ndim != 2:
            raise ValueError(f"expected a 2-D weight matrix, got shape {w.shape}")
        counts.append(int(w.shape[0]) * int(w.shape[1]))
    return counts


# ============================================================
# Output divergence metrics
# ============================================================

def _logit_nrmse(y: np.ndarray, y_ref: np.ndarray) -> float:
    """RMSE of ``y`` against ``y_ref``, normalised by the RMS of ``y_ref``."""
    diff = (y - y_ref).astype(np.float64)
    denom = np.sqrt(np.mean(y_ref.astype(np.float64) ** 2)) + 1e-12
    return float(np.sqrt(np.mean(diff ** 2)) / denom)


def output_divergence(
    y: np.ndarray,
    y_ref: np.ndarray,
    *,
    metric: SensitivityMetric = "logit_nrmse",
) -> float:
    """Scalar degradation of an approximate output ``y`` vs the exact ``y_ref``.

    * ``logit_nrmse`` — normalised RMSE in logit space; continuous, so it
      gives a usable signal even when the prediction does not yet flip.
    * ``argmax`` — fraction of inputs whose argmax prediction changed.
    """
    if metric == "logit_nrmse":
        return _logit_nrmse(y, y_ref)
    if metric == "argmax":
        return 1.0 - float(np.mean(y.argmax(axis=1) == y_ref.argmax(axis=1)))
    raise ValueError(f"unknown metric {metric!r}")


# ============================================================
# Per-layer sensitivity probe
# ============================================================

def layer_sensitivity(
    x: np.ndarray,
    layers: Layers,
    *,
    fmt: IntFormat,
    K_probe: int = 4,
    metric: SensitivityMetric = "logit_nrmse",
) -> list[float]:
    """Per-layer sensitivity = output degradation when only that layer is truncated.

    Layer ``i`` is run at ``K = K_probe`` while every other layer stays exact
    (``K = 0``); the output is compared against the all-exact baseline. A
    larger score means the layer tolerates truncation less well. This is the
    one-layer-at-a-time perturbation analysis HAWQ / ALWANN perform, done here
    directly because the truncated-multiplier perturbation is exact and cheap.
    """
    n = len(layers)
    if n == 0:
        raise ValueError("need at least one layer")
    if K_probe <= 0:
        raise ValueError("K_probe must be >= 1")
    y_ref = tiny_mlp_forward(x, layers, fmt=fmt, K=0)
    scores: list[float] = []
    for i in range(n):
        K_vec = [0] * n
        K_vec[i] = K_probe
        y = tiny_mlp_forward(x, layers, fmt=fmt, K=K_vec)
        scores.append(output_divergence(y, y_ref, metric=metric))
    return scores


# ============================================================
# Budget allocators
# ============================================================

def uniform_K(n_layers: int, total_budget: int) -> list[int]:
    """Spread ``total_budget`` truncation units evenly — the baseline allocation.

    Any remainder is handed to the first layers, so the result sums to exactly
    ``total_budget`` and stays within one unit of perfectly uniform. This is
    the global-K policy Contribution B is measured against, at matched budget.
    """
    if n_layers <= 0:
        raise ValueError("n_layers must be >= 1")
    if total_budget < 0:
        raise ValueError("total_budget must be >= 0")
    base, extra = divmod(total_budget, n_layers)
    return [base + (1 if i < extra else 0) for i in range(n_layers)]


def allocate_K(
    sensitivities: Sequence[float],
    mac_counts: Sequence[int],
    *,
    total_budget: int,
    K_max: int,
) -> list[int]:
    """Greedy marginal allocation of a truncation budget across layers.

    ``total_budget`` truncation-bit units are handed out one at a time. Each
    unit goes to the layer that currently maximises

        benefit / cost  =  mac_counts[i] / (sensitivities[i] * (K[i] + 1))

    — the most MAC-heavy (largest energy win), least-sensitive (smallest
    accuracy loss) layer. The ``(K[i] + 1)`` factor models the rising marginal
    cost of deepening an already-truncated layer, so the budget spreads
    instead of dumping onto one layer. ``K[i]`` is capped at ``K_max``.

    This is the greedy knapsack heuristic behind HAWQ's sensitivity-ordered
    bit assignment and ALWANN's per-layer approximate-multiplier selection.
    With equal sensitivities and equal MAC counts it reduces to :func:`uniform_K`.
    """
    n = len(sensitivities)
    if n == 0:
        raise ValueError("need at least one layer")
    if len(mac_counts) != n:
        raise ValueError("sensitivities and mac_counts must have equal length")
    if total_budget < 0:
        raise ValueError("total_budget must be >= 0")
    if K_max < 0:
        raise ValueError("K_max must be >= 0")

    K = [0] * n
    eps = 1e-12
    for _ in range(total_budget):
        best_i, best_score = -1, -1.0
        for i in range(n):
            if K[i] >= K_max:
                continue
            score = mac_counts[i] / (sensitivities[i] * (K[i] + 1) + eps)
            if score > best_score:
                best_score, best_i = score, i
        if best_i < 0:
            break  # every layer has saturated at K_max
        K[best_i] += 1
    return K


def allocate_K_for_mlp(
    x: np.ndarray,
    layers: Layers,
    *,
    fmt: IntFormat,
    total_budget: int,
    K_max: int,
    K_probe: int = 4,
    metric: SensitivityMetric = "logit_nrmse",
) -> tuple[list[int], list[float], list[int]]:
    """End-to-end Contribution B: probe sensitivity, then allocate the budget.

    Returns ``(K_allocation, sensitivities, mac_counts)`` so a caller can both
    use the allocation and report why each layer got the K it did.
    """
    sens = layer_sensitivity(x, layers, fmt=fmt, K_probe=K_probe, metric=metric)
    macs = layer_mac_counts(layers)
    K = allocate_K(sens, macs, total_budget=total_budget, K_max=K_max)
    return K, sens, macs
