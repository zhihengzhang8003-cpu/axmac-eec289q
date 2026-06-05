"""DNN inference with AxMAC injected as a custom torch.autograd.Function.

Stage 5 deliverable. Two interchangeable backends:

* **NumPy** (default) — vectorized integer matmul/conv2d that reproduces the
  scalar :mod:`axmac.approx_mac` semantics across whole tensors. Fast enough
  to run real inference on MNIST-sized data.
* **PyTorch** (optional) — :func:`make_approx_linear_torch` returns a
  ``torch.autograd.Function`` that calls the NumPy backend on the forward
  pass and uses a straight-through gradient on backward. Imported lazily
  so this module loads without ``torch`` installed.

Scope:

* INT path is fully vectorized (matmul + conv2d via im2col).
* FP path is provided as a scalar reference loop that calls
  :func:`approx_mac_fp` — accurate but slow; intended for cross-checks,
  not for running CIFAR-sized models.
* Bias is added as an exact 32-bit integer (INT) or in the format (FP).
* No ACA on layer-level accumulation: ACA applies per-MAC and would be
  meaningless after a wide reduction. K-bit truncation is per-MAC.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

import random as _random

from .approx_mac import RoundingMode, approx_mac_fp, approx_mac_int
from .exact_mac import FPFormat, IntFormat, decode_fp, encode_fp

_VALID_ROUNDING = ("trunc", "round", "stochastic")


# ============================================================
# Quantization helpers
# ============================================================

def quantize_to_int(x: np.ndarray, fmt: IntFormat) -> np.ndarray:
    """Round-and-clip a float array into the integer range of ``fmt``."""
    out = np.rint(x).astype(np.int64)
    return np.clip(out, fmt.min_val, fmt.max_val)


def _wrap_int32(arr: np.ndarray, acc_bits: int) -> np.ndarray:
    """Two's-complement wrap an int64 array to a signed ``acc_bits`` window."""
    mask = (1 << acc_bits) - 1
    half = 1 << (acc_bits - 1)
    wrapped = arr & mask
    return np.where(wrapped >= half, wrapped - (1 << acc_bits), wrapped)


def truncate_products(
    prod: np.ndarray,
    K: int,
    *,
    rounding: RoundingMode = "trunc",
    rng: "np.random.Generator | None" = None,
) -> np.ndarray:
    """Drop the low K bits of every product in ``prod`` under ``rounding``.

    Vectorized counterpart of :func:`axmac.approx_mac._apply_rounding`:

    * ``trunc``      — ``prod & ~(2^K - 1)``; rounds toward -inf, biased.
    * ``round``      — add the correction constant 2^(K-1) first
                       (Schulte & Swartzlander 1993); near-zero-mean error.
    * ``stochastic`` — add a fresh uniform offset in [0, 2^K) per element
                       (Gupta et al. 2015); needs a NumPy ``rng``.

    ``& ~mask`` on NumPy int64 rounds toward -inf for signed products just as
    Python's arbitrary-precision ints do. ``K <= 0`` is the identity.
    """
    if K <= 0:
        return prod
    if rounding not in _VALID_ROUNDING:
        raise ValueError(f"rounding must be one of {_VALID_ROUNDING}, got {rounding!r}")
    mask = ~((1 << K) - 1)
    if rounding == "round":
        prod = prod + (1 << (K - 1))
    elif rounding == "stochastic":
        gen = rng if rng is not None else np.random.default_rng()
        prod = prod + gen.integers(0, 1 << K, size=prod.shape, dtype=np.int64)
    return prod & mask


# ============================================================
# INT vectorized backend
# ============================================================

def int_matmul_approx(
    x: np.ndarray,
    w: np.ndarray,
    *,
    fmt: IntFormat,
    K: int = 0,
    acc_bits: int = 32,
    bias: np.ndarray | None = None,
    rounding: RoundingMode = "trunc",
    rng: "np.random.Generator | None" = None,
) -> np.ndarray:
    """Integer matmul ``x @ w + bias`` with K-bit per-MAC truncation.

    Both ``x`` and ``w`` must already be in ``fmt``'s integer range. Returns
    an int64 array (caller may further quantize). With ``rounding="trunc"``
    this is bit-equivalent to looping :func:`approx_mac_int` over the
    reduction axis with the same K; ``round`` / ``stochastic`` apply the
    error-compensated variants element-wise (see :func:`truncate_products`).
    """
    if K < 0:
        raise ValueError("K must be >= 0")
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError(f"matmul expects 2-D inputs, got {x.shape} and {w.shape}")
    if x.shape[1] != w.shape[0]:
        raise ValueError(f"reduction-dim mismatch: {x.shape} vs {w.shape}")

    a = x.astype(np.int64)
    b = w.astype(np.int64)

    # Per-MAC product: (M, K_dim, N) — could be large; reduce one row at a time
    # for big inputs. For now keep it simple and rely on int64 capacity.
    prod = a[:, :, None] * b[None, :, :]  # (M, K_dim, N)
    prod = truncate_products(prod, K, rounding=rounding, rng=rng)

    accum = prod.sum(axis=1)  # (M, N)
    if bias is not None:
        if bias.shape != (w.shape[1],):
            raise ValueError(f"bias shape {bias.shape} != ({w.shape[1]},)")
        accum = accum + bias.astype(np.int64)
    return _wrap_int32(accum, acc_bits)


def int_linear_approx(
    x: np.ndarray,
    w: np.ndarray,
    *,
    fmt: IntFormat,
    K: int = 0,
    bias: np.ndarray | None = None,
    acc_bits: int = 32,
    rounding: RoundingMode = "trunc",
    rng: "np.random.Generator | None" = None,
) -> np.ndarray:
    """Linear (fully-connected) layer: ``y = x @ w + bias`` with K-bit truncation."""
    return int_matmul_approx(
        x, w, fmt=fmt, K=K, bias=bias, acc_bits=acc_bits,
        rounding=rounding, rng=rng,
    )


def _im2col(
    x: np.ndarray,
    kh: int,
    kw: int,
    stride: int,
    padding: int,
) -> tuple[np.ndarray, int, int]:
    """Lay out conv input patches as rows. Returns (patches, out_h, out_w).

    ``x`` shape: (N, C, H, W). Output ``patches`` shape: (N*out_h*out_w, C*kh*kw).
    """
    if x.ndim != 4:
        raise ValueError(f"conv2d expects 4-D input, got {x.shape}")
    n, c, h, w = x.shape
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out_h = (h + 2 * padding - kh) // stride + 1
    out_w = (w + 2 * padding - kw) // stride + 1
    cols = np.empty((n, c, kh, kw, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j, :, :] = x[:, :, i : i + stride * out_h : stride,
                                       j : j + stride * out_w : stride]
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, c * kh * kw)
    return cols, out_h, out_w


def int_conv2d_approx(
    x: np.ndarray,
    w: np.ndarray,
    *,
    fmt: IntFormat,
    K: int = 0,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
    acc_bits: int = 32,
    rounding: RoundingMode = "trunc",
    rng: "np.random.Generator | None" = None,
) -> np.ndarray:
    """2-D conv via im2col + :func:`int_matmul_approx`.

    Shapes: ``x`` (N, C_in, H, W), ``w`` (C_out, C_in, kH, kW), output
    (N, C_out, H_out, W_out).
    """
    if w.ndim != 4:
        raise ValueError(f"conv2d weights must be 4-D, got {w.shape}")
    c_out, c_in, kh, kw = w.shape
    if x.shape[1] != c_in:
        raise ValueError(f"input channels {x.shape[1]} != weight C_in {c_in}")

    cols, out_h, out_w = _im2col(x, kh, kw, stride, padding)
    w_mat = w.reshape(c_out, c_in * kh * kw).T  # (C_in*kh*kw, C_out)
    flat_out = int_matmul_approx(
        cols, w_mat, fmt=fmt, K=K, bias=bias, acc_bits=acc_bits,
        rounding=rounding, rng=rng,
    )
    return flat_out.reshape(x.shape[0], out_h, out_w, c_out).transpose(0, 3, 1, 2)


# ============================================================
# FP scalar reference (for cross-checks)
# ============================================================

def fp_linear_approx_scalar(
    x_bits: np.ndarray,
    w_bits: np.ndarray,
    *,
    fmt: FPFormat,
    K: int = 0,
    bias_bits: np.ndarray | None = None,
    rounding: RoundingMode = "trunc",
    rng: "_random.Random | None" = None,
) -> np.ndarray:
    """Slow per-MAC FP linear that calls :func:`approx_mac_fp` directly.

    Inputs are bit-pattern arrays in ``fmt``'s encoding. Used to verify
    correctness on small tensors; not for production-scale inference.
    ``rounding`` selects trunc / round / stochastic on the mantissa product.
    """
    if K < 0:
        raise ValueError("K must be >= 0")
    if x_bits.ndim != 2 or w_bits.ndim != 2:
        raise ValueError(f"linear expects 2-D inputs, got {x_bits.shape} and {w_bits.shape}")
    if x_bits.shape[1] != w_bits.shape[0]:
        raise ValueError("reduction-dim mismatch")

    m, kdim = x_bits.shape
    _, n = w_bits.shape
    zero_bits = encode_fp(0.0, fmt)
    out_bits = np.zeros((m, n), dtype=np.int64)

    for i in range(m):
        for j in range(n):
            acc = zero_bits if bias_bits is None else int(bias_bits[j])
            for k in range(kdim):
                acc = approx_mac_fp(
                    int(x_bits[i, k]), int(w_bits[k, j]), acc, fmt,
                    K=K, rounding=rounding, rng=rng,
                )
            out_bits[i, j] = acc
    return out_bits


# ============================================================
# Optional PyTorch backend (lazy import)
# ============================================================

def _try_import_torch():
    try:
        import torch  # type: ignore[import-not-found]
        return torch
    except ImportError:
        return None


def make_approx_linear_torch(fmt: IntFormat, K: int = 0):
    """Build a ``torch.autograd.Function`` that calls the NumPy INT linear forward.

    Backward uses a straight-through estimator (gradient flows through
    truncation unchanged), the standard choice in approx-arith DNN papers.
    Raises :class:`ImportError` if PyTorch is not installed.
    """
    torch = _try_import_torch()
    if torch is None:
        raise ImportError("PyTorch is required for make_approx_linear_torch")

    class ApproxLinearFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: "torch.Tensor", w: "torch.Tensor",
                    bias: "torch.Tensor | None" = None):
            ctx.save_for_backward(x, w)
            x_np = quantize_to_int(x.detach().cpu().numpy(), fmt)
            w_np = quantize_to_int(w.detach().cpu().numpy(), fmt)
            bias_np = (
                quantize_to_int(bias.detach().cpu().numpy(), fmt)
                if bias is not None
                else None
            )
            out = int_linear_approx(x_np, w_np, fmt=fmt, K=K, bias=bias_np)
            return torch.from_numpy(out.astype(np.float32)).to(x.device)

        @staticmethod
        def backward(ctx, grad_out: "torch.Tensor"):
            x, w = ctx.saved_tensors
            grad_x = grad_out @ w.T
            grad_w = x.T @ grad_out
            grad_bias = grad_out.sum(dim=0) if grad_out.ndim > 1 else grad_out
            return grad_x, grad_w, grad_bias

    return ApproxLinearFn


# ============================================================
# Demo: tiny MLP forward
# ============================================================

def _per_layer_K(K: "int | Sequence[int]", n_layers: int) -> list[int]:
    """Normalise a scalar-or-sequence ``K`` argument to a per-layer list.

    A bare int broadcasts to every layer (uniform K); a sequence is taken
    layer-by-layer and must match the layer count. Contribution B
    (:mod:`axmac.sensitivity`) feeds the per-layer form.
    """
    if isinstance(K, (int, np.integer)):
        return [int(K)] * n_layers
    Ks = [int(k) for k in K]
    if len(Ks) != n_layers:
        raise ValueError(
            f"per-layer K has {len(Ks)} entries but the MLP has {n_layers} layers"
        )
    return Ks


def tiny_mlp_forward(
    x: np.ndarray,
    layers: Sequence[tuple[np.ndarray, np.ndarray | None]],
    *,
    fmt: IntFormat,
    K: "int | Sequence[int]" = 0,
    rounding: RoundingMode = "trunc",
    rng: "np.random.Generator | None" = None,
) -> np.ndarray:
    """Run an N-layer ReLU MLP forward with approximate MAC at every layer.

    ``layers`` is a list of ``(W, b)`` pairs; ``b`` may be ``None``. Inputs
    and weights must already be quantized to ``fmt``'s integer range. ReLU
    is applied to all layers except the last.

    ``K`` is either a single int (the same truncation depth everywhere) or a
    per-layer sequence — Contribution B's non-uniform allocation feeds the
    latter (see :mod:`axmac.sensitivity`). ``rounding`` selects the
    error-compensation mode for every layer (Contribution A).
    """
    h = x
    last = len(layers) - 1
    Ks = _per_layer_K(K, len(layers))
    for i, (w, b) in enumerate(layers):
        h = int_linear_approx(
            h, w, fmt=fmt, K=Ks[i], bias=b, rounding=rounding, rng=rng,
        )
        if i != last:
            h = np.clip(h, 0, fmt.max_val).astype(np.int64)
    return h


# ============================================================
# DRUM-k vectorized backend
# Hashemi et al., ICCAD 2015 — zero-mean approximate multiply
# ============================================================

def _drum_quantize_arr(x: np.ndarray, k: int) -> np.ndarray:
    """Vectorized DRUM-k operand quantization: keep k most significant bits.

    For each element, zeroes all bits below position (msb - k + 1) where
    msb = floor(log2(|x|)). Zero inputs remain zero. Uses element-wise
    numpy bitwise shifts (int64) for an efficient, loop-free implementation.
    """
    x = x.astype(np.int64)
    abs_x = np.abs(x)
    sign = np.where(x >= 0, np.int64(1), np.int64(-1))
    nonzero = abs_x > 0

    msb = np.zeros(abs_x.shape, dtype=np.int64)
    if np.any(nonzero):
        msb[nonzero] = np.floor(
            np.log2(abs_x[nonzero].astype(np.float64))
        ).astype(np.int64)

    shift = np.maximum(np.int64(0), msb - np.int64(k - 1))
    drum_abs = np.where(nonzero, (abs_x >> shift) << shift, np.int64(0))
    return np.where(nonzero, sign * drum_abs, np.int64(0))


def int_matmul_drum(
    x: np.ndarray,
    w: np.ndarray,
    *,
    fmt: IntFormat,
    k: int = 4,
    acc_bits: int = 32,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Integer matmul ``x @ w + bias`` with DRUM-k per-operand quantization.

    Unlike :func:`int_matmul_approx` which truncates the *product*, DRUM
    quantizes each *operand* to k MSBs before multiplying. The zero-mean
    error property (Hashemi et al. 2015) means accumulated bias stays O(√N)
    rather than O(N) — at the cost of a leading-bit detector + barrel shift
    per operand vs. a plain mask for trunc/round.
    """
    if k < 1:
        raise ValueError("k must be >= 1 for DRUM")
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError(f"matmul expects 2-D inputs, got {x.shape} and {w.shape}")
    if x.shape[1] != w.shape[0]:
        raise ValueError(f"reduction-dim mismatch: {x.shape} vs {w.shape}")

    a = _drum_quantize_arr(x, k)          # (M, K_dim)
    b = _drum_quantize_arr(w, k)          # (K_dim, N)

    prod = a[:, :, None] * b[None, :, :]  # (M, K_dim, N)
    accum = prod.sum(axis=1)              # (M, N)
    if bias is not None:
        if bias.shape != (w.shape[1],):
            raise ValueError(f"bias shape {bias.shape} != ({w.shape[1]},)")
        accum = accum + bias.astype(np.int64)
    return _wrap_int32(accum, acc_bits)


def int_linear_drum(
    x: np.ndarray,
    w: np.ndarray,
    *,
    fmt: IntFormat,
    k: int = 4,
    bias: np.ndarray | None = None,
    acc_bits: int = 32,
) -> np.ndarray:
    """Linear layer with DRUM-k approximation."""
    return int_matmul_drum(x, w, fmt=fmt, k=k, bias=bias, acc_bits=acc_bits)


def int_conv2d_drum(
    x: np.ndarray,
    w: np.ndarray,
    *,
    fmt: IntFormat,
    k: int = 4,
    bias: np.ndarray | None = None,
    stride: int = 1,
    padding: int = 0,
    acc_bits: int = 32,
) -> np.ndarray:
    """2-D conv via im2col + :func:`int_matmul_drum` (DRUM-k approximation).

    Shapes: ``x`` (N, C_in, H, W), ``w`` (C_out, C_in, kH, kW), output
    (N, C_out, H_out, W_out). Zero-mean error, same im2col layout as
    :func:`int_conv2d_approx`.
    """
    if w.ndim != 4:
        raise ValueError(f"conv2d weights must be 4-D, got {w.shape}")
    c_out, c_in, kh, kw = w.shape
    if x.shape[1] != c_in:
        raise ValueError(f"input channels {x.shape[1]} != weight C_in {c_in}")

    cols, out_h, out_w = _im2col(x, kh, kw, stride, padding)
    w_mat = w.reshape(c_out, c_in * kh * kw).T   # (C_in*kh*kw, C_out)
    flat_out = int_matmul_drum(
        cols, w_mat, fmt=fmt, k=k, bias=bias, acc_bits=acc_bits,
    )
    return flat_out.reshape(x.shape[0], out_h, out_w, c_out).transpose(0, 3, 1, 2)
