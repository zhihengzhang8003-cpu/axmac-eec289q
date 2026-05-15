"""W5 tests for axmac.dnn_inference.

Contracts:
  1. K=0 matches the exact integer matmul/conv2d reference (numpy @).
  2. K>0 matches the scalar approx_mac_int loop on tiny tensors.
  3. Conv2d shape correctness for various stride/padding.
  4. FP scalar linear matches scalar mac_fp at K=0.
  5. Torch wrapper smoke test if torch is importable; otherwise skipped.
"""

from __future__ import annotations

import numpy as np
import pytest

from axmac.approx_mac import approx_mac_int
from axmac.dnn_inference import (
    fp_linear_approx_scalar,
    int_conv2d_approx,
    int_linear_approx,
    int_matmul_approx,
    make_approx_linear_torch,
    quantize_to_int,
    tiny_mlp_forward,
)
from axmac.exact_mac import FP16, FP32, INT4, INT8, INT16, encode_fp, mac_fp


# ============================================================
# K=0 equivalence to exact reference
# ============================================================

@pytest.mark.parametrize("fmt", [INT4, INT8, INT16])
def test_int_matmul_k0_matches_numpy(fmt):
    rng = np.random.default_rng(0xA0)
    x = rng.integers(fmt.min_val, fmt.max_val + 1, size=(8, 5))
    w = rng.integers(fmt.min_val, fmt.max_val + 1, size=(5, 7))
    got = int_matmul_approx(x, w, fmt=fmt, K=0)
    ref = x.astype(np.int64) @ w.astype(np.int64)
    # Both should fit in int32 for these small sizes.
    assert got.shape == ref.shape
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("fmt", [INT4, INT8])
def test_int_linear_k0_with_bias(fmt):
    rng = np.random.default_rng(0xA1)
    x = rng.integers(fmt.min_val, fmt.max_val + 1, size=(4, 6))
    w = rng.integers(fmt.min_val, fmt.max_val + 1, size=(6, 3))
    bias = rng.integers(-1000, 1000, size=(3,))
    got = int_linear_approx(x, w, fmt=fmt, K=0, bias=bias)
    ref = x.astype(np.int64) @ w.astype(np.int64) + bias.astype(np.int64)
    np.testing.assert_array_equal(got, ref)


# ============================================================
# K>0: vectorized matches scalar per-element loop
# ============================================================

@pytest.mark.parametrize("fmt", [INT8])
@pytest.mark.parametrize("K", [1, 2, 4])
def test_int_matmul_matches_scalar_loop(fmt, K):
    rng = np.random.default_rng(0xB0 + K)
    x = rng.integers(fmt.min_val, fmt.max_val + 1, size=(3, 5))
    w = rng.integers(fmt.min_val, fmt.max_val + 1, size=(5, 4))
    vec = int_matmul_approx(x, w, fmt=fmt, K=K)
    # Build the scalar reference: for each output entry, accumulate via
    # approx_mac_int over the reduction axis.
    ref = np.zeros((3, 4), dtype=np.int64)
    for i in range(3):
        for j in range(4):
            acc = 0
            for k in range(5):
                acc = approx_mac_int(
                    int(x[i, k]), int(w[k, j]), acc, fmt, K=K, aca_window=None
                )
            ref[i, j] = acc
    np.testing.assert_array_equal(vec, ref)


# ============================================================
# Conv2d shape + K=0 equivalence
# ============================================================

@pytest.mark.parametrize("stride,padding,expected_hw",
                         [(1, 0, (6, 6)), (1, 1, (8, 8)), (2, 1, (4, 4))])
def test_conv2d_shapes(stride, padding, expected_hw):
    rng = np.random.default_rng(0xC0)
    x = rng.integers(-8, 8, size=(2, 3, 8, 8))
    w = rng.integers(-4, 4, size=(5, 3, 3, 3))
    out = int_conv2d_approx(x, w, fmt=INT8, K=0, stride=stride, padding=padding)
    assert out.shape == (2, 5, *expected_hw)


def test_conv2d_k0_matches_naive():
    rng = np.random.default_rng(0xC1)
    x = rng.integers(-8, 8, size=(1, 2, 4, 4))
    w = rng.integers(-4, 4, size=(3, 2, 2, 2))
    got = int_conv2d_approx(x, w, fmt=INT8, K=0, stride=1, padding=0)
    # Naive nested-loop reference.
    ref = np.zeros((1, 3, 3, 3), dtype=np.int64)
    for n in range(1):
        for co in range(3):
            for oh in range(3):
                for ow in range(3):
                    s = 0
                    for ci in range(2):
                        for kh in range(2):
                            for kw in range(2):
                                s += int(x[n, ci, oh + kh, ow + kw]) * int(w[co, ci, kh, kw])
                    ref[n, co, oh, ow] = s
    np.testing.assert_array_equal(got, ref)


def test_conv2d_with_bias_and_padding():
    rng = np.random.default_rng(0xC2)
    x = rng.integers(-4, 4, size=(2, 2, 5, 5))
    w = rng.integers(-2, 2, size=(3, 2, 3, 3))
    bias = rng.integers(-50, 50, size=(3,))
    got = int_conv2d_approx(x, w, fmt=INT8, K=0, bias=bias, stride=1, padding=1)
    # Verify center pixel matches a manual computation.
    center = got[0, 0, 2, 2]
    manual = 0
    for ci in range(2):
        for kh in range(3):
            for kw in range(3):
                manual += int(x[0, ci, 1 + kh, 1 + kw]) * int(w[0, ci, kh, kw])
    manual += int(bias[0])
    assert center == manual


# ============================================================
# Validation
# ============================================================

def test_matmul_rejects_negative_K():
    x = np.zeros((2, 3), dtype=np.int64)
    w = np.zeros((3, 4), dtype=np.int64)
    with pytest.raises(ValueError):
        int_matmul_approx(x, w, fmt=INT8, K=-1)


def test_matmul_rejects_shape_mismatch():
    x = np.zeros((2, 3), dtype=np.int64)
    w = np.zeros((4, 5), dtype=np.int64)
    with pytest.raises(ValueError):
        int_matmul_approx(x, w, fmt=INT8, K=0)


def test_matmul_rejects_wrong_ndim():
    with pytest.raises(ValueError):
        int_matmul_approx(np.zeros(3), np.zeros((3, 4)), fmt=INT8)


def test_conv2d_rejects_wrong_channels():
    x = np.zeros((1, 3, 4, 4), dtype=np.int64)
    w = np.zeros((2, 4, 3, 3), dtype=np.int64)  # C_in mismatch
    with pytest.raises(ValueError):
        int_conv2d_approx(x, w, fmt=INT8)


# ============================================================
# Quantize helper
# ============================================================

def test_quantize_to_int_clips_and_rounds():
    fmt = INT8
    x = np.array([-200.0, -127.7, -0.3, 0.0, 0.6, 127.5, 200.0])
    got = quantize_to_int(x, fmt)
    expected = np.array([-128, -128, 0, 0, 1, 127, 127])
    np.testing.assert_array_equal(got, expected)


# ============================================================
# FP scalar path
# ============================================================

def test_fp_linear_k0_matches_scalar_mac():
    fmt = FP32
    rng = np.random.default_rng(0xD0)
    xs = rng.uniform(-1.0, 1.0, size=(2, 3)).astype(np.float32)
    ws = rng.uniform(-1.0, 1.0, size=(3, 4)).astype(np.float32)
    x_bits = np.vectorize(lambda v: encode_fp(float(v), fmt))(xs).astype(np.int64)
    w_bits = np.vectorize(lambda v: encode_fp(float(v), fmt))(ws).astype(np.int64)
    got = fp_linear_approx_scalar(x_bits, w_bits, fmt=fmt, K=0)
    # Reference: accumulate mac_fp without truncation.
    zero = encode_fp(0.0, fmt)
    for i in range(2):
        for j in range(4):
            acc = zero
            for k in range(3):
                acc = mac_fp(int(x_bits[i, k]), int(w_bits[k, j]), acc, fmt)
            assert got[i, j] == acc


# ============================================================
# Tiny MLP demo
# ============================================================

def test_tiny_mlp_forward_runs():
    fmt = INT8
    rng = np.random.default_rng(0xE0)
    x = rng.integers(0, 64, size=(4, 16))  # batch=4, 16 features
    w1 = rng.integers(-8, 8, size=(16, 8))
    b1 = rng.integers(-32, 32, size=(8,))
    w2 = rng.integers(-8, 8, size=(8, 4))
    b2 = rng.integers(-32, 32, size=(4,))
    out_exact = tiny_mlp_forward(x, [(w1, b1), (w2, b2)], fmt=fmt, K=0)
    out_approx = tiny_mlp_forward(x, [(w1, b1), (w2, b2)], fmt=fmt, K=2)
    assert out_exact.shape == (4, 4)
    assert out_approx.shape == (4, 4)
    # Approximation shouldn't blow up the magnitudes — values stay in a sane range.
    assert np.max(np.abs(out_exact - out_approx)) < 1e6


# ============================================================
# PyTorch backend (skip if torch unavailable)
# ============================================================

def test_torch_backend_smoke():
    torch = pytest.importorskip("torch")
    fn = make_approx_linear_torch(INT8, K=0)
    x = torch.randn(2, 5) * 10
    w = torch.randn(5, 3) * 10
    bias = torch.randn(3) * 10
    out = fn.apply(x, w, bias)
    assert out.shape == (2, 3)


def test_torch_factory_raises_without_torch(monkeypatch):
    """If torch import fails, factory must raise ImportError with a helpful message."""
    import axmac.dnn_inference as mod
    monkeypatch.setattr(mod, "_try_import_torch", lambda: None)
    with pytest.raises(ImportError, match="PyTorch"):
        make_approx_linear_torch(INT8, K=0)
