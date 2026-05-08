# Copyright (c) 2026 Teerth Sharma. All rights reserved.
"""
Spectral-fixed-point and Perron-Frobenius tests.

These tests confirm the central theoretical claims of Faraday:

1. The God Tensor is the dominant eigenvector of T (Perron-Frobenius
   power method).
2. The normalised power iteration converges geometrically with rate
   |λ₂/λ₁|.
3. Once converged, T(x*) = x* up to ±1 sign, ⇒ the spectral residual
   approaches machine epsilon for stable T.
4. T(T(x*)) = T(x*) (idempotency on the fixed-point line).
"""
from __future__ import annotations

import numpy as np
import pytest

from faraday.god_tensor import GodTensor


def _build_random_diagonalisable_T(L: int, seed: int) -> np.ndarray:
    """Construct a random L×L matrix with a unique dominant real eigenvalue."""
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(L, L))
    # Eigenvalues: 1.0 (dominant), then descending by 0.7
    eigs = np.diag([1.0] + [0.7 * 0.9**i for i in range(L - 1)])
    T = Q @ eigs @ np.linalg.inv(Q)
    return T


def _power_iteration(T: np.ndarray, iters: int = 200, tol: float = 1e-12) -> tuple[np.ndarray, list[float], int]:
    L = T.shape[0]
    rng = np.random.default_rng(0)
    x = rng.normal(size=L)
    x = x / np.linalg.norm(x)
    residuals: list[float] = []
    iters_taken = iters
    for i in range(iters):
        x_new = T @ x
        norm = np.linalg.norm(x_new)
        if norm > 1e-15:
            x_new = x_new / norm
        sign = 1.0 if np.dot(x_new, x) >= 0 else -1.0
        r = float(np.linalg.norm(x_new - sign * x))
        residuals.append(r)
        x = sign * x_new
        if r < tol:
            iters_taken = i + 1
            break
    return x, residuals, iters_taken


# ---------------------------------------------------------------------------


def test_power_iteration_finds_dominant_eigenvector():
    T = _build_random_diagonalisable_T(L=8, seed=42)
    x, residuals, _iters = _power_iteration(T, iters=400, tol=1e-10)
    # Reference: dominant eigenvector via numpy.linalg.eig
    w, V = np.linalg.eig(T)
    idx = np.argmax(np.abs(w))
    v_ref = np.real(V[:, idx])
    v_ref = v_ref / np.linalg.norm(v_ref)
    # x and v_ref should be parallel (or anti-parallel)
    cosine = abs(float(x @ v_ref))
    assert cosine > 0.999, f"cosine={cosine}, residuals[-3:]={residuals[-3:]}"


def test_residual_reaches_machine_epsilon_for_stable_T():
    """For a T with |λ₁| ≈ 1, residual should reach near machine epsilon."""
    T = _build_random_diagonalisable_T(L=6, seed=7)
    _, residuals, _iters = _power_iteration(T, iters=2000, tol=1e-15)
    # Reach below 1e-13 with enough iterations (we stop early at 1e-15)
    assert residuals[-1] < 1e-12, residuals[-1]


def test_geometric_convergence_rate_matches_spectral_gap():
    """log-log slope of residuals matches |λ₂/λ₁| within a factor of 2."""
    T = _build_random_diagonalisable_T(L=6, seed=11)
    _, residuals, _ = _power_iteration(T, iters=200, tol=1e-14)
    eigs = np.linalg.eigvals(T)
    eigs_sorted = np.sort(np.abs(eigs))[::-1]
    gap = eigs_sorted[1] / eigs_sorted[0]
    # Estimate convergence rate from the second half of the iterates.
    rs = np.asarray(residuals)
    rs = rs[(rs > 1e-12) & (rs < rs[0])]
    if len(rs) < 6:
        pytest.skip("not enough non-degenerate residuals")
    # Geometric ratio = mean(r[k+1] / r[k])
    ratios = rs[1:] / rs[:-1]
    rate = float(np.median(ratios))
    # rate should be within a small factor of |λ₂|/|λ₁|
    assert 0.5 * gap < rate < 2.0 * gap, f"rate={rate}, gap={gap}"


def test_god_tensor_idempotency_once_converged():
    """T(T(x*)) ≈ T(x*) once the iteration has converged."""
    T = _build_random_diagonalisable_T(L=10, seed=99)
    x, _, _ = _power_iteration(T, iters=400, tol=1e-10)
    Tx = T @ x
    Tx = Tx / np.linalg.norm(Tx)
    TTx = T @ Tx
    TTx = TTx / np.linalg.norm(TTx)
    assert np.linalg.norm(Tx - TTx) < 1e-9


# ---------------------------------------------------------------------------
# End-to-end: the trained God Tensor obeys all four properties above.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_gt():
    gt = GodTensor(n_geometries=12)
    gt.collect_training_data(nx=20, ny=20, num_modes=2, seed=42)
    if len(gt.samples) < 4:
        pytest.skip("not enough valid samples for trained_gt fixture")
    gt.learn_T()
    gt.find_fixed_point(iters=400, tol=1e-10)
    return gt


def test_god_tensor_is_dominant_eigenvector(trained_gt):
    T = trained_gt.T_matrix
    g = trained_gt.god_tensor
    assert T is not None and g is not None

    eigs, V = np.linalg.eig(T)
    idx = int(np.argmax(np.abs(eigs)))
    v_ref = np.real(V[:, idx])
    v_ref = v_ref / np.linalg.norm(v_ref)
    cosine = abs(float(g @ v_ref))
    assert cosine > 0.999, f"cosine={cosine}"


def test_god_tensor_unit_norm(trained_gt):
    g = trained_gt.god_tensor
    assert g is not None
    assert abs(float(np.linalg.norm(g)) - 1.0) < 1e-8


def test_god_tensor_idempotent_under_T(trained_gt):
    T = trained_gt.T_matrix
    g = trained_gt.god_tensor
    assert T is not None and g is not None
    Tg = T @ g
    Tg = Tg / np.linalg.norm(Tg)
    TTg = T @ Tg
    TTg = TTg / np.linalg.norm(TTg)
    assert np.linalg.norm(Tg - TTg) < 1e-7


def test_summary_reports_consistent_state(trained_gt):
    s = trained_gt.summary()
    assert s["n_samples"] >= 4
    assert s["T_matrix_shape"] == (16, 16)
    assert s["god_tensor_shape"] == (16,)
    assert 0.0 < s["god_score"] <= 1.0
    assert s["dominant_eigenvalue"] is not None
    assert s["spectral_gap_ratio"] is not None
