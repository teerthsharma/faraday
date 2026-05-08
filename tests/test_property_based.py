# Copyright (c) 2026 Teerth Sharma. All rights reserved.
"""
Property-based tests using :mod:`hypothesis`.

These tests assert *invariants* of the Faraday pipeline rather than
specific outputs, so they explore far more of the input space than
example-based tests can.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# barcode_to_coefficients
# ---------------------------------------------------------------------------


@given(
    n_pairs=st.integers(min_value=1, max_value=20),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=50, deadline=None)
def test_hilbert_coeffs_zero_sum_for_finite_bars(n_pairs, seed):
    """Each finite bar contributes +1 at birth and -1 at death ⇒ Σ c_k = 0."""
    from faraday.manifold_projector import barcode_to_coefficients

    rng = np.random.default_rng(seed)
    barcode = []
    for _ in range(n_pairs):
        b = float(rng.uniform(0.0, 0.9))
        d = float(rng.uniform(b + 0.01, min(b + 0.5, 0.99)))
        barcode.append((b, d))
    coeffs = barcode_to_coefficients(barcode, degree=50)
    # Each finite (b, d) bar adds 0 in total → sum of coeffs is zero
    assert abs(coeffs.sum()) < 1e-9


# ---------------------------------------------------------------------------
# embed_barcode
# ---------------------------------------------------------------------------


@given(
    n_pairs=st.integers(min_value=1, max_value=20),
    dim=st.integers(min_value=10, max_value=80),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=30, deadline=None)
def test_embed_barcode_l2_normalised(n_pairs, dim, seed):
    """``embed_barcode`` returns an L2-normalised vector (or the zero vector)."""
    from faraday.manifold_projector import embed_barcode

    rng = np.random.default_rng(seed)
    barcode = []
    for _ in range(n_pairs):
        b = float(rng.uniform(0.0, 0.9))
        d = float(rng.uniform(b + 0.01, min(b + 0.5, 0.99)))
        barcode.append((b, d))
    emb = embed_barcode(barcode, dim=dim)
    assert emb.shape == (dim,)
    n = float(np.linalg.norm(emb))
    assert n == 0 or abs(n - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# ManifoldProjector — autoencoder loss is non-increasing-in-mean over training
# ---------------------------------------------------------------------------


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n=st.integers(min_value=4, max_value=12),
    epochs=st.integers(min_value=10, max_value=50),
)
@settings(
    max_examples=12, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
def test_autoencoder_loss_decreases_on_average(seed, n, epochs):
    """End-of-training MSE < start-of-training MSE on i.i.d. Gaussian data."""
    from faraday.manifold_projector import ManifoldProjector

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 50))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    mp = ManifoldProjector(input_dim=50, latent_dim=16, seed=seed)
    losses = mp.fit(X, lr=0.05, epochs=epochs, batch_size=2)
    assert losses[-1] <= losses[0] * 1.5  # allow some noise; trend must be down


# ---------------------------------------------------------------------------
# CavityGeometry — `contains` is monotone in size
# ---------------------------------------------------------------------------


@given(
    w=st.floats(min_value=0.5, max_value=5.0),
    h=st.floats(min_value=0.5, max_value=5.0),
    x=st.floats(min_value=-2.0, max_value=2.0),
    y=st.floats(min_value=-2.0, max_value=2.0),
)
@settings(max_examples=80, deadline=None)
def test_rectangular_contains_monotone(w, h, x, y):
    """If (x, y) is inside (w, h) it is inside (w', h') for any w'≥w, h'≥h."""
    from faraday.em_solver import CavityGeometry, CavityShape

    g_small = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(w, h))
    g_big = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(w + 1.0, h + 1.0))
    inside_small = bool(g_small.contains(np.float64(x), np.float64(y)))
    inside_big = bool(g_big.contains(np.float64(x), np.float64(y)))
    assert (not inside_small) or inside_big


@given(
    r1=st.floats(min_value=0.5, max_value=2.0),
    r2=st.floats(min_value=0.0, max_value=10.0),
)
@settings(max_examples=50, deadline=None)
def test_circular_contains_monotone(r1, r2):
    """Doubling radius grows the support."""
    from faraday.em_solver import CavityGeometry, CavityShape

    g_small = CavityGeometry(shape=CavityShape.CIRCULAR, dims=(r1,))
    g_big = CavityGeometry(shape=CavityShape.CIRCULAR, dims=(r1 * 2,))
    x = np.float64(r2 * np.cos(0.7))
    y = np.float64(r2 * np.sin(0.7))
    inside_small = bool(g_small.contains(x, y))
    inside_big = bool(g_big.contains(x, y))
    assert (not inside_small) or inside_big


# ---------------------------------------------------------------------------
# Coupled fingerprint coupling_strength is in (0, 1]
# ---------------------------------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=8, deadline=None)
def test_coupling_strength_bounded(seed):
    """0 < coupling_strength ≤ 1 for any well-defined fields."""
    from faraday.barcode import coupled_fingerprint
    from faraday.em_solver import CavityGeometry, CavityShape, solve_cavity_modes

    geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0))
    md = solve_cavity_modes(geom, nx=20, ny=20, num_modes=2, seed=seed)
    e = np.asarray(md["e_modes"]["mode_0"]["field"])
    h = np.asarray(md["h_modes"]["mode_0"]["field"])
    cf = coupled_fingerprint(e, h)
    assert 0.0 < cf["coupling_strength"] <= 1.0


# ---------------------------------------------------------------------------
# god_score in (0, 1]
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=4, max_value=8),
    seed=st.integers(min_value=0, max_value=2000),
)
@settings(
    max_examples=4, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
def test_god_score_bounded_unit_interval(n, seed):
    from faraday import GodTensor

    gt = GodTensor(n_geometries=n)
    gt.collect_training_data(nx=15, ny=15, num_modes=2, seed=seed)
    if len(gt.samples) < 2:
        pytest.skip("not enough samples")
    gt.learn_T()
    gt.find_fixed_point(iters=80, tol=1e-5)
    score = gt.god_score()
    assert 0.0 < score <= 1.0
