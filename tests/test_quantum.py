# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
Tests for the quantum sector (faraday.quantum) and the duality-based God
Tensor: each pins an integer or Z₂ invariant that a wrong implementation of
the physics would violate.
"""

import numpy as np
import pytest

from faraday.duality import mode_topology
from faraday.em_solver import solve_interior_modes
from faraday.god_tensor import (
    GodTensor,
    TrainingSample,
    duality_views,
    fixed_point_basis,
    fixed_point_operator,
)
from faraday.quantum import berry_sign, phase_charges, wall_windings


def _grid(R, n):
    c = np.linspace(-R, R, n)
    return np.meshgrid(c, c)


def _modes(f, R=1.05, n=121, num_modes=10):
    X, Y = _grid(R, n)
    mask = f(X, Y)
    return solve_interior_modes(X, Y, mask, num_modes=num_modes, seed=0), mask


def _field(modes, j):
    return np.array(modes["e_modes"][f"mode_{j}"]["field"])


DISK = lambda X, Y: np.hypot(X, Y) < 1.0  # noqa: E731
ANNULUS = lambda X, Y: (np.hypot(X, Y) < 1.0) & (np.hypot(X, Y) > 0.35)  # noqa: E731
TWO_HOLES = lambda X, Y: (  # noqa: E731
    (np.hypot(X, Y) < 1.1)
    & (np.hypot(X - 0.45, Y) > 0.22)
    & (np.hypot(X + 0.4, Y + 0.3) > 0.18)
)


# ---------------------------------------------------------------------------
# Quantum sector
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def disk():
    return _modes(DISK)


def _degenerate_pairs(modes, rel=2e-3):
    k = modes["k_values"]
    return [(j, j + 1) for j in range(len(k) - 1) if abs(k[j + 1] - k[j]) < rel * k[j]]


def test_angular_momentum_states_saturate_the_bound(disk):
    """Disk: ψ = u + iv of a degenerate pair has |V| = ℓ = N_wall / 2."""
    modes, mask = disk
    pairs = _degenerate_pairs(modes)
    assert len(pairs) >= 3
    for a, b in pairs:
        u, v = _field(modes, a), _field(modes, b)
        V = int(phase_charges(u + 1j * v, mask).sum())
        n_wall = min(mode_topology(modes, a).n_wall, mode_topology(modes, b).n_wall)
        assert abs(V) == n_wall // 2 >= 1


def test_phase_charge_is_gauge_invariant(disk):
    modes, mask = disk
    a, b = _degenerate_pairs(modes)[0]
    psi = _field(modes, a) + 1j * _field(modes, b)
    V = phase_charges(psi, mask).sum()
    for alpha in (0.3, 1.7, 2.9):
        assert phase_charges(np.exp(1j * alpha) * psi, mask).sum() == V


def test_phase_charge_bounded_by_classical_wall_count():
    """Ellipse (no degeneracy): any complex superposition obeys the bound."""
    modes, mask = _modes(lambda X, Y: (X / 1.2) ** 2 + (Y / 0.8) ** 2 < 1, R=1.25)
    for j in range(9):
        V = int(phase_charges(_field(modes, j) + 1j * _field(modes, j + 1), mask).sum())
        bound = min(mode_topology(modes, j).n_wall, mode_topology(modes, j + 1).n_wall) // 2
        assert abs(V) <= bound


@pytest.mark.parametrize("shape", [DISK, ANNULUS, TWO_HOLES])
def test_bulk_charge_is_outer_minus_hole_windings(shape):
    modes, mask = _modes(shape, R=1.15, n=151)
    for j in range(8):
        psi = _field(modes, j) + 1j * _field(modes, j + 1)
        w = wall_windings(psi, mask)
        assert phase_charges(psi, mask).sum() == w[0] - sum(w[1:])


def test_annulus_charge_lives_on_the_hole():
    """Concentric annulus: angular-momentum pairs carry no bulk vortex; the
    winding ℓ sits on both walls (quantised circulation around the hole)."""
    modes, mask = _modes(ANNULUS, n=151)
    pairs = _degenerate_pairs(modes)
    assert pairs
    for a, b in pairs:
        psi = _field(modes, a) + 1j * _field(modes, b)
        outer, hole = wall_windings(psi, mask)
        assert phase_charges(psi, mask).sum() == 0
        assert outer == hole != 0


def test_berry_phase_flips_only_around_the_diabolical_point():
    """Rectangle family (w, tilt δ): TM31/TM12 cross only at (√(8/3), 0)."""
    X, Y = _grid(1.0, 81)

    def mode2(w, d):
        mask = (abs(Y) < 0.5) & (abs(X) < w / 2 + d * Y)
        return _field(solve_interior_modes(X, Y, mask, num_modes=3, seed=0), 2)

    def loop(cw, cd):
        ts = np.linspace(0, 2 * np.pi, 41)
        return berry_sign(mode2(cw + 0.12 * np.cos(t), cd + 0.15 * np.sin(t)) for t in ts)

    w_star = np.sqrt(8 / 3)
    sign_in, ov_in = loop(w_star, 0.0)
    sign_out, ov_out = loop(w_star + 0.3, 0.0)
    assert min(ov_in, ov_out) > 0.9  # transport is resolved
    assert sign_in == -1
    assert sign_out == +1


def test_berry_sign_rejects_empty_loop():
    with pytest.raises(ValueError):
        berry_sign([])


# ---------------------------------------------------------------------------
# God Tensor on duality signatures
# ---------------------------------------------------------------------------


def _samples(n_modes=8):
    shapes = [
        (lambda X, Y: (abs(X) < 1.0) & (abs(Y) < 0.6), 1.05),
        (lambda X, Y: (abs(X) < 0.7) & (abs(Y) < 0.6), 1.05),
        (DISK, 1.05),
        (lambda X, Y: (X / 1.2) ** 2 + (Y / 0.8) ** 2 < 1, 1.25),
        (ANNULUS, 1.05),
        (TWO_HOLES, 1.15),
    ]
    out = []
    for f, R in shapes:
        modes, _ = _modes(f, R=R, n=151, num_modes=n_modes)
        e, h = duality_views(modes, n_modes)
        out.append(TrainingSample((), {}, {}, e, h, modes["k_values"]))
    return out


@pytest.fixture(scope="module")
def duality_samples():
    return _samples()


def test_views_agree_on_resolved_grids(duality_samples):
    for s in duality_samples:
        np.testing.assert_array_equal(s.e_embedding, s.h_embedding)


def test_learned_T_is_an_orthogonal_projector(duality_samples):
    gt = GodTensor(signature_modes=8, samples=list(duality_samples))
    T = gt.learn_T()
    E = np.array([s.e_embedding for s in duality_samples])
    np.testing.assert_allclose(T @ T, T, atol=1e-10)
    np.testing.assert_allclose(T, T.T, atol=1e-10)
    assert gt.idempotency_defect < 1e-10
    assert np.linalg.matrix_rank(T) == np.linalg.matrix_rank(E)


def test_god_tensor_is_the_realisable_subspace(duality_samples):
    gt = GodTensor(signature_modes=8, samples=list(duality_samples))
    gt.learn_T()
    x = gt.find_fixed_point()
    E = np.array([s.e_embedding for s in duality_samples])
    assert gt.fixed_point_converged
    assert gt.contraction_factor < 1e-4
    assert gt.god_basis.shape[1] == np.linalg.matrix_rank(E)
    np.testing.assert_allclose(gt.T_matrix @ x, x, atol=1e-9)
    assert gt.god_score() == pytest.approx(1.0)
    # every training signature is a fixed point of T
    for e in E:
        assert gt.distance_to_god(e / np.linalg.norm(e)) == pytest.approx(0.0, abs=1e-7)
    summary = gt.summary()
    assert summary["fixed_point_multiplicity"] == gt.god_basis.shape[1]
    assert summary["idempotency_defect"] < 1e-10


def test_law_violation_breaks_idempotency(duality_samples):
    """Negative control: one corrupted H-view entry and T is no projector."""
    samples = [
        TrainingSample(s.geometry_params, {}, {}, s.e_embedding.copy(), s.h_embedding.copy(), [])
        for s in duality_samples
    ]
    samples[2].h_embedding[3] += 1
    gt = GodTensor(signature_modes=8, samples=samples)
    gt.learn_T()
    gt.find_fixed_point()
    assert gt.idempotency_defect > 1e-3
    assert gt.god_score() < 1.0


def test_distance_to_god_geometry():
    gt = GodTensor()
    gt.god_basis = np.eye(4)[:, :2]
    assert gt.distance_to_god(np.array([0.6, -0.8, 0, 0])) == pytest.approx(0.0)
    assert gt.distance_to_god(np.array([0, 0, 1.0, 0])) == pytest.approx(1.0)
    rows = np.array([[1.0, 0, 0, 0], [0, 0, 0, -1.0]])
    np.testing.assert_allclose(gt.distance_to_god(rows), [0.0, 1.0])


def test_distance_to_god_requires_fixed_point():
    with pytest.raises(ValueError):
        GodTensor().distance_to_god(np.ones(3))


def test_degenerate_eigenvalue_one_is_a_subspace_not_a_tie():
    T = np.diag([1.0, 1.0, 0.0, 0.3])
    _, lam, q = fixed_point_operator(T)
    assert lam.real == pytest.approx(1.0)
    assert q < 1e-4
    assert fixed_point_basis(T).shape[1] == 2


def test_god_tensor_pipeline_on_real_training_data():
    gt = GodTensor(n_geometries=6, signature_modes=8)
    gt.collect_training_data(nx=30, ny=30, num_modes=8, seed=42)
    gt.find_fixed_point()
    assert gt.idempotency_defect < 1e-8
    assert gt.god_score() == pytest.approx(1.0)


def test_fixed_basis_of_a_high_multiplicity_projector():
    """Regression: eig() eigenvectors are ill-conditioned inside a degenerate
    eigenspace; the basis must still span all of it."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((16, 13))
    P = A @ np.linalg.pinv(A)  # rank-13 orthogonal projector
    B = fixed_point_basis(P)
    assert B.shape == (16, 13)
    np.testing.assert_allclose(B.T @ B, np.eye(13), atol=1e-10)
    np.testing.assert_allclose(P @ B, B, atol=1e-10)
