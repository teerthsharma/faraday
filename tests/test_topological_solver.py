# Copyright (c) 2026 Teerth Sharma. All rights reserved.
"""Tests for faraday.topological_solver — TopologicalFDFD and TopologicalFDTD."""

from __future__ import annotations

import numpy as np
import pytest

from faraday.exceptions import ConvergenceError
from faraday.god_tensor import GodTensor
from faraday.topological_solver import TopologicalFDFD, TopologicalFDTD


@pytest.fixture(scope="module")
def trained_gt() -> GodTensor:
    """A small trained GodTensor for surrogate solver tests."""
    gt = GodTensor(n_geometries=8)
    gt.collect_training_data(nx=15, ny=15, num_modes=2, seed=42)
    gt.learn_T()
    gt.find_fixed_point(iters=50)
    return gt


class TestTopologicalFDFD:
    def test_raises_without_trained_T(self) -> None:
        gt = GodTensor(n_geometries=5)
        with pytest.raises(ConvergenceError):
            TopologicalFDFD(gt)

    def test_solve_rect(self, trained_gt: GodTensor) -> None:
        fdfd = TopologicalFDFD(trained_gt)
        result = fdfd.solve((2.0, 1.0), shape="rect")
        assert "predicted_coupling_score" in result
        assert "e_betti_0" in result
        assert "h_betti_0" in result
        assert "e_latent_vector" in result
        assert "h_latent_vector" in result
        assert result["solver"] == "TopologicalFDFD"

    def test_solve_circle(self, trained_gt: GodTensor) -> None:
        fdfd = TopologicalFDFD(trained_gt)
        result = fdfd.solve((1.0,), shape="circle")
        assert result["solver"] == "TopologicalFDFD"
        assert isinstance(result["predicted_coupling_score"], float)


class TestTopologicalFDTD:
    def test_raises_without_trained_T(self) -> None:
        gt = GodTensor(n_geometries=5)
        with pytest.raises(ConvergenceError):
            TopologicalFDTD(gt, dt=0.01)

    def test_step_preserves_unit_norm(self, trained_gt: GodTensor) -> None:
        fdtd = TopologicalFDTD(trained_gt, dt=0.01)
        latent_dim = trained_gt.T_matrix.shape[0]
        x0 = np.random.default_rng(42).normal(size=latent_dim)
        x0 = x0 / np.linalg.norm(x0)
        x1 = fdtd.step(x0)
        assert abs(np.linalg.norm(x1) - 1.0) < 1e-10

    def test_step_changes_state(self, trained_gt: GodTensor) -> None:
        fdtd = TopologicalFDTD(trained_gt, dt=0.1)
        latent_dim = trained_gt.T_matrix.shape[0]
        x0 = np.random.default_rng(42).normal(size=latent_dim)
        x0 = x0 / np.linalg.norm(x0)
        x1 = fdtd.step(x0)
        # The state should change (unless x0 is a fixed point, which is vanishingly unlikely)
        assert not np.allclose(x0, x1, atol=1e-8)

    def test_simulate_trajectory(self, trained_gt: GodTensor) -> None:
        fdtd = TopologicalFDTD(trained_gt, dt=0.01)
        latent_dim = trained_gt.T_matrix.shape[0]
        x0 = np.random.default_rng(42).normal(size=latent_dim)
        x0 = x0 / np.linalg.norm(x0)
        history = fdtd.simulate(x0, steps=10)
        # history has initial state + 10 steps = 11 entries
        assert len(history) == 11
        # All states should be unit-norm
        for state in history:
            assert abs(np.linalg.norm(state) - 1.0) < 1e-10

    def test_simulate_from_fdfd_initial_condition(self, trained_gt: GodTensor) -> None:
        """End-to-end: FDFD initial condition -> FDTD trajectory."""
        fdfd = TopologicalFDFD(trained_gt)
        res = fdfd.solve((2.0, 1.0))
        initial = np.asarray(res["e_latent_vector"], dtype=np.float64)

        fdtd = TopologicalFDTD(trained_gt, dt=0.01)
        history = fdtd.simulate(initial, steps=5)
        assert len(history) == 6
        for state in history:
            assert np.isfinite(state).all()
