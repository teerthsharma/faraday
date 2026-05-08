# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.topological_solver — Topological FDFD and FDTD Surrogate Solvers.

Both solvers operate entirely in the latent-manifold space and consume
the trained :class:`GodTensor` operator :math:`T` rather than the FDFD
sparse Helmholtz matrix.

* :class:`TopologicalFDFD` — *frequency-domain* surrogate.  Given a new
  geometry it asks the trained operator for the dominant E and H latent
  vectors and the associated coupling score.

* :class:`TopologicalFDTD` — *time-domain* surrogate.  The latent
  evolution operator

  .. math::

     U(\\Delta t) = \\Re\\, \\exp(i\\, \\Delta t \\, T)

  is applied each step.  Because the dynamics live on the 16-D learned
  manifold rather than on the full 60×60 FDTD grid, the time step is no
  longer bound by the Courant–Friedrichs–Lewy stability criterion of the
  spatial discretisation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from faraday.exceptions import ConvergenceError
from faraday.god_tensor import GodTensor
from faraday.logging import get_logger
from faraday.predict import predict_eh_barcode

log = get_logger(__name__)


class TopologicalFDFD:
    """Frequency-domain surrogate: predict the cavity mode topology."""

    def __init__(self, god_tensor: GodTensor) -> None:
        if god_tensor.T_matrix is None:
            raise ConvergenceError(
                "GodTensor must be trained before using TopologicalFDFD"
            )
        self.gt = god_tensor

    def solve(
        self, geometry_params: tuple[float, ...], shape: str = "rect"
    ) -> dict[str, Any]:
        """Infer the topological signature of the fundamental mode."""
        log.info(
            "topological_fdfd_solve", params=geometry_params, shape=shape
        )
        prediction = predict_eh_barcode(self.gt, geometry_params, shape)
        return {
            "predicted_coupling_score": prediction["coupling_score"],
            "e_betti_0": prediction["knn_e_fingerprint"].get("betti_0", 0),
            "h_betti_0": prediction["knn_h_fingerprint"].get("betti_0", 0),
            "e_latent_vector": prediction["inferred_e_latent"],
            "h_latent_vector": prediction["inferred_h_latent"],
            "solver": "TopologicalFDFD",
        }


class TopologicalFDTD:
    """Time-domain surrogate: evolve the latent state via :math:`U(\\Delta t)`."""

    def __init__(self, god_tensor: GodTensor, dt: float) -> None:
        if god_tensor.T_matrix is None:
            raise ConvergenceError(
                "GodTensor must be fully trained before using TopologicalFDTD"
            )
        self.gt = god_tensor
        self.dt = dt
        from scipy.linalg import expm

        log.info("building_topological_time_operator", dt=dt)
        T = np.asarray(god_tensor.T_matrix)
        self.T_time = np.real(expm(1j * dt * T))

    def step(self, current_latent_state: np.ndarray) -> np.ndarray:
        """Apply :math:`U(\\Delta t)` once to a latent state."""
        x = self.T_time @ np.asarray(current_latent_state, dtype=np.float64)
        norm = float(np.linalg.norm(x))
        if norm > 1e-12:
            x = x / norm
        return x

    def simulate(
        self, initial_latent_state: np.ndarray, steps: int
    ) -> list[np.ndarray]:
        """Run the topological FDTD for ``steps`` steps; returns the trajectory."""
        log.info(
            "topological_fdtd_simulate",
            steps=steps,
            dt=self.dt,
            total_time=steps * self.dt,
        )
        history: list[np.ndarray] = [np.asarray(initial_latent_state, dtype=np.float64)]
        current = history[0]
        for _ in range(steps):
            current = self.step(current)
            history.append(current)
        return history
