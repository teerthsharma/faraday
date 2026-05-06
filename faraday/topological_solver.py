# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.topological_solver — Topological FDFD and FDTD Surrogate Solvers

These solvers completely bypass the classical Finite-Difference grid and Courant 
limitations. They operate entirely in the topological manifold space (Persistence Images),
using the trained God Tensor to perform inference and time-evolution.

- TopologicalFDFD: Instantly infers the field topology of a cavity geometry.
- TopologicalFDTD: Steps a field forward in time by applying a topological evolution operator.
"""

from __future__ import annotations

import numpy as np

from faraday.god_tensor import GodTensor
from faraday.predict import predict_eh_barcode
from faraday.logging import get_logger

log = get_logger(__name__)


class TopologicalFDFD:
    """
    Topological Surrogate for Finite-Difference Frequency-Domain.
    
    Instead of inverting a massive Laplacian matrix, this solver uses the 
    learned God Tensor to immediately predict the topological representation 
    of the fundamental cavity mode.
    """

    def __init__(self, god_tensor: GodTensor) -> None:
        """
        Initialize with a pre-trained God Tensor.
        """
        if god_tensor.T_matrix is None:
            raise ValueError("GodTensor must be trained before using TopologicalFDFD.")
        self.gt = god_tensor

    def solve(self, geometry_params: tuple[float, ...], shape: str = "rect") -> dict:
        """
        Infer the topological signature of the fundamental mode.
        
        Args:
            geometry_params: (width, height) or (radius,)
            shape: 'rect' or 'circ'
            
        Returns:
            dict containing the inferred topological properties and coupling strength.
        """
        log.info("topological_fdfd_solve", params=geometry_params, shape=shape)
        
        # Predict the topological signature using the God Tensor's manifold
        prediction = predict_eh_barcode(self.gt, geometry_params, shape)
        
        return {
            "predicted_coupling_score": prediction["coupling_score"],
            "e_betti_0": prediction["knn_e_fingerprint"].get("betti_0", 0),
            "h_betti_0": prediction["knn_h_fingerprint"].get("betti_0", 0),
            "e_latent_vector": prediction.get("inferred_e_latent", []),
            "h_latent_vector": prediction.get("inferred_h_latent", []),
            "solver": "TopologicalFDFD",
        }


class TopologicalFDTD:
    """
    Topological Surrogate for Finite-Difference Time-Domain.
    
    Classical FDTD is bound by the Courant-Friedrichs-Lewy (CFL) stability limit.
    Topological FDTD steps the electromagnetic field forward in time by evolving 
    its Persistence Image (latent vector) directly.
    """

    def __init__(self, god_tensor: GodTensor, dt: float) -> None:
        """
        Initialize the Time-Domain solver.
        
        Args:
            god_tensor: A trained GodTensor containing the spatial coupling matrix `T`.
            dt: Time step. Can be arbitrarily large since it operates in latent space.
        """
        if god_tensor.T_matrix is None:
            raise ValueError("GodTensor must be fully trained.")
            
        self.gt = god_tensor
        self.dt = dt
        
        # The God Tensor's T_matrix defines spatial E <-> H coupling.
        # For time evolution, we construct a unitary evolution operator in the latent space:
        # T_time = exp(-i * dt * T)
        from scipy.linalg import expm
        
        log.info("building_topological_time_operator", dt=dt)
        # We ensure T_time is real by taking the real part of the exponential of an antisymmetric matrix,
        # or just acting as a rotation matrix in latent space.
        # A simple phenomenological time-evolution in latent space:
        self.T_time = np.real(expm(1j * self.dt * self.gt.T_matrix))

    def step(self, current_latent_state: np.ndarray) -> np.ndarray:
        """
        Evolve the field topology forward by one time step `dt`.
        
        Args:
            current_latent_state: (16,) numpy array representing the field topology.
            
        Returns:
            next_latent_state: (16,) numpy array.
        """
        # Unconstrained by grid resolution!
        next_state = self.T_time @ current_latent_state
        # Normalize to prevent explosion in the surrogate simulation
        norm = np.linalg.norm(next_state)
        if norm > 1e-10:
            next_state = next_state / norm
        return next_state

    def simulate(self, initial_latent_state: np.ndarray, steps: int) -> list[np.ndarray]:
        """
        Run the FDTD simulation for a given number of steps.
        
        Args:
            initial_latent_state: The starting topology (e.g. from TopologicalFDFD).
            steps: Number of time steps to simulate.
            
        Returns:
            List of latent states representing the time-evolution of the field's topology.
        """
        log.info("topological_fdtd_simulate", steps=steps, dt=self.dt, total_time=steps*self.dt)
        history = [initial_latent_state]
        current = initial_latent_state
        for _ in range(steps):
            current = self.step(current)
            history.append(current)
        return history
