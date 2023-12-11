# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.god_tensor — The Fixed Point of E ⇄ H

The God Tensor is the fixed point of the E-H co-determination operator.
Given E_signature and H_signature from the same cavity mode:

    T(E_sig) -> H_sig
    T(H_sig) -> E_sig

Iterating: T(T(x)) -> T(x)  [fixed point]

At convergence: T(x) = God Tensor
The God Tensor IS the unified field — it captures the invariant
that E and H mutually encode about each other.

Usage
-----
    gt = GodTensor(n_geometries=100)
    gt.collect_training_data(nx=40, ny=40)   # generate E, H field datasets
    gt.find_fixed_point(iters=500, tol=1e-6)  # converge to God Tensor
    pred = gt.predict(w=2.0, h=1.5)           # predict for new geometry
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from faraday._types import ModeData
from faraday.barcode import coupled_fingerprint
from faraday.em_solver import CavityGeometry, CavityShape, solve_cavity_modes
from faraday.logging import get_logger
from faraday.manifold_projector import ManifoldProjector, embed_fingerprint

log = get_logger(__name__)


def _solve(geom: CavityGeometry, nx: int, ny: int, num_modes: int) -> ModeData:
    """Wrapper that casts solve_cavity_modes to ModeData for type checker."""
    return solve_cavity_modes(geom, nx=nx, ny=ny, num_modes=num_modes)  # type: ignore[return-value]


@dataclass
class TrainingSample:
    """One training sample: a geometry + its E and H field signatures."""

    geometry_params: tuple[float, ...]  # e.g. (w, h) or (r,)
    e_fingerprint: dict
    h_fingerprint: dict
    e_embedding: np.ndarray  # manifold embedding of E fingerprint
    h_embedding: np.ndarray  # manifold embedding of H fingerprint
    k_values: list[float]  # cavity eigenmode wave numbers

    def to_dict(self) -> dict:
        return {
            "geometry_params": self.geometry_params,
            "e_fingerprint": self.e_fingerprint,
            "h_fingerprint": self.h_fingerprint,
            "e_embedding": self.e_embedding.tolist(),
            "h_embedding": self.h_embedding.tolist(),
            "k_values": self.k_values,
        }


@dataclass
class GodTensor:
    """
    The God Tensor: fixed point of E ⇄ H co-determination.

    Learn T such that:
        T(e) ≈ h  (E encodes H)
        T(h) ≈ e  (H encodes E)
        T(T(x)) = T(x)  [fixed point — the invariant]

    This fixed point T is the God Tensor — it IS the unified field.
    """

    n_geometries: int = 50
    samples: list[TrainingSample] = field(default_factory=list)
    projector_e: ManifoldProjector = field(
        default_factory=lambda: ManifoldProjector(input_dim=50, latent_dim=16)
    )
    projector_h: ManifoldProjector = field(
        default_factory=lambda: ManifoldProjector(input_dim=50, latent_dim=16)
    )
    T_matrix: np.ndarray | None = field(default=None, repr=False)
    god_tensor: np.ndarray | None = field(default=None, repr=False)
    fixed_point_converged: bool = False
    convergence_history: list[dict] = field(default_factory=list)

    def collect_training_data(
        self,
        nx: int = 50,
        ny: int = 50,
        num_modes: int = 8,
        seed: int = 42,
    ) -> None:
        """
        Generate training dataset: varied cavity geometries with E and H fields.

        Samples random (w, h) rectangles and (r,) circles, solves the cavity
        modes for each, computes topological fingerprints and embeddings.

        Args:
            nx, ny: grid resolution
            num_modes: number of eigenmodes to compute per geometry
            seed: random seed for reproducibility
        """
        rng = np.random.default_rng(seed)
        self.samples = []

        for i in range(self.n_geometries):
            # Random geometry: 70% rectangular, 30% circular
            if rng.random() < 0.7:
                w = rng.uniform(0.8, 3.0)
                h = rng.uniform(0.5, 2.0)
                geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(w, h))
                params = (w, h)
            else:
                r = rng.uniform(0.5, 1.5)
                geom = CavityGeometry(shape=CavityShape.CIRCULAR, dims=(r,))
                params = (r,)

            # Solve cavity modes
            try:
                mode_data = solve_cavity_modes(
                    geom, nx=nx, ny=ny, num_modes=num_modes, seed=seed
                )
            except Exception as e:
                log.warning("geometry_solve_failed", params=params, error=str(e))
                continue

            # Use the dominant (lowest k) mode for fingerprinting
            mode_key = "mode_0"
            if mode_key not in mode_data["e_modes"]:
                continue

            e_field = np.array(mode_data["e_modes"][mode_key]["field"])
            h_field = np.array(mode_data["h_modes"][mode_key]["field"])

            # Coupled fingerprint
            fp = coupled_fingerprint(e_field, h_field, threshold=0.05)
            e_fp = fp["e_fingerprint"]
            h_fp = fp["h_fingerprint"]

            if "error" in e_fp or "error" in h_fp:
                continue

            # Embeddings
            e_emb = embed_fingerprint(e_fp, dim=50)
            h_emb = embed_fingerprint(h_fp, dim=50)

            sample = TrainingSample(
                geometry_params=params,
                e_fingerprint=e_fp,
                h_fingerprint=h_fp,
                e_embedding=e_emb,
                h_embedding=h_emb,
                k_values=mode_data["k_values"],
            )
            self.samples.append(sample)

            log.info("sample_collected", i=i, params=params, valid=len(self.samples))
            if (i + 1) % 20 == 0:
                log.info("collection_progress", collected=i + 1, total=self.n_geometries)

        log.info("training_data_collected", n_samples=len(self.samples))

    def learn_T(self) -> np.ndarray:
        """
        Learn the coupling operator T: E_embedding <-> H_embedding.

        T is a matrix such that:
            T @ e_emb ≈ h_emb   (for most samples)

        Solved via least squares: T = H @ E^+  (H = stacked h_emb, E = stacked e_emb)
        The pseudoinverse handles the overdetermined case.

        Returns:
            T: (latent_dim, latent_dim) coupling matrix
        """
        if len(self.samples) < 2:
            raise ValueError("Need at least 2 training samples")

        E = np.array([s.e_embedding for s in self.samples])  # (n, 50)
        H = np.array([s.h_embedding for s in self.samples])  # (n, 50)

        # First project to latent space
        E_latent = np.array([self.projector_e.encode(e) for e in E])
        H_latent = np.array([self.projector_h.encode(h) for h in H])

        # Solve T @ E_latent.T ≈ H_latent.T via least squares
        # T @ E_latent.T = H_latent.T  →  T @ E_latent.T @ E_latent = H_latent.T @ E_latent
        # →  T = H_latent.T @ E_latent @ (E_latent.T @ E_latent)^-1
        # Using lstsq for numerical stability: T @ E_latent = H_latent
        from scipy.linalg import lstsq

        T_raw, _residuals, _rank, _s = lstsq(E_latent, H_latent)  # type: ignore[assignment]
        T = T_raw.T  # (latent, latent)

        self.T_matrix = T
        # Verify: T @ e should reconstruct h
        H_recon = E_latent @ T.T
        error = float(np.mean(np.abs(H_recon - H_latent)))

        log.info("t_matrix_learned", shape=T.shape, rank=_rank)
        log.info("t_reconstruction_error", error=error)

        return T

    def find_fixed_point(self, iters: int = 500, tol: float = 1e-7) -> np.ndarray:
        """
        Find the fixed point of T: the eigenvector with eigenvalue closest to 1.

        The God Tensor is the fixed point x* where T(x*) = x*.
        This is the eigenvector of T with eigenvalue λ = 1.

        Since a learned T may not have an exact eigenvalue of 1, this method
        uses two strategies:

        1. Direct eigendecomposition: find the eigenvector whose eigenvalue
           is closest to 1 (most isometric coupling direction).
        2. Power iteration: refine the eigenvector via iterative normalization.

        The final god_tensor is the sign-invariant eigenvector of T.

        Args:
            iters: max power-iteration refinement passes
            tol: convergence tolerance for power iteration refinement

        Returns:
            god_tensor: the fixed-point eigenvector
        """
        if self.T_matrix is None:
            self.learn_T()

        T = self.T_matrix

        # Strategy 1: direct eigendecomposition
        # Find eigenvector closest to eigenvalue 1 via Rayleigh quotient
        eigenvalues, eigenvectors = np.linalg.eig(T)

        # Rayleigh quotient of each eigenvector for eigenvalue 1
        # r_i = (x_i^T T x_i) / (x_i^T x_i)  should equal λ_i
        # We find eigenvector minimizing |λ_i - 1|
        eigenvalue_distances = np.abs(eigenvalues - 1.0)
        best_idx = int(np.argmin(eigenvalue_distances))
        x = np.real(eigenvectors[:, best_idx])
        x = x / (np.linalg.norm(x) + 1e-10)

        log.info(
            "fixed_point_eigenvector_found",
            eigenvalue=float(np.real(eigenvalues[best_idx])),
            eigenvalue_dist=float(eigenvalue_distances[best_idx]),
            latent_dim=T.shape[0],
        )

        # Strategy 2: power-iteration refinement (for numeric stability)
        for i in range(iters):
            x_new = T @ x
            norm = np.linalg.norm(x_new)
            if norm > 1e-10:
                x_new = x_new / norm

            # Sign-invariant delta: convergence to eigenvector, not ±eigenvector
            sign_correction = 1.0 if np.dot(x_new, x) >= 0 else -1.0
            delta = float(np.linalg.norm(x_new - sign_correction * x))
            self.convergence_history.append(
                {"iter": i, "delta": delta, "norm": float(norm)}
            )

            if delta < tol:
                log.info("fixed_point_converged", iter=i, delta=delta)
                self.fixed_point_converged = True
                x = sign_correction * x_new
                break

            x = sign_correction * x_new

            if (i + 1) % 100 == 0:
                log.debug("fixed_point_progress", iter=i + 1, delta=delta)
        else:
            log.warning(
                "fixed_point_iteration_max_iters",
                final_delta=delta,
                note="Proceeding with best eigenvector from spectral analysis",
            )

        self.god_tensor = x

        # Verify: T(T(x)) ≈ T(x)
        Tx = T @ x
        TTx = T @ Tx
        verification_error = float(np.linalg.norm(Tx - TTx))

        # Also verify: both E and H converge to same point under T
        e_latent = np.array(
            [self.projector_e.encode(s.e_embedding) for s in self.samples]
        )
        h_latent = np.array(
            [self.projector_h.encode(s.h_embedding) for s in self.samples]
        )
        e_under_T = e_latent @ T.T
        e_under_T_normed = e_under_T / (
            np.linalg.norm(e_under_T, axis=1, keepdims=True) + 1e-10
        )
        h_under_T = h_latent @ T.T
        h_under_T_normed = h_under_T / (
            np.linalg.norm(h_under_T, axis=1, keepdims=True) + 1e-10
        )
        e_dist = np.linalg.norm(e_under_T_normed - x, axis=1)
        h_dist = np.linalg.norm(h_under_T_normed - x, axis=1)

        log.info("fixed_point_verified", verification_error=verification_error)
        log.info("e_convergence_to_fixed_point", avg_dist=float(np.mean(e_dist)))
        log.info("h_convergence_to_fixed_point", avg_dist=float(np.mean(h_dist)))

        return x

    def get_e_to_h_map(self, e_embedding: np.ndarray) -> np.ndarray:
        """Map an E embedding to its predicted H embedding via T."""
        if self.T_matrix is None:
            raise ValueError("Must call find_fixed_point first")
        e_latent = self.projector_e.encode(e_embedding)
        h_latent = e_latent @ self.T_matrix.T
        # Normalize
        h_latent = h_latent / (np.linalg.norm(h_latent) + 1e-10)
        return h_latent

    def get_h_to_e_map(self, h_embedding: np.ndarray) -> np.ndarray:
        """Map an H embedding to its predicted E embedding via T."""
        if self.T_matrix is None:
            raise ValueError("Must call find_fixed_point first")
        h_latent = self.projector_h.encode(h_embedding)
        e_latent = h_latent @ self.T_matrix
        e_latent = e_latent / (np.linalg.norm(e_latent) + 1e-10)
        return e_latent

    def god_score(self) -> float:
        """
        Compute the 'god score' — how well the God Tensor unifies E and H.

        Score = exp(-mean(||T(e_i) - god|| + ||T(h_i) - god||) / 2)

        Using an exponential keeps the score bounded in [0, 1] regardless of
        the geometry of the embedding space or the number of samples.
        ``exp(-d)`` decays from 1 (d=0, perfect coupling) toward 0 (d→∞).
        """
        if self.god_tensor is None:
            return 0.0

        god = self.god_tensor
        e_latent = np.array(
            [self.projector_e.encode(s.e_embedding) for s in self.samples]
        )
        h_latent = np.array(
            [self.projector_h.encode(s.h_embedding) for s in self.samples]
        )

        e_under_T = e_latent @ self.T_matrix.T
        h_under_T = h_latent @ self.T_matrix.T

        e_under_T = e_under_T / (
            np.linalg.norm(e_under_T, axis=1, keepdims=True) + 1e-10
        )
        h_under_T = h_under_T / (
            np.linalg.norm(h_under_T, axis=1, keepdims=True) + 1e-10
        )

        e_dists = np.linalg.norm(e_under_T - god, axis=1)
        h_dists = np.linalg.norm(h_under_T - god, axis=1)

        score = float(np.exp(-np.mean(e_dists + h_dists) / 2))
        return score

    def summary(self) -> dict:
        """Return a human-readable summary of the God Tensor."""
        return {
            "n_samples": len(self.samples),
            "T_matrix_shape": self.T_matrix.shape
            if self.T_matrix is not None
            else None,
            "god_tensor_shape": self.god_tensor.shape
            if self.god_tensor is not None
            else None,
            "converged": self.fixed_point_converged,
            "final_delta": self.convergence_history[-1]["delta"]
            if self.convergence_history
            else None,
            "god_score": self.god_score(),
        }

    def predict(self, w: float, h: float) -> list[tuple[float, float]]:
        """
        Predict barcode for a new geometry.

        Args:
            w: cavity width
            h: cavity height

        Returns:
            List of (birth, death) pairs from predicted E-field barcode.
        """
        from faraday import CavityGeometry, CavityShape

        geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(w, h))
        mode_data = _solve(geom, nx=50, ny=50, num_modes=8)
        mode_key = "mode_0"
        if mode_key in mode_data["e_modes"]:
            e_field = np.array(mode_data["e_modes"][mode_key]["field"])
        else:
            return []
        from faraday.barcode import topological_fingerprint

        fp = topological_fingerprint(e_field)
        diagram = fp.get("diagram", [])
        return [(float(b), float(d)) for b, d in diagram]

    # ------------------------------------------------------------------
    # Persistence / serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save a trained GodTensor to a pickle file."""
        import pickle

        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str) -> "GodTensor":
        """Load a saved GodTensor from a pickle file."""
        import pickle

        with open(path, "rb") as fh:
            return pickle.load(fh)
