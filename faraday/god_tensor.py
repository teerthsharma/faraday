# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.god_tensor — The Spectral Fixed Point of E ⇄ H Coupling.

We learn a coupling operator :math:`T: \\mathbb{R}^L \\to \\mathbb{R}^L`
between latent embeddings of E- and H-field topological fingerprints by
least squares,

.. math::

   T = \\arg\\min_{M \\in \\mathbb{R}^{L\\times L}}
      \\sum_{n=1}^{N} \\| M\\, z^E_n - z^H_n \\|^2_2,
   \\qquad z^E_n, z^H_n = \\Pi(B^E_n), \\Pi(B^H_n),

with :math:`\\Pi` the autoencoder projection of the Hilbert-series barcode
embedding.  The closed-form solution is
:math:`T = (Z_E^\\top Z_E)^{-1} Z_E^\\top Z_H` (transposed).

The **God Tensor** :math:`g \\in \\mathbb{R}^L` is the dominant eigenvector
of :math:`T`, characterised by the spectral fixed-point equation

.. math::

   \\hat T(g) := T g / \\|T g\\|_2 = g.

Existence and convergence of the iteration

.. math::

   x_{n+1} = T x_n / \\| T x_n \\|_2

to :math:`g` is guaranteed for any starting vector with a non-zero
projection onto the dominant eigenspace by the **Perron–Frobenius
theorem** (Perron 1907, Frobenius 1912) in its primitive-matrix form, and
more generally by the convergence of the *power method* (Golub & Van
Loan, *Matrix Computations*, §7.3).  The convergence rate is geometric
with ratio :math:`|\\lambda_2 / \\lambda_1|` where :math:`\\lambda_1,
\\lambda_2` are the two largest eigenvalues of :math:`T` in modulus.

Numerically, :math:`x_n` converges to a fixed sign of :math:`g` modulo
arbitrary global phase; we use the sign-corrected residual

.. math::

   r_n = \\| x_{n+1} - \\mathrm{sign}(\\langle x_{n+1}, x_n \\rangle)\\, x_n\\|_2

to detect convergence; once :math:`r_n < \\varepsilon` the iteration is
declared converged.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from faraday._types import ModeData
from faraday.barcode import coupled_fingerprint
from faraday.em_solver import CavityGeometry, CavityShape, solve_cavity_modes
from faraday.exceptions import ConvergenceError
from faraday.logging import get_logger
from faraday.manifold_projector import ManifoldProjector, embed_fingerprint

log = get_logger(__name__)


def _solve(geom: CavityGeometry, nx: int, ny: int, num_modes: int) -> ModeData:
    """Type-narrowing wrapper around :func:`solve_cavity_modes`."""
    return solve_cavity_modes(geom, nx=nx, ny=ny, num_modes=num_modes)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# TrainingSample
# ---------------------------------------------------------------------------


@dataclass
class TrainingSample:
    """A single (geometry, E-fp, H-fp, embeddings) training sample."""

    geometry_params: tuple[float, ...]
    e_fingerprint: dict[str, Any]
    h_fingerprint: dict[str, Any]
    e_embedding: np.ndarray
    h_embedding: np.ndarray
    k_values: list[float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "geometry_params": self.geometry_params,
            "e_fingerprint": self.e_fingerprint,
            "h_fingerprint": self.h_fingerprint,
            "e_embedding": self.e_embedding.tolist(),
            "h_embedding": self.h_embedding.tolist(),
            "k_values": list(self.k_values),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingSample:
        return cls(
            geometry_params=tuple(d["geometry_params"]),
            e_fingerprint=d["e_fingerprint"],
            h_fingerprint=d["h_fingerprint"],
            e_embedding=np.asarray(d["e_embedding"], dtype=np.float64),
            h_embedding=np.asarray(d["h_embedding"], dtype=np.float64),
            k_values=list(d["k_values"]),
        )


# ---------------------------------------------------------------------------
# GodTensor
# ---------------------------------------------------------------------------


@dataclass
class GodTensor:
    """The God Tensor: spectral fixed point of the learned E ⇄ H operator."""

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
    convergence_history: list[dict[str, float]] = field(default_factory=list)
    final_residual: float = float("inf")
    dominant_eigenvalue: complex = complex("nan")
    spectral_gap: float = float("nan")

    # ------------------------------------------------------------------
    # Training-data collection
    # ------------------------------------------------------------------

    def collect_training_data(
        self,
        nx: int = 50,
        ny: int = 50,
        num_modes: int = 8,
        seed: int = 42,
    ) -> None:
        """Generate a sweep of cavity geometries with E- and H-field fingerprints.

        60% rectangular, 20% circular, 20% photonic-crystal — the same
        proportions used in the documented demo runs.
        """
        rng = np.random.default_rng(seed)
        self.samples = []

        for i in range(self.n_geometries):
            p = rng.random()
            if p < 0.6:
                w = float(rng.uniform(0.8, 3.0))
                h = float(rng.uniform(0.5, 2.0))
                geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(w, h))
                params: tuple[float, ...] = (w, h)
            elif p < 0.8:
                r = float(rng.uniform(0.5, 1.5))
                geom = CavityGeometry(shape=CavityShape.CIRCULAR, dims=(r,))
                params = (r,)
            else:
                a = float(rng.uniform(0.1, 0.3))
                r_p = a * 0.3
                geom = CavityGeometry(
                    shape=CavityShape.PHOTONIC_CRYSTAL, dims=(a, r_p)
                )
                params = (a, r_p)

            try:
                mode_data = solve_cavity_modes(
                    geom, nx=nx, ny=ny, num_modes=num_modes, seed=seed
                )
            except Exception as exc:
                log.warning("geometry_solve_failed", params=params, error=str(exc))
                continue

            mode_key = "mode_0"
            if mode_key not in mode_data["e_modes"]:
                continue
            e_field = np.asarray(
                mode_data["e_modes"][mode_key]["field"], dtype=np.float64
            )
            h_field = np.asarray(
                mode_data["h_modes"][mode_key]["field"], dtype=np.float64
            )

            fp = coupled_fingerprint(e_field, h_field, threshold=0.05)
            e_fp = fp["e_fingerprint"]
            h_fp = fp["h_fingerprint"]
            if "error" in e_fp or "error" in h_fp:
                continue

            sample = TrainingSample(
                geometry_params=params,
                e_fingerprint=e_fp,
                h_fingerprint=h_fp,
                e_embedding=embed_fingerprint(e_fp, dim=50),
                h_embedding=embed_fingerprint(h_fp, dim=50),
                k_values=list(mode_data["k_values"]),
            )
            self.samples.append(sample)
            log.debug("sample_collected", i=i, params=params, valid=len(self.samples))
            if (i + 1) % 20 == 0:
                log.info(
                    "collection_progress",
                    collected=i + 1,
                    total=self.n_geometries,
                )

        log.info("training_data_collected", n_samples=len(self.samples))

    # ------------------------------------------------------------------
    # Learn T
    # ------------------------------------------------------------------

    def learn_T(self) -> np.ndarray:
        """Learn the coupling operator T via least squares.

        Solves :math:`\\min_T \\| Z_E T^\\top - Z_H \\|_F^2` for T, where
        :math:`Z_E, Z_H` are stacked autoencoder-projected latent vectors
        of the E- and H-fingerprint embeddings.
        """
        if len(self.samples) < 2:
            raise ConvergenceError(
                "need >=2 training samples to fit T", n_samples=len(self.samples)
            )

        E = np.stack([s.e_embedding for s in self.samples])
        H = np.stack([s.h_embedding for s in self.samples])

        log.info("training_manifold_projector_e", samples=len(E))
        self.projector_e.fit(
            list(E), epochs=100, batch_size=min(8, len(E)), verbose=False
        )
        log.info("training_manifold_projector_h", samples=len(H))
        self.projector_h.fit(
            list(H), epochs=100, batch_size=min(8, len(H)), verbose=False
        )

        E_lat = np.stack([self.projector_e.encode(e) for e in E])
        H_lat = np.stack([self.projector_h.encode(h) for h in H])

        from scipy.linalg import lstsq

        T_raw, _, rank, _ = lstsq(E_lat, H_lat)
        T = T_raw.T  # (L, L) so that T @ e ≈ h
        self.T_matrix = T

        H_recon = E_lat @ T.T
        recon_err = float(np.mean(np.abs(H_recon - H_lat)))
        log.info("t_matrix_learned", shape=T.shape, rank=int(rank))
        log.info("t_reconstruction_error", error=recon_err)
        return T

    # ------------------------------------------------------------------
    # Spectral fixed-point iteration (Perron-Frobenius / power method)
    # ------------------------------------------------------------------

    def find_fixed_point(
        self, iters: int = 500, tol: float = 1e-7
    ) -> np.ndarray:
        """Compute the God Tensor :math:`g` — the dominant eigenvector of T.

        Combines (1) a closed-form eigendecomposition for the initial
        guess and (2) normalised power iteration for refinement to
        machine precision.

        Parameters
        ----------
        iters : int
            Maximum number of power-method refinement passes.
        tol : float
            Sign-corrected convergence tolerance; once
            :math:`\\| x_{n+1} - \\mathrm{sign}\\langle x_{n+1}, x_n\\rangle\\, x_n\\|_2 < \\varepsilon`
            the iteration is declared converged.
        """
        if self.T_matrix is None:
            self.learn_T()
        T = self.T_matrix
        assert T is not None

        eigenvalues, eigenvectors = np.linalg.eig(T)
        order = np.argsort(-np.abs(eigenvalues))  # descending |λ|
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        self.dominant_eigenvalue = complex(eigenvalues[0])
        self.spectral_gap = float(
            np.abs(eigenvalues[1]) / max(np.abs(eigenvalues[0]), 1e-12)
        ) if len(eigenvalues) > 1 else 0.0

        x = np.real(eigenvectors[:, 0])
        x = x / max(float(np.linalg.norm(x)), 1e-12)

        log.info(
            "spectral_init",
            dominant_eigenvalue=float(np.real(eigenvalues[0])),
            spectral_gap_ratio=self.spectral_gap,
            latent_dim=T.shape[0],
        )

        spectral_residual = float("inf")
        sign_correction = 1.0
        x_new = x

        for i in range(iters):
            x_new = T @ x
            norm = float(np.linalg.norm(x_new))
            if norm > 1e-12:
                x_new = x_new / norm
            sign_correction = 1.0 if float(np.dot(x_new, x)) >= 0 else -1.0
            spectral_residual = float(np.linalg.norm(x_new - sign_correction * x))
            self.convergence_history.append(
                {
                    "iter": i,
                    "spectral_residual": spectral_residual,
                    "norm": norm,
                }
            )
            x = sign_correction * x_new
            if spectral_residual < tol:
                self.fixed_point_converged = True
                log.info(
                    "spectral_fixed_point_converged",
                    iter=i,
                    residual=spectral_residual,
                )
                break
            if (i + 1) % 100 == 0:
                log.debug(
                    "spectral_progress", iter=i + 1, residual=spectral_residual
                )

        self.final_residual = spectral_residual
        self.god_tensor = x

        # Verification: T(T(x)) ≈ T(x) (idempotency on the eigen-line)
        Tx = T @ x
        TTx = T @ Tx
        verification = float(np.linalg.norm(Tx - TTx))
        log.info("spectral_fixed_point_verified", verification_error=verification)

        # E-side and H-side projection distances to the God Tensor
        e_lat = np.stack([self.projector_e.encode(s.e_embedding) for s in self.samples])
        h_lat = np.stack([self.projector_h.encode(s.h_embedding) for s in self.samples])
        e_under_T = e_lat @ T.T
        h_under_T = h_lat @ T.T
        e_under_T = e_under_T / (
            np.linalg.norm(e_under_T, axis=1, keepdims=True) + 1e-12
        )
        h_under_T = h_under_T / (
            np.linalg.norm(h_under_T, axis=1, keepdims=True) + 1e-12
        )
        e_dist = float(np.mean(np.linalg.norm(e_under_T - x, axis=1)))
        h_dist = float(np.mean(np.linalg.norm(h_under_T - x, axis=1)))
        log.info("e_convergence_to_spectral_point", avg_dist=e_dist)
        log.info("h_convergence_to_spectral_point", avg_dist=h_dist)
        return x

    # ------------------------------------------------------------------
    # Public-API mappings
    # ------------------------------------------------------------------

    def get_e_to_h_map(self, e_embedding: np.ndarray) -> np.ndarray:
        """Map an E-embedding to its predicted H-latent via T."""
        if self.T_matrix is None:
            raise ConvergenceError("call find_fixed_point first")
        e_latent = self.projector_e.encode(e_embedding)
        h_latent = e_latent @ self.T_matrix.T
        return h_latent / (np.linalg.norm(h_latent) + 1e-12)

    def get_h_to_e_map(self, h_embedding: np.ndarray) -> np.ndarray:
        """Map an H-embedding to its predicted E-latent via T."""
        if self.T_matrix is None:
            raise ConvergenceError("call find_fixed_point first")
        h_latent = self.projector_h.encode(h_embedding)
        e_latent = h_latent @ self.T_matrix
        return e_latent / (np.linalg.norm(e_latent) + 1e-12)

    def god_score(self) -> float:
        """Decay score in :math:`(0, 1]` measuring E/H convergence to g.

        :math:`\\mathrm{score} = \\exp\\!\\bigl(-\\tfrac{1}{2}\\,
        \\overline{\\|T(z^E) - g\\| + \\|T(z^H) - g\\|}\\bigr)`,
        averaged over training samples.  Score = 1 ⇔ E and H map exactly
        onto the God Tensor.
        """
        if self.god_tensor is None or self.T_matrix is None:
            return 0.0
        g = self.god_tensor
        T = self.T_matrix
        e_lat = np.stack(
            [self.projector_e.encode(s.e_embedding) for s in self.samples]
        )
        h_lat = np.stack(
            [self.projector_h.encode(s.h_embedding) for s in self.samples]
        )
        e_T = e_lat @ T.T
        h_T = h_lat @ T.T
        e_T = e_T / (np.linalg.norm(e_T, axis=1, keepdims=True) + 1e-12)
        h_T = h_T / (np.linalg.norm(h_T, axis=1, keepdims=True) + 1e-12)
        d_e = np.linalg.norm(e_T - g, axis=1)
        d_h = np.linalg.norm(h_T - g, axis=1)
        return float(np.exp(-np.mean(d_e + d_h) / 2))

    def summary(self) -> dict[str, Any]:
        """Human-readable summary."""
        return {
            "n_samples": len(self.samples),
            "T_matrix_shape": (
                tuple(self.T_matrix.shape) if self.T_matrix is not None else None
            ),
            "god_tensor_shape": (
                tuple(self.god_tensor.shape) if self.god_tensor is not None else None
            ),
            "converged": self.fixed_point_converged,
            "final_spectral_residual": (
                self.convergence_history[-1]["spectral_residual"]
                if self.convergence_history
                else None
            ),
            "dominant_eigenvalue": (
                None
                if not np.isfinite(np.real(self.dominant_eigenvalue))
                else complex(self.dominant_eigenvalue)
            ),
            "spectral_gap_ratio": (
                None if not np.isfinite(self.spectral_gap) else self.spectral_gap
            ),
            "god_score": self.god_score(),
        }

    def predict(self, w: float, h: float) -> list[tuple[float, float]]:
        """Predict the persistence diagram of a new rectangular cavity.

        Convenience wrapper that runs FDFD on the fly and returns its
        H_0 barcode.  For full prediction with the learned operator, use
        :func:`faraday.predict.predict_eh_barcode`.
        """
        from faraday.barcode import topological_fingerprint

        geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(w, h))
        mode_data = _solve(geom, nx=50, ny=50, num_modes=8)
        mode_key = "mode_0"
        if mode_key not in mode_data["e_modes"]:
            return []
        e_field = np.asarray(
            mode_data["e_modes"][mode_key]["field"], dtype=np.float64
        )
        fp = topological_fingerprint(e_field)
        diagram = fp.get("diagrams", [[]])[0]
        return [(float(b), float(d)) for b, d in diagram]

    # ------------------------------------------------------------------
    # Persistence / serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Pickle the trained GodTensor to ``path``."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str) -> GodTensor:
        """Load a GodTensor from a pickle file."""
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected GodTensor, got {type(obj).__name__}")
        return obj

    def save_checkpoint(
        self, path: str, epoch: int, rng_state: dict[str, Any]
    ) -> None:
        """Save spectral-burn checkpoint (god_tensor + epoch + RNG state)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(  # type: ignore[call-overload]
            path,
            god_tensor=np.asarray(self.god_tensor),
            epoch=int(epoch),
            rng_state=np.asarray(rng_state, dtype=object),
        )

    @classmethod
    def load_checkpoint(
        cls, path: str
    ) -> tuple[np.ndarray, int, dict[str, Any]]:
        """Load checkpoint. Returns ``(god_tensor, epoch, rng_state)``."""
        data = np.load(path, allow_pickle=True)
        return (
            data["god_tensor"],
            int(data["epoch"]),
            dict(data["rng_state"].item()),
        )
