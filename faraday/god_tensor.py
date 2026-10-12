# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.god_tensor — The Fixed Point of E ⇄ H

The God Tensor is the fixed point of the E-H co-determination operator.
Given E_signature and H_signature from the same cavity mode:

    T(E_sig)  -> H_sig
    T⁺(H_sig) -> E_sig

The invariant is T x* = x*: the eigenvector of T with eigenvalue nearest 1,
reached by a Banach contraction (shifted inverse iteration) from a generic
start vector.

At convergence: x* = God Tensor
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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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


def initial_vector(dim: int, seed: int = 0) -> np.ndarray:
    """Deterministic generic unit start vector for the fixed-point iteration.

    Generic means: not an eigenvector of T, so the iteration genuinely has to
    contract onto the fixed point instead of sitting on whatever eigenvector
    it was handed.
    """
    x = np.random.default_rng(seed).standard_normal(dim)
    return x / np.linalg.norm(x)


def fixed_point_operator(
    T: np.ndarray, target: float = 1.0
) -> tuple[Callable[[np.ndarray], np.ndarray], complex, float]:
    """Build the Banach contraction whose fixed point solves T x = λ* x, λ* ≈ 1.

    The God Tensor is the invariant of the coupling, T g = g, i.e. the
    eigenvector whose eigenvalue λ* is nearest ``target`` = 1. Plain power
    iteration x → Tx/||Tx|| cannot find it: it only ever converges to the
    eigenvector of largest |λ|, and a learned T need not have λ_max = 1.
    Shifted inverse iteration can:

        G(x) = (T - μI)⁻¹ x / ||(T - μI)⁻¹ x||,   μ = target + ε

    The eigenvalues of (T - μI)⁻¹ are 1/(λ_j - μ), so the eigenvector with λ
    nearest μ dominates and G contracts onto it with Lipschitz factor
    q = |λ* - μ| / |λ₂ - μ| < 1 (λ₂ = next-nearest eigenvalue). ε = 1e-6·||T||
    keeps T - μI invertible when λ* = 1 exactly; it only makes q smaller.

    Returns:
        (G, λ*, q). q = 1 means no contraction: the eigenvalue nearest the
        target is a complex pair (no real fixed point) or a tie.
    """
    from scipy.linalg import lu_factor, lu_solve

    n = T.shape[0]
    mu = target + 1e-6 * max(1.0, float(np.linalg.norm(T, 2)))
    eigenvalues = np.linalg.eigvals(T)
    dist = np.abs(eigenvalues - mu)
    order = np.argsort(dist)
    lam_star = complex(eigenvalues[order[0]])
    q = float(dist[order[0]] / dist[order[1]]) if n > 1 else 0.0

    lu = lu_factor(T - mu * np.eye(n))

    def G(x: np.ndarray) -> np.ndarray:
        return lu_solve(lu, x)

    return G, lam_star, q


def banach_step(
    G: Callable[[np.ndarray], np.ndarray], x: np.ndarray
) -> tuple[np.ndarray, float]:
    """One normalised step x → G(x)/||G(x)||, aligned to the sign of ``x``.

    Eigenvectors are defined up to ±1; aligning the sign keeps a negative
    shifted eigenvalue from flipping the iterate every step.

    Returns:
        (x_new, delta): next iterate and the sign-invariant step ||x_new - x||.
    """
    y = G(x)
    y = y / np.linalg.norm(y)
    if np.dot(y, x) < 0:
        y = -y
    return y, float(np.linalg.norm(y - x))


def sign_invariant_distance(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Row-wise min(||a - x||, ||a + x||): distance to the line spanned by ±x."""
    return np.minimum(
        np.linalg.norm(a - x, axis=-1), np.linalg.norm(a + x, axis=-1)
    )


@dataclass
class TrainingSample:
    """One training sample: a geometry + its E and H field signatures."""

    geometry_params: tuple[float, ...]  # e.g. (w, h) or (r,)
    e_fingerprint: dict[str, Any]
    h_fingerprint: dict[str, Any]
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

    @classmethod
    def from_dict(cls, d: dict) -> TrainingSample:
        return cls(
            geometry_params=tuple(d["geometry_params"]),
            e_fingerprint=d["e_fingerprint"],
            h_fingerprint=d["h_fingerprint"],
            e_embedding=np.array(d["e_embedding"], dtype=np.float64),
            h_embedding=np.array(d["h_embedding"], dtype=np.float64),
            k_values=list(d["k_values"]),
        )


@dataclass
class GodTensor:
    """
    The God Tensor: fixed point of E ⇄ H co-determination.

    Learn T such that:
        T(e) ≈ h   (E encodes H)
        T⁺(h) ≈ e  (H encodes E)
        T(x*) = x*  [fixed point — the invariant; eigenvalue nearest 1]

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
    fixed_point_eigenvalue: complex | None = None
    contraction_factor: float | None = None

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
            params: tuple[float, ...]
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

        # Train manifold projectors on the data before encoding
        log.info("training_manifold_projector_e", samples=len(E))
        self.projector_e.fit([s.e_embedding for s in self.samples], epochs=100, batch_size=8, verbose=False)
        log.info("training_manifold_projector_h", samples=len(H))
        self.projector_h.fit([s.h_embedding for s in self.samples], epochs=100, batch_size=8, verbose=False)

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
        Find the God Tensor: the fixed point x* of the coupling, T x* = λ* x*
        with λ* the eigenvalue of T nearest 1 (exactly T x* = x* when λ* = 1).

        Found by Banach iteration of the contraction G (shifted inverse
        iteration, see :func:`fixed_point_operator`) from a generic start
        vector, so convergence is earned, not assumed. Certified by:

        - ``fixed_point_eigenvalue``: λ*
        - ``contraction_factor``: Lipschitz constant q of G (< 1 ⇒ Banach)
        - logged ``eigen_residual`` ||T x - λ* x|| and ``fixed_point_defect``
          ||T x - x|| = |λ* - 1|

        Args:
            iters: max iterations of G
            tol: convergence tolerance on the sign-invariant step ||x_{n+1} - x_n||

        Returns:
            god_tensor: the unit-norm fixed point x*
        """
        if self.T_matrix is None:
            self.learn_T()

        T = self.T_matrix
        assert T is not None

        G, lam_star, q = fixed_point_operator(T)
        self.fixed_point_eigenvalue = lam_star
        self.contraction_factor = q

        log.info(
            "fixed_point_operator",
            eigenvalue=lam_star.real,
            eigenvalue_imag=lam_star.imag,
            contraction_factor=q,
            latent_dim=T.shape[0],
        )

        self.convergence_history = []
        self.fixed_point_converged = False
        x = initial_vector(T.shape[0])
        delta = float("inf")
        for i in range(iters):
            x, delta = banach_step(G, x)
            self.convergence_history.append({"iter": i, "delta": delta})
            if delta < tol:
                log.info("fixed_point_converged", iter=i, delta=delta)
                self.fixed_point_converged = True
                break
            if (i + 1) % 100 == 0:
                log.debug("fixed_point_progress", iter=i + 1, delta=delta)
        else:
            log.warning(
                "fixed_point_iteration_max_iters",
                final_delta=delta,
                contraction_factor=q,
                note="q ≈ 1: eigenvalue nearest 1 is a complex pair or a tie",
            )

        self.god_tensor = x

        Tx = T @ x
        eigen_residual = float(np.linalg.norm(Tx - lam_star.real * x))
        fixed_point_defect = float(np.linalg.norm(Tx - x))

        # How close E and H land to the fixed point under T
        e_dist, h_dist = self._distances_to_god(x)

        log.info(
            "fixed_point_verified",
            eigen_residual=eigen_residual,
            fixed_point_defect=fixed_point_defect,
        )
        log.info("e_convergence_to_fixed_point", avg_dist=float(np.mean(e_dist)))
        log.info("h_convergence_to_fixed_point", avg_dist=float(np.mean(h_dist)))

        return x

    def _distances_to_god(self, god: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sign-invariant distances of normalised T(e_i), T(h_i) to ±god."""
        assert self.T_matrix is not None
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
        return (
            sign_invariant_distance(e_under_T, god),
            sign_invariant_distance(h_under_T, god),
        )

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
        """Map an H embedding to its predicted E embedding via T⁺.

        T maps E → H, so the reverse map is its Moore-Penrose pseudo-inverse
        (the least-squares inverse; T is often rank-deficient). Tᵀ is not an
        inverse unless T is orthogonal.
        """
        if self.T_matrix is None:
            raise ValueError("Must call find_fixed_point first")
        h_latent = self.projector_h.encode(h_embedding)
        e_latent = np.linalg.pinv(self.T_matrix) @ h_latent
        e_latent = e_latent / (np.linalg.norm(e_latent) + 1e-10)
        return e_latent

    def god_score(self) -> float:
        """
        Compute the 'god score' — how well the God Tensor unifies E and H.

        Score = exp(-mean(d(T(e_i), god) + d(T(h_i), god)) / 2)

        with d the sign-invariant distance min(||a - g||, ||a + g||): the fixed
        point is an eigenvector, defined only up to ±1, so the score must not
        depend on which sign the solver returned.

        Using an exponential keeps the score bounded in [0, 1] regardless of
        the geometry of the embedding space or the number of samples.
        ``exp(-d)`` decays from 1 (d=0, perfect coupling) toward 0 (d→∞).
        """
        if self.god_tensor is None:
            return 0.0

        e_dists, h_dists = self._distances_to_god(self.god_tensor)
        return float(np.exp(-np.mean(e_dists + h_dists) / 2))

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
            "fixed_point_eigenvalue": self.fixed_point_eigenvalue,
            "contraction_factor": self.contraction_factor,
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
            List of (birth, death) pairs of the fundamental mode's E-field H0
            barcode (the essential class has death = inf).
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
        diagrams = fp.get("diagrams", [])
        h0 = diagrams[0] if diagrams else []
        return [(float(b), float(d)) for b, d in h0]

    # ------------------------------------------------------------------
    # Persistence / serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save a trained GodTensor to a pickle file."""
        import pickle

        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str) -> GodTensor:
        """Load a saved GodTensor from a pickle file."""
        import pickle

        with open(path, "rb") as fh:
            return pickle.load(fh)

    def save_checkpoint(self, path: str, epoch: int, rng_state: dict) -> None:
        """Save burn-loop checkpoint: god_tensor + epoch + RNG state (NumPy format)."""
        np.savez(
            path,
            god_tensor=self.god_tensor,  # type: ignore[arg-type]
            epoch=epoch,
            rng_state=rng_state,  # type: ignore[arg-type]
        )

    @classmethod
    def load_checkpoint(cls, path: str) -> tuple[np.ndarray, int, dict]:
        """Load burn-loop checkpoint. Returns (god_tensor, epoch, rng_state)."""
        data = np.load(path, allow_pickle=True)
        god_tensor = data["god_tensor"]
        epoch = int(data["epoch"])
        rng_state = dict(data["rng_state"].item())
        return god_tensor, epoch, rng_state
