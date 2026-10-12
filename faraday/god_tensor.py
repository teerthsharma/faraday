# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.god_tensor — The Fixed Point of E ⇄ H

Every training geometry is described twice, by two fields that never see
each other:

    E-view_j = χ(D) + N_wall_j / 2   from E alone (its nodal points on the wall)
    H-view_j = Q_j                   from H alone (net vortex winding)

for its first ``signature_modes`` TM modes j. The coupling T is learned by
least squares, T e_i ≈ h_i. The bulk-boundary law of :mod:`faraday.duality`
says the two views are equal, so the minimum-norm T is the orthogonal
projector onto the subspace of physically realisable signatures:

    T(T(x)) = T(x)        (idempotent; eigenvalues 1 and 0)

The God Tensor is that fixed subspace — every signature a real cavity can
produce is left unchanged by E → H → E. It is reached by a Banach
contraction (shifted inverse iteration) from a generic start vector and
certified by the idempotency defect ||T² - T|| / ||T||, the multiplicity of
λ* = 1 and the contraction factor q. A violation of the duality law in the
data (e.g. an under-resolved grid) shows up as T drifting from a projector.

Usage
-----
    gt = GodTensor(n_geometries=50)
    gt.collect_training_data(nx=40, ny=40)    # FDFD → duality signatures
    gt.find_fixed_point(iters=500, tol=1e-7)  # learn T, contract onto x*
    gt.summary()["idempotency_defect"]        # ≈ 0 when the law holds
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from faraday._types import ModeData
from faraday.barcode import coupled_fingerprint
from faraday.duality import euler_characteristic, vortex_charges, wall_nodal_count
from faraday.em_solver import CavityGeometry, CavityShape, solve_cavity_modes
from faraday.logging import get_logger

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


def _nearest_cluster(
    eigenvalues: np.ndarray, mu: float, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Order eigenvalues by distance to mu; mask the cluster nearest to it.

    The cluster holds every eigenvalue within 1e-5·||T|| of the nearest one,
    so an exactly or nearly degenerate λ* (e.g. T a projector learned from
    data that obeys the duality law up to a few grid-limited modes) is one
    fixed subspace rather than a tie. The tolerance sits above the inverse
    iteration shift ε = 1e-6·||T||: eigenvalues closer than that cannot be
    separated by the iteration anyway.
    """
    dist = np.abs(eigenvalues - mu)
    return dist, dist <= dist.min() + 1e-5 * scale


def fixed_point_operator(
    T: np.ndarray, target: float = 1.0
) -> tuple[Callable[[np.ndarray], np.ndarray], complex, float]:
    """Build the Banach contraction whose fixed points solve T x = λ* x, λ* ≈ 1.

    The God Tensor is the invariant of the coupling, T g = g: the eigenspace
    whose eigenvalue λ* is nearest ``target`` = 1. Plain power iteration
    x → Tx/||Tx|| cannot find it: it only converges to the eigenvector of
    largest |λ|. Shifted inverse iteration can:

        G(x) = (T - μI)⁻¹ x / ||(T - μI)⁻¹ x||,   μ = target + ε

    The eigenvalues of (T - μI)⁻¹ are 1/(λ_j - μ), so the eigenspace with λ
    nearest μ dominates and G contracts onto it with Lipschitz factor
    q = |λ* - μ| / |λ₂ - μ| < 1, λ₂ the nearest eigenvalue outside that
    eigenspace. An exactly degenerate λ* (multiplicity > 1) is one fixed
    subspace, not a tie. ε = 1e-6·||T|| keeps T - μI invertible when λ* = 1
    exactly; it only makes q smaller.

    Returns:
        (G, λ*, q). q = 1 means no contraction: λ* is complex (no real fixed
        point) or every eigenvalue is equally near the target.
    """
    from scipy.linalg import lu_factor, lu_solve

    n = T.shape[0]
    scale = max(1.0, float(np.linalg.norm(T, 2)))
    mu = target + 1e-6 * scale
    eigenvalues = np.linalg.eigvals(T)
    dist, cluster = _nearest_cluster(eigenvalues, mu, scale)
    lam_star = complex(eigenvalues[int(np.argmin(dist))])
    if abs(lam_star.imag) > 1e-8 * scale or cluster.all():
        q = 1.0
    else:
        q = float(dist[cluster].max() / dist[~cluster].min())

    lu = lu_factor(T - mu * np.eye(n))

    def G(x: np.ndarray) -> np.ndarray:
        return lu_solve(lu, x)

    return G, lam_star, q


def fixed_point_basis(T: np.ndarray, target: float = 1.0) -> np.ndarray:
    """Orthonormal basis (columns) of the eigenspace of T nearest ``target``.

    The eigenspace of the cluster around λ* (multiplicity m) is the near-null
    space of T - λ*I: its m smallest right singular vectors. This is
    orthonormal and exactly m-dimensional by construction; eigenvectors from
    ``np.linalg.eig`` are ill-conditioned inside a degenerate eigenspace.
    """
    scale = max(1.0, float(np.linalg.norm(T, 2)))
    eigenvalues = np.linalg.eigvals(T)
    dist, cluster = _nearest_cluster(eigenvalues, target + 1e-6 * scale, scale)
    lam = float(np.real(eigenvalues[int(np.argmin(dist))]))
    _, _, vh = np.linalg.svd(T - lam * np.eye(T.shape[0]))
    return vh[-int(cluster.sum()) :].T


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


def duality_views(mode_data: ModeData, num_modes: int) -> tuple[np.ndarray, np.ndarray]:
    """E-view and H-view charge signatures of the first ``num_modes`` modes.

    ``E-view_j = χ(D) + N_wall_j / 2`` uses only E (and the cavity's shape);
    ``H-view_j = Q_j`` uses only H. The bulk-boundary law says they are equal.
    Modes beyond those solved are 0 in both views.
    """
    interior = np.array(mode_data["interior"], dtype=bool)
    chi = euler_characteristic(interior)
    e_view = np.zeros(num_modes)
    h_view = np.zeros(num_modes)
    for j in range(min(num_modes, mode_data["num_modes_found"])):
        key = f"mode_{j}"
        e = np.array(mode_data["e_modes"][key]["field"])
        hx = np.array(mode_data["h_modes"][key]["hx"])
        hy = np.array(mode_data["h_modes"][key]["hy"])
        e_view[j] = chi + wall_nodal_count(e, interior) / 2
        h_view[j] = vortex_charges(hx, hy, interior).sum()
    return e_view, h_view


def make_sample(
    geom: CavityGeometry,
    params: tuple[float, ...],
    nx: int,
    ny: int,
    num_modes: int,
    signature_modes: int,
    seed: int | None = None,
    fingerprint_threshold: float = 0.1,
) -> TrainingSample | None:
    """Solve one cavity and build its training sample (None if unusable).

    The fundamental mode's barcode fingerprints feed the KNN predictor; the
    duality signatures of the first ``signature_modes`` modes feed T.
    """
    mode_data = solve_cavity_modes(
        geom, nx=nx, ny=ny, num_modes=max(num_modes, signature_modes), seed=seed
    )
    if "mode_0" not in mode_data["e_modes"]:
        return None
    fp = coupled_fingerprint(
        np.array(mode_data["e_modes"]["mode_0"]["field"]),
        np.array(mode_data["h_modes"]["mode_0"]["field"]),
        threshold=fingerprint_threshold,
    )
    if "error" in fp["e_fingerprint"] or "error" in fp["h_fingerprint"]:
        return None
    e_view, h_view = duality_views(mode_data, signature_modes)
    return TrainingSample(
        geometry_params=params,
        e_fingerprint=fp["e_fingerprint"],
        h_fingerprint=fp["h_fingerprint"],
        e_embedding=e_view,
        h_embedding=h_view,
        k_values=mode_data["k_values"],
    )


@dataclass
class TrainingSample:
    """One training sample: a geometry + its E and H field signatures."""

    geometry_params: tuple[float, ...]  # e.g. (w, h) or (r,)
    e_fingerprint: dict[str, Any]
    h_fingerprint: dict[str, Any]
    e_embedding: np.ndarray  # E-view duality signature (χ + N_wall/2 per mode)
    h_embedding: np.ndarray  # H-view duality signature (vortex charge per mode)
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
        T(e) ≈ h    (E's wall data encodes H's vortex charge)
        T⁺(h) ≈ e   (H encodes E)
        T(x*) = x*  [fixed subspace — the invariant; eigenvalue 1]

    When the duality law holds T is an orthogonal projector, T² = T, and the
    God Tensor is its range: the subspace of realisable charge signatures.
    """

    n_geometries: int = 50
    signature_modes: int = 16
    samples: list[TrainingSample] = field(default_factory=list)
    T_matrix: np.ndarray | None = field(default=None, repr=False)
    god_tensor: np.ndarray | None = field(default=None, repr=False)
    god_basis: np.ndarray | None = field(default=None, repr=False)
    fixed_point_converged: bool = False
    convergence_history: list[dict] = field(default_factory=list)
    fixed_point_eigenvalue: complex | None = None
    contraction_factor: float | None = None
    idempotency_defect: float | None = None

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
        modes for each, and records the fundamental mode's barcode
        fingerprints and the E-view / H-view duality signatures.

        Args:
            nx, ny: grid resolution
            num_modes: eigenmodes to solve per geometry (at least
                ``signature_modes`` are always solved)
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

            try:
                sample = make_sample(
                    geom,
                    params,
                    nx,
                    ny,
                    num_modes,
                    self.signature_modes,
                    seed=seed,
                    fingerprint_threshold=0.05,
                )
            except Exception as e:  # pragma: no cover - degenerate geometry
                log.warning("geometry_solve_failed", params=params, error=str(e))
                continue
            if sample is None:
                continue
            self.samples.append(sample)

            log.info("sample_collected", i=i, params=params, valid=len(self.samples))
            if (i + 1) % 20 == 0:
                log.info("collection_progress", collected=i + 1, total=self.n_geometries)

        log.info("training_data_collected", n_samples=len(self.samples))

    def learn_T(self) -> np.ndarray:
        """
        Learn the coupling operator T: E-view → H-view.

        Minimum-norm least squares T = H Eᵀ⁺ over the stacked signatures. When
        every sample obeys the duality law (H = E), T is the orthogonal
        projector onto the span of the signatures, so T² = T exactly; the
        relative idempotency defect ||T² - T|| / ||T|| measures how far the
        data departs from the law.

        Returns:
            T: (signature_modes, signature_modes) coupling matrix
        """
        if len(self.samples) < 2:
            raise ValueError("Need at least 2 training samples")

        E = np.array([s.e_embedding for s in self.samples])
        H = np.array([s.h_embedding for s in self.samples])

        X, *_, rank, _ = np.linalg.lstsq(E, H, rcond=None)
        T = X.T
        self.T_matrix = T

        norm = float(np.linalg.norm(T))
        self.idempotency_defect = (
            float(np.linalg.norm(T @ T - T)) / norm if norm > 0 else 0.0
        )
        law_violations = int(np.sum(np.any(np.abs(E - H) > 1e-9, axis=1)))
        log.info(
            "t_matrix_learned",
            shape=T.shape,
            rank=int(rank),
            idempotency_defect=self.idempotency_defect,
            reconstruction_error=float(np.mean(np.abs(E @ T.T - H))),
            samples_violating_law=law_violations,
        )
        return T

    def find_fixed_point(self, iters: int = 500, tol: float = 1e-7) -> np.ndarray:
        """
        Find the God Tensor: the fixed subspace of the coupling, T x = λ* x
        with λ* the eigenvalue of T nearest 1 (exactly T x = x when λ* = 1).

        Found by Banach iteration of the contraction G (shifted inverse
        iteration, see :func:`fixed_point_operator`) from a generic start
        vector, so convergence is earned, not assumed. ``god_tensor`` is the
        fixed point reached; ``god_basis`` spans the whole fixed subspace
        (its dimension is the multiplicity of λ*). Certified by:

        - ``fixed_point_eigenvalue``: λ*
        - ``contraction_factor``: Lipschitz constant q of G (< 1 ⇒ Banach)
        - ``idempotency_defect``: ||T² - T|| / ||T|| (0 ⇔ T is a projector)
        - logged ``eigen_residual`` ||T x - λ* x|| and ``fixed_point_defect``
          ||T x - x|| = |λ* - 1|

        Args:
            iters: max iterations of G
            tol: convergence tolerance on the sign-invariant step
                ||x_{n+1} - x_n|| or the eigen-residual ||T x - λ* x||

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
        self.god_basis = fixed_point_basis(T)

        log.info(
            "fixed_point_operator",
            eigenvalue=lam_star.real,
            eigenvalue_imag=lam_star.imag,
            multiplicity=self.god_basis.shape[1],
            contraction_factor=q,
            dim=T.shape[0],
        )

        self.convergence_history = []
        self.fixed_point_converged = False
        x = initial_vector(T.shape[0])
        delta = float("inf")
        for i in range(iters):
            x, delta = banach_step(G, x)
            # A degenerate λ* has neutral directions: the iterate drifts inside
            # the fixed subspace at the solver's noise floor, so stop on the
            # eigen-residual as well as on the step.
            residual = float(np.linalg.norm(T @ x - lam_star.real * x))
            self.convergence_history.append(
                {"iter": i, "delta": delta, "residual": residual}
            )
            if delta < tol or residual < tol:
                log.info("fixed_point_converged", iter=i, delta=delta, residual=residual)
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

        # How close E and H land to the fixed subspace under T
        e_dist, h_dist = self._distances_to_god()

        log.info(
            "fixed_point_verified",
            eigen_residual=eigen_residual,
            fixed_point_defect=fixed_point_defect,
        )
        log.info("e_convergence_to_fixed_point", avg_dist=float(np.mean(e_dist)))
        log.info("h_convergence_to_fixed_point", avg_dist=float(np.mean(h_dist)))

        return x

    def distance_to_god(self, v: np.ndarray) -> np.ndarray:
        """Distance of the direction(s) ``v`` to the fixed subspace (sign-invariant).

        ``sqrt(1 - ||Bᵀu||²)`` for u = v/||v|| and an orthonormal basis B: 0
        inside the subspace, 1 orthogonal to it (and for a zero vector).
        """
        basis = self.god_basis
        if basis is None:
            if self.god_tensor is None:
                raise ValueError("Must call find_fixed_point first")
            basis = self.god_tensor[:, None]
        rows = np.atleast_2d(np.asarray(v, dtype=float))
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        units = np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)
        captured = np.sum((units @ basis) ** 2, axis=1)
        d = np.sqrt(np.clip(1.0 - captured, 0.0, None))
        return d if np.ndim(v) > 1 else d[0]

    def _distances_to_god(self) -> tuple[np.ndarray, np.ndarray]:
        """Distances of the directions T(e_i), T(h_i) to the fixed subspace."""
        assert self.T_matrix is not None
        out = []
        for views in (
            [s.e_embedding for s in self.samples],
            [s.h_embedding for s in self.samples],
        ):
            out.append(self.distance_to_god(np.array(views) @ self.T_matrix.T))
        return out[0], out[1]

    def get_e_to_h_map(self, e_embedding: np.ndarray) -> np.ndarray:
        """Map an E-view signature to its predicted H-view via T (unit norm)."""
        if self.T_matrix is None:
            raise ValueError("Must call find_fixed_point first")
        h = self.T_matrix @ e_embedding
        return h / (np.linalg.norm(h) + 1e-10)

    def get_h_to_e_map(self, h_embedding: np.ndarray) -> np.ndarray:
        """Map an H-view signature to its predicted E-view via T⁺ (unit norm).

        T maps E → H, so the reverse map is its Moore-Penrose pseudo-inverse
        (the least-squares inverse; T is rank-deficient). Tᵀ is not an
        inverse unless T is orthogonal.
        """
        if self.T_matrix is None:
            raise ValueError("Must call find_fixed_point first")
        e = np.linalg.pinv(self.T_matrix) @ h_embedding
        return e / (np.linalg.norm(e) + 1e-10)

    def god_score(self) -> float:
        """
        Compute the 'god score' — how well the God Tensor unifies E and H.

        Score = exp(-mean(d(T(e_i)) + d(T(h_i))) / 2)

        with d the distance of the normalised vector to the fixed subspace
        (see :meth:`distance_to_god`). It is sign-invariant: an eigenvector is
        defined only up to ±1. Score 1 ⇔ T maps every E and H signature into
        the fixed subspace, i.e. E and H are unified by the duality law.

        Using an exponential keeps the score bounded in [0, 1] regardless of
        the geometry of the embedding space or the number of samples.
        ``exp(-d)`` decays from 1 (d=0, perfect coupling) toward 0 (d→∞).
        """
        if self.god_tensor is None:
            return 0.0

        e_dists, h_dists = self._distances_to_god()
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
            "fixed_point_multiplicity": self.god_basis.shape[1]
            if self.god_basis is not None
            else None,
            "contraction_factor": self.contraction_factor,
            "idempotency_defect": self.idempotency_defect,
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
