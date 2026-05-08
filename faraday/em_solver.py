# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.em_solver — FDFD Cavity Solver for E and H Fields.

Computes TM eigenmodes of a 2D PEC cavity::

    ∇²E_z + k² E_z = 0,     E_z|∂Ω = 0  (PEC, Dirichlet)

The H-field is derived from Maxwell's curl equation::

    H_x = (i / ω μ) ∂E_z/∂y,    H_y = -(i / ω μ) ∂E_z/∂x

For benchmarking against analytic theory, the rectangular cavity has the
closed-form eigenvalues

.. math::

   k_{mn}^2 = (m \\pi / w)^2 + (n \\pi / h)^2,
   \\quad m, n = 1, 2, 3, \\ldots

which we validate in ``tests/test_em_solver_analytic.py``.

Implementation notes
--------------------

* The Laplacian is built fully vectorised via Kronecker products of the
  1-D Dirichlet stencils, eliminating the previous O(N²) Python double-loop.
* The H-field is computed with ``numpy.gradient`` rather than a per-cell
  finite-difference loop — both vectorised and boundary-aware.
* The eigensolver uses ``eigsh`` in **shift-invert mode** (``sigma=0``)
  which targets the eigenvalues nearest zero (the physical low-frequency
  cavity modes) and converges far faster than ``which="SM"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import eigsh

from faraday.exceptions import GeometryError, SolverError
from faraday.logging import get_logger

if TYPE_CHECKING:
    from faraday._types import ModeData

log = get_logger(__name__)


class CavityShape(Enum):
    """Supported cavity shapes."""

    RECTANGULAR = "rectangular"
    CIRCULAR = "circular"
    PHOTONIC_CRYSTAL = "photonic_crystal"


@dataclass
class CavityGeometry:
    """Cavity geometry descriptor.

    Parameters
    ----------
    shape : CavityShape
        The shape of the cavity.
    dims : tuple[float, ...]
        Dimensions: ``(width, height)`` for rectangular, ``(radius,)`` for
        circular, or ``(a, r_pillar)`` for photonic_crystal where ``a`` is
        the lattice constant.
    boundary_conditions : str
        Only ``"pec"`` (perfect electric conductor) is currently supported.
    """

    shape: CavityShape
    dims: tuple[float, ...]
    boundary_conditions: str = "pec"

    def __post_init__(self) -> None:
        if self.boundary_conditions != "pec":
            raise GeometryError(
                f"only PEC boundary supported, got {self.boundary_conditions!r}",
                bc=self.boundary_conditions,
            )
        if self.shape == CavityShape.RECTANGULAR:
            if len(self.dims) != 2:
                raise GeometryError("rectangular dims must be (w, h)", dims=self.dims)
            w, h = self.dims
            if w <= 0 or h <= 0:
                raise GeometryError("w, h must be positive", w=w, h=h)
        elif self.shape == CavityShape.CIRCULAR:
            if len(self.dims) != 1:
                raise GeometryError("circular dims must be (r,)", dims=self.dims)
            (r,) = self.dims
            if r <= 0:
                raise GeometryError("r must be positive", r=r)
        elif self.shape == CavityShape.PHOTONIC_CRYSTAL:
            if len(self.dims) != 2:
                raise GeometryError(
                    "photonic_crystal dims must be (a, r_pillar)", dims=self.dims
                )
            a, r_p = self.dims
            if a <= 0 or r_p <= 0:
                raise GeometryError("a, r_p must be positive", a=a, r_p=r_p)

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Boolean mask of points inside the cavity.

        Both 1-D and 2-D ``(x, y)`` inputs are accepted; a meshgrid is the
        common case for FDFD discretisation.
        """
        if self.shape == CavityShape.RECTANGULAR:
            w, h = self.dims
            return (np.abs(x) < w / 2) & (np.abs(y) < h / 2)
        if self.shape == CavityShape.CIRCULAR:
            (r,) = self.dims
            return (x * x + y * y) < r * r
        if self.shape == CavityShape.PHOTONIC_CRYSTAL:
            a, r_p = self.dims
            w, h = a * 15, a * 10
            interior = (np.abs(x) < w / 2) & (np.abs(y) < h / 2)
            # Vectorised hexagonal pillar lattice with an L3 line-defect.
            i_idx = np.arange(-10, 11)
            j_idx = np.arange(-8, 9)
            ii, jj = np.meshgrid(i_idx, j_idx, indexing="ij")
            px = ii * a + (jj % 2) * (a / 2)
            py = jj * a * (np.sqrt(3) / 2)
            # L3 defect: skip the 3 central pillars on the j=0 row.
            keep = ~((jj == 0) & (np.isin(ii, [-1, 0, 1])))
            px = px[keep].ravel()
            py = py[keep].ravel()
            x_e = x[..., None]
            y_e = y[..., None]
            dist_sq = (x_e - px) ** 2 + (y_e - py) ** 2
            pillars_mask = np.any(dist_sq < r_p * r_p, axis=-1)
            return interior & ~pillars_mask
        raise GeometryError(f"Unsupported cavity shape: {self.shape}", shape=self.shape)

    def interior_mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Alias for :meth:`contains` — useful with meshgrid inputs."""
        return self.contains(X, Y)


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def make_rectangular_grid(
    w: float, h: float, nx: int, ny: int
) -> tuple[np.ndarray, np.ndarray]:
    """Centred meshgrid of shape ``(ny, nx)``."""
    x = np.linspace(-w / 2, w / 2, nx)
    y = np.linspace(-h / 2, h / 2, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y


def make_circular_grid(
    r: float, nx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centred meshgrid + interior mask for a circular cavity."""
    coord = np.linspace(-r, r, nx)
    X, Y = np.meshgrid(coord, coord)
    return X, Y, r * r > X * X + Y * Y


# ---------------------------------------------------------------------------
# Vectorised Laplacian
# ---------------------------------------------------------------------------

# The Dirichlet (PEC) Laplacian on a regular ``ny × nx`` grid factors as
#
#     L = I_y ⊗ D_x  +  D_y ⊗ I_x ,
#
# where ``D_x`` and ``D_y`` are tridiagonal 1-D second-difference matrices
# scaled by 1/dx² and 1/dy². Exterior cells are zeroed out by a row mask.

_PENALTY = 1e6  # exterior diagonal value (so SM/sigma=0 ignores them)


def _dirichlet_1d(n: int, h: float) -> csr_matrix:
    """Return the ``n × n`` second-difference operator with Dirichlet BC.

    Stencil = (1, -2, 1) / h². Boundary rows already implement the
    Dirichlet condition because they only see one neighbour.
    """
    diagonals = [
        np.full(n - 1, 1.0 / (h * h)),
        np.full(n, -2.0 / (h * h)),
        np.full(n - 1, 1.0 / (h * h)),
    ]
    return diags(diagonals, offsets=[-1, 0, 1], format="csr")


def build_laplacian_2d(
    nx: int, ny: int, dx: float, dy: float, interior: np.ndarray
) -> csr_matrix:
    """Build the 5-point Dirichlet Laplacian for a 2-D Helmholtz problem.

    The exterior of ``interior`` is suppressed by a large diagonal penalty
    so that the small-magnitude eigenpairs returned by ``eigsh`` belong to
    the cavity interior. This is equivalent to a soft Dirichlet boundary
    and converges to a hard Dirichlet boundary as the penalty grows.
    """
    if nx < 3 or ny < 3:
        raise SolverError("nx, ny must be >= 3 for a 5-point stencil", nx=nx, ny=ny)

    Dx = _dirichlet_1d(nx, dx)
    Dy = _dirichlet_1d(ny, dy)
    L = kron(eye(ny, format="csr"), Dx, format="csr") + kron(
        Dy, eye(nx, format="csr"), format="csr"
    )

    # Mask exterior with a large diagonal penalty.
    exterior = (~interior).ravel()
    if exterior.any():
        # Zero out exterior rows/cols, then set diagonal penalty.
        L = L.tolil()
        ext_idx = np.flatnonzero(exterior)
        L[ext_idx, :] = 0.0
        L[:, ext_idx] = 0.0
        for i in ext_idx:
            L[i, i] = -_PENALTY
        L = L.tocsr()
    L.eliminate_zeros()
    return L


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def _curl_h_from_ez(
    ez: np.ndarray, dx: float, dy: float, k: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised H-field magnitudes from a TM E_z mode.

    Returns ``(|H_x|, |H_y|, |H|)``. For TM modes the magnitudes are real;
    we drop the ``i / ω μ`` global phase as it does not affect topology.
    """
    omega = max(k, 1e-6)  # ω = c·k, c=1 in normalised units
    # np.gradient is centred in the interior and one-sided at the edges,
    # which is exactly what we want next to a Dirichlet boundary.
    dez_dy, dez_dx = np.gradient(ez, dy, dx, edge_order=2)
    hx = np.abs(dez_dy) / omega
    hy = np.abs(dez_dx) / omega
    h_mag = np.hypot(hx, hy)
    return hx, hy, h_mag


def solve_cavity_modes(
    geometry: CavityGeometry,
    nx: int = 60,
    ny: int = 60,
    num_modes: int = 12,
    seed: int | None = None,
) -> ModeData:
    """Solve TM eigenmodes of a PEC cavity.

    Parameters
    ----------
    geometry : CavityGeometry
        The cavity descriptor.
    nx, ny : int
        Grid resolution. The grid is rectangular and contains ``nx * ny``
        cells. For circular cavities ``ny = nx`` is enforced.
    num_modes : int
        Number of physical modes to return (after filtering spurious
        zero-eigenvalue artefacts).
    seed : int, optional
        Seed for the ARPACK random initial vector. ``None`` uses unseeded
        randomness which produces tiny run-to-run variation.

    Returns
    -------
    ModeData
        Dict with keys ``geometry``, ``dims``, ``nx``, ``ny``,
        ``num_modes_found``, ``k_values``, ``e_modes``, ``h_modes``,
        ``s_modes``, ``X``, ``Y``, ``interior``.
    """
    if num_modes < 1:
        raise SolverError("num_modes must be >= 1", num_modes=num_modes)

    if geometry.shape == CavityShape.RECTANGULAR:
        w, h = geometry.dims
        X, Y = make_rectangular_grid(w, h, nx, ny)
        dx = w / (nx - 1)
        dy = h / (ny - 1)
        interior = geometry.contains(X, Y)
    elif geometry.shape == CavityShape.CIRCULAR:
        (r,) = geometry.dims
        ny = nx  # square grid for circular cavities
        X, Y, interior = make_circular_grid(r, nx)
        dx = 2 * r / (nx - 1)
        dy = dx
    elif geometry.shape == CavityShape.PHOTONIC_CRYSTAL:
        a, _ = geometry.dims
        w, h = a * 15, a * 10
        X, Y = make_rectangular_grid(w, h, nx, ny)
        dx = w / (nx - 1)
        dy = h / (ny - 1)
        interior = geometry.contains(X, Y)
    else:
        raise GeometryError(
            f"Unsupported shape: {geometry.shape}", shape=geometry.shape
        )

    n_interior = int(interior.sum())
    if n_interior < 4:
        raise SolverError(
            "interior has too few points to support a stencil",
            n_interior=n_interior,
            nx=nx,
            ny=ny,
        )

    log.info(
        "building_laplacian",
        shape=str(geometry.shape.value),
        dims=geometry.dims,
        nx=nx,
        ny=ny,
        interior_points=n_interior,
    )

    L = build_laplacian_2d(nx, ny, dx, dy, interior)

    # Shift-invert mode (sigma=0) targets eigenvalues nearest zero, which
    # for the negative-semidefinite Laplacian are exactly the lowest-|k|
    # physical cavity modes. Far faster than ``which="SM"``.
    k_request = min(num_modes + 2, max(1, n_interior - 2))
    rng = np.random.default_rng(seed) if seed is not None else None
    v0 = rng.normal(size=L.shape[0]) if rng is not None else None
    try:
        k_raw, v = eigsh(
            L,
            k=k_request,
            sigma=0.0,
            which="LM",
            tol=1e-8,
            maxiter=20000,
            v0=v0,
        )
    except Exception as exc:  # ARPACK failure: fall back to SM mode
        log.warning("eigsh_shift_invert_failed", error=str(exc))
        k_raw, v = eigsh(
            L,
            k=k_request,
            which="SM",
            tol=1e-6,
            maxiter=20000,
            v0=v0,
        )

    # L is negative-semidefinite, eigenvalues = -k². We want k = sqrt(-λ).
    k_squared = -k_raw
    k_values = np.sqrt(np.maximum(k_squared, 0.0))

    # Sort ascending (fundamental mode first) and drop spurious zero-modes
    # plus PEC-penalty modes that appear with k² ≈ _PENALTY.
    sort = np.argsort(k_values)
    k_values = k_values[sort]
    v = v[:, sort]
    physical = (k_values > 1e-6) & (k_values < np.sqrt(_PENALTY) * 0.5)
    valid_idx = np.flatnonzero(physical)[:num_modes].tolist()

    if not valid_idx:
        raise SolverError(
            "no physical modes found — increase num_modes or grid resolution",
            requested=num_modes,
            k_values=k_values.tolist(),
        )

    log.info(
        "eigsh_complete",
        requested_modes=num_modes,
        found_modes=len(valid_idx),
        k_values=[float(x) for x in k_values[valid_idx][:3]],
    )

    e_modes: dict[str, dict] = {}
    h_modes: dict[str, dict] = {}
    s_modes: dict[str, dict] = {}

    interior_2d = interior  # already shape (ny, nx)
    for count, i in enumerate(valid_idx):
        kk = float(k_values[i])
        e_map = v[:, i].reshape(ny, nx)
        e_map = np.where(interior_2d, e_map, 0.0)
        # Normalise sign so the mode has a definite orientation.
        peak = e_map.flat[np.argmax(np.abs(e_map))]
        if peak < 0:
            e_map = -e_map
        hx, hy, h_mag = _curl_h_from_ez(e_map, dx, dy, kk)
        # H is non-zero on the boundary in general; no zeroing.
        s_map = np.abs(e_map) * h_mag

        e_modes[f"mode_{count}"] = {
            "k": kk,
            "wavelength": float(2 * np.pi / kk),
            "field": e_map.tolist(),
            "nx": nx,
            "ny": ny,
        }
        h_modes[f"mode_{count}"] = {
            "k": kk,
            "wavelength": float(2 * np.pi / kk),
            "field": h_mag.tolist(),
            "hx": hx.tolist(),
            "hy": hy.tolist(),
            "nx": nx,
            "ny": ny,
        }
        s_modes[f"mode_{count}"] = {
            "k": kk,
            "field": s_map.tolist(),
            "nx": nx,
            "ny": ny,
        }

    result: ModeData = {  # type: ignore[valid-type, assignment]
        "geometry": str(geometry.shape.value),
        "dims": geometry.dims,
        "nx": nx,
        "ny": ny,
        "num_modes_found": len(e_modes),
        "k_values": [float(k_values[i]) for i in valid_idx],
        "e_modes": e_modes,
        "h_modes": h_modes,
        "s_modes": s_modes,
        "X": X.tolist(),
        "Y": Y.tolist(),
        "interior": interior.tolist(),
    }
    return result


# ---------------------------------------------------------------------------
# Analytic reference (used by tests)
# ---------------------------------------------------------------------------


def rectangular_analytic_k(
    w: float, h: float, num_modes: int = 6
) -> list[float]:
    """Closed-form ascending eigen-wavenumbers of a rectangular PEC cavity.

    For a TM_mn mode of a perfect rectangular cavity the eigenvalues are

    .. math::

       k_{mn}^2 = \\left(\\frac{m \\pi}{w}\\right)^2
                 + \\left(\\frac{n \\pi}{h}\\right)^2,
       \\quad m, n \\in \\{1, 2, 3, \\ldots\\}.

    Returns the first ``num_modes`` values sorted ascending.
    """
    if w <= 0 or h <= 0 or num_modes < 1:
        raise GeometryError("w,h>0 and num_modes>=1", w=w, h=h, num_modes=num_modes)
    # Generate enough candidate (m,n) so we can pick the smallest num_modes.
    span = max(4, int(np.ceil(np.sqrt(num_modes))) + 2)
    ms = np.arange(1, span + 1)
    ns = np.arange(1, span + 1)
    mm, nn = np.meshgrid(ms, ns, indexing="ij")
    ks = np.sqrt((mm * np.pi / w) ** 2 + (nn * np.pi / h) ** 2)
    return sorted(ks.ravel().tolist())[:num_modes]


# ---------------------------------------------------------------------------
# Wave superposition (kept for backwards compatibility)
# ---------------------------------------------------------------------------


@dataclass
class EMWave:
    """A plane wave for injection into a cavity."""

    amplitude: complex
    kx: float
    ky: float
    phase: float = 0.0
    omega: float = 1.0

    @property
    def k(self) -> float:
        """Total wave-number magnitude."""
        return float(np.sqrt(self.kx**2 + self.ky**2))

    def field_at(
        self, x: np.ndarray, y: np.ndarray, t: float = 0.0
    ) -> np.ndarray:
        """Evaluate the plane-wave field at positions and time."""
        phase = self.kx * x + self.ky * y - self.omega * t + self.phase
        return self.amplitude * np.exp(1j * phase)


class WaveSuperposer:
    """Superpose multiple plane waves and eigenmodes in a cavity."""

    def __init__(self, geometry: CavityGeometry, mode_data: dict) -> None:
        self.geometry = geometry
        self.mode_data = mode_data
        self.e_waves: list[EMWave] = []
        self.h_waves: list[EMWave] = []
        self.active_mode_idx: int | None = None
        self.e_mode_amp: complex = 1.0 + 0j
        self.h_mode_amp: complex = 1.0 + 0j

    def add_e_mode(
        self, mode_idx: int, amplitude: complex = 1.0 + 0j
    ) -> WaveSuperposer:
        """Add an E-field eigenmode to the superposition."""
        modes = self.mode_data["e_modes"]
        key = f"mode_{mode_idx}"
        if key not in modes:
            raise KeyError(f"Mode {mode_idx} not found in e_modes")
        self.active_mode_idx = mode_idx
        self.e_mode_amp = amplitude
        return self

    def add_h_mode(
        self, mode_idx: int, amplitude: complex = 1.0 + 0j
    ) -> WaveSuperposer:
        """Add an H-field eigenmode to the superposition."""
        modes = self.mode_data["h_modes"]
        key = f"mode_{mode_idx}"
        if key not in modes:
            raise KeyError(f"Mode {mode_idx} not found in h_modes")
        self.h_mode_amp = amplitude
        return self

    def add_e_wave(
        self, amplitude: complex, angle: float, wavelength: float, phase: float = 0.0
    ) -> WaveSuperposer:
        """Add a custom E-field plane wave."""
        k = 2 * np.pi / wavelength
        self.e_waves.append(
            EMWave(
                amplitude=amplitude,
                kx=k * np.cos(angle),
                ky=k * np.sin(angle),
                phase=phase,
                omega=k,
            )
        )
        return self

    def add_h_wave(
        self, amplitude: complex, angle: float, wavelength: float, phase: float = 0.0
    ) -> WaveSuperposer:
        """Add a custom H-field plane wave."""
        k = 2 * np.pi / wavelength
        self.h_waves.append(
            EMWave(
                amplitude=amplitude,
                kx=k * np.cos(angle),
                ky=k * np.sin(angle),
                phase=phase,
                omega=k,
            )
        )
        return self

    def e_field_at(
        self, X: np.ndarray, Y: np.ndarray, t: float = 0.0
    ) -> np.ndarray:
        """Total E-field at grid positions and time."""
        total = np.zeros_like(X, dtype=complex)
        if self.active_mode_idx is not None:
            mode_key = f"mode_{self.active_mode_idx}"
            e_map = np.array(self.mode_data["e_modes"][mode_key]["field"])
            total = total + self.e_mode_amp * e_map
        for wave in self.e_waves:
            total = total + wave.field_at(X, Y, t)
        return total

    def h_field_at(
        self, X: np.ndarray, Y: np.ndarray, t: float = 0.0
    ) -> np.ndarray:
        """Total H-field at grid positions and time."""
        total = np.zeros_like(X, dtype=complex)
        if self.active_mode_idx is not None:
            mode_key = f"mode_{self.active_mode_idx}"
            h_map = np.array(self.mode_data["h_modes"][mode_key]["field"])
            total = total + self.h_mode_amp * h_map
        for wave in self.h_waves:
            total = total + wave.field_at(X, Y, t)
        return total

    def poynting_vector(
        self, X: np.ndarray, Y: np.ndarray, t: float = 0.0
    ) -> np.ndarray:
        """Poynting vector magnitude ``|S| = |E| · |H|``."""
        E = self.e_field_at(X, Y, t)
        H = self.h_field_at(X, Y, t)
        return np.abs(E) * np.abs(H)

    def coupled_field_at(
        self, X: np.ndarray, Y: np.ndarray, t: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(E, H)`` coupled fields together."""
        return self.e_field_at(X, Y, t), self.h_field_at(X, Y, t)
