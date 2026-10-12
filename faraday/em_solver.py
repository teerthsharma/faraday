# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.em_solver — FDFD Cavity Solver for E and H Fields

Computes TE and TM eigenmodes of a hollow PEC cavity.
For TM modes:  ∇²E_z + k²E_z = 0
For TE modes:  ∇²H_z + k²H_z = 0

Both E_z and H_z share the same eigenvalue k — they are COUPLED.
This coupling is what the God Tensor learns to capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from faraday.exceptions import SolverError
from faraday.logging import get_logger

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

    from faraday._types import ModeData

log = get_logger(__name__)


class CavityShape(Enum):
    RECTANGULAR = "rectangular"
    CIRCULAR = "circular"


@dataclass
class CavityGeometry:
    """Cavity geometry descriptor.

    Attributes
    ----------
    shape : CavityShape
        The shape of the cavity (rectangular or circular).
    dims : tuple[float, ...]
        Dimensions: ``(width, height)`` for rectangular, ``(radius,)`` for circular.
    boundary_conditions : str
        Boundary condition type. Only ``"pec"`` (perfect electric conductor) is
        currently supported.
    """

    shape: CavityShape
    dims: tuple[float, ...]
    boundary_conditions: str = "pec"

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return a boolean mask for points inside the cavity.

        Parameters
        ----------
        x, y : np.ndarray
            Coordinate arrays. Can be 1D or 2D (meshgrid output).

        Returns
        -------
        np.ndarray
            Boolean array. For 1D ``x, y`` of the same length returns a 1D
            mask (element-wise AND after broadcasting). For 2D meshgrid input
            returns the corresponding 2D mask.
        """
        if self.shape == CavityShape.RECTANGULAR:
            w, h = self.dims
            return (np.abs(x) < w / 2) & (np.abs(y) < h / 2)
        elif self.shape == CavityShape.CIRCULAR:
            (r,) = self.dims
            return (x**2 + y**2) < r**2
        msg = f"Unsupported cavity shape: {self.shape}"
        raise NotImplementedError(msg)

    def interior_mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Alias for :meth:`contains` with meshgrid inputs."""
        return self.contains(X, Y)


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def make_rectangular_grid(
    w: float, h: float, nx: int, ny: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create a centred meshgrid for a rectangular cavity.

    Parameters
    ----------
    w, h : float
        Physical width and height of the cavity.
    nx, ny : int
        Number of grid points in the x and y directions.

    Returns
    -------
    X, Y : np.ndarray
        2D coordinate arrays of shape ``(ny, nx)``.
    """
    x = np.linspace(-w / 2, w / 2, nx)
    y = np.linspace(-h / 2, h / 2, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y


def make_circular_grid(
    r: float, nx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a centred meshgrid for a circular cavity.

    Parameters
    ----------
    r : float
        Physical radius of the cavity.
    nx : int
        Number of grid points along each axis (square grid).

    Returns
    -------
    X, Y : np.ndarray
        2D coordinate arrays of shape ``(nx, nx)``.
    interior : np.ndarray
        Boolean mask, ``True`` for points strictly inside the circle. Points on
        the circle itself are PEC boundary nodes (E_z = 0).
    """
    coord = np.linspace(-r, r, nx)
    X, Y = np.meshgrid(coord, coord)
    interior = r**2 > (X**2 + Y**2)
    return X, Y, interior


# ---------------------------------------------------------------------------
# Laplacian builder
# ---------------------------------------------------------------------------


def build_laplacian_2d(
    nx: int, ny: int, dx: float, dy: float, interior: np.ndarray
) -> csr_matrix:  # type: ignore[name-defined]
    """Build the 5-point Dirichlet Laplacian restricted to the interior nodes.

    The Laplacian applies the discrete approximation:

        ∇²f ≈ (f_{i+1,j} - 2f_{i,j} + f_{i-1,j}) / dx²
             + (f_{i,j+1} - 2f_{i,j} + f_{i,j-1}) / dy²

    PEC (Dirichlet) boundary: every non-interior node is pinned to f = 0, so
    its column contributes nothing and it is eliminated from the system. The
    result is symmetric negative-definite, which is what ``eigsh`` assumes.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    dx, dy : float
        Grid spacing in x and y.
    interior : np.ndarray
        Boolean mask of shape ``(ny, nx)``, ``True`` for interior (cavity) points.

    Returns
    -------
    csr_matrix
        Sparse matrix of shape ``(n_int, n_int)`` with ``n_int = interior.sum()``.
        Unknowns are ordered as ``X[interior]`` (row-major over the grid).
    """
    from scipy.sparse import csr_matrix

    interior = np.asarray(interior, dtype=bool)
    if interior.shape != (ny, nx):
        msg = f"interior mask shape {interior.shape} != (ny, nx) = {(ny, nx)}"
        raise ValueError(msg)

    n_int = int(interior.sum())
    index = np.full(interior.shape, -1, dtype=np.int64)
    index[interior] = np.arange(n_int)

    diag = np.arange(n_int)
    rows = [diag]
    cols = [diag]
    data = [np.full(n_int, -2.0 / dx**2 - 2.0 / dy**2)]

    # Couple every pair of neighbouring interior nodes (both directions → symmetric)
    for p, q, step in (
        (index[:, :-1], index[:, 1:], dx),
        (index[:-1, :], index[1:, :], dy),
    ):
        both = (p >= 0) & (q >= 0)
        a, b = p[both], q[both]
        w = np.full(a.size, 1.0 / step**2)
        rows += [a, b]
        cols += [b, a]
        data += [w, w]

    return csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_int, n_int),
    )


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def solve_cavity_modes(
    geometry: CavityGeometry,
    nx: int = 60,
    ny: int = 60,
    num_modes: int = 12,
    seed: int | None = None,
) -> ModeData:
    """Solve for E and H eigenmodes of the cavity.

    Both share the same wave number *k* and spatial pattern structure —
    they are linked by Maxwell's equations. We solve for E_z (TM modes)
    and derive H_x, H_y (transverse components) from the curl relation.

    .. math::

        H_x = \\frac{{i}}{{\\omega\\mu}} \\frac{{\\partial E_z}}{{\\partial y}}
        \\quad
        H_y = -\\frac{{i}}{{\\omega\\mu}} \\frac{{\\partial E_z}}{{\\partial x}}

    The Poynting vector magnitude ``|S| = |E| | |H|`` is used as the
    physically meaningful H-field proxy for topological analysis.

    Parameters
    ----------
    geometry : CavityGeometry
        The cavity shape and dimensions.
    nx, ny : int
        Grid resolution. Circular cavities use a square ``nx x nx`` grid and
        ignore ``ny``.
    num_modes : int
        Maximum number of eigenmodes to compute. Modes are returned in
        ascending ``k``: ``mode_0`` is the fundamental (lowest-frequency) mode.
    seed : int, optional
        Random seed for the ARPACK eigenvalue solver (eigsh). When None
        (default), the solver uses unseeded random initialization which
        produces slight eigenvalue variation across runs. Pass an explicit
        seed for reproducible results. Internally a random initial guess
        vector is generated from the seed and passed via the ``v0``
        parameter of ``eigsh``.

    Returns
    -------
    ModeData
        Dict with keys: ``geometry``, ``dims``, ``nx``, ``ny``, ``num_modes_found``,
        ``k_values``, ``e_modes``, ``h_modes``, ``s_modes``, ``X``, ``Y``, ``interior``.
    """
    if geometry.shape == CavityShape.RECTANGULAR:
        w, h = geometry.dims
        X, Y = make_rectangular_grid(w, h, nx, ny)
        dx = w / (nx - 1)
        dy = h / (ny - 1)
        interior = geometry.contains(X, Y)
    elif geometry.shape == CavityShape.CIRCULAR:
        (r,) = geometry.dims
        ny = nx  # square grid; ny is meaningless for a circle
        X, Y, interior = make_circular_grid(r, nx)
        dx = dy = 2 * r / (nx - 1)
    else:
        msg = f"Unsupported shape: {geometry.shape}"
        raise NotImplementedError(msg)

    log.info(
        "building_laplacian",
        shape=str(geometry.shape.value),
        dims=geometry.dims,
        nx=nx,
        ny=ny,
        interior_points=int(interior.sum()),
    )

    L = build_laplacian_2d(nx, ny, dx, dy, interior)
    n_int = L.shape[0]
    n_eig = min(num_modes, n_int - 1)
    if n_eig < 1:
        raise SolverError(
            "too few interior points for an eigenmode",
            n_interior=n_int,
            nx=nx,
            ny=ny,
        )

    # L is symmetric negative-definite with eigenvalues -k². The fundamental
    # modes are the eigenvalues closest to 0, so use shift-invert about
    # sigma=0: it maps them to the largest-magnitude eigenvalues of L⁻¹, which
    # Lanczos resolves in a few iterations. (which="SM" without a shift
    # converges poorly and skips modes.) Seeding v0 makes ARPACK reproducible.
    from scipy.sparse.linalg import eigsh

    v0 = np.random.default_rng(seed).random(n_int) if seed is not None else None
    lam, vecs = eigsh(L, k=n_eig, sigma=0.0, which="LM", v0=v0)

    order = np.argsort(-lam)  # -k² descending  ⇔  k ascending
    k_values = np.sqrt(np.maximum(-lam[order], 0.0))
    vecs = vecs[:, order]

    log.info(
        "eigsh_complete",
        requested_modes=num_modes,
        found_modes=len(k_values),
        k_values=k_values[:3].tolist(),
    )

    e_modes: dict[str, dict] = {}
    h_modes: dict[str, dict] = {}
    s_modes: dict[str, dict] = {}

    c = 1.0  # natural units: c = μ = 1  →  ω = k
    for count, kk in enumerate(k_values):
        e_map = np.zeros((ny, nx))
        e_map[interior] = vecs[:, count]

        # H from E via Maxwell's curl (TM): H_x = (i/ωμ) ∂E_z/∂y,
        # H_y = -(i/ωμ) ∂E_z/∂x. Central differences; the zero padding is the
        # PEC wall. The common factor i is a 90° phase and is dropped.
        e_pad = np.pad(e_map, 1)
        de_dx = (e_pad[1:-1, 2:] - e_pad[1:-1, :-2]) / (2 * dx)
        de_dy = (e_pad[2:, 1:-1] - e_pad[:-2, 1:-1]) / (2 * dy)
        omega = kk * c
        hx_map = np.where(interior, de_dy / omega, 0.0)
        hy_map = np.where(interior, -de_dx / omega, 0.0)

        h_mag = np.hypot(hx_map, hy_map)
        # Poynting vector magnitude: |S| = |E| |H| (energy flux density)
        s_map = np.abs(e_map) * h_mag

        wavelength = float(2 * np.pi / kk)
        e_modes[f"mode_{count}"] = {
            "k": float(kk),
            "wavelength": wavelength,
            "field": e_map.tolist(),
            "nx": nx,
            "ny": ny,
        }
        h_modes[f"mode_{count}"] = {
            "k": float(kk),
            "wavelength": wavelength,
            "field": h_mag.tolist(),
            "hx": hx_map.tolist(),
            "hy": hy_map.tolist(),
            "nx": nx,
            "ny": ny,
        }
        s_modes[f"mode_{count}"] = {
            "k": float(kk),
            "field": s_map.tolist(),
            "nx": nx,
            "ny": ny,
        }

    result: dict[str, Any] = {  # type: ignore[assignment]
        "geometry": str(geometry.shape.value),
        "dims": geometry.dims,
        "nx": nx,
        "ny": ny,
        "num_modes_found": len(e_modes),
        "k_values": [float(kk) for kk in k_values],
        "e_modes": e_modes,
        "h_modes": h_modes,
        "s_modes": s_modes,
        "X": X.tolist(),
        "Y": Y.tolist(),
        "interior": interior.tolist(),
    }
    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Wave superposition
# ---------------------------------------------------------------------------


@dataclass
class EMWave:
    """A plane wave for injection into a cavity.

    Attributes
    ----------
    amplitude : complex
        Complex amplitude of the wave.
    kx, ky : float
        Wave numbers in x and y directions.
    phase : float
        Initial phase offset (radians).
    omega : float
        Angular frequency.
    """

    amplitude: complex
    kx: float
    ky: float
    phase: float = 0.0
    omega: float = 1.0

    @property
    def k(self) -> float:
        """Total wave number magnitude."""
        return np.sqrt(self.kx**2 + self.ky**2)

    def field_at(
        self, x: np.ndarray, y: np.ndarray, t: float = 0.0
    ) -> np.ndarray:
        """Evaluate the wave field at positions and time.

        Parameters
        ----------
        x, y : np.ndarray
            Spatial coordinates.
        t : float
            Time.

        Returns
        -------
        np.ndarray
            Complex field values.
        """
        phase = self.kx * x + self.ky * y - self.omega * t + self.phase
        return self.amplitude * np.exp(1j * phase)


class WaveSuperposer:
    """Superposes multiple EM waves and eigenmodes in a cavity.

    ``E_total = Σ a_n * E_n`` (eigenmode expansion)
    ``+ Σ w_m`` (custom plane wave injection)

    Both E and H fields are superposed together.
    """

    def __init__(self, geometry: CavityGeometry, mode_data: dict) -> None:
        self.geometry = geometry
        self.mode_data = mode_data
        self.e_waves: list[EMWave] = []
        self.h_waves: list[EMWave] = []
        self.active_mode_idx: int | None = None
        self.active_h_mode_idx: int | None = None
        self.e_mode_amp: complex = 1.0 + 0j
        self.h_mode_amp: complex = 1.0 + 0j

    def add_e_mode(
        self, mode_idx: int, amplitude: complex = 1.0 + 0j
    ) -> WaveSuperposer:
        """Add an E-field eigenmode to the superposition."""
        modes = self.mode_data["e_modes"]
        key = f"mode_{mode_idx}"
        if key not in modes:
            msg = f"Mode {mode_idx} not found in e_modes"
            raise KeyError(msg)
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
            msg = f"Mode {mode_idx} not found in h_modes"
            raise KeyError(msg)
        self.active_h_mode_idx = mode_idx
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
        """Compute total E-field at grid positions and time."""
        total = np.zeros_like(X, dtype=complex)
        if self.active_mode_idx is not None:
            mode_key = f"mode_{self.active_mode_idx}"
            e_map = np.array(self.mode_data["e_modes"][mode_key]["field"])
            total += self.e_mode_amp * e_map
        for wave in self.e_waves:
            total += wave.field_at(X, Y, t)
        return total

    def h_field_at(
        self, X: np.ndarray, Y: np.ndarray, t: float = 0.0
    ) -> np.ndarray:
        """Compute total H-field at grid positions and time."""
        total = np.zeros_like(X, dtype=complex)
        if self.active_h_mode_idx is not None:
            mode_key = f"mode_{self.active_h_mode_idx}"
            h_map = np.array(self.mode_data["h_modes"][mode_key]["field"])
            total += self.h_mode_amp * h_map
        for wave in self.h_waves:
            total += wave.field_at(X, Y, t)
        return total

    def poynting_vector(
        self, X: np.ndarray, Y: np.ndarray, t: float = 0.0
    ) -> np.ndarray:
        """Poynting vector magnitude ``|S| = |E| | |H|``."""
        E = self.e_field_at(X, Y, t)
        H = self.h_field_at(X, Y, t)
        return np.abs(E) * np.abs(H)

    def coupled_field_at(
        self, X: np.ndarray, Y: np.ndarray, t: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(E, H)`` coupled fields together."""
        return self.e_field_at(X, Y, t), self.h_field_at(X, Y, t)
