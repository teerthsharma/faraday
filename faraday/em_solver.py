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
from typing import TYPE_CHECKING

import numpy as np

from faraday.logging import get_logger

if TYPE_CHECKING:
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
        Boolean mask, ``True`` for points inside the circle.
    """
    coord = np.linspace(-r, r, nx)
    X, Y = np.meshgrid(coord, coord)
    interior = r**2 >= (X**2 + Y**2)
    return X, Y, interior


# ---------------------------------------------------------------------------
# Laplacian builder
# ---------------------------------------------------------------------------


def build_laplacian_2d(
    nx: int, ny: int, dx: float, dy: float, interior: np.ndarray
) -> np.ndarray:
    """Build the 5-point finite-difference Laplacian for the 2D Helmholtz problem.

    The Laplacian applies the discrete approximation:

        ∇²f ≈ (f_{i+1,j} - 2f_{i,j} + f_{i-1,j}) / dx²
             + (f_{i,j+1} - 2f_{i,j} + f_{i,j-1}) / dy²

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    dx, dy : float
        Grid spacing in x and y.
    interior : np.ndarray
        Boolean mask, ``True`` for interior (cavity) points.

    Returns
    -------
    np.ndarray
        Sparse CSR matrix of shape ``(nx * ny, nx * ny)`` representing the Laplacian.
    """
    n = nx * ny
    row_idx: list[int] = []
    col_idx: list[int] = []
    data: list[float] = []

    for j in range(ny):
        for i in range(nx):
            idx = i + j * nx
            if not interior[j, i]:
                row_idx.append(idx)
                col_idx.append(idx)
                data.append(1.0)
                continue
            row_idx.append(idx)
            col_idx.append(idx)
            data.append(-2.0 / dx**2 - 2.0 / dy**2)
            if i + 1 < nx:
                row_idx.append(idx)
                col_idx.append(idx + 1)
                data.append(1.0 / dx**2)
            if i - 1 >= 0:
                row_idx.append(idx)
                col_idx.append(idx - 1)
                data.append(1.0 / dx**2)
            if j + 1 < ny:
                row_idx.append(idx)
                col_idx.append(idx + nx)
                data.append(1.0 / dy**2)
            if j - 1 >= 0:
                row_idx.append(idx)
                col_idx.append(idx - nx)
                data.append(1.0 / dy**2)

    from scipy.sparse import csr_matrix

    return csr_matrix((data, (row_idx, col_idx)), shape=(n, n))


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------


def solve_cavity_modes(
    geometry: CavityGeometry,
    nx: int = 60,
    ny: int = 60,
    num_modes: int = 12,
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
        Grid resolution.
    num_modes : int
        Maximum number of eigenmodes to compute.

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
        X, Y, interior = make_circular_grid(r, nx)
        dx = 2 * r / (nx - 1)
        dy = 2 * r / (nx - 1)
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
    n_interior = interior.sum()

    from scipy.sparse.linalg import eigsh

    # "LM" = Largest Magnitude eigenvalues of L.
    # L is negative-semidefinite (eigenvalues = -k² ≤ 0), so "LM" gives the
    # most negative eigenvalues → largest k² → highest-frequency modes.
    # These are the dominant structural modes best suited for topological
    # fingerprinting (most loops, nodes, antinodes per unit area).
    #
    # The PEC Dirichlet BC zero-modes (k=0, eigenvalue=0) are the smallest
    # magnitude eigenvalues — correctly excluded by "LM".
    k_raw, v = eigsh(L, k=min(num_modes, max(1, n_interior - 1)), which="LM")
    k_squared = -k_raw
    k_values = np.sqrt(np.maximum(k_squared, 0))

    # Filter spurious PEC Dirichlet zero-modes (k ≈ 0)
    valid_idx = [i for i, kk in enumerate(k_values) if kk > 1e-6]

    log.info(
        "eigsh_complete",
        requested_modes=num_modes,
        found_modes=len(valid_idx),
        k_values=k_values[valid_idx][:3].tolist(),
    )

    e_modes: dict[str, dict] = {}
    h_modes: dict[str, dict] = {}
    s_modes: dict[str, dict] = {}

    for count, i in enumerate(valid_idx):
        kk = k_values[i]
        e_map = np.zeros((ny, nx))
        hx_map = np.zeros((ny, nx))
        hy_map = np.zeros((ny, nx))

        for j in range(ny):
            for ii in range(nx):
                idx = ii + j * nx
                if not interior[j, ii]:
                    continue
                e_val = v[idx, i]
                e_map[j, ii] = e_val
                # H-field from E-field via Maxwell's curl equations (finite-difference)
                if 0 < ii < nx - 1 and 0 < j < ny - 1:
                    dex_dy = (v[idx + nx, i] - v[idx - nx, i]) / (2 * dy)
                    dex_dx = (v[idx + 1, i] - v[idx - 1, i]) / (2 * dx)
                    omega = kk if kk > 1e-6 else 1.0
                    hx_map[j, ii] = abs(dex_dy) / omega
                    hy_map[j, ii] = abs(dex_dx) / omega

        # Poynting vector magnitude: |S| = |E| | |H| (energy flux density)
        h_mag = np.sqrt(hx_map**2 + hy_map**2)
        s_map = np.abs(e_map) * h_mag

        e_modes[f"mode_{count}"] = {
            "k": float(kk),
            "wavelength": float(2 * np.pi / kk) if kk > 0 else float("inf"),
            "field": e_map.tolist(),
            "nx": nx,
            "ny": ny,
        }
        h_modes[f"mode_{count}"] = {
            "k": float(kk),
            "wavelength": float(2 * np.pi / kk) if kk > 0 else float("inf"),
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

    result: dict = {
        "geometry": str(geometry.shape.value),
        "dims": geometry.dims,
        "nx": nx,
        "ny": ny,
        "num_modes_found": len(e_modes),
        "k_values": [float(kk) for kk in k_values[valid_idx]],
        "e_modes": e_modes,
        "h_modes": h_modes,
        "s_modes": s_modes,
        "X": X.tolist(),
        "Y": Y.tolist(),
        "interior": interior.tolist(),
    }
    return result


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
        if hasattr(self, "h_mode_amp"):
            mode_key = f"mode_{self.active_mode_idx or 0}"
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
