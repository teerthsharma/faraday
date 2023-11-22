"""
faraday.em_solver — FDFD Cavity Solver for E and H Fields

Computes TE and TM eigenmodes of a hollow PEC cavity.
For TM modes:  ∇²E_z + k²E_z = 0
For TE modes:  ∇²H_z + k²H_z = 0

Both E_z and H_z share the same eigenvalue k — they are COUPLED.
This coupling is what the God Tensor learns to capture.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class CavityShape(Enum):
    RECTANGULAR = "rectangular"
    CIRCULAR = "circular"


@dataclass
class CavityGeometry:
    shape: CavityShape
    dims: Tuple[float, ...]  # (w, h) for rect, (r,) for circle
    boundary_conditions: str = "pec"  # perfect electric conductor

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.shape == CavityShape.RECTANGULAR:
            w, h = self.dims
            return (np.abs(x) < w / 2) & (np.abs(y) < h / 2)
        elif self.shape == CavityShape.CIRCULAR:
            r, = self.dims
            return (x**2 + y**2) < r**2

    def interior_mask(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self.contains(X, Y)


def make_rectangular_grid(w: float, h: float, nx: int, ny: int):
    x = np.linspace(-w / 2, w / 2, nx)
    y = np.linspace(-h / 2, h / 2, ny)
    X, Y = np.meshgrid(x, y)
    return X, Y


def make_circular_grid(r: float, nx: int):
    x = np.linspace(-r, r, nx)
    y = np.linspace(-r, r, nx)
    X, Y = np.meshgrid(x, y)
    mask = (X**2 + Y**2) <= r**2
    return X, Y, mask


def build_laplacian_2d(
    nx: int, ny: int, dx: float, dy: float, interior: np.ndarray
) -> np.ndarray:
    """Build 5-point stencil Laplacian for 2D Helmholtz problem."""
    n = nx * ny
    row_idx, col_idx, data = [], [], []

    for j in range(ny):
        for i in range(nx):
            idx = i + j * nx
            if not interior[j, i]:
                row_idx.append(idx); col_idx.append(idx); data.append(1.0)
                continue
            row_idx.append(idx); col_idx.append(idx)
            data.append(-2.0 / dx**2 - 2.0 / dy**2)
            if i + 1 < nx:
                row_idx.append(idx); col_idx.append(idx + 1); data.append(1.0 / dx**2)
            if i - 1 >= 0:
                row_idx.append(idx); col_idx.append(idx - 1); data.append(1.0 / dx**2)
            if j + 1 < ny:
                row_idx.append(idx); col_idx.append(idx + nx); data.append(1.0 / dy**2)
            if j - 1 >= 0:
                row_idx.append(idx); col_idx.append(idx - nx); data.append(1.0 / dy**2)

    from scipy.sparse import csr_matrix
    return csr_matrix((data, (row_idx, col_idx)), shape=(n, n))


def solve_cavity_modes(
    geometry: CavityGeometry,
    nx: int = 60, ny: int = 60,
    num_modes: int = 12,
) -> Dict:
    """
    Solve for E and H eigenmodes of the cavity.

    Both share the same wave number k and spatial pattern structure —
    they are linked by Maxwell's equations. We solve for E_z (TM modes)
    and derive H_z from the curl relation.

    Returns dict with:
      - k_values: list of wave numbers (shared by E and H)
      - e_modes: dict of E_z field patterns
      - h_modes: dict of H_z field patterns (derived)
      - geometry: cavity shape description
      - dims: physical dimensions
    """
    if geometry.shape == CavityShape.RECTANGULAR:
        w, h = geometry.dims
        X, Y = make_rectangular_grid(w, h, nx, ny)
        dx, dy = w / (nx - 1), h / (ny - 1)
        interior = geometry.contains(X, Y)
    elif geometry.shape == CavityShape.CIRCULAR:
        r, = geometry.dims
        X, Y, interior = make_circular_grid(r, nx)
        dx = dy = 2 * r / (nx - 1)
    else:
        raise NotImplementedError(f"Shape {geometry.shape} not yet supported")

    L = build_laplacian_2d(nx, ny, dx, dy, interior)
    n_interior = interior.sum()

    from scipy.sparse.linalg import eigsh
    k_raw, v = eigsh(L, k=min(num_modes, max(1, n_interior - 1)), which="SM", sigma=0.0)
    k_squared = -k_raw
    k_values = np.sqrt(np.maximum(k_squared, 0))

    # Filter spurious PEC Dirichlet zero-modes
    valid_idx = [i for i, kk in enumerate(k_values) if kk > 1e-6]

    e_modes = {}
    h_modes = {}
    for count, i in enumerate(valid_idx):
        kk = k_values[i]
        e_map = np.zeros((ny, nx), dtype=complex)
        hx_map = np.zeros((ny, nx), dtype=complex)
        hy_map = np.zeros((ny, nx), dtype=complex)
        for j in range(ny):
            for ii in range(nx):
                idx = ii + j * nx
                if interior[j, ii]:
                    e_map[j, ii] = v[idx, i]
                    # H_x and H_y (transverse) derived from E_z via Maxwell's curl:
                    # H_x = (i/ωμ) ∂E_z/∂y,  H_y = -(i/ωμ) ∂E_z/∂x
                    if ii > 0 and ii < nx - 1 and j > 0 and j < ny - 1:
                        dex_dy = (v[idx + nx, i] - v[idx - nx, i]) / (2 * dy)
                        dex_dx = (v[idx + 1, i] - v[idx - 1, i]) / (2 * dx)
                        omega = kk if kk > 1e-6 else 1.0
                        hx_map[j, ii] = 1j * dex_dy / omega
                        hy_map[j, ii] = -1j * dex_dx / omega
                    else:
                        hx_map[j, ii] = 0
                        hy_map[j, ii] = 0

        e_modes[f"mode_{count}"] = {
            "k": float(kk),
            "wavelength": float(2 * np.pi / kk) if kk > 0 else float("inf"),
            "field": e_map.real.tolist(),
            "nx": nx, "ny": ny,
        }
        h_modes[f"mode_{count}"] = {
            "k": float(kk),
            "wavelength": float(2 * np.pi / kk) if kk > 0 else float("inf"),
            "field": hx_map.imag.tolist(),  # eigsh is real, 1j*real is imaginary
            "hx": hx_map.imag.tolist(),
            "hy": hy_map.imag.tolist(),
            "nx": nx, "ny": ny,
        }

    return {
        "geometry": str(geometry.shape.value),
        "dims": geometry.dims,
        "nx": nx, "ny": ny,
        "num_modes_found": len(e_modes),
        "k_values": [float(kk) for kk in k_values[valid_idx]],
        "e_modes": e_modes,
        "h_modes": h_modes,
        "X": X.tolist(),
        "Y": Y.tolist(),
        "interior": interior.tolist(),
    }


@dataclass
class EMWave:
    amplitude: complex
    kx: float
    ky: float
    phase: float = 0.0
    omega: float = 1.0

    @property
    def k(self) -> float:
        return np.sqrt(self.kx**2 + self.ky**2)

    def field_at(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        phase = self.kx * x + self.ky * y - self.omega * t + self.phase
        return self.amplitude * np.exp(1j * phase)


class WaveSuperposer:
    """
    Superposes multiple EM waves in a cavity.
    E_total = Σ a_n * E_n (eigenmode expansion)
             + Σ w_m (custom plane wave injection)

    Both E and H fields are superposed together.
    """

    def __init__(self, geometry: CavityGeometry, mode_data: Dict):
        self.geometry = geometry
        self.mode_data = mode_data
        self.e_waves: List[EMWave] = []
        self.h_waves: List[EMWave] = []
        self.active_mode_idx: Optional[int] = None

    def add_e_mode(self, mode_idx: int, amplitude: complex = 1.0 + 0j) -> "WaveSuperposer":
        """Add an E-field eigenmode to the superposition."""
        modes = self.mode_data["e_modes"]
        key = f"mode_{mode_idx}"
        if key not in modes:
            raise ValueError(f"Mode {mode_idx} not found")
        self.active_mode_idx = mode_idx
        self.e_mode_amp = amplitude
        return self

    def add_h_mode(self, mode_idx: int, amplitude: complex = 1.0 + 0j) -> "WaveSuperposer":
        """Add an H-field eigenmode to the superposition."""
        modes = self.mode_data["h_modes"]
        key = f"mode_{mode_idx}"
        if key not in modes:
            raise ValueError(f"Mode {mode_idx} not found")
        self.h_mode_amp = amplitude
        return self

    def add_e_wave(self, amplitude: complex, angle: float, wavelength: float, phase: float = 0.0) -> "WaveSuperposer":
        k = 2 * np.pi / wavelength
        self.e_waves.append(EMWave(amplitude=amplitude, kx=k * np.cos(angle), ky=k * np.sin(angle), phase=phase, omega=k))
        return self

    def add_h_wave(self, amplitude: complex, angle: float, wavelength: float, phase: float = 0.0) -> "WaveSuperposer":
        k = 2 * np.pi / wavelength
        self.h_waves.append(EMWave(amplitude=amplitude, kx=k * np.cos(angle), ky=k * np.sin(angle), phase=phase, omega=k))
        return self

    def e_field_at(self, X: np.ndarray, Y: np.ndarray, t: float = 0.0) -> np.ndarray:
        total = np.zeros_like(X, dtype=complex)
        # Eigenmode
        if hasattr(self, "active_mode_idx") and self.active_mode_idx is not None:
            mode_key = f"mode_{self.active_mode_idx}"
            e_map = np.array(self.mode_data["e_modes"][mode_key]["field"])
            total += self.e_mode_amp * e_map
        # Custom waves
        for wave in self.e_waves:
            total += wave.field_at(X, Y, t)
        return total

    def h_field_at(self, X: np.ndarray, Y: np.ndarray, t: float = 0.0) -> np.ndarray:
        total = np.zeros_like(X, dtype=complex)
        # H eigenmode
        if hasattr(self, "h_mode_amp"):
            mode_key = f"mode_{self.active_mode_idx or 0}"
            h_map = np.array(self.mode_data["h_modes"][mode_key]["field"])
            total += self.h_mode_amp * h_map
        # Custom waves
        for wave in self.h_waves:
            total += wave.field_at(X, Y, t)
        return total

    def poynting_vector(self, X: np.ndarray, Y: np.ndarray, t: float = 0.0) -> np.ndarray:
        """
        Poynting vector S = E x H (cross product).
        In 2D TE/TM approximation: S_z ~ E_x * H_y - E_y * H_x.
        Here we use the real magnitude as a scalar proxy.
        """
        E = self.e_field_at(X, Y, t)
        H = self.h_field_at(X, Y, t)
        # |S| = |E| * |H| (simplified 2D scalar case)
        return np.abs(E) * np.abs(H)

    def coupled_field_at(self, X: np.ndarray, Y: np.ndarray, t: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Return (E, H) coupled fields together."""
        return self.e_field_at(X, Y, t), self.h_field_at(X, Y, t)
