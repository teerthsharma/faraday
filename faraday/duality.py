# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.duality — Topological E ⇄ H duality of TM cavity modes.

For a TM mode, E = E_z solves -ΔE = k²E with E = 0 on the PEC wall, and the
in-plane magnetic field is H = (∂_y E, -∂_x E) / k: ∇E rotated by 90°, tangent
to the wall. Zeros of H are critical points of E, and they carry integer
winding numbers (vortex charge: +1 at extrema, -1 at saddles). Three exact
relations tie the topology of E to that of H:

1. Domain law: inside every nodal domain Ω of E the net vortex charge of H is
   the Euler characteristic χ(Ω) (Poincaré-Hopf; ∇E is normal and one-signed
   on ∂Ω by the Hopf boundary lemma).
2. Bulk-boundary law: the total charge of H in the cavity D is

       Q = χ(D) + N_wall / 2

   where N_wall is the number of points where nodal lines of E meet the wall
   (each acts as a boundary zero of index -1/2) and χ(D) = 1 - #holes.
3. Ampère-Maxwell: the circulation of H around Ω is k ∫_Ω E dA, so each
   vortex's chirality is the sign of E in its domain.

Consequences: the bulk H topology is fixed by 1-D wall data, and in generic
(symmetry-broken) cavities the charge of a mode changes only when a nodal line
attaches to or detaches from the wall (N_wall changes by 2).

Prior art: (1) and (3) are classical (Morse 1929; Pugh 1968; Stokes). Close
analogues of (2) exist for Neumann eigenfunctions on polygons (Judge &
Mondal, Comm. PDE 2022) and for Steklov eigenfunctions (Battaglia, Pistoia &
Provenzano, 2024); this module states and tests it for Dirichlet TM cavities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from faraday._types import ModeData


def vortex_charges(hx: np.ndarray, hy: np.ndarray, interior: np.ndarray) -> np.ndarray:
    """Winding number of H around every grid plaquette.

    Each edge's wrapped angle difference is computed once and enters its two
    plaquettes with opposite signs, so the sum over any region telescopes
    exactly to the winding along the region's boundary (discrete Stokes). This
    stays correct when a zero lies on a grid line and an edge difference is
    exactly ±π, where per-plaquette wrapping double-counts.

    Returns:
        int array of shape ``(ny-1, nx-1)``; 0 on plaquettes not fully interior.
    """
    a = np.arctan2(hy, hx)
    ex = _wrap(a[:, 1:] - a[:, :-1])  # edge (j, i) → (j, i+1)
    ey = _wrap(a[1:, :] - a[:-1, :])  # edge (j, i) → (j+1, i)
    w = (ex[:-1, :] + ey[:, 1:] - ex[1:, :] - ey[:, :-1]) / (2 * np.pi)
    return np.where(_plaquettes_inside(interior), np.rint(w), 0).astype(int)


def euler_characteristic(mask: np.ndarray) -> int:
    """χ = connected components - holes of a 2-D region (4-connected)."""
    components = ndimage.label(mask)[1]
    holes = ndimage.label(ndimage.binary_fill_holes(mask) & ~mask)[1]
    return int(components - holes)


def nodal_domains(
    e: np.ndarray, interior: np.ndarray, rel_tol: float = 1e-3
) -> list[tuple[int, np.ndarray]]:
    """Nodal domains of E as ``(sign, mask)`` pairs.

    Pixels with ``|E| <= rel_tol * max|E|`` are treated as nodal. A finite
    tolerance keeps same-sign domains that only touch at a singular nodal
    point (e.g. the centre of a TM31 mode, where E ~ r³) from being glued.
    """
    t = rel_tol * np.abs(e).max()
    domains = []
    for sign, region in ((1, (e > t) & interior), (-1, (e < -t) & interior)):
        labels, n = ndimage.label(region)
        domains += [(sign, labels == i) for i in range(1, n + 1)]
    return domains


def wall_nodal_count(e: np.ndarray, interior: np.ndarray, rel_tol: float = 1e-5) -> int:
    """N_wall: points where nodal lines of E meet the wall, over all walls.

    Each boundary component (outer wall and one per hole) is walked as a loop
    of interior pixels touching the PEC region, ordered by angle about the
    loop's centroid (exact for star-shaped walls), and sign changes of E are
    counted.
    """
    edge = interior & ~ndimage.binary_erosion(interior, border_value=0)
    labels, n = ndimage.label(edge, structure=np.ones((3, 3)))
    t = rel_tol * np.abs(e).max()
    total = 0
    for i in range(1, n + 1):
        yy, xx = np.nonzero(labels == i)
        order = np.argsort(np.arctan2(yy - yy.mean(), xx - xx.mean()))
        v = e[yy[order], xx[order]]
        s = np.sign(v[np.abs(v) > t])
        total += int(np.sum(s != np.roll(s, 1)))
    return total


@dataclass(frozen=True)
class NodalDomain:
    """One nodal domain Ω of E and the H topology inside it."""

    sign: int  # sign of E in Ω = chirality of the H circulation
    area: float
    euler: int  # χ(Ω)
    charge: int  # net vortex charge of H inside Ω
    circulation: float  # ∮_∂Ω H·dl = k ∫_Ω E dA


@dataclass(frozen=True)
class ModeTopology:
    """Integer topological signature of one TM mode, E side and H side."""

    k: float
    euler_cavity: int  # χ(D)
    n_wall: int  # N_wall (E side, boundary)
    charge: int  # Q, net H vortex charge (H side, bulk)
    vortices: int  # +1 zeros of H
    antivortices: int  # -1 zeros of H
    domains: tuple[NodalDomain, ...]

    @property
    def n_domains(self) -> int:
        return len(self.domains)

    @property
    def boundary_law_holds(self) -> bool:
        """Q = χ(D) + N_wall / 2."""
        return 2 * self.charge == 2 * self.euler_cavity + self.n_wall

    @property
    def domain_law_holds(self) -> bool:
        """charge(Ω) = χ(Ω) for every nodal domain."""
        return all(d.charge == d.euler for d in self.domains)

    def signature(self) -> np.ndarray:
        """Fixed-length invariant vector: an embedding that carries topology.

        ``[k, χ(D), N_wall, n_domains, Σχ(Ω), vortices, antivortices, Q,
        sign balance]``. Everything but ``k`` is an integer and scale-invariant.
        """
        return np.array(
            [
                self.k,
                self.euler_cavity,
                self.n_wall,
                self.n_domains,
                sum(d.euler for d in self.domains),
                self.vortices,
                self.antivortices,
                self.charge,
                sum(d.sign for d in self.domains),
            ],
            dtype=float,
        )


def mode_topology(mode_data: ModeData, mode_index: int = 0) -> ModeTopology:
    """Measure the topological signature of one solved mode.

    Args:
        mode_data: output of :func:`faraday.solve_cavity_modes` or
            :func:`faraday.em_solver.solve_interior_modes`
        mode_index: which mode (0 = fundamental)
    """
    key = f"mode_{mode_index}"
    interior = np.array(mode_data["interior"], dtype=bool)
    X = np.array(mode_data["X"])
    Y = np.array(mode_data["Y"])
    cell = float((X[0, 1] - X[0, 0]) * (Y[1, 0] - Y[0, 0]))
    k = float(mode_data["k_values"][mode_index])
    e = np.array(mode_data["e_modes"][key]["field"])
    hx = np.array(mode_data["h_modes"][key]["hx"])
    hy = np.array(mode_data["h_modes"][key]["hy"])

    w = vortex_charges(hx, hy, interior)
    domains = tuple(
        NodalDomain(
            sign=sign,
            area=float(mask.sum() * cell),
            euler=euler_characteristic(mask),
            charge=int(w[_plaquettes_inside(mask)].sum()),
            circulation=float(k * e[mask].sum() * cell),
        )
        for sign, mask in nodal_domains(e, interior)
    )
    return ModeTopology(
        k=k,
        euler_cavity=euler_characteristic(interior),
        n_wall=wall_nodal_count(e, interior),
        charge=int(w.sum()),
        vortices=int((w == 1).sum()),
        antivortices=int((w == -1).sum()),
        domains=domains,
    )


def predict_rectangle_topology(w: float, h: float, num_modes: int) -> list[dict]:
    """Predict mode topology of a w x h rectangle from geometry alone.

    Geometry → analytic ordering of k_mn = π √((m/w)² + (n/h)²) → (m, n) →
    topology: m·n nodal domains, N_wall = 2(m-1) + 2(n-1), Q = m + n - 1 and
    (m-1)(n-1) antivortices (one per interior nodal crossing). No field is
    solved. Predictions can fail only near a level crossing, where ``gap``
    (relative distance to the nearest other level) is small.
    """
    levels = sorted(
        (float(np.pi * np.hypot(m / w, n / h)), m, n)
        for m in range(1, num_modes + 1)
        for n in range(1, num_modes + 1)
    )[:num_modes]
    out = []
    for i, (k, m, n) in enumerate(levels):
        others = [abs(k - lv[0]) for j, lv in enumerate(levels) if j != i]
        out.append(
            {
                "k": k,
                "mn": (m, n),
                "n_domains": m * n,
                "n_wall": 2 * (m - 1) + 2 * (n - 1),
                "charge": m + n - 1,
                "antivortices": (m - 1) * (n - 1),
                "gap": min(others) / k if others else float("inf"),
            }
        )
    return out


def discover_coupling(topologies: list[ModeTopology]) -> dict[str, float]:
    """Learn the linear E → H topological coupling from data.

    Least-squares fit of the H-side bulk charge Q on E-side invariants
    ``[1, χ(D), N_wall, n_domains]``. If the bulk-boundary law is the true
    coupling, the fit recovers ``χ(D): 1, N_wall: 0.5`` with zero residual and
    puts no weight on the intercept or on n_domains (a distractor).
    """
    A = np.array(
        [[1.0, t.euler_cavity, t.n_wall, t.n_domains] for t in topologies]
    )
    q = np.array([t.charge for t in topologies], dtype=float)
    coef, *_ = np.linalg.lstsq(A, q, rcond=None)
    residual = float(np.max(np.abs(A @ coef - q))) if len(q) else 0.0
    return {
        "intercept": float(coef[0]),
        "euler_cavity": float(coef[1]),
        "n_wall": float(coef[2]),
        "n_domains": float(coef[3]),
        "max_residual": residual,
    }


def _wrap(d: np.ndarray) -> np.ndarray:
    return (d + np.pi) % (2 * np.pi) - np.pi


def _plaquettes_inside(mask: np.ndarray) -> np.ndarray:
    return mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, 1:] & mask[1:, :-1]
