# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""Tests for photons in a Higgs background (faraday.higgs)."""

import numpy as np
import pytest

from faraday.duality import euler_characteristic, wall_nodal_count
from faraday.em_solver import solve_interior_modes
from faraday.higgs import (
    box_interior,
    higgs_wall_mass2,
    region_charge,
    trapped,
    vortex_condensate,
)


def _grid(R, n):
    c = np.linspace(-R, R, n)
    return np.meshgrid(c, c)


def _higgs_modes(X, Y, mass2, num_modes):
    return solve_interior_modes(X, Y, box_interior(X), num_modes=num_modes, seed=0, mass2=mass2)


def _field(modes, j):
    return np.array(modes["e_modes"][f"mode_{j}"]["field"])


def test_zero_mass_reduces_to_the_cavity_solver():
    X, Y = _grid(1.0, 41)
    inside = box_interior(X)
    a = solve_interior_modes(X, Y, inside, num_modes=3, seed=0)
    b = solve_interior_modes(X, Y, inside, num_modes=3, seed=0, mass2=np.zeros_like(X))
    np.testing.assert_allclose(a["k_values"], b["k_values"])


def test_uniform_mass_shifts_omega_squared():
    """Uniform condensate: ω² = k² + m², eigenfunctions unchanged."""
    X, Y = _grid(1.0, 41)
    inside = box_interior(X)
    k = np.array(solve_interior_modes(X, Y, inside, num_modes=3, seed=0)["k_values"])
    w = np.array(_higgs_modes(X, Y, np.full_like(X, 9.0), 3)["k_values"])
    np.testing.assert_allclose(w**2, k**2 + 9.0, rtol=1e-8)


def test_pec_wall_is_the_infinite_mass_limit():
    X, Y = _grid(2.0, 161)
    disk = np.hypot(X, Y) < 1.0
    k_pec = solve_interior_modes(X, Y, disk, num_modes=1, seed=0)["k_values"][0]
    w = [_higgs_modes(X, Y, higgs_wall_mass2(disk, m), 1)["k_values"][0] for m in (3, 6, 12)]
    assert w[0] < w[1] < w[2] < k_pec
    # O(1/m) convergence: quadrupling m cuts the gap to about a quarter
    assert k_pec - w[2] < 0.35 * (k_pec - w[0])


@pytest.mark.parametrize(
    "shape",
    [
        lambda X, Y: np.hypot(X, Y) < 1.0,
        lambda X, Y: ((abs(X) < 0.6) & (abs(Y) < 0.6)) | ((abs(X) - 0.6) ** 2 + Y**2 < 0.36),
        lambda X, Y: (np.hypot(X, Y) < 1.0) & (np.hypot(X, Y) > 0.35),
    ],
    ids=["disk", "stadium", "annulus"],
)
def test_mass_gap_protects_the_bulk_boundary_law(shape):
    X, Y = _grid(2.0, 201)  # at 161 one stadium mode's pair straddles the staircase
    D = shape(X, Y)
    m = 5.0
    modes = _higgs_modes(X, Y, higgs_wall_mass2(D, m), 30)
    bound = trapped(modes, m)
    assert len(bound) >= 4
    chi = euler_characteristic(D)

    def obeys(j):
        return 2 * region_charge(modes, j, D) == 2 * chi + wall_nodal_count(_field(modes, j), D)

    assert all(obeys(j) for j in bound)
    unbound = [j for j in range(len(modes["k_values"])) if j not in bound]
    assert sum(obeys(j) for j in unbound) < 0.6 * len(unbound)


def test_condensate_filled_hole_carries_index_one_minus_l():
    """Annulus: engulfing the Higgs-filled hole adds 1 - l = 1 - N_wall/4.

    Inside the hole E grows like I_l(κr), so the hole wall's effective Robin
    coefficient is only ~κ²r_h/2 for l = 0: near the gap edge (κ·r_h small)
    the hole wall is nearly Neumann and the protection is lost. The rule is
    asserted where κ·r_h > 0.8.
    """
    X, Y = _grid(2.0, 161)
    r = np.hypot(X, Y)
    r_hole = 0.35
    D = (r < 1.0) & (r > r_hole)
    filled = r < 1.0
    m = 6.0
    modes = _higgs_modes(X, Y, higgs_wall_mass2(D, m), 10)
    checked = 0
    for j in trapped(modes, m):
        if np.sqrt(m**2 - modes["k_values"][j] ** 2) * r_hole <= 0.8:
            continue
        checked += 1
        n_wall = wall_nodal_count(_field(modes, j), D)
        jump = region_charge(modes, j, filled) - region_charge(modes, j, D)
        assert jump == 1 - n_wall // 4
    assert checked >= 5


@pytest.mark.parametrize("L", [1, 2, 3])
def test_each_vortex_traps_exactly_one_photon_mode(L):
    """g = m0·ξ = 2: one s-shell per vortex, so #trapped = winding number L."""
    X, Y = _grid(3.0, 161)
    cores = [(-1.5, -0.8), (1.4, 0.9), (0.0, 1.6)][:L]
    xi, m0 = 0.35, 2.0 / 0.35
    modes = _higgs_modes(X, Y, m0**2 * vortex_condensate(X, Y, cores, xi), L + 3)
    bound = trapped(modes, m0)
    assert len(bound) == L
    near = np.zeros_like(X, dtype=bool)
    for cx, cy in cores:
        near |= np.hypot(X - cx, Y - cy) < 3 * xi
    for j in bound:
        E = _field(modes, j)
        assert (E[near] ** 2).sum() / (E**2).sum() > 0.99


def test_no_vortex_no_trap():
    X, Y = _grid(3.0, 81)
    modes = _higgs_modes(X, Y, 16.0 * vortex_condensate(X, Y, [], 0.35), 3)
    assert trapped(modes, 4.0) == []


def test_stronger_core_fills_the_p_shell():
    """g = 3: s shell plus a degenerate p pair (l = ±1) per vortex."""
    X, Y = _grid(8.0, 161)
    modes = _higgs_modes(X, Y, 9.0 * vortex_condensate(X, Y, [(0.0, 0.0)], 1.0), 6)
    w = [modes["k_values"][j] for j in trapped(modes, 3.0)]
    assert len(w) == 3
    assert w[1] == pytest.approx(w[2], rel=1e-6)
    assert w[0] < w[1]
