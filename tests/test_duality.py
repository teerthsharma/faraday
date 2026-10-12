# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
Tests for the topological E ⇄ H duality (faraday.duality).

The laws are exact theorems for the continuum fields; the tests check that the
discrete measurement reproduces them on cavities of every topological type.
"""

import numpy as np
import pytest

from faraday import CavityGeometry, CavityShape, solve_cavity_modes
from faraday.duality import (
    discover_coupling,
    euler_characteristic,
    mode_topology,
    predict_rectangle_topology,
    vortex_charges,
    wall_nodal_count,
)
from faraday.em_solver import solve_interior_modes


def _grid(R=1.2, n=121):
    c = np.linspace(-R, R, n)
    return np.meshgrid(c, c)


def _solve_mask(f, R=1.2, n=121, num_modes=8):
    X, Y = _grid(R, n)
    return solve_interior_modes(X, Y, f(X, Y), num_modes=num_modes, seed=0)


CAVITIES = {
    "rectangle": lambda: solve_cavity_modes(
        CavityGeometry(CavityShape.RECTANGULAR, (2.0, 1.2)), nx=81, ny=81, num_modes=8, seed=0
    ),
    "circle": lambda: solve_cavity_modes(
        CavityGeometry(CavityShape.CIRCULAR, (1.0,)), nx=101, num_modes=8, seed=0
    ),
    "stadium": lambda: _solve_mask(
        lambda X, Y: ((abs(X) < 0.6) & (abs(Y) < 0.6)) | ((abs(X) - 0.6) ** 2 + Y**2 < 0.36)
    ),
    "eccentric annulus": lambda: _solve_mask(
        lambda X, Y: (np.hypot(X, Y) < 1.0) & (np.hypot(X - 0.3, Y + 0.1) > 0.3), n=151
    ),
    "square with square hole": lambda: _solve_mask(
        lambda X, Y: (abs(X) < 1.0) & (abs(Y) < 1.0)
        & ~((abs(X - 0.2) < 0.3) & (abs(Y + 0.1) < 0.3)),
        n=151,
    ),
}


@pytest.fixture(scope="module", params=list(CAVITIES))
def cavity(request):
    modes = CAVITIES[request.param]()
    return request.param, [mode_topology(modes, j) for j in range(modes["num_modes_found"])]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class TestPrimitives:
    @pytest.mark.parametrize(("sx", "sy", "charge"), [(1, 1, 1), (1, -1, -1), (-1, -1, 1)])
    def test_vortex_charge_of_linear_fields(self, sx, sy, charge):
        X, Y = _grid(1.0, 21)
        w = vortex_charges(sx * X, sy * Y, np.ones_like(X, dtype=bool))
        assert w.sum() == charge
        assert np.abs(w).sum() == 1

    def test_zero_on_grid_line_is_counted_once(self):
        """Regression: a vortex on a grid edge (edge angle jump exactly π)."""
        X, Y = _grid(1.0, 21)  # y = 0 is a grid row; put the zero mid-edge
        x0 = (X[0, 10] + X[0, 11]) / 2
        w = vortex_charges(-Y, X - x0, np.ones_like(X, dtype=bool))
        assert w.sum() == 1

    def test_euler_characteristic(self):
        X, Y = _grid(1.2, 121)
        r = np.hypot(X, Y)
        assert euler_characteristic(r < 1) == 1
        assert euler_characteristic((r < 1) & (r > 0.4)) == 0
        two_holes = (r < 1.1) & (np.hypot(X - 0.45, Y) > 0.2) & (np.hypot(X + 0.4, Y) > 0.2)
        assert euler_characteristic(two_holes) == -1

    def test_wall_nodal_count_of_sign_pattern(self):
        X, Y = _grid(1.0, 41)
        mask = np.hypot(X, Y) < 0.9
        # cos 2θ changes sign 4 times around any circle
        e = np.where(mask, (X**2 - Y**2), 0.0)
        assert wall_nodal_count(e, mask) == 4


# ---------------------------------------------------------------------------
# The two laws, on every topological type of cavity
# ---------------------------------------------------------------------------


class TestDualityLaws:
    def test_bulk_boundary_law(self, cavity):
        name, tops = cavity
        bad = [(j, t.charge, t.euler_cavity, t.n_wall) for j, t in enumerate(tops)
               if not t.boundary_law_holds]
        assert not bad, f"{name}: Q != χ(D) + N_wall/2 for {bad}"

    def test_domain_law(self, cavity):
        name, tops = cavity
        for j, t in enumerate(tops):
            assert t.domain_law_holds, f"{name} mode {j}: {t.domains}"

    def test_circulation_chirality_is_sign_of_e(self, cavity):
        _, tops = cavity
        for t in tops:
            for d in t.domains:
                assert np.sign(d.circulation) == d.sign

    def test_fundamental_mode(self, cavity):
        """One nodal domain, no wall nodal points: Q = χ(D)."""
        _, tops = cavity
        t = tops[0]
        assert t.n_domains == 1
        assert t.n_wall == 0
        assert t.charge == t.euler_cavity

    def test_signature_is_integer_topology_plus_k(self, cavity):
        _, tops = cavity
        sig = tops[1].signature()
        assert sig.shape == (9,)
        assert sig[0] == pytest.approx(tops[1].k)
        np.testing.assert_array_equal(sig[1:], np.rint(sig[1:]))


def test_two_hole_fundamental_carries_an_antivortex():
    """χ(D) = -1: the single-domain fundamental mode must have Q = -1."""
    modes = _solve_mask(
        lambda X, Y: (np.hypot(X, Y) < 1.1)
        & (np.hypot(X - 0.45, Y) > 0.22)
        & (np.hypot(X + 0.4, Y + 0.3) > 0.18),
        n=151,
        num_modes=1,
    )
    t = mode_topology(modes, 0)
    assert t.euler_cavity == -1
    assert t.charge == -1
    assert t.boundary_law_holds and t.domain_law_holds


def test_interior_solver_matches_named_shape():
    geom = CavityGeometry(CavityShape.RECTANGULAR, (2.0, 1.0))
    a = solve_cavity_modes(geom, nx=41, ny=41, num_modes=3, seed=0)
    X, Y = np.array(a["X"]), np.array(a["Y"])
    b = solve_interior_modes(X, Y, np.array(a["interior"]), num_modes=3, seed=0)
    np.testing.assert_allclose(a["k_values"], b["k_values"])
    assert b["geometry"] == "mask"


# ---------------------------------------------------------------------------
# Causality: geometry → topology, and how topology changes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("aspect", [1.37, 2.31])
def test_rectangle_topology_predicted_from_geometry(aspect):
    """Away from level crossings, geometry alone fixes every mode's topology."""
    w, h = aspect, 1.0
    modes = solve_cavity_modes(
        CavityGeometry(CavityShape.RECTANGULAR, (w, h)), nx=91, ny=91, num_modes=8, seed=0
    )
    for j, p in enumerate(predict_rectangle_topology(w, h, 8)):
        if p["gap"] < 0.01:
            continue
        t = mode_topology(modes, j)
        assert (t.n_domains, t.n_wall, t.charge) == (
            p["n_domains"], p["n_wall"], p["charge"]
        ), f"mode {j} (m,n)={p['mn']}"


def test_prediction_flags_degeneracy():
    """2x1 box: TM22 and TM41 are exactly degenerate (k = π√5)."""
    preds = predict_rectangle_topology(2.0, 1.0, 6)
    deg = [p["mn"] for p in preds if p["gap"] < 1e-9]
    assert sorted(deg) == [(2, 2), (4, 1)]


def test_avoided_crossing_changes_charge_through_the_wall():
    """Parity-broken cavity: modes 2 and 3 never cross, yet both pass through
    Q = 2 — a vortex leaves via the wall (N_wall 4 → 2), it is not swapped."""
    X, Y = _grid(1.0, 121)

    def top(w, delta):
        mask = (abs(Y) < 0.5) & (abs(X) < w / 2 + delta * Y)
        modes = solve_interior_modes(X, Y, mask, num_modes=4, seed=0)
        return mode_topology(modes, 2), mode_topology(modes, 3)

    lo2, lo3 = top(1.50, 0.15)
    mid2, mid3 = top(1.62, 0.15)
    hi2, hi3 = top(1.78, 0.15)
    assert (lo2.charge, lo3.charge) == (2, 3)
    assert (mid2.charge, mid3.charge) == (2, 2)
    assert (hi2.charge, hi3.charge) == (3, 2)
    assert mid3.k - mid2.k > 0.01 * mid2.k  # the gap stays open
    for t in (lo2, lo3, mid2, mid3, hi2, hi3):
        assert t.boundary_law_holds


def test_least_squares_rediscovers_the_law():
    tops = []
    for build in CAVITIES.values():
        modes = build()
        tops += [mode_topology(modes, j) for j in range(modes["num_modes_found"])]
    fit = discover_coupling(tops)
    assert fit["euler_cavity"] == pytest.approx(1.0, abs=1e-9)
    assert fit["n_wall"] == pytest.approx(0.5, abs=1e-9)
    assert fit["intercept"] == pytest.approx(0.0, abs=1e-9)
    assert fit["n_domains"] == pytest.approx(0.0, abs=1e-9)
    assert fit["max_residual"] < 1e-9


def test_topology_cli():
    from click.testing import CliRunner

    from faraday.cli import main

    out = CliRunner().invoke(main, ["topology", "-w", "2", "-h", "1", "-n", "6", "--nx", "61"])
    assert out.exit_code == 0, out.output
    assert "near-degenerate" in out.output
    assert "✗" not in out.output
    out = CliRunner().invoke(main, ["topology", "-r", "1", "-n", "3", "--nx", "61"])
    assert out.exit_code == 0, out.output
