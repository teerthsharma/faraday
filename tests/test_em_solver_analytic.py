# Copyright (c) 2026 Teerth Sharma. All rights reserved.
"""
Analytic reference tests for the FDFD eigensolver.

For a rectangular PEC cavity of width *w*, height *h*, the TM modes have
the closed-form eigenvalues

    k_{mn}^2 = (m π / w)^2 + (n π / h)^2,    m, n = 1, 2, 3, …

This file checks that :func:`solve_cavity_modes` reproduces the
analytical spectrum to leading order in 1/N (the grid resolution).
"""
from __future__ import annotations

import numpy as np
import pytest

from faraday.em_solver import (
    CavityGeometry,
    CavityShape,
    rectangular_analytic_k,
    solve_cavity_modes,
)


@pytest.mark.parametrize("dims", [(2.0, 1.0), (1.5, 1.5), (3.0, 0.5)])
def test_rectangular_eigenvalues_match_analytic(dims):
    w, h = dims
    geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=dims)
    md = solve_cavity_modes(geom, nx=80, ny=80, num_modes=4, seed=42)
    numerical = np.asarray(md["k_values"][:4])
    analytic = np.asarray(rectangular_analytic_k(w, h, num_modes=4))
    rel = np.abs(numerical - analytic) / analytic
    # Discrete-grid error scales like O(1/N²). At N=80 this is ~1e-3.
    assert np.all(rel < 5e-3), f"rel errors {rel} on dims {dims}"


def test_rectangular_low_resolution_still_reasonable():
    geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0))
    md = solve_cavity_modes(geom, nx=20, ny=20, num_modes=4, seed=0)
    numerical = np.asarray(md["k_values"][:4])
    analytic = np.asarray(rectangular_analytic_k(2.0, 1.0, num_modes=4))
    rel = np.abs(numerical - analytic) / analytic
    # 20×20 is very coarse — accept up to 5%
    assert np.all(rel < 5e-2), rel


def test_eigenvalues_increase_with_smaller_cavity():
    """Smaller cavity ⇒ higher eigenvalues (compactness scaling)."""
    g_big = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0))
    g_sml = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(1.0, 0.5))
    big = solve_cavity_modes(g_big, nx=40, ny=40, num_modes=2, seed=1)["k_values"]
    sml = solve_cavity_modes(g_sml, nx=40, ny=40, num_modes=2, seed=1)["k_values"]
    # k_{mn} for the smaller cavity is exactly 2x bigger
    ratio = sml[0] / big[0]
    assert abs(ratio - 2.0) < 0.05, f"expected ratio 2.0, got {ratio}"


def test_h_field_curl_sanity():
    """∇×E = iωμH ⇒ |H| support overlaps with |∇E| support."""
    geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0))
    md = solve_cavity_modes(geom, nx=50, ny=50, num_modes=2, seed=42)
    e = np.asarray(md["e_modes"]["mode_0"]["field"])
    h = np.asarray(md["h_modes"]["mode_0"]["field"])
    # |H| should be non-trivially correlated with |∇E|
    grad_e = np.hypot(*np.gradient(e))
    corr = np.corrcoef(grad_e.ravel(), h.ravel())[0, 1]
    assert corr > 0.6, f"H-field should track |∇E|, got corr={corr}"


def test_modes_are_orthogonal_in_l2():
    """Distinct cavity eigenmodes are L²-orthogonal."""
    geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0))
    md = solve_cavity_modes(geom, nx=40, ny=40, num_modes=3, seed=42)
    e0 = np.asarray(md["e_modes"]["mode_0"]["field"]).ravel()
    e1 = np.asarray(md["e_modes"]["mode_1"]["field"]).ravel()
    e2 = np.asarray(md["e_modes"]["mode_2"]["field"]).ravel()
    cross_01 = abs(float(e0 @ e1)) / (np.linalg.norm(e0) * np.linalg.norm(e1))
    cross_02 = abs(float(e0 @ e2)) / (np.linalg.norm(e0) * np.linalg.norm(e2))
    cross_12 = abs(float(e1 @ e2)) / (np.linalg.norm(e1) * np.linalg.norm(e2))
    assert cross_01 < 0.05
    assert cross_02 < 0.05
    assert cross_12 < 0.05


def test_pec_dirichlet_boundary():
    """E_z must vanish at the PEC boundary (interior=False region)."""
    geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0))
    md = solve_cavity_modes(geom, nx=40, ny=40, num_modes=2, seed=42)
    e = np.asarray(md["e_modes"]["mode_0"]["field"])
    interior = np.asarray(md["interior"])
    np.testing.assert_allclose(e[~interior], 0.0, atol=1e-10)


def test_circular_cavity_first_root_of_J0():
    """First eigenvalue of a circular PEC cavity is k = j_{0,1}/r ≈ 2.4048/r."""
    r = 1.0
    geom = CavityGeometry(shape=CavityShape.CIRCULAR, dims=(r,))
    md = solve_cavity_modes(geom, nx=80, ny=80, num_modes=1, seed=42)
    k0 = md["k_values"][0]
    # First zero of J_0 is 2.4048256
    assert abs(k0 - 2.4048256 / r) < 0.05, f"got k0={k0}"


def test_validation_rejects_invalid_geometry():
    """CavityGeometry rejects degenerate input."""
    from faraday.exceptions import GeometryError

    with pytest.raises(GeometryError):
        CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(0.0, 1.0))
    with pytest.raises(GeometryError):
        CavityGeometry(shape=CavityShape.CIRCULAR, dims=(-1.0,))
    with pytest.raises(GeometryError):
        CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(1.0,))  # wrong arity
