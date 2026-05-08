# Copyright (c) 2026 Teerth Sharma. All rights reserved.
"""Extended tests for em_solver — WaveSuperposer, grid construction, edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from faraday.em_solver import (
    CavityGeometry,
    CavityShape,
    EMWave,
    WaveSuperposer,
    _curl_h_from_ez,
    build_laplacian_2d,
    make_circular_grid,
    make_rectangular_grid,
    solve_cavity_modes,
)
from faraday.exceptions import GeometryError, SolverError


class TestGridConstruction:
    def test_rectangular_grid_shape(self) -> None:
        X, Y = make_rectangular_grid(2.0, 1.0, 30, 20)
        assert X.shape == (20, 30)
        assert Y.shape == (20, 30)

    def test_rectangular_grid_bounds(self) -> None:
        X, Y = make_rectangular_grid(2.0, 1.0, 30, 20)
        assert X.min() == pytest.approx(-1.0)
        assert X.max() == pytest.approx(1.0)
        assert Y.min() == pytest.approx(-0.5)
        assert Y.max() == pytest.approx(0.5)

    def test_circular_grid(self) -> None:
        X, _Y, mask = make_circular_grid(1.0, 30)
        assert X.shape == (30, 30)
        assert mask.shape == (30, 30)
        assert mask.sum() > 0
        # Centre should be inside
        cx, cy = 15, 15
        assert mask[cy, cx]


class TestBuildLaplacian:
    def test_small_grid_raises(self) -> None:
        with pytest.raises(SolverError):
            build_laplacian_2d(2, 2, 0.1, 0.1, np.ones((2, 2), dtype=bool))

    def test_shape(self) -> None:
        nx, ny = 10, 10
        interior = np.ones((ny, nx), dtype=bool)
        L = build_laplacian_2d(nx, ny, 0.1, 0.1, interior)
        assert L.shape == (100, 100)

    def test_exterior_masking(self) -> None:
        nx, ny = 10, 10
        interior = np.ones((ny, nx), dtype=bool)
        interior[0, :] = False  # top row is exterior
        L = build_laplacian_2d(nx, ny, 0.1, 0.1, interior)
        # Exterior rows should have the penalty diagonal
        for j in range(nx):
            assert L[j, j] == pytest.approx(-1e6)


class TestCurlH:
    def test_curl_h_shape(self) -> None:
        ez = np.random.default_rng(42).normal(size=(20, 30))
        hx, hy, h_mag = _curl_h_from_ez(ez, 0.1, 0.1, 3.0)
        assert hx.shape == ez.shape
        assert hy.shape == ez.shape
        assert h_mag.shape == ez.shape

    def test_curl_h_zero_field(self) -> None:
        ez = np.zeros((20, 30))
        _hx, _hy, h_mag = _curl_h_from_ez(ez, 0.1, 0.1, 3.0)
        np.testing.assert_allclose(h_mag, 0.0)


class TestEMWave:
    def test_wave_k(self) -> None:
        w = EMWave(amplitude=1.0 + 0j, kx=3.0, ky=4.0)
        assert w.k == pytest.approx(5.0)

    def test_field_at(self) -> None:
        w = EMWave(amplitude=1.0 + 0j, kx=1.0, ky=0.0, omega=1.0)
        x = np.array([0.0, np.pi])
        y = np.array([0.0, 0.0])
        field = w.field_at(x, y, t=0.0)
        assert field.shape == x.shape
        assert np.isfinite(field).all()


class TestWaveSuperposer:
    @pytest.fixture
    def superposer(self) -> WaveSuperposer:
        geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0))
        mode_data = solve_cavity_modes(geom, nx=15, ny=15, num_modes=2)
        return WaveSuperposer(geom, mode_data)

    def test_add_e_mode(self, superposer: WaveSuperposer) -> None:
        superposer.add_e_mode(0)
        assert superposer.active_mode_idx == 0

    def test_add_h_mode(self, superposer: WaveSuperposer) -> None:
        superposer.add_h_mode(0)
        assert superposer.h_mode_amp == 1.0 + 0j

    def test_add_invalid_mode_raises(self, superposer: WaveSuperposer) -> None:
        with pytest.raises(KeyError):
            superposer.add_e_mode(999)
        with pytest.raises(KeyError):
            superposer.add_h_mode(999)

    def test_add_e_wave(self, superposer: WaveSuperposer) -> None:
        superposer.add_e_wave(amplitude=1.0 + 0j, angle=0.0, wavelength=0.5)
        assert len(superposer.e_waves) == 1

    def test_add_h_wave(self, superposer: WaveSuperposer) -> None:
        superposer.add_h_wave(amplitude=1.0 + 0j, angle=np.pi / 4, wavelength=1.0)
        assert len(superposer.h_waves) == 1

    def test_e_field_at(self, superposer: WaveSuperposer) -> None:
        superposer.add_e_mode(0)
        X = np.linspace(-1, 1, 15)
        Y = np.linspace(-0.5, 0.5, 15)
        XX, YY = np.meshgrid(X, Y)
        field = superposer.e_field_at(XX, YY, t=0.0)
        assert field.shape == XX.shape
        assert np.isfinite(np.abs(field)).all()

    def test_h_field_at(self, superposer: WaveSuperposer) -> None:
        superposer.add_e_mode(0).add_h_mode(0)
        X = np.linspace(-1, 1, 15)
        Y = np.linspace(-0.5, 0.5, 15)
        XX, YY = np.meshgrid(X, Y)
        field = superposer.h_field_at(XX, YY, t=0.0)
        assert field.shape == XX.shape

    def test_poynting_vector(self, superposer: WaveSuperposer) -> None:
        superposer.add_e_mode(0).add_h_mode(0)
        X = np.linspace(-1, 1, 15)
        Y = np.linspace(-0.5, 0.5, 15)
        XX, YY = np.meshgrid(X, Y)
        S = superposer.poynting_vector(XX, YY)
        assert S.shape == XX.shape
        assert (S >= 0).all()

    def test_coupled_field_at(self, superposer: WaveSuperposer) -> None:
        superposer.add_e_mode(0)
        X = np.linspace(-1, 1, 15)
        Y = np.linspace(-0.5, 0.5, 15)
        XX, YY = np.meshgrid(X, Y)
        E, H = superposer.coupled_field_at(XX, YY)
        assert E.shape == XX.shape
        assert H.shape == XX.shape

    def test_superposition_with_waves(self, superposer: WaveSuperposer) -> None:
        superposer.add_e_wave(1.0 + 0j, 0.0, 0.5)
        superposer.add_h_wave(0.5 + 0j, np.pi / 4, 1.0)
        X = np.linspace(-1, 1, 10)
        Y = np.linspace(-0.5, 0.5, 10)
        XX, YY = np.meshgrid(X, Y)
        E = superposer.e_field_at(XX, YY)
        H = superposer.h_field_at(XX, YY)
        assert np.abs(E).max() > 0
        assert np.abs(H).max() > 0


class TestSolverEdgeCases:
    def test_photonic_crystal(self) -> None:
        geom = CavityGeometry(shape=CavityShape.PHOTONIC_CRYSTAL, dims=(0.2, 0.06))
        result = solve_cavity_modes(geom, nx=30, ny=30, num_modes=2)
        assert result["num_modes_found"] >= 1

    def test_num_modes_zero_raises(self) -> None:
        geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0))
        with pytest.raises(SolverError):
            solve_cavity_modes(geom, nx=15, ny=15, num_modes=0)

    def test_photonic_crystal_geometry_validation(self) -> None:
        with pytest.raises(GeometryError):
            CavityGeometry(shape=CavityShape.PHOTONIC_CRYSTAL, dims=(0.2,))
        with pytest.raises(GeometryError):
            CavityGeometry(shape=CavityShape.PHOTONIC_CRYSTAL, dims=(-0.1, 0.03))

    def test_pec_only(self) -> None:
        with pytest.raises(GeometryError, match="PEC"):
            CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0), boundary_conditions="pmc")

    def test_circular_dims_validation(self) -> None:
        with pytest.raises(GeometryError):
            CavityGeometry(shape=CavityShape.CIRCULAR, dims=(1.0, 2.0))
        with pytest.raises(GeometryError):
            CavityGeometry(shape=CavityShape.CIRCULAR, dims=(-1.0,))
