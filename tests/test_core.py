# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
Tests for faraday core functionality.
"""

import numpy as np


class TestCavityGeometry:
    def test_rectangular_contains(self):
        from faraday import CavityGeometry, CavityShape

        geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0))
        # w=2, h=1 → half-width=1, half-height=0.5
        # 1D arrays: broadcast & → 1D result
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 0.5, 1.0])
        mask = geom.contains(x, y)
        # x: |0|<1 ✓, |1|<1 ✓, |2|≥1 ✗  → [True, True, False]
        # y: |0|<0.5 ✓, |0.5|≥0.5 ✗, |1|≥0.5 ✗  → [True, False, False]
        # broadcast & with same length → element-wise AND
        assert mask.shape == (3,)
        assert np.all(mask == [True, False, False])

    def test_rectangular_contains_2d(self):
        from faraday import CavityGeometry, CavityShape

        geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0))
        # 2D case: meshgrid → proper 2D mask
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-0.8, 0.8, 3)
        X, Y = np.meshgrid(x, y)
        mask = geom.contains(X, Y)
        # Corner (1.5, 0.8) is outside both → False
        # Center (0, 0) is inside both → True
        assert mask.shape == (3, 4)
        assert mask[1, 2]   # center
        assert not mask[0, 3]  # top-right corner outside

    def test_circular_contains(self):
        from faraday import CavityGeometry, CavityShape

        geom = CavityGeometry(shape=CavityShape.CIRCULAR, dims=(1.0,))
        x = np.array([0.0, 0.5, 1.5])
        y = np.array([0.0, 0.0, 0.0])
        mask = geom.contains(x, y)
        assert np.all(mask == [True, True, False])

    def test_interior_mask(self):
        from faraday import CavityGeometry, CavityShape

        geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 2.0))
        X = np.linspace(-1.5, 1.5, 4)
        Y = np.linspace(-1.5, 1.5, 4)
        X2d, Y2d = np.meshgrid(X, Y)
        mask = geom.interior_mask(X2d, Y2d)
        assert mask.shape == (4, 4)


class TestEMSolver:
    def test_solve_rectangular_cavity(self):
        from faraday import CavityGeometry, CavityShape, solve_cavity_modes

        geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0))
        result = solve_cavity_modes(geom, nx=20, ny=20, num_modes=3)

        assert "k_values" in result
        assert "e_modes" in result
        assert "h_modes" in result
        assert "s_modes" in result
        assert result["num_modes_found"] >= 1
        assert len(result["k_values"]) >= 1
        # Wave numbers should be positive
        assert all(k > 0 for k in result["k_values"])

    def test_solve_circular_cavity(self):
        from faraday import CavityGeometry, CavityShape, solve_cavity_modes

        geom = CavityGeometry(shape=CavityShape.CIRCULAR, dims=(1.0,))
        result = solve_cavity_modes(geom, nx=25, ny=25, num_modes=3)

        assert result["num_modes_found"] >= 1
        assert len(result["k_values"]) >= 1

    def test_eigenmode_physical_constraints(self):
        from faraday import CavityGeometry, CavityShape, solve_cavity_modes

        geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(1.5, 1.0))
        result = solve_cavity_modes(geom, nx=30, ny=30, num_modes=4)

        for key, mode in result["e_modes"].items():
            field = np.array(mode["field"])
            # E-field should be real (eigsh returns real for symmetric L)
            assert np.isrealobj(field), f"{key} E-field should be real"
            # E-field should be zero at PEC boundary
            interior = np.array(result["interior"])
            _ny, _nx = field.shape
            assert np.allclose(field[~interior], 0.0), f"{key} should be zero outside"

    def test_h_field_from_e_via_curl(self):
        from faraday import CavityGeometry, CavityShape, solve_cavity_modes

        geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0))
        result = solve_cavity_modes(geom, nx=25, ny=25, num_modes=2)

        for key in result["e_modes"]:
            e_field = np.array(result["e_modes"][key]["field"])
            h_field = np.array(result["h_modes"][key]["field"])
            # H field should be non-negative (magnitude)
            assert np.all(h_field >= 0), f"{key} H magnitude should be non-negative"
            # E and H should be non-trivially correlated
            assert np.corrcoef(e_field.flatten(), h_field.flatten())[0, 1] != 0


class TestBarcode:
    def test_field_to_pointcloud(self):
        from faraday import field_to_pointcloud

        field = np.random.rand(20, 20) + 0.1j * np.random.rand(20, 20)
        points = field_to_pointcloud(field, threshold=0.1)
        assert points.ndim == 2
        assert points.shape[1] == 3  # (x, y, phase)
        assert points.shape[0] >= 1

    def test_field_to_pointcloud_no_phase(self):
        from faraday import field_to_pointcloud

        field = np.random.rand(20, 20)
        points = field_to_pointcloud(field, threshold=0.1, add_phase=False)
        assert points.shape[1] == 3
        assert points.shape[0] >= 1

    def test_compute_barcodes(self):
        from faraday import compute_barcodes, field_to_pointcloud

        np.random.seed(42)
        field = np.random.rand(30, 30)
        points = field_to_pointcloud(field, threshold=0.1)
        result = compute_barcodes(points, max_dim=1)
        assert "betti_0" in result
        assert "betti_1" in result
        assert "diagrams" in result
        assert isinstance(result["betti_0"], int)
        assert isinstance(result["betti_1"], int)

    def test_topological_fingerprint(self):
        from faraday import topological_fingerprint

        np.random.seed(42)
        field = np.random.rand(30, 30)
        fp = topological_fingerprint(field, threshold=0.1)
        assert "betti_0" in fp
        assert "betti_1" in fp
        assert "topological_score" in fp
        assert 0 <= fp["confinement_ratio"] <= 1

    def test_coupled_fingerprint(self):
        from faraday import coupled_fingerprint

        np.random.seed(42)
        e_field = np.random.rand(30, 30)
        h_field = np.random.rand(30, 30)
        result = coupled_fingerprint(e_field, h_field, threshold=0.1)

        assert "e_fingerprint" in result
        assert "h_fingerprint" in result
        assert "coupling_strength" in result
        assert 0 <= result["coupling_strength"] <= 1
        assert "emd_S" in result


class TestManifoldProjector:
    def test_barcode_to_coefficients(self):
        from faraday.manifold_projector import barcode_to_coefficients

        barcode = [(0.1, 0.5), (0.2, 0.8), (0.3, float("inf"))]
        coeffs = barcode_to_coefficients(barcode, degree=50)
        assert coeffs.shape == (50,)
        assert np.abs(coeffs).sum() > 0

    def test_embed_barcode(self):
        from faraday.manifold_projector import embed_barcode

        barcode = [(0.1, 0.5), (0.2, 0.8)]
        emb = embed_barcode(barcode, dim=50)
        assert emb.shape == (50,)
        # Should be L2-normalized
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-6)

    def test_embed_fingerprint(self):
        from faraday.manifold_projector import embed_fingerprint

        fp = {
            "betti_0": 5,
            "betti_1": 3,
            "h0_bars": 5,
            "h1_bars": 3,
            "h0_lifetimes": [0.1, 0.2, 0.3],
            "h1_lifetimes": [0.05, 0.15],
            "topological_score": 1.5,
            "confinement_ratio": 0.8,
            "field_max": 0.9,
            "field_mean": 0.4,
            "field_std": 0.2,
            "num_grid_points": 100,
        }
        emb = embed_fingerprint(fp, dim=50)
        assert emb.shape == (50,)

    def test_manifold_projector_encode_decode(self):
        from faraday.manifold_projector import ManifoldProjector

        mp = ManifoldProjector(input_dim=50, latent_dim=16)
        x = np.random.rand(50)
        x = x / np.linalg.norm(x)
        z = mp.encode(x)
        assert z.shape == (16,)
        x_recon = mp.decode(z)
        assert x_recon.shape == (50,)

    def test_manifold_projector_fit(self):
        from faraday.manifold_projector import ManifoldProjector

        np.random.seed(42)
        mp = ManifoldProjector(input_dim=50, latent_dim=16)
        embeddings = [np.random.rand(50) for _ in range(10)]
        for e in embeddings:
            e /= np.linalg.norm(e)
        losses = mp.fit(embeddings, lr=0.01, epochs=20, batch_size=4)
        assert len(losses) == 20
        assert mp.is_trained
        # Loss should generally decrease
        assert losses[-1] < losses[0] * 2  # loose check

    def test_manifold_projector_reconstruct(self):
        from faraday.manifold_projector import ManifoldProjector

        np.random.seed(42)
        mp = ManifoldProjector(input_dim=50, latent_dim=16)
        x = np.random.rand(50)
        x /= np.linalg.norm(x)
        mse1 = mp.reconstruct(x)
        mse2 = mp.reconstruct(x)
        # Should be deterministic with same seed
        assert np.isclose(mse1, mse2, rtol=1e-3)
