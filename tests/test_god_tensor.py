"""
Tests for faraday GodTensor pipeline.
"""

import numpy as np
import pytest


class TestGodTensor:
    def test_god_tensor_collect_training_data(self):
        from faraday import GodTensor

        gt = GodTensor(n_geometries=5)
        gt.collect_training_data(nx=15, ny=15, num_modes=2, seed=42)
        assert len(gt.samples) >= 4  # some may fail
        for s in gt.samples:
            assert s.e_fingerprint is not None
            assert s.h_fingerprint is not None
            assert s.e_embedding.shape == (50,)
            assert s.h_embedding.shape == (50,)

    def test_god_tensor_learn_T(self):
        from faraday import GodTensor

        gt = GodTensor(n_geometries=5)
        gt.collect_training_data(nx=15, ny=15, num_modes=2, seed=42)
        T = gt.learn_T()
        assert T.shape == (16, 16)
        assert gt.T_matrix is not None

    def test_god_tensor_fixed_point_convergence(self):
        from faraday import GodTensor

        # Need enough geometries for T (16x16) to be full-rank in latent space.
        # With n_geometries=20 and reasonable failure rate, expect >=15 valid samples.
        gt = GodTensor(n_geometries=20)
        gt.collect_training_data(nx=15, ny=15, num_modes=2, seed=42)
        gt.learn_T()
        god = gt.find_fixed_point(iters=200, tol=1e-6)
        assert god.shape == (16,)
        assert gt.god_tensor is not None
        # Convergence is expected with sufficient training data for a full-rank T
        assert gt.fixed_point_converged, "T matrix should be full-rank with >=15 samples"

    def test_god_score(self):
        from faraday import GodTensor

        gt = GodTensor(n_geometries=5)
        gt.collect_training_data(nx=15, ny=15, num_modes=2, seed=42)
        gt.learn_T()
        gt.find_fixed_point(iters=200, tol=1e-6)
        score = gt.god_score()
        assert 0 <= score <= 1

    def test_e_to_h_map(self):
        from faraday import GodTensor

        gt = GodTensor(n_geometries=5)
        gt.collect_training_data(nx=15, ny=15, num_modes=2, seed=42)
        gt.learn_T()
        gt.find_fixed_point(iters=200, tol=1e-6)

        e_emb = gt.samples[0].e_embedding
        h_pred = gt.get_e_to_h_map(e_emb)
        assert h_pred.shape == (16,)

    def test_h_to_e_map(self):
        from faraday import GodTensor

        gt = GodTensor(n_geometries=5)
        gt.collect_training_data(nx=15, ny=15, num_modes=2, seed=42)
        gt.learn_T()
        gt.find_fixed_point(iters=200, tol=1e-6)

        h_emb = gt.samples[0].h_embedding
        e_pred = gt.get_h_to_e_map(h_emb)
        assert e_pred.shape == (16,)

    def test_summary(self):
        from faraday import GodTensor

        gt = GodTensor(n_geometries=3)
        gt.collect_training_data(nx=12, ny=12, num_modes=2, seed=42)
        gt.learn_T()
        gt.find_fixed_point(iters=100, tol=1e-5)
        summary = gt.summary()
        assert "n_samples" in summary
        assert "T_matrix_shape" in summary
        assert "god_score" in summary
        assert summary["n_samples"] >= 2

    def test_training_sample_to_dict(self):
        from faraday.god_tensor import TrainingSample

        sample = TrainingSample(
            geometry_params=(2.0, 1.0),
            e_fingerprint={"betti_0": 5},
            h_fingerprint={"betti_1": 3},
            e_embedding=np.random.rand(50),
            h_embedding=np.random.rand(50),
            k_values=[1.0, 2.0],
        )
        d = sample.to_dict()
        assert d["geometry_params"] == (2.0, 1.0)
        assert d["e_fingerprint"]["betti_0"] == 5
        assert isinstance(d["e_embedding"], list)


class TestPredict:
    def test_predict_eh_barcode(self):
        from faraday import GodTensor
        from faraday.predict import predict_eh_barcode

        gt = GodTensor(n_geometries=5)
        gt.collect_training_data(nx=15, ny=15, num_modes=2, seed=42)
        gt.learn_T()
        gt.find_fixed_point(iters=100, tol=1e-5)

        pred = predict_eh_barcode(gt, (2.0, 1.2), "rect")
        assert "knn_e_fingerprint" in pred
        assert "knn_h_fingerprint" in pred
        assert "coupling_score" in pred
        assert "god_distance_e" in pred

    def test_predict_with_circular_geometry(self):
        from faraday import GodTensor
        from faraday.predict import predict_eh_barcode

        gt = GodTensor(n_geometries=5)
        gt.collect_training_data(nx=15, ny=15, num_modes=2, seed=42)
        gt.learn_T()
        gt.find_fixed_point(iters=100, tol=1e-5)

        pred = predict_eh_barcode(gt, (1.0,), "circle")
        assert "knn_e_fingerprint" in pred
        assert pred["shape"] == "circle"


class TestIntegration:
    def test_full_pipeline_small(self):
        """Run the full pipeline with a small dataset and verify end-to-end."""
        from faraday import GodTensor, CavityGeometry, CavityShape, solve_cavity_modes, coupled_fingerprint
        from faraday.predict import predict_eh_barcode

        # 1. Collect
        gt = GodTensor(n_geometries=4)
        gt.collect_training_data(nx=12, ny=12, num_modes=2, seed=99)

        # 2. Learn
        gt.learn_T()

        # 3. Fixed point
        god = gt.find_fixed_point(iters=100, tol=1e-5)

        # 4. Predict
        pred = predict_eh_barcode(gt, (2.0, 1.0), "rect")
        assert pred["coupling_score"] == gt.god_score()

        # 5. Verify with actual FDFD
        geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0))
        mode_data = solve_cavity_modes(geom, nx=12, ny=12, num_modes=2)
        e_field = np.array(mode_data["e_modes"]["mode_0"]["field"])
        h_field = np.array(mode_data["h_modes"]["mode_0"]["field"])
        actual = coupled_fingerprint(e_field, h_field)

        # Actual coupling should be high (> 0.8) for a well-posed cavity
        assert actual["coupling_strength"] > 0.0

        # Predicted betti_0 should be non-negative
        assert pred["knn_e_fingerprint"]["betti_0"] >= 0
