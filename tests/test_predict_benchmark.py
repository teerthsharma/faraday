# Copyright (c) 2026 Teerth Sharma. All rights reserved.
"""Tests for faraday.predict.benchmark and edge-case coverage."""

from __future__ import annotations

import pytest

from faraday.god_tensor import GodTensor
from faraday.predict import _average_fingerprints, benchmark, predict_eh_barcode


@pytest.fixture(scope="module")
def trained_gt() -> GodTensor:
    """Small trained GodTensor for predict tests."""
    gt = GodTensor(n_geometries=8)
    gt.collect_training_data(nx=15, ny=15, num_modes=2, seed=42)
    gt.learn_T()
    gt.find_fixed_point(iters=50)
    return gt


class TestAverageFingerprints:
    def test_empty_list(self) -> None:
        result = _average_fingerprints([], [], 0.0)
        assert result == {}

    def test_single_fingerprint(self) -> None:
        fp = {
            "betti_0": 3,
            "betti_1": 1,
            "h0_bars": 5,
            "h1_bars": 2,
            "topological_score": 1.5,
            "confinement_ratio": 0.8,
            "field_max": 1.0,
            "field_mean": 0.5,
            "field_std": 0.2,
            "num_grid_points": 100,
            "h0_lifetimes": [0.5, 0.3],
            "h1_lifetimes": [0.1],
        }
        result = _average_fingerprints([fp], [1.0], 1.0)
        assert result["betti_0"] == 3.0
        assert result["betti_1"] == 1.0
        assert len(result["h0_lifetimes"]) == 2
        assert result["h0_lifetimes"][0] == pytest.approx(0.5)

    def test_weighted_average(self) -> None:
        fp1 = {"betti_0": 2, "betti_1": 0, "h0_lifetimes": [1.0], "h1_lifetimes": []}
        fp2 = {"betti_0": 4, "betti_1": 2, "h0_lifetimes": [2.0], "h1_lifetimes": []}
        result = _average_fingerprints([fp1, fp2], [0.5, 0.5], 1.0)
        assert result["betti_0"] == pytest.approx(3.0)
        assert result["betti_1"] == pytest.approx(1.0)


class TestPredictEhBarcode:
    def test_raises_without_training(self) -> None:
        gt = GodTensor(n_geometries=5)
        with pytest.raises(ValueError, match="trained"):
            predict_eh_barcode(gt, (2.0, 1.0))

    def test_returns_all_fields(self, trained_gt: GodTensor) -> None:
        pred = predict_eh_barcode(trained_gt, (2.0, 1.0), shape="rect")
        expected_keys = {
            "geometry_params",
            "shape",
            "knn_e_fingerprint",
            "knn_h_fingerprint",
            "baseline_h_fingerprint",
            "god_distance_e",
            "god_distance_h",
            "coupling_score",
            "inferred_e_latent",
            "inferred_h_latent",
        }
        assert expected_keys.issubset(pred.keys())
        assert pred["shape"] == "rect"
        assert isinstance(pred["god_distance_e"], float)
        assert isinstance(pred["god_distance_h"], float)

    def test_circle_geometry(self, trained_gt: GodTensor) -> None:
        pred = predict_eh_barcode(trained_gt, (1.0,), shape="circle")
        assert pred["shape"] == "circle"


class TestBenchmark:
    def test_benchmark_small(self, trained_gt: GodTensor) -> None:
        result = benchmark(
            trained_gt,
            test_geometries=[(2.0, 1.0)],
            shapes=["rect"],
            nx=15,
            ny=15,
        )
        assert "avg_e_betti0_error" in result
        assert "avg_h_betti0_error" in result
        assert "per_geometry" in result
        assert len(result["per_geometry"]) == 1

    def test_benchmark_multiple_geometries(self, trained_gt: GodTensor) -> None:
        result = benchmark(
            trained_gt,
            test_geometries=[(1.5, 1.0), (2.5, 1.5)],
            shapes=["rect", "rect"],
            nx=15,
            ny=15,
        )
        assert result["n_test"] == 2
