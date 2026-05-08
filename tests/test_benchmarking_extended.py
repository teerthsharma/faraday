# Copyright (c) 2026 Teerth Sharma. All rights reserved.
"""Extended tests for faraday.benchmarking — run_suite, save_report, EpochTelemetry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from faraday.benchmarking import (
    BenchmarkReport,
    BenchmarkResult,
    EpochTelemetry,
    ValidationReport,
    run_benchmark,
    run_suite,
    save_report,
)


class TestEpochTelemetry:
    def test_to_dict(self) -> None:
        t = EpochTelemetry(
            epoch=1,
            spectral_residual=0.1,
            betti_0_err=0.05,
            betti_1_err=0.03,
            betti_2_err=0.01,
            timestamp="2026-05-08T00:00:00.000Z",
        )
        d = t.to_dict()
        assert d["epoch"] == 1
        assert d["spectral_residual"] == 0.1


class TestBenchmarkResult:
    def test_creation(self) -> None:
        r = BenchmarkResult(name="test", duration_s=0.5, memory_mb=10.0)
        assert r.name == "test"
        assert r.duration_s == 0.5


class TestRunBenchmark:
    def test_run_benchmark_timing(self) -> None:
        r = run_benchmark("noop", lambda: None)
        assert r.name == "noop"
        assert r.duration_s >= 0


class TestRunSuite:
    def test_micro_suite(self) -> None:
        report = run_suite("micro", n_runs=1)
        assert isinstance(report, BenchmarkReport)
        assert report.suite == "micro"
        assert len(report.results) > 0
        assert report.total_duration_s > 0

    def test_unknown_suite_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown suite"):
            run_suite("nonexistent")

    def test_suite_with_validation(self) -> None:
        result = run_suite("micro", n_runs=1, include_validation=True)
        assert isinstance(result, tuple)
        bench, val = result
        assert isinstance(bench, BenchmarkReport)
        assert isinstance(val, ValidationReport)
        assert val.n_test > 0


class TestSaveReport:
    def test_save_json(self, tmp_path: Path) -> None:
        report = BenchmarkReport(
            suite="test",
            timestamp="2026-05-08",
            total_duration_s=1.0,
            results=[BenchmarkResult(name="a", duration_s=0.5)],
        )
        save_report(report, output_dir=str(tmp_path), formats=["json"])
        json_file = tmp_path / "benchmark_test.json"
        assert json_file.exists()
        data = json.loads(json_file.read_text())
        assert data["suite"] == "test"

    def test_save_csv(self, tmp_path: Path) -> None:
        report = BenchmarkReport(
            suite="test",
            timestamp="2026-05-08",
            total_duration_s=1.0,
            results=[
                BenchmarkResult(name="a", duration_s=0.5),
                BenchmarkResult(name="b", duration_s=0.3),
            ],
        )
        save_report(report, output_dir=str(tmp_path), formats=["csv"])
        csv_file = tmp_path / "benchmark_test.csv"
        assert csv_file.exists()
        lines = csv_file.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows

    def test_save_default_format(self, tmp_path: Path) -> None:
        report = BenchmarkReport(
            suite="test",
            timestamp="2026-05-08",
            total_duration_s=0.1,
            results=[],
        )
        save_report(report, output_dir=str(tmp_path))
        assert (tmp_path / "benchmark_test.json").exists()


class TestValidationReport:
    def test_summary(self) -> None:
        vr = ValidationReport(
            suite="test",
            timestamp="now",
            n_train=10,
            n_test=3,
            n_total=13,
            train_god_score=0.85,
            mean_e_betti0_error=0.1,
            mean_h_betti0_error=0.2,
            mean_coupling_error=0.05,
            mean_god_distance=0.5,
            std_god_distance=0.1,
            max_god_distance=0.8,
            convergence_rate=0.9,
            knn_vs_gt_agreement=0.0,
            per_geometry=[],
        )
        s = vr.summary()
        assert "god_score=0.8500" in s
        assert "convergence_rate=90.0%" in s

    def test_to_dict(self) -> None:
        vr = ValidationReport(
            suite="test",
            timestamp="now",
            n_train=10,
            n_test=3,
            n_total=13,
            train_god_score=0.85,
            mean_e_betti0_error=0.1,
            mean_h_betti0_error=0.2,
            mean_coupling_error=0.05,
            mean_god_distance=0.5,
            std_god_distance=0.1,
            max_god_distance=0.8,
            convergence_rate=0.9,
            knn_vs_gt_agreement=0.0,
            per_geometry=[],
        )
        d = vr.to_dict()
        assert d["suite"] == "test"
        assert d["n_train"] == 10
