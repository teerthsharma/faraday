"""
Benchmark suite for faraday.

Provides reproducible micro-benchmarks and end-to-end benchmarks with
JSON, CSV, and W&B reporters.

Usage
-----
    from faraday.benchmarking import run_suite, MICRO

    results = run_suite(MICRO)

    # Or via CLI:
    #   faraday benchmark --suite micro --output benchmarks/
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from faraday.barcode import coupled_fingerprint, topological_fingerprint
from faraday.em_solver import (
    CavityGeometry,
    CavityShape,
    solve_cavity_modes,
)
from faraday.god_tensor import GodTensor
from faraday.manifold_projector import ManifoldProjector, embed_fingerprint
from faraday.predict import predict_eh_barcode

# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

# Aliases for convenient CLI use
MICRO = "micro"
SMALL = "small"
MEDIUM = "medium"

BENCHMARK_SUITES = {
    "micro": {
        "description": "Quick sanity check — 15 geometries, low resolution",
        "n_geometries": 15,
        "nx": 15,
        "ny": 15,
        "num_modes": 2,
        "iters": 50,
        "test_n": 3,
    },
    "small": {
        "description": "Development suite — 20 geometries, medium resolution",
        "n_geometries": 20,
        "nx": 30,
        "ny": 30,
        "num_modes": 4,
        "iters": 100,
        "test_n": 5,
    },
    "medium": {
        "description": "CI / default suite — 50 geometries, production resolution",
        "n_geometries": 50,
        "nx": 60,
        "ny": 60,
        "num_modes": 8,
        "iters": 500,
        "test_n": 10,
    },
}


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """A single timed benchmark result."""

    name: str
    duration_s: float
    memory_mb: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """A full benchmark suite report."""

    suite: str
    timestamp: str
    total_duration_s: float
    results: list[BenchmarkResult]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def run_benchmark(
    name: str, fn, *args, **kwargs
) -> BenchmarkResult:
    """Time a single benchmark function."""
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    duration_s = time.perf_counter() - t0
    return BenchmarkResult(name=name, duration_s=duration_s)


@dataclass
class ValidationReport:
    """Report from a held-out generalization experiment."""

    suite: str
    timestamp: str
    n_train: int
    n_test: int
    n_total: int
    train_god_score: float
    mean_e_betti0_error: float
    mean_h_betti0_error: float
    mean_coupling_error: float
    mean_god_distance: float
    std_god_distance: float
    max_god_distance: float
    convergence_rate: float  # fraction of test geometries with god_dist < 1.0
    # Aggregate: how well does KNN baseline vs God Tensor compare
    knn_vs_gt_agreement: float
    per_geometry: list[dict]
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """One-line summary for CI / human reading."""
        return (
            f"ValidationReport: {self.n_train} train / {self.n_test} test geometries | "
            f"god_score={self.train_god_score:.4f} | "
            f"mean_E_err={self.mean_e_betti0_error:.3f} | "
            f"mean_H_err={self.mean_h_betti0_error:.3f} | "
            f"convergence_rate={self.convergence_rate:.1%}"
        )


def run_validation_experiment(
    n_total: int = 50,
    train_fraction: float = 0.8,
    nx: int = 40,
    ny: int = 40,
    num_modes: int = 4,
    seed: int = 99,
) -> ValidationReport:
    """Run a held-out generalization experiment for the God Tensor.

    Procedure
    ---------
    1. Generate ``n_total`` cavity geometries with FDFD + fingerprints.
    2. Randomly split into ``train_fraction`` train / ``(1-train_fraction)`` test.
    3. Train God Tensor on training set only.
    4. For every test geometry: run FDFD to get ground-truth fingerprint,
       then ask the God Tensor (via ``predict_eh_barcode``) for a prediction.
    5. Compare: E/H Betti-0 error, coupling error, God Tensor convergence quality.

    This is the experiment that answers: *"Does the God Tensor generalize
    to unseen geometries, or is it just memorising the training set?"*

    Parameters
    ----------
    n_total : int
        Total number of geometries to generate (train + test combined).
    train_fraction : float
        Fraction of geometries for training. Default 0.8 (80/20 split).
    nx, ny : int
        Grid resolution for FDFD solver.
    num_modes : int
        Number of eigenmodes per geometry.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    ValidationReport
        Dataclass with per-metric means, stds, and per-geometry raw results.
    """
    rng = np.random.default_rng(seed)
    n_train = round(n_total * train_fraction)
    n_total - n_train

    # ── Step 1: generate all geometries ──────────────────────────────
    all_params: list[tuple[tuple[float, ...], str]] = []
    for _ in range(n_total):
        if rng.random() < 0.7:
            w = rng.uniform(0.8, 3.0)
            h = rng.uniform(0.5, 2.0)
            all_params.append(((w, h), "rect"))
        else:
            r = rng.uniform(0.5, 1.5)
            all_params.append(((r,), "circle"))

    # ── Step 2: split ─────────────────────────────────────────────────
    rng.shuffle(all_params)
    train_params = all_params[:n_train]
    test_params = all_params[n_train:]

    # ── Step 3: build training set ───────────────────────────────────
    gt = GodTensor(n_geometries=n_train)
    for params, shape in train_params:
        if shape == "rect":
            geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=params)
        else:
            geom = CavityGeometry(shape=CavityShape.CIRCULAR, dims=params)

        try:
            mode_data = solve_cavity_modes(
                geom, nx=nx, ny=ny, num_modes=num_modes, seed=seed
            )
        except Exception:
            continue

        e_field = np.array(next(iter(mode_data["e_modes"].values()))["field"])
        h_field = np.array(next(iter(mode_data["h_modes"].values()))["field"])
        fp = coupled_fingerprint(e_field, h_field)
        if "error" in fp.get("e_fingerprint", {}) or "error" in fp.get(
            "h_fingerprint", {}
        ):
            continue

        e_emb = embed_fingerprint(fp["e_fingerprint"], dim=50)
        h_emb = embed_fingerprint(fp["h_fingerprint"], dim=50)

        from faraday.god_tensor import TrainingSample

        sample = TrainingSample(
            geometry_params=params,
            e_fingerprint=fp["e_fingerprint"],
            h_fingerprint=fp["h_fingerprint"],
            e_embedding=e_emb,
            h_embedding=h_emb,
            k_values=mode_data["k_values"],
        )
        gt.samples.append(sample)

    if len(gt.samples) < 5:
        raise RuntimeError(
            f"Not enough valid training samples ({len(gt.samples)}). "
            "Try increasing n_total or adjusting grid resolution."
        )

    gt.learn_T()
    gt.find_fixed_point(iters=500, tol=1e-7)
    train_god_score = gt.god_score()

    # ── Step 4 & 5: evaluate on held-out geometries ───────────────────
    per_geometry: list[dict] = []
    god_distances: list[float] = []

    for params, shape in test_params:
        if shape == "rect":
            geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=params)
        else:
            geom = CavityGeometry(shape=CavityShape.CIRCULAR, dims=params)

        try:
            mode_data = solve_cavity_modes(
                geom, nx=nx, ny=ny, num_modes=num_modes, seed=seed
            )
        except Exception:
            continue

        e_field = np.array(next(iter(mode_data["e_modes"].values()))["field"])
        h_field = np.array(next(iter(mode_data["h_modes"].values()))["field"])
        actual = coupled_fingerprint(e_field, h_field)
        if "error" in actual.get("e_fingerprint", {}) or "error" in actual.get(
            "h_fingerprint", {}
        ):
            continue

        pred = predict_eh_barcode(gt, params, shape)

        e_betti0_actual = actual["e_fingerprint"]["betti_0"]
        e_betti0_pred = pred["knn_e_fingerprint"]["betti_0"]
        h_betti0_actual = actual["h_fingerprint"]["betti_0"]
        h_betti0_pred = pred["knn_h_fingerprint"]["betti_0"]
        coupling_actual = actual["coupling_strength"]
        coupling_pred = pred["coupling_score"]

        e_error = abs(e_betti0_actual - e_betti0_pred)
        h_error = abs(h_betti0_actual - h_betti0_pred)
        coupling_error = abs(coupling_actual - coupling_pred)

        per_geometry.append(
            {
                "geometry": params,
                "shape": shape,
                "e_betti0_actual": e_betti0_actual,
                "e_betti0_predicted": e_betti0_pred,
                "e_error": e_error,
                "h_betti0_actual": h_betti0_actual,
                "h_betti0_predicted": h_betti0_pred,
                "h_error": h_error,
                "coupling_actual": coupling_actual,
                "coupling_predicted": coupling_pred,
                "coupling_error": coupling_error,
                "god_distance": pred["god_distance_e"],
            }
        )
        god_distances.append(pred["god_distance_e"])

    if not per_geometry:
        raise RuntimeError("No valid test results — all test geometries failed.")

    god_distances_arr = np.array(god_distances)

    return ValidationReport(
        suite="generalization",
        timestamp=str(np.datetime64("now")),
        n_train=len(gt.samples),
        n_test=len(per_geometry),
        n_total=n_total,
        train_god_score=train_god_score,
        mean_e_betti0_error=float(np.mean([r["e_error"] for r in per_geometry])),
        mean_h_betti0_error=float(np.mean([r["h_error"] for r in per_geometry])),
        mean_coupling_error=float(np.mean([r["coupling_error"] for r in per_geometry])),
        mean_god_distance=float(np.mean(god_distances_arr)),
        std_god_distance=float(np.std(god_distances_arr)),
        max_god_distance=float(np.max(god_distances_arr)),
        convergence_rate=float(np.mean(god_distances_arr < 1.0)),
        knn_vs_gt_agreement=0.0,  # placeholder for KNN-vs-GT comparison
        metadata={"nx": nx, "ny": ny, "num_modes": num_modes, "seed": seed},
        per_geometry=per_geometry,
    )


def run_suite(
    suite_name: str = "small",
    n_runs: int = 1,
    include_validation: bool = False,
) -> BenchmarkReport | ValidationReport | tuple[BenchmarkReport, ValidationReport]:
    """Run a named benchmark suite.

    Parameters
    ----------
    suite_name : str
        One of "micro", "small", "medium".
    n_runs : int
        Number of repetitions for timing stability.
    include_validation : bool
        If True, also run the held-out generalization experiment
        and return a tuple (BenchmarkReport, ValidationReport).

    Returns
    -------
    BenchmarkReport
        Timing benchmarks.
    ValidationReport (if include_validation=True)
        Held-out generalization metrics.
    """
    if suite_name not in BENCHMARK_SUITES:
        raise ValueError(
            f"Unknown suite {suite_name!r}. Available: {list(BENCHMARK_SUITES)}"
        )

    cfg = BENCHMARK_SUITES[suite_name]
    results: list[BenchmarkResult] = []
    t0_total = time.perf_counter()

    # ── EMSolver ────────────────────────────────────────────────────────
    geom_rect = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0))
    geom_circ = CavityGeometry(shape=CavityShape.CIRCULAR, dims=(1.0,))

    for _run in range(n_runs):
        r = run_benchmark(
            "solve_rectangular_cavity",
            solve_cavity_modes,
            geom_rect,
            nx=cfg["nx"],
            ny=cfg["ny"],
            num_modes=cfg["num_modes"],
        )
        results.append(r)

        r = run_benchmark(
            "solve_circular_cavity",
            solve_cavity_modes,
            geom_circ,
            nx=cfg["nx"],
            ny=cfg["ny"],
            num_modes=cfg["num_modes"],
        )
        results.append(r)

    # ── Barcode ─────────────────────────────────────────────────────────
    mode_data = solve_cavity_modes(
        geom_rect, nx=cfg["nx"], ny=cfg["ny"], num_modes=cfg["num_modes"]
    )
    e_field = np.array(next(iter(mode_data["e_modes"].values()))["field"])
    h_field = np.array(next(iter(mode_data["h_modes"].values()))["field"])

    for _run in range(n_runs):
        r = run_benchmark(
            "topological_fingerprint",
            topological_fingerprint,
            e_field,
            threshold=0.1,
        )
        results.append(r)

        r = run_benchmark(
            "coupled_fingerprint",
            coupled_fingerprint,
            e_field,
            h_field,
            threshold=0.1,
        )
        results.append(r)

    # ── ManifoldProjector ────────────────────────────────────────────────
    fp = coupled_fingerprint(e_field, h_field)
    emb = embed_fingerprint(fp, dim=50)
    mp = ManifoldProjector(input_dim=50, latent_dim=16)

    for _run in range(n_runs):
        r = run_benchmark(
            "manifold_projector_encode",
            mp.encode,
            emb,
        )
        results.append(r)

    # ── GodTensor end-to-end ────────────────────────────────────────────
    def run_gt():
        gt = GodTensor(n_geometries=cfg["n_geometries"])
        gt.collect_training_data(
            nx=cfg["nx"],
            ny=cfg["ny"],
            num_modes=cfg["num_modes"],
            seed=42,
        )
        gt.learn_T()
        gt.find_fixed_point(iters=cfg["iters"])

    for _run in range(n_runs):
        r = run_benchmark("god_tensor_full_pipeline", run_gt)
        results.append(r)

    total_duration = time.perf_counter() - t0_total

    benchmark_report = BenchmarkReport(
        suite=suite_name,
        timestamp=str(np.datetime64("now")),
        total_duration_s=total_duration,
        results=results,
        metadata={
            "n_runs": n_runs,
            "config": cfg,
        },
    )

    if include_validation:
        val_report = run_validation_experiment(
            n_total=cfg.get("n_geometries", 30),
            nx=cfg["nx"],
            ny=cfg["ny"],
            num_modes=cfg["num_modes"],
        )
        return benchmark_report, val_report

    return benchmark_report


def save_report(
    report: BenchmarkReport,
    output_dir: str = ".",
    formats: list[str] | None = None,
) -> None:
    """Save a benchmark report to disk.

    Parameters
    ----------
    report : BenchmarkReport
    output_dir : str
        Directory to write output files.
    formats : list of str, optional
        List of formats: ``["json", "csv"]``.
    """
    if formats is None:
        formats = ["json"]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "json" in formats:
        path = out / f"benchmark_{report.suite}.json"
        with open(path, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"Wrote {path}")

    if "csv" in formats:
        path = out / f"benchmark_{report.suite}.csv"
        with open(path, "w") as fh:
            fh.write("name,duration_s\n")
            for r in report.results:
                fh.write(f"{r.name},{r.duration_s:.6f}\n")
        print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# CLI entry point (importable)
# ---------------------------------------------------------------------------

def cli_main():
    """CLI entry point for benchmarking."""
    import argparse

    parser = argparse.ArgumentParser(description="Run faraday benchmarks")
    parser.add_argument(
        "--suite",
        choices=list(BENCHMARK_SUITES.keys()),
        default="small",
        help="Benchmark suite to run",
    )
    parser.add_argument(
        "--output",
        default=".",
        help="Output directory for results",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repetitions",
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Also run the held-out generalization experiment and print the ValidationReport summary.",
    )
    args = parser.parse_args()

    if args.validate:
        bench, val = run_suite(suite_name=args.suite, n_runs=args.runs, include_validation=True)
        save_report(bench, output_dir=args.output, formats=[args.format])
        for r in bench.results:
            print(f"  {r.name}: {r.duration_s:.4f}s")
        print()
        print(val.summary())
    else:
        report = run_suite(suite_name=args.suite, n_runs=args.runs)
        save_report(report, output_dir=args.output, formats=[args.format])
        for r in report.results:
            print(f"  {r.name}: {r.duration_s:.4f}s")


if __name__ == "__main__":
    cli_main()
