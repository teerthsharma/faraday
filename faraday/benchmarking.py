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
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from faraday.barcode import coupled_fingerprint, topological_fingerprint
from faraday.em_solver import (
    CavityGeometry,
    CavityShape,
    solve_cavity_modes,
)
from faraday.god_tensor import GodTensor
from faraday.manifold_projector import ManifoldProjector, embed_fingerprint
from faraday.predict import benchmark as predict_benchmark

# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

# Aliases for convenient CLI use
MICRO = "micro"
SMALL = "small"
MEDIUM = "medium"

BENCHMARK_SUITES = {
    "micro": {
        "description": "Quick sanity check — 2 geometries, low resolution",
        "n_geometries": 3,
        "nx": 15,
        "ny": 15,
        "num_modes": 2,
        "iters": 50,
        "test_n": 2,
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
    memory_mb: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """A full benchmark suite report."""

    suite: str
    timestamp: str
    total_duration_s: float
    results: List[BenchmarkResult]
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


def run_benchmark(
    name: str, fn, *args, **kwargs
) -> BenchmarkResult:
    """Time a single benchmark function."""
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    duration_s = time.perf_counter() - t0
    return BenchmarkResult(name=name, duration_s=duration_s)


def run_suite(
    suite_name: str = "small",
    n_runs: int = 1,
) -> BenchmarkReport:
    """Run a named benchmark suite.

    Parameters
    ----------
    suite_name : str
        One of "micro", "small", "medium".
    n_runs : int
        Number of repetitions for timing stability.

    Returns
    -------
    BenchmarkReport
    """
    if suite_name not in BENCHMARK_SUITES:
        raise ValueError(
            f"Unknown suite {suite_name!r}. Available: {list(BENCHMARK_SUITES)}"
        )

    cfg = BENCHMARK_SUITES[suite_name]
    results: List[BenchmarkResult] = []
    t0_total = time.perf_counter()

    # ── EMSolver ────────────────────────────────────────────────────────
    geom_rect = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0))
    geom_circ = CavityGeometry(shape=CavityShape.CIRCULAR, dims=(1.0,))

    for run in range(n_runs):
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
    e_field = np.array(list(mode_data["e_modes"].values())[0]["field"])
    h_field = np.array(list(mode_data["h_modes"].values())[0]["field"])

    for run in range(n_runs):
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

    for run in range(n_runs):
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

    for run in range(n_runs):
        r = run_benchmark("god_tensor_full_pipeline", run_gt)
        results.append(r)

    total_duration = time.perf_counter() - t0_total

    return BenchmarkReport(
        suite=suite_name,
        timestamp=np.datetime64("now").astype(str),
        total_duration_s=total_duration,
        results=results,
        metadata={
            "n_runs": n_runs,
            "config": cfg,
        },
    )


def save_report(
    report: BenchmarkReport,
    output_dir: str = ".",
    formats: List[str] = None,
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
    args = parser.parse_args()

    report = run_suite(suite_name=args.suite, n_runs=args.runs)
    save_report(report, output_dir=args.output, formats=[args.format])

    # Print summary
    for r in report.results:
        print(f"  {r.name}: {r.duration_s:.4f}s")


if __name__ == "__main__":
    cli_main()
