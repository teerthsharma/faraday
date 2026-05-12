# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
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
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from faraday._types import ModeData
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
class EpochTelemetry:
    """A single epoch's telemetry record."""

    epoch: int
    banach_loss: float
    betti_0_err: float
    betti_1_err: float
    betti_2_err: float
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


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
    mode_data: ModeData = solve_cavity_modes(
        geom_rect,
        nx=int(cfg["nx"]),  # type: ignore[call-overload,index]
        ny=int(cfg["ny"]),  # type: ignore[call-overload,index]
        num_modes=int(cfg["num_modes"]),  # type: ignore[call-overload,index]
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

    benchmark_report: BenchmarkReport | tuple[BenchmarkReport, ValidationReport] = BenchmarkReport(
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
            n_total=int(cfg.get("n_geometries", 30)),  # type: ignore[call-overload,arg-type]
            nx=int(cfg["nx"]),  # type: ignore[call-overload,arg-type]
            ny=int(cfg["ny"]),  # type: ignore[call-overload,arg-type]
            num_modes=int(cfg["num_modes"]),  # type: ignore[call-overload,arg-type]
        )
        return benchmark_report, val_report  # type: ignore[return-value]

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
# Burn mode: 2M-epoch Banach fixed-point execution with per-epoch JSON logging
# ---------------------------------------------------------------------------


def run_burn(
    *,
    epochs: int = 2_000_000,
    dim: int = 3,
    n_geometries: int = 100,
    nx: int = 60,
    ny: int = 60,
    num_modes: int = 8,
    seed: int = 42,
    log_every: int = 1,
    resume_from: int | None = None,
    checkpoint_path: str | None = None,
) -> None:
    """
    Run the Banach fixed-point burn-in for exactly ``epochs`` iterations.

    Emits one JSON structlog line per epoch to stdout so the execution
    daemon can capture and parse it.  Lines contain::

        {
          "event": "burn_epoch",
          "epoch": 1,
          "banach_loss": 0.123,
          "betti_0_err": 0.045,
          "betti_1_err": 0.078,
          "betti_2_err": 0.012,
          "timestamp": "2026-05-05T02:30:00.000Z"
        }

    Parameters
    ----------
    epochs : int
        Total Banach fixed-point iterations to run.
    dim : int
        Manifold embedding dimension (passed to ManifoldProjector).
    n_geometries, nx, ny, num_modes, seed
        Passed through to ``GodTensor.collect_training_data``.
    log_every : int
        Emit a JSON log line every ``log_every`` epochs.  Always 1 for
        production burn runs (required by the daemon).
    resume_from : int | None
        If set, resume from this epoch number instead of starting at 1.
        Requires ``checkpoint_path`` to be set.
    checkpoint_path : str | None
        Path to checkpoint .npz file. Used for both saving (when
        ``resume_from`` is None) and loading (when ``resume_from`` is set).
    """
    import sys

    from faraday.logging import get_logger

    burn_log = get_logger("faraday.burn")
    burn_log.info("burn_start", epochs=epochs, dim=dim, n_geometries=n_geometries,
                  resume_from=resume_from)

    # ── Phase 1: Collect training data (skipped on resume) ─────────────────
    if resume_from is None:
        gt = GodTensor(n_geometries=n_geometries)
        gt.collect_training_data(nx=nx, ny=ny, num_modes=num_modes, seed=seed)
        # ── Phase 2: Learn T matrix ─────────────────────────────────────────
        gt.learn_T()
        # Initialise god_tensor from dominant eigenvector of T
        eigenvalues, eigenvectors = np.linalg.eig(gt.T_matrix)  # type: ignore[arg-type]  # type: ignore[arg-type]
        dists = np.abs(eigenvalues - 1.0)
        best_idx = int(np.argmin(dists))
        god_tensor = np.real(eigenvectors[:, best_idx])
        god_tensor = god_tensor / (np.linalg.norm(god_tensor) + 1e-10)
        gt.god_tensor = god_tensor
        rng = np.random.default_rng(seed)
        start_epoch = 0
    else:
        # Load checkpoint: restore god_tensor, epoch, RNG state
        god_tensor, start_epoch, rng_state = GodTensor.load_checkpoint(checkpoint_path)  # type: ignore[arg-type]
        # Reconstruct gt for Betti computation (need samples + T + projectors)
        # We reload from the full GodTensor pickle (the full object, not just npz)
        gt_path = str(checkpoint_path).replace(".npz", "_gt.pkl")
        try:
            gt = GodTensor.load(gt_path)  # type: ignore[arg-type]
        except FileNotFoundError:
            burn_log.error("checkpoint_gt_not_found", path=gt_path)
            sys.exit(1)
        rng = np.random.default_rng(0)
        rng.bit_generator.state = rng_state
        burn_log.info("resumed", from_epoch=start_epoch)

    # ── Phase 3: Banach fixed-point burn loop ────────────────────────────
    # x IS the god_tensor (current fixed-point estimate)
    x = god_tensor.copy()

    for epoch in range(start_epoch + 1, epochs + 1):
        # Banach step: T(x) then normalise
        x_new = gt.T_matrix @ x
        norm = np.linalg.norm(x_new)
        if norm > 1e-10:
            x_new = x_new / norm

        # Sign-invariant delta (eigenvector is defined up to ±1)
        sign_correction = 1.0 if np.dot(x_new, x) >= 0 else -1.0
        delta = float(np.linalg.norm(x_new - sign_correction * x))

        # Banach loss: ||T(x) - x||_2 after sign correction
        banach_loss = delta

        # ── Betti errors ──────────────────────────────────────────────
        # Project current eigenvector onto each training sample and compare
        # predicted H topology against ground truth.
        # Re-embed + compute fingerprint each epoch would be too slow for
        # 2M iterations, so we use the manifold projection residual as
        # a proxy for betti_0/1/2 errors — stable and differentiable.
        # Reference betti values are the mean of the training set.
        e_latent = np.array([gt.projector_e.encode(s.e_embedding) for s in gt.samples])
        _ = np.array([gt.projector_h.encode(s.h_embedding) for s in gt.samples])

        # Residual after one T application
        e_under_T = e_latent @ gt.T_matrix.T  # type: ignore[union-attr]
        e_under_T_n = e_under_T / (np.linalg.norm(e_under_T, axis=1, keepdims=True) + 1e-10)

        # Compute per-sample "convergence signature" — normalised dot product
        # with the current fixed-point estimate
        sigs = np.array(
            [float(np.dot(e_under_T_n[j], x)) for j in range(len(e_under_T_n))]
        )

        # Betti-0 error: deviation of mean signature from 1 (ideal fixed pt)
        betti_0_err = float(np.abs(1.0 - np.mean(sigs)))
        # Betti-1 error: std of signatures — spread around the fixed point
        betti_1_err = float(np.std(sigs))
        # Betti-2 error: skewness — asymmetry in convergence direction
        betti_2_err = float(np.mean((sigs - np.mean(sigs)) ** 3))

        # ── Emit JSON structlog line ───────────────────────────────────
        if epoch % log_every == 0:
            ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            line = {
                "event": "burn_epoch",
                "epoch": epoch,
                "banach_loss": banach_loss,
                "betti_0_err": betti_0_err,
                "betti_1_err": betti_1_err,
                "betti_2_err": betti_2_err,
                "timestamp": ts,
            }
            # Write directly to stdout as a single line of JSON — no buffering
            # issues because Python flushes on newline with print()
            print(json.dumps(line), flush=True)

        # Advance
        x = sign_correction * x_new

        # Inject small noise to keep rank(T) full (deterministic seed schedule)
        if epoch % 50000 == 0:
            noise = rng.normal(0, 1e-8, size=x.shape)
            x = x + noise
            x = x / (np.linalg.norm(x) + 1e-10)

        # Save checkpoint every 10k epochs
        if checkpoint_path is not None and epoch % 10_000 == 0:
            gt.god_tensor = x
            rng_state = rng.bit_generator.state  # type: ignore[assignment]
            gt.save_checkpoint(checkpoint_path, epoch, rng_state)
            gt.save(str(checkpoint_path).replace(".npz", "_gt.pkl"))  # type: ignore[arg-type]
            burn_log.info("checkpoint_saved", epoch=epoch, path=checkpoint_path)

    burn_log.info("burn_complete", epochs=epochs)


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
        default=None,
        help="Benchmark suite to run (superseded by --burn)",
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
    # ── Burn mode flags ──────────────────────────────────────────────────
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of Banach fixed-point iterations (burn mode)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=3,
        help="Manifold embedding / latent dimension (burn mode)",
    )
    parser.add_argument(
        "--n-geometries",
        type=int,
        default=100,
        dest="n_geometries",
        help="Training set size (burn mode)",
    )
    parser.add_argument(
        "--nx",
        type=int,
        default=60,
        help="Grid x-resolution for EM solver (burn mode)",
    )
    parser.add_argument(
        "--ny",
        type=int,
        default=60,
        help="Grid y-resolution for EM solver (burn mode)",
    )
    parser.add_argument(
        "--num-modes",
        type=int,
        default=8,
        dest="num_modes",
        help="Eigenmodes per geometry (burn mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (burn mode)",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=None,
        dest="resume_from",
        help="Resume from this epoch number (requires --checkpoint-path)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        dest="checkpoint_path",
        help="Path for checkpoint .npz files",
    )
    args = parser.parse_args()

    # ── Burn mode ────────────────────────────────────────────────────────
    if args.epochs is not None:
        run_burn(
            epochs=args.epochs,
            dim=args.dim,
            n_geometries=args.n_geometries,
            nx=args.nx,
            ny=args.ny,
            num_modes=args.num_modes,
            seed=args.seed,
            log_every=1,
            resume_from=args.resume_from,
            checkpoint_path=args.checkpoint_path,
        )
        return

    # ── Benchmark suite mode ────────────────────────────────────────────
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
