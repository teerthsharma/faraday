#!/usr/bin/env python3
# Copyright (c) 2026 Teerth Sharma. All rights reserved.
"""
Experimental Studies for the Faraday Computational Topology Paper.

Runs four studies and saves JSON results + matplotlib figures:

1. Quantitative comparison: predicted vs ground-truth FDFD Betti numbers
2. Convergence rate: spectral gap vs number of training geometries
3. Ablation study: Hilbert embedding vs raw, cubical vs Rips
4. Scaling study: accuracy vs grid resolution (nx, ny)

Usage:
    python experiments/run_all_studies.py [--output-dir figures/]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Matplotlib setup (Agg backend for headless environments)
# ---------------------------------------------------------------------------
import matplotlib
import numpy as np

from faraday.barcode import coupled_fingerprint, topological_fingerprint
from faraday.em_solver import CavityGeometry, CavityShape, solve_cavity_modes
from faraday.god_tensor import GodTensor
from faraday.manifold_projector import embed_fingerprint
from faraday.predict import predict_eh_barcode

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
})


# =========================================================================
# Study 1: Quantitative Comparison — Predicted vs Ground-Truth FDFD
# =========================================================================

def study_1_quantitative_comparison(
    n_train: int = 25,
    n_test: int = 10,
    nx: int = 30,
    ny: int = 30,
    num_modes: int = 4,
    seed: int = 42,
) -> dict:
    """Train a GodTensor on n_train geometries, predict n_test held-out."""
    print("\n" + "=" * 60)
    print("  STUDY 1: Quantitative Comparison — Predicted vs FDFD")
    print("=" * 60)

    rng = np.random.default_rng(seed)

    # Train
    gt = GodTensor(n_geometries=n_train)
    gt.collect_training_data(nx=nx, ny=ny, num_modes=num_modes, seed=seed)
    gt.learn_T()
    gt.find_fixed_point(iters=500, tol=1e-10)

    # Generate test geometries
    test_geoms: list[tuple[tuple[float, ...], str]] = []
    for _ in range(n_test * 2):  # generate extra in case some fail
        if rng.random() < 0.7:
            w = float(rng.uniform(0.8, 3.0))
            h = float(rng.uniform(0.5, 2.0))
            test_geoms.append(((w, h), "rect"))
        else:
            r = float(rng.uniform(0.5, 1.5))
            test_geoms.append(((r,), "circle"))

    results = []
    for params, shape in test_geoms:
        if len(results) >= n_test:
            break
        try:
            if shape == "rect":
                geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=params)
            else:
                geom = CavityGeometry(shape=CavityShape.CIRCULAR, dims=params)
            mode_data = solve_cavity_modes(geom, nx=nx, ny=ny, num_modes=num_modes, seed=seed)
            e_field = np.array(next(iter(mode_data["e_modes"].values()))["field"])
            h_field = np.array(next(iter(mode_data["h_modes"].values()))["field"])
            actual = coupled_fingerprint(e_field, h_field)
            if "error" in actual.get("e_fingerprint", {}) or "error" in actual.get("h_fingerprint", {}):
                continue

            pred = predict_eh_barcode(gt, params, shape)
            results.append({
                "geometry": list(params),
                "shape": shape,
                "actual_e_betti0": int(actual["e_fingerprint"]["betti_0"]),
                "predicted_e_betti0": float(pred["knn_e_fingerprint"]["betti_0"]),
                "actual_h_betti0": int(actual["h_fingerprint"]["betti_0"]),
                "predicted_h_betti0": float(pred["knn_h_fingerprint"]["betti_0"]),
                "actual_coupling": float(actual["coupling_strength"]),
                "predicted_coupling": float(pred["coupling_score"]),
                "god_distance_e": float(pred["god_distance_e"]),
                "god_distance_h": float(pred["god_distance_h"]),
            })
        except Exception as exc:
            print(f"  [skip] {params}: {exc}")
            continue

    e_errors = [abs(r["actual_e_betti0"] - r["predicted_e_betti0"]) for r in results]
    h_errors = [abs(r["actual_h_betti0"] - r["predicted_h_betti0"]) for r in results]
    coupling_errors = [abs(r["actual_coupling"] - r["predicted_coupling"]) for r in results]

    summary = {
        "n_train": n_train,
        "n_test": len(results),
        "mean_e_betti0_error": float(np.mean(e_errors)) if e_errors else 0.0,
        "mean_h_betti0_error": float(np.mean(h_errors)) if h_errors else 0.0,
        "mean_coupling_error": float(np.mean(coupling_errors)) if coupling_errors else 0.0,
        "god_score": float(gt.god_score()),
        "per_geometry": results,
    }

    print(f"  n_train={n_train}, n_test={len(results)}")
    print(f"  mean E Betti-0 error: {summary['mean_e_betti0_error']:.3f}")
    print(f"  mean H Betti-0 error: {summary['mean_h_betti0_error']:.3f}")
    print(f"  mean coupling error:  {summary['mean_coupling_error']:.4f}")
    print(f"  god_score:            {summary['god_score']:.4f}")

    return summary


def plot_study_1(data: dict, output_dir: str) -> None:
    """Plot predicted vs actual Betti numbers and coupling strength."""
    results = data["per_geometry"]
    if not results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # --- Scatter: E Betti-0 ---
    actual_e = [r["actual_e_betti0"] for r in results]
    pred_e = [r["predicted_e_betti0"] for r in results]
    axes[0].scatter(actual_e, pred_e, c="#2563eb", s=60, alpha=0.8, edgecolors="white", linewidth=0.5, zorder=3)
    lim = max(max(actual_e), max(pred_e)) + 1
    axes[0].plot([0, lim], [0, lim], "k--", alpha=0.4, linewidth=1, label="y = x")
    axes[0].set_xlabel("Actual E Betti-0 (FDFD)")
    axes[0].set_ylabel("Predicted E Betti-0 (God Tensor)")
    axes[0].set_title("E-field Betti-0")
    axes[0].legend()
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    # --- Scatter: H Betti-0 ---
    actual_h = [r["actual_h_betti0"] for r in results]
    pred_h = [r["predicted_h_betti0"] for r in results]
    axes[1].scatter(actual_h, pred_h, c="#dc2626", s=60, alpha=0.8, edgecolors="white", linewidth=0.5, zorder=3)
    lim = max(max(actual_h), max(pred_h)) + 1
    axes[1].plot([0, lim], [0, lim], "k--", alpha=0.4, linewidth=1, label="y = x")
    axes[1].set_xlabel("Actual H Betti-0 (FDFD)")
    axes[1].set_ylabel("Predicted H Betti-0 (God Tensor)")
    axes[1].set_title("H-field Betti-0")
    axes[1].legend()
    axes[1].set_aspect("equal")
    axes[1].grid(True, alpha=0.3)

    # --- Bar: God Distance ---
    god_dists = [r["god_distance_e"] for r in results]
    x_pos = np.arange(len(results))
    colors = ["#16a34a" if d < 1.0 else "#dc2626" for d in god_dists]
    axes[2].bar(x_pos, god_dists, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
    axes[2].axhline(1.0, color="black", linestyle="--", alpha=0.4, linewidth=1)
    axes[2].set_xlabel("Test Geometry Index")
    axes[2].set_ylabel("God Distance (E-side)")
    axes[2].set_title("Convergence to God Tensor")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Study 1: Predicted vs Ground-Truth FDFD ({data['n_train']} train / {data['n_test']} test)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "study1_quantitative_comparison.png"))
    fig.savefig(os.path.join(output_dir, "study1_quantitative_comparison.pdf"))
    plt.close(fig)
    print("  [saved] study1_quantitative_comparison.png/pdf")


# =========================================================================
# Study 2: Convergence Rate — Spectral Gap vs Training Set Size
# =========================================================================

def study_2_convergence_rate(
    sizes: list[int] | None = None,
    nx: int = 25,
    ny: int = 25,
    num_modes: int = 3,
    seed: int = 42,
) -> dict:
    """How does spectral gap and god_score change with more training geometries?"""
    print("\n" + "=" * 60)
    print("  STUDY 2: Convergence Rate — Spectral Gap vs N_train")
    print("=" * 60)

    if sizes is None:
        sizes = [5, 8, 12, 16, 20, 25, 30, 40, 50]

    results = []
    for n in sizes:
        t0 = time.perf_counter()
        gt = GodTensor(n_geometries=n)
        gt.collect_training_data(nx=nx, ny=ny, num_modes=num_modes, seed=seed)
        if len(gt.samples) < 3:
            print(f"  [skip] n={n}: only {len(gt.samples)} valid samples")
            continue
        gt.learn_T()
        gt.find_fixed_point(iters=500, tol=1e-12)
        elapsed = time.perf_counter() - t0

        entry = {
            "n_geometries": n,
            "n_samples": len(gt.samples),
            "spectral_gap": float(gt.spectral_gap),
            "dominant_eigenvalue_real": float(np.real(gt.dominant_eigenvalue)),
            "dominant_eigenvalue_imag": float(np.imag(gt.dominant_eigenvalue)),
            "final_residual": float(gt.final_residual),
            "converged": gt.fixed_point_converged,
            "god_score": float(gt.god_score()),
            "elapsed_s": elapsed,
        }
        results.append(entry)
        print(
            f"  n={n:3d} -> samples={len(gt.samples):3d}, "
            f"gap={gt.spectral_gap:.4f}, score={gt.god_score():.4f}, "
            f"residual={gt.final_residual:.2e}, {elapsed:.1f}s"
        )

    return {"sizes": sizes, "results": results}


def plot_study_2(data: dict, output_dir: str) -> None:
    """Plot spectral gap, god_score, and residual vs training set size."""
    results = data["results"]
    if not results:
        return

    ns = [r["n_samples"] for r in results]
    gaps = [r["spectral_gap"] for r in results]
    scores = [r["god_score"] for r in results]
    residuals = [r["final_residual"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Spectral gap
    axes[0].plot(ns, gaps, "o-", color="#2563eb", markersize=6, linewidth=1.5)
    axes[0].set_xlabel("Number of Training Samples")
    axes[0].set_ylabel(r"Spectral Gap $|\lambda_2/\lambda_1|$")
    axes[0].set_title("Spectral Gap vs Training Size")
    axes[0].grid(True, alpha=0.3)

    # God score
    axes[1].plot(ns, scores, "s-", color="#16a34a", markersize=6, linewidth=1.5)
    axes[1].set_xlabel("Number of Training Samples")
    axes[1].set_ylabel("God Score")
    axes[1].set_title("God Score vs Training Size")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    # Final residual
    axes[2].semilogy(ns, residuals, "^-", color="#dc2626", markersize=6, linewidth=1.5)
    axes[2].set_xlabel("Number of Training Samples")
    axes[2].set_ylabel("Final Spectral Residual")
    axes[2].set_title("Convergence Residual vs Training Size")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Study 2: Convergence Rate Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "study2_convergence_rate.png"))
    fig.savefig(os.path.join(output_dir, "study2_convergence_rate.pdf"))
    plt.close(fig)
    print("  [saved] study2_convergence_rate.png/pdf")


# =========================================================================
# Study 3: Ablation Study
# =========================================================================

def study_3_ablation(
    n_train: int = 20,
    nx: int = 25,
    ny: int = 25,
    num_modes: int = 3,
    seed: int = 42,
) -> dict:
    """Compare cubical vs Rips filtration, and Hilbert embedding vs raw fingerprint."""
    print("\n" + "=" * 60)
    print("  STUDY 3: Ablation Study")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    configs = {
        "cubical+hilbert": {"filtration": "cubical", "use_hilbert": True},
        "rips+hilbert": {"filtration": "rips", "use_hilbert": True},
        "cubical+raw": {"filtration": "cubical", "use_hilbert": False},
        "rips+raw": {"filtration": "rips", "use_hilbert": False},
    }

    # Generate a fixed set of geometries
    geom_params: list[tuple[float, ...]] = []
    for _ in range(n_train):
        w = float(rng.uniform(0.8, 3.0))
        h = float(rng.uniform(0.5, 2.0))
        geom_params.append((w, h))

    # Precompute all modes once
    precomputed: list[tuple[tuple[float, ...], np.ndarray, np.ndarray]] = []
    for params in geom_params:
        try:
            geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=params)
            mode_data = solve_cavity_modes(geom, nx=nx, ny=ny, num_modes=num_modes, seed=seed)
            e_field = np.array(next(iter(mode_data["e_modes"].values()))["field"])
            h_field = np.array(next(iter(mode_data["h_modes"].values()))["field"])
            precomputed.append((params, e_field, h_field))
        except Exception:
            continue

    results = {}
    for name, cfg in configs.items():
        t0 = time.perf_counter()
        filtration = cfg["filtration"]
        use_hilbert = cfg["use_hilbert"]

        # Build training samples manually
        from faraday.god_tensor import TrainingSample

        samples: list[TrainingSample] = []
        for params, e_field, h_field in precomputed:
            try:
                e_fp = topological_fingerprint(e_field, filtration=filtration)
                h_fp = topological_fingerprint(h_field, filtration=filtration)
                if "error" in e_fp or "error" in h_fp:
                    continue

                if use_hilbert:
                    # Use the Hilbert-series embedding
                    e_emb = embed_fingerprint(e_fp, dim=50)
                    h_emb = embed_fingerprint(h_fp, dim=50)
                else:
                    # Raw: just stack the fingerprint scalars into a vector
                    e_emb = embed_fingerprint(e_fp, dim=50)  # same function but skip Hilbert
                    h_emb = embed_fingerprint(h_fp, dim=50)
                    # To actually ablate Hilbert, zero out the lifetime-statistics slots
                    # and keep only Betti / topological score (pure counting features)
                    e_emb[:10] = 0.0
                    e_emb[10:20] = 0.0
                    h_emb[:10] = 0.0
                    h_emb[10:20] = 0.0
                    # Re-normalise
                    en = float(np.linalg.norm(e_emb))
                    hn = float(np.linalg.norm(h_emb))
                    if en > 1e-12:
                        e_emb = e_emb / en
                    if hn > 1e-12:
                        h_emb = h_emb / hn

                sample = TrainingSample(
                    geometry_params=params,
                    e_fingerprint=e_fp,
                    h_fingerprint=h_fp,
                    e_embedding=e_emb,
                    h_embedding=h_emb,
                    k_values=[],
                )
                samples.append(sample)
            except Exception:
                continue

        if len(samples) < 5:
            results[name] = {"error": "too few samples", "n_samples": len(samples)}
            continue

        gt = GodTensor(n_geometries=len(samples))
        gt.samples = samples
        try:
            gt.learn_T()
            gt.find_fixed_point(iters=500, tol=1e-10)
        except Exception as exc:
            results[name] = {"error": str(exc), "n_samples": len(samples)}
            continue

        elapsed = time.perf_counter() - t0
        entry = {
            "n_samples": len(samples),
            "spectral_gap": float(gt.spectral_gap),
            "god_score": float(gt.god_score()),
            "final_residual": float(gt.final_residual),
            "converged": gt.fixed_point_converged,
            "elapsed_s": elapsed,
        }
        results[name] = entry
        print(f"  {name:20s}: score={entry['god_score']:.4f}, gap={entry['spectral_gap']:.4f}, {elapsed:.1f}s")

    return {"configs": list(configs.keys()), "results": results}


def plot_study_3(data: dict, output_dir: str) -> None:
    """Bar chart comparing ablation configurations."""
    results = data["results"]
    names = [k for k in data["configs"] if "error" not in results.get(k, {})]
    if not names:
        return

    scores = [results[n]["god_score"] for n in names]
    gaps = [results[n]["spectral_gap"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    colors = ["#2563eb", "#dc2626", "#16a34a", "#f59e0b"]

    x = np.arange(len(names))
    axes[0].bar(x, scores, color=colors[:len(names)], alpha=0.85, edgecolor="white", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].set_ylabel("God Score")
    axes[0].set_title("God Score by Configuration")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, gaps, color=colors[:len(names)], alpha=0.85, edgecolor="white", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_ylabel(r"Spectral Gap $|\lambda_2/\lambda_1|$")
    axes[1].set_title("Spectral Gap by Configuration")
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Study 3: Ablation Study", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "study3_ablation.png"))
    fig.savefig(os.path.join(output_dir, "study3_ablation.pdf"))
    plt.close(fig)
    print("  [saved] study3_ablation.png/pdf")


# =========================================================================
# Study 4: Scaling Study — Accuracy vs Grid Resolution
# =========================================================================

def study_4_scaling(
    resolutions: list[int] | None = None,
    n_train: int = 15,
    num_modes: int = 3,
    seed: int = 42,
) -> dict:
    """How does eigenvalue accuracy and god_score change with grid resolution?"""
    print("\n" + "=" * 60)
    print("  STUDY 4: Scaling Study — Accuracy vs Grid Resolution")
    print("=" * 60)

    if resolutions is None:
        resolutions = [10, 15, 20, 30, 40, 60, 80]

    from faraday.em_solver import rectangular_analytic_k

    test_geom = (2.0, 1.0)  # fixed rectangular cavity
    analytic_k = rectangular_analytic_k(test_geom[0], test_geom[1], num_modes=num_modes)

    results = []
    for res in resolutions:
        t0 = time.perf_counter()

        # Eigenvalue accuracy
        geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=test_geom)
        try:
            mode_data = solve_cavity_modes(geom, nx=res, ny=res, num_modes=num_modes, seed=seed)
        except Exception as exc:
            print(f"  [skip] nx=ny={res}: {exc}")
            continue

        computed_k = mode_data["k_values"][:num_modes]
        n_compare = min(len(computed_k), len(analytic_k))
        rel_errors = [
            abs(computed_k[i] - analytic_k[i]) / max(abs(analytic_k[i]), 1e-12)
            for i in range(n_compare)
        ]
        mean_rel_error = float(np.mean(rel_errors)) if rel_errors else 1.0
        max_rel_error = float(np.max(rel_errors)) if rel_errors else 1.0

        # God score at this resolution
        gt = GodTensor(n_geometries=n_train)
        gt.collect_training_data(nx=res, ny=res, num_modes=num_modes, seed=seed)
        if len(gt.samples) < 3:
            continue
        gt.learn_T()
        gt.find_fixed_point(iters=300, tol=1e-10)

        elapsed = time.perf_counter() - t0
        entry = {
            "resolution": res,
            "n_dof": res * res,
            "mean_k_relative_error": mean_rel_error,
            "max_k_relative_error": max_rel_error,
            "god_score": float(gt.god_score()),
            "spectral_gap": float(gt.spectral_gap),
            "final_residual": float(gt.final_residual),
            "elapsed_s": elapsed,
        }
        results.append(entry)
        print(
            f"  nx=ny={res:3d}: mean_k_err={mean_rel_error:.2e}, "
            f"score={gt.god_score():.4f}, {elapsed:.1f}s"
        )

    return {"resolutions": resolutions, "test_geometry": list(test_geom), "results": results}


def plot_study_4(data: dict, output_dir: str) -> None:
    """Plot eigenvalue error and god_score vs grid resolution."""
    results = data["results"]
    if not results:
        return

    res = [r["resolution"] for r in results]
    k_err = [r["mean_k_relative_error"] for r in results]
    scores = [r["god_score"] for r in results]
    ndof = [r["n_dof"] for r in results]
    times = [r["elapsed_s"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].loglog(ndof, k_err, "o-", color="#2563eb", markersize=6, linewidth=1.5)
    axes[0].set_xlabel("Degrees of Freedom (N = nx * ny)")
    axes[0].set_ylabel("Mean Relative Eigenvalue Error")
    axes[0].set_title(r"FDFD Eigenvalue Error vs Grid Size")
    axes[0].grid(True, alpha=0.3, which="both")
    # Add O(h^2) reference line
    if len(ndof) > 2:
        n0, e0 = ndof[0], k_err[0]
        ref_ndof = np.array(ndof)
        ref_err = e0 * (n0 / ref_ndof)
        axes[0].plot(ref_ndof, ref_err, "k--", alpha=0.4, label=r"$O(h^2)$ reference")
        axes[0].legend()

    axes[1].plot(res, scores, "s-", color="#16a34a", markersize=6, linewidth=1.5)
    axes[1].set_xlabel("Grid Resolution (nx = ny)")
    axes[1].set_ylabel("God Score")
    axes[1].set_title("God Score vs Resolution")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(res, times, "^-", color="#f59e0b", markersize=6, linewidth=1.5)
    axes[2].set_xlabel("Grid Resolution (nx = ny)")
    axes[2].set_ylabel("Wall-Clock Time (s)")
    axes[2].set_title("Computation Time vs Resolution")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Study 4: Scaling Study — Geometry ({data['test_geometry'][0]} x {data['test_geometry'][1]})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "study4_scaling.png"))
    fig.savefig(os.path.join(output_dir, "study4_scaling.pdf"))
    plt.close(fig)
    print("  [saved] study4_scaling.png/pdf")


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run all Faraday experimental studies")
    parser.add_argument("--output-dir", default="figures", help="Output directory for figures and data")
    parser.add_argument("--fast", action="store_true", help="Use smaller parameters for quick testing")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.fast:
        # Smaller parameters for CI / quick testing
        s1_kw = {"n_train": 10, "n_test": 5, "nx": 15, "ny": 15, "num_modes": 2}
        s2_kw = {"sizes": [5, 8, 12, 16, 20], "nx": 15, "ny": 15, "num_modes": 2}
        s3_kw = {"n_train": 10, "nx": 15, "ny": 15, "num_modes": 2}
        s4_kw = {"resolutions": [10, 15, 20, 30], "n_train": 8, "num_modes": 2}
    else:
        s1_kw = {"n_train": 25, "n_test": 10, "nx": 30, "ny": 30, "num_modes": 4}
        s2_kw = {"nx": 25, "ny": 25, "num_modes": 3}
        s3_kw = {"n_train": 20, "nx": 25, "ny": 25, "num_modes": 3}
        s4_kw = {"n_train": 15, "num_modes": 3}

    all_results = {}

    # Study 1
    data1 = study_1_quantitative_comparison(**s1_kw)
    plot_study_1(data1, output_dir)
    all_results["study_1_quantitative_comparison"] = data1

    # Study 2
    data2 = study_2_convergence_rate(**s2_kw)
    plot_study_2(data2, output_dir)
    all_results["study_2_convergence_rate"] = data2

    # Study 3
    data3 = study_3_ablation(**s3_kw)
    plot_study_3(data3, output_dir)
    all_results["study_3_ablation"] = data3

    # Study 4
    data4 = study_4_scaling(**s4_kw)
    plot_study_4(data4, output_dir)
    all_results["study_4_scaling"] = data4

    # Save all results as JSON
    json_path = os.path.join(output_dir, "all_studies_results.json")
    with open(json_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\n  [saved] {json_path}")

    print("\n" + "=" * 60)
    print("  ALL STUDIES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
