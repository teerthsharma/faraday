# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.predict — Predict E/H Topology for New Geometries.

Given a trained :class:`GodTensor` we predict the topological signature
of a *new* cavity geometry without running FDFD.  The prediction has
four components, two of which act as baselines:

1. **Geometry-space KNN** (baseline #1) — Gaussian-weighted average of
   the actual E and H fingerprints of the :math:`k`-nearest training
   geometries.

2. **Barcode-space KNN** (baseline #2) — :math:`k=1` nearest E
   embedding, return its paired H fingerprint.

3. **God Tensor projection** — push the interpolated E-embedding through
   the operator :math:`T` and report the distance from the dominant
   eigen-vector :math:`g`.  Smaller distance ⇒ higher fidelity to the
   learned coupling law.

4. **Coupling score** — :func:`GodTensor.god_score` evaluated on the
   training set (proxy for the global tightness of the coupling).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from faraday.barcode import coupled_fingerprint
from faraday.em_solver import CavityGeometry, CavityShape, solve_cavity_modes
from faraday.god_tensor import GodTensor
from faraday.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def predict_h_from_e_baseline(
    gt: GodTensor, e_embedding: np.ndarray, k: int = 1
) -> dict[str, Any]:
    """Barcode-space :math:`k`-NN baseline.

    Find the :math:`k` E-embeddings nearest to ``e_embedding`` in the
    training set, then return the (uniform) average of their H
    fingerprints.
    """
    e_embedding = np.asarray(e_embedding, dtype=np.float64)
    distances = sorted(
        (
            (float(np.linalg.norm(e_embedding - s.e_embedding)), s)
            for s in gt.samples
        ),
        key=lambda x: x[0],
    )
    top_k = distances[:k]
    fingerprints = [s.h_fingerprint for _, s in top_k]
    weights = [1.0] * len(top_k)
    total = float(len(top_k))
    return _average_fingerprints(fingerprints, weights, total)


# ---------------------------------------------------------------------------
# Main predictor
# ---------------------------------------------------------------------------


def predict_eh_barcode(
    gt: GodTensor,
    geometry_params: tuple[float, ...],
    shape: str = "rect",
    k: int = 5,
    bandwidth: float = 0.5,
) -> dict[str, Any]:
    """Predict E/H barcode signatures for a *new* geometry.

    Parameters
    ----------
    gt : GodTensor
        A trained God Tensor (must have ``T_matrix`` and ``god_tensor``).
    geometry_params : tuple[float, ...]
        ``(w, h)`` for rectangular, ``(r,)`` for circular.
    shape : {"rect", "circle"}
    k : int
        Number of nearest training geometries to use for KNN.
    bandwidth : float
        Gaussian-kernel bandwidth in geometry space.
    """
    if gt.T_matrix is None or gt.god_tensor is None:
        raise ValueError(
            "GodTensor must be trained (call learn_T then find_fixed_point)"
        )
    god = np.asarray(gt.god_tensor, dtype=np.float64)
    params = np.asarray(geometry_params, dtype=np.float64)

    # ── KNN on geometry params ─────────────────────────────────────
    similarities = sorted(
        (
            (
                math.exp(
                    -float(np.linalg.norm(params - np.asarray(s.geometry_params))) ** 2
                    / max(2 * bandwidth, 1e-12)
                ),
                s,
            )
            for s in gt.samples
        ),
        key=lambda x: -x[0],
    )
    top_k = similarities[:k]
    total_w = sum(sim for sim, _ in top_k) or 1.0

    knn_e_fp = _average_fingerprints(
        [s.e_fingerprint for _, s in top_k],
        [sim for sim, _ in top_k],
        total_w,
    )
    knn_h_fp = _average_fingerprints(
        [s.h_fingerprint for _, s in top_k],
        [sim for sim, _ in top_k],
        total_w,
    )

    # ── God Tensor projection ──────────────────────────────────────
    e_interp = np.zeros(50, dtype=np.float64)
    h_interp = np.zeros(50, dtype=np.float64)
    for sim, s in top_k:
        e_interp = e_interp + (sim / total_w) * s.e_embedding
        h_interp = h_interp + (sim / total_w) * s.h_embedding
    e_interp_norm = e_interp / (float(np.linalg.norm(e_interp)) + 1e-12)
    h_interp_norm = h_interp / (float(np.linalg.norm(h_interp)) + 1e-12)

    e_via_T = np.asarray(gt.get_e_to_h_map(e_interp_norm))
    h_via_T = np.asarray(gt.get_h_to_e_map(h_interp_norm))

    god_dist_e = float(np.linalg.norm(e_via_T - god))
    god_dist_h = float(np.linalg.norm(h_via_T - god))

    # Inferred latent vectors (16D, used by the topological surrogate solver)
    e_latent = gt.projector_e.encode(e_interp_norm)
    h_latent = gt.projector_h.encode(h_interp_norm)

    # ── Barcode-space baseline ────────────────────────────────────
    baseline_h_fp = predict_h_from_e_baseline(gt, e_interp_norm, k=1)

    return {
        "geometry_params": geometry_params,
        "shape": shape,
        "knn_e_fingerprint": knn_e_fp,
        "knn_h_fingerprint": knn_h_fp,
        "baseline_h_fingerprint": baseline_h_fp,
        "god_distance_e": god_dist_e,
        "god_distance_h": god_dist_h,
        "coupling_score": gt.god_score(),
        "inferred_e_latent": e_latent.tolist(),
        "inferred_h_latent": h_latent.tolist(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _average_fingerprints(
    fingerprints: list[dict[str, Any]],
    weights: list[float],
    total_weight: float,
) -> dict[str, Any]:
    """Weighted average of topological fingerprints.

    Scalar fields are averaged directly.  Lifetime lists are padded with
    zeros to a common length and averaged element-wise (keeping at most
    the first 10 entries — sufficient for the dominant homology bars).
    """
    if not fingerprints or total_weight <= 0:
        return {}

    result: dict[str, Any] = {}
    scalar_keys = (
        "betti_0",
        "betti_1",
        "h0_bars",
        "h1_bars",
        "topological_score",
        "confinement_ratio",
        "field_max",
        "field_mean",
        "field_std",
        "num_grid_points",
    )
    for key in scalar_keys:
        vals = [float(fp.get(key, 0) or 0) for fp in fingerprints]
        result[key] = float(
            sum(w * v for w, v in zip(weights, vals, strict=True)) / total_weight
        )

    for dim in ("h0_lifetimes", "h1_lifetimes"):
        all_lts = [list(fp.get(dim, [])) for fp in fingerprints]
        max_len = max((len(lt) for lt in all_lts), default=0)
        n = min(max_len, 10)
        avg_lts: list[float] = []
        for i in range(n):
            vals = [float(lt[i]) if i < len(lt) else 0.0 for lt in all_lts]
            avg_lts.append(
                float(
                    sum(w * v for w, v in zip(weights, vals, strict=True))
                    / total_weight
                )
            )
        result[dim] = avg_lts

    result["diagrams"] = []
    return result


# ---------------------------------------------------------------------------
# Benchmark vs FDFD
# ---------------------------------------------------------------------------


def benchmark(
    gt: GodTensor,
    test_geometries: list[tuple[float, ...]],
    shapes: list[str],
    nx: int = 50,
    ny: int = 50,
) -> dict[str, Any]:
    """Compare God Tensor predictions vs ground-truth FDFD."""
    results: list[dict[str, Any]] = []
    for params, shape in zip(test_geometries, shapes, strict=True):
        if shape == "rect":
            geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=params)
        else:
            geom = CavityGeometry(shape=CavityShape.CIRCULAR, dims=params)

        try:
            mode_data = solve_cavity_modes(geom, nx=nx, ny=ny, num_modes=6)
        except Exception:
            continue

        e_field = np.asarray(
            next(iter(mode_data["e_modes"].values()))["field"], dtype=np.float64
        )
        h_field = np.asarray(
            next(iter(mode_data["h_modes"].values()))["field"], dtype=np.float64
        )
        actual = coupled_fingerprint(e_field, h_field)
        pred = predict_eh_barcode(gt, params, shape)

        e_err = abs(
            actual["e_fingerprint"]["betti_0"] - pred["knn_e_fingerprint"]["betti_0"]
        )
        h_err = abs(
            actual["h_fingerprint"]["betti_0"] - pred["knn_h_fingerprint"]["betti_0"]
        )
        baseline_h_err = abs(
            actual["h_fingerprint"]["betti_0"]
            - pred["baseline_h_fingerprint"].get("betti_0", 0)
        )
        coupling_err = abs(actual["coupling_strength"] - pred["coupling_score"])

        results.append(
            {
                "geometry": params,
                "shape": shape,
                "e_betti0_actual": actual["e_fingerprint"]["betti_0"],
                "e_betti0_predicted": pred["knn_e_fingerprint"]["betti_0"],
                "e_error": e_err,
                "h_error": h_err,
                "baseline_h_error": baseline_h_err,
                "coupling_error": coupling_err,
                "god_distance": pred["god_distance_e"],
                "actual_coupling_strength": actual["coupling_strength"],
            }
        )

    if not results:
        return {"error": "No valid results"}

    return {
        "n_test": len(results),
        "avg_e_betti0_error": float(np.mean([r["e_error"] for r in results])),
        "avg_h_betti0_error": float(np.mean([r["h_error"] for r in results])),
        "avg_baseline_h_betti0_error": float(
            np.mean([r["baseline_h_error"] for r in results])
        ),
        "avg_coupling_error": float(np.mean([r["coupling_error"] for r in results])),
        "per_geometry": results,
    }
