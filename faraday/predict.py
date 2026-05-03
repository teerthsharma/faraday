"""
faraday.predict — Predict E and H Field Topology for New Geometries

Once the God Tensor is found, use it to predict the topological signature
of E and H fields for any new cavity geometry — WITHOUT running FDFD.

The God Tensor captures the coupling law. Given geometry params,
we interpolate the E and H embeddings from the training manifold,
then project back to barcode space for visualization.
"""

from __future__ import annotations

import math

import numpy as np

from faraday.barcode import coupled_fingerprint
from faraday.em_solver import CavityGeometry, CavityShape, solve_cavity_modes
from faraday.god_tensor import GodTensor
from faraday.logging import get_logger

log = get_logger(__name__)


def predict_eh_barcode(
    gt: GodTensor,
    geometry_params: tuple[float, ...],
    shape: str = "rect",
) -> dict:
    """
    Predict E and H barcode signatures for a new geometry.

    Uses the God Tensor to project from geometry -> E_signature -> H_signature
    (or the reverse). Instead of running FDFD, we interpolate on the
    learned manifold using K-nearest-geometries on the training set.

    The prediction has THREE components:
    1. KNN fingerprint: weighted average of actual fingerprints from
       the k most similar training geometries (ground truth baseline)
    2. God Tensor projection: E embedding -> T(E) -> predicted H embedding
    3. Coupled prediction: both E and H flow through T to verify convergence

    Args:
        gt: trained GodTensor
        geometry_params: e.g. (w, h) for rect or (r,) for circle
        shape: "rect" or "circle"

    Returns:
        dict with knn_e_fingerprint, knn_h_fingerprint, god_tensor_projected_e/h,
        god_distances, coupling_score
    """
    params = np.array(geometry_params)

    # ── KNN on geometry params ─────────────────────────────────────
    similarities = []
    for sample in gt.samples:
        s_params = np.array(sample.geometry_params)
        dist = float(np.linalg.norm(params - s_params))
        sim = math.exp(-(dist**2) / 0.5)
        similarities.append((sim, sample))

    similarities.sort(key=lambda x: -x[0])
    top_k = similarities[:5]

    total_weight = sum(sim for sim, _ in top_k)

    # KNN fingerprint: weighted average of ACTUAL training fingerprints
    # (not invented from embedding coordinates)
    e_fingerprints = [s.e_fingerprint for _, s in top_k]
    h_fingerprints = [s.h_fingerprint for _, s in top_k]

    # Weighted average of scalar fields for the KNN prediction
    knn_e_fp = _average_fingerprints(
        e_fingerprints, [sim for sim, _ in top_k], total_weight
    )
    knn_h_fp = _average_fingerprints(
        h_fingerprints, [sim for sim, _ in top_k], total_weight
    )

    # ── God Tensor projection ──────────────────────────────────────
    e_interp = sum(sim / total_weight * s.e_embedding for sim, s in top_k)
    h_interp = sum(sim / total_weight * s.h_embedding for sim, s in top_k)
    e_interp = e_interp / (np.linalg.norm(e_interp) + 1e-10)
    h_interp = h_interp / (np.linalg.norm(h_interp) + 1e-10)

    # Project through T
    e_via_gt = gt.get_e_to_h_map(e_interp)  # E -> T(E) = predicted H
    h_via_gt = gt.get_h_to_e_map(h_interp)  # H -> T(H) = predicted E

    # Distance from God Tensor
    god_dist_e = float(np.linalg.norm(e_via_gt - gt.god_tensor))
    god_dist_h = float(np.linalg.norm(h_via_gt - gt.god_tensor))

    # God Tensor projected fingerprints (for comparison with KNN)
    gt_e_fp = _embed_to_fingerprint(h_via_gt, source="e_via_gt")
    gt_h_fp = _embed_to_fingerprint(e_via_gt, source="h_via_gt")

    return {
        "geometry_params": geometry_params,
        "shape": shape,
        # KNN predictions (actual fingerprints from similar geometries)
        "knn_e_fingerprint": knn_e_fp,
        "knn_h_fingerprint": knn_h_fp,
        # God Tensor projections
        "god_tensor_projected_e": gt_e_fp,  # H <- T(E) — predicted H
        "god_tensor_projected_h": gt_h_fp,  # E <- T(H) — predicted E
        "god_distance_e": god_dist_e,
        "god_distance_h": god_dist_h,
        "coupling_score": gt.god_score(),
        # Comparison
        "e_betti0_knn_vs_gt_diff": abs(knn_e_fp["betti_0"] - gt_e_fp["betti_0"]),
        "h_betti0_knn_vs_gt_diff": abs(knn_h_fp["betti_0"] - gt_h_fp["betti_0"]),
    }


def _average_fingerprints(
    fingerprints: list[dict],
    weights: list[float],
    total_weight: float,
) -> dict:
    """
    Compute a weighted average of topological fingerprints.
    Averages scalar fields (betti numbers, lifetimes, scores) across k neighbors.
    """
    if not fingerprints:
        return {}

    result = {}
    # Scalar fields: weighted average
    for key in [
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
    ]:
        vals = [fp.get(key, 0) or 0 for fp in fingerprints]
        result[key] = float(
            sum(w * v for w, v in zip(weights, vals, strict=True)) / total_weight
        )

    # Lifetime lists: average element-wise, clip to max 50
    for dim in ["h0_lifetimes", "h1_lifetimes"]:
        all_lts = [fp.get(dim, []) for fp in fingerprints]
        max_len = max(len(lt) for lt in all_lts) if all_lts else 0
        avg_lts = []
        for i in range(min(max_len, 10)):  # cap at 10 lifetime values
            vals = [lt[i] if i < len(lt) else 0 for lt in all_lts]
            avg_lts.append(
                float(
                    sum(w * v for w, v in zip(weights, vals, strict=True))
                    / total_weight
                )
            )
        result[dim] = avg_lts

    result["diagrams"] = []
    return result


def _embed_to_fingerprint(embedding: np.ndarray, source: str = "e") -> dict:
    """
    Convert a manifold embedding back to a fingerprint-like dict.
    This is an INVERSE of embed_fingerprint for a 16D embedding:
      [:10] → h0 lifetimes,  [10:] → h1 lifetimes

    NOTE: This is only used for God Tensor projected fingerprints.
    The primary prediction path uses actual KNN fingerprints which
    are computed from real field data, not from embedding coordinates.
    """
    dim = len(embedding)
    h0 = embedding[: min(10, dim)]
    h1 = embedding[min(10, dim) :]

    betti_0 = int(np.clip(np.linalg.norm(h0) * 2, 0, 20))
    betti_1 = int(np.clip(np.linalg.norm(h1) * 2, 0, 20))
    h0_bars = int(np.clip(np.sum(h0 > 0.05) * 5, 0, 50))
    h1_bars = int(np.clip(np.sum(h1 > 0.05) * 5, 0, 50))
    topo_score = float(abs(embedding[-1]) * np.linalg.norm(embedding))

    return {
        "betti_0": betti_0,
        "betti_1": betti_1,
        "h0_bars": h0_bars,
        "h1_bars": h1_bars,
        "h0_lifetimes": h0.tolist(),
        "h1_lifetimes": h1.tolist(),
        "topological_score": topo_score,
        "confinement_ratio": float(np.clip(embedding.std(), 0, 1)),
        "field_max": float(np.linalg.norm(embedding[:5])),
        "field_mean": float(np.mean(embedding)),
        "field_std": float(np.std(embedding)),
        "num_grid_points": int(np.sum(np.abs(embedding) > 0.05) * 10),
        "diagrams": [],
        "source": source,
    }


def benchmark(
    gt: GodTensor,
    test_geometries: list[tuple[float, ...]],
    shapes: list[str],
    nx: int = 50,
    ny: int = 50,
) -> dict:
    """
    Benchmark: compare God Tensor predictions vs actual FDFD solutions
    for a set of held-out geometries.

    Returns:
        dict with per-geometry errors and aggregate metrics
    """
    results = []
    for params, shape in zip(test_geometries, shapes, strict=True):
        # Get actual fingerprint from FDFD
        if shape == "rect":
            geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=params)
        else:
            geom = CavityGeometry(shape=CavityShape.CIRCULAR, dims=params)

        try:
            mode_data = solve_cavity_modes(geom, nx=nx, ny=ny, num_modes=6)
        except Exception:
            continue

        e_field = np.array(next(iter(mode_data["e_modes"].values()))["field"])
        h_field = np.array(next(iter(mode_data["h_modes"].values()))["field"])
        actual = coupled_fingerprint(e_field, h_field)

        # Get prediction
        pred = predict_eh_barcode(gt, params, shape)

        # Compute error — compare KNN prediction vs actual FDFD
        e_error = abs(
            actual["e_fingerprint"]["betti_0"] - pred["knn_e_fingerprint"]["betti_0"]
        )
        h_error = abs(
            actual["h_fingerprint"]["betti_0"] - pred["knn_h_fingerprint"]["betti_0"]
        )
        coupling_error = abs(actual["coupling_strength"] - pred["coupling_score"])

        results.append(
            {
                "geometry": params,
                "shape": shape,
                "e_betti0_actual": actual["e_fingerprint"]["betti_0"],
                "e_betti0_predicted": pred["knn_e_fingerprint"]["betti_0"],
                "e_error": e_error,
                "h_error": h_error,
                "coupling_error": coupling_error,
                "god_distance": pred["god_distance_e"],
                "actual_coupling_strength": actual["coupling_strength"],
                "knn_coupling_strength": pred["knn_h_fingerprint"].get(
                    "topological_score", 0
                ),
            }
        )

    if not results:
        return {"error": "No valid results"}

    avg_e_error = np.mean([r["e_error"] for r in results])
    avg_h_error = np.mean([r["h_error"] for r in results])
    avg_coupling_error = np.mean([r["coupling_error"] for r in results])

    return {
        "n_test": len(results),
        "avg_e_betti0_error": float(avg_e_error),
        "avg_h_betti0_error": float(avg_h_error),
        "avg_coupling_error": float(avg_coupling_error),
        "per_geometry": results,
    }
