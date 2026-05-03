"""
faraday.predict — Predict E and H Field Topology for New Geometries

Once the God Tensor is found, use it to predict the topological signature
of E and H fields for any new cavity geometry — WITHOUT running FDFD.

The God Tensor captures the coupling law. Given geometry params,
we interpolate the E and H embeddings from the training manifold,
then project back to barcode space for visualization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import math

from .god_tensor import GodTensor, TrainingSample
from .em_solver import CavityGeometry, CavityShape, solve_cavity_modes
from .barcode import coupled_fingerprint, topological_fingerprint
from .manifold_projector import embed_fingerprint


def predict_eh_barcode(
    gt: GodTensor,
    geometry_params: Tuple[float, ...],
    shape: str = "rect",
) -> Dict:
    """
    Predict E and H barcode signatures for a new geometry.

    Uses the God Tensor to project from geometry -> E_signature -> H_signature
    (or the reverse). Instead of running FDFD, we interpolate on the
    learned manifold.

    Args:
        gt: trained GodTensor
        geometry_params: e.g. (w, h) for rect or (r,) for circle
        shape: "rect" or "circle"

    Returns:
        dict with predicted E and H fingerprints, God Tensor projection
    """
    params = np.array(geometry_params)

    # Interpolate E and H embeddings from training data via geometry similarity
    similarities = []
    for sample in gt.samples:
        s_params = np.array(sample.geometry_params)
        dist = float(np.linalg.norm(params - s_params))
        sim = math.exp(-dist ** 2 / 0.5)  # RBF weight
        similarities.append((sim, sample))

    similarities.sort(key=lambda x: -x[0])
    top_k = similarities[:5]  # 5 nearest neighbors

    total_weight = sum(sim for sim, _ in top_k)
    e_interp = sum(sim / total_weight * s.e_embedding for sim, s in top_k)
    h_interp = sum(sim / total_weight * s.h_embedding for sim, s in top_k)

    # Normalize
    e_interp = e_interp / (np.linalg.norm(e_interp) + 1e-10)
    h_interp = h_interp / (np.linalg.norm(h_interp) + 1e-10)

    # Project through God Tensor
    e_via_gt = gt.get_e_to_h_map(e_interp)  # E -> T(E) = predicted H
    h_via_gt = gt.get_h_to_e_map(h_interp)  # H -> T(H) = predicted E

    # Decode from latent back to fingerprint (simplified: reconstruct)
    # We store the embedding directly as the predicted signature
    e_pred_fp = _embed_to_fingerprint(e_via_gt, source="e")
    h_pred_fp = _embed_to_fingerprint(h_via_gt, source="h")

    # Distance from God Tensor
    god_dist_e = float(np.linalg.norm(e_via_gt - gt.god_tensor))
    god_dist_h = float(np.linalg.norm(h_via_gt - gt.god_tensor))

    return {
        "geometry_params": geometry_params,
        "shape": shape,
        "predicted_e_fingerprint": e_pred_fp,
        "predicted_h_fingerprint": h_pred_fp,
        "e_via_god_tensor": e_via_gt.tolist(),
        "h_via_god_tensor": h_via_gt.tolist(),
        "god_tensor": gt.god_tensor.tolist(),
        "god_distance_e": god_dist_e,
        "god_distance_h": god_dist_h,
        "coupling_score": gt.god_score(),
    }


def _embed_to_fingerprint(embedding: np.ndarray, source: str = "e") -> Dict:
    """
    Convert a manifold embedding back to a fingerprint-like dict.
    Inverse of embed_fingerprint for a 16D embedding:
      [:10] → h0 lifetimes,  [10:] → h1 lifetimes
    """
    dim = len(embedding)
    h0 = embedding[:min(10, dim)]
    h1 = embedding[min(10, dim):]

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
    test_geometries: List[Tuple[float, ...]],
    shapes: List[str],
    nx: int = 50,
    ny: int = 50,
) -> Dict:
    """
    Benchmark: compare God Tensor predictions vs actual FDFD solutions
    for a set of held-out geometries.

    Returns:
        dict with per-geometry errors and aggregate metrics
    """
    results = []
    for params, shape in zip(test_geometries, shapes):
        # Get actual fingerprint from FDFD
        if shape == "rect":
            geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=params)
        else:
            geom = CavityGeometry(shape=CavityShape.CIRCULAR, dims=params)

        try:
            mode_data = solve_cavity_modes(geom, nx=nx, ny=ny, num_modes=6)
        except Exception as e:
            continue

        e_field = np.array(list(mode_data["e_modes"].values())[0]["field"])
        h_field = np.array(list(mode_data["h_modes"].values())[0]["field"])
        actual = coupled_fingerprint(e_field, h_field)

        # Get prediction
        pred = predict_eh_barcode(gt, params, shape)

        # Compute error
        e_error = abs(actual["e_fingerprint"]["betti_0"] - pred["predicted_e_fingerprint"]["betti_0"])
        h_error = abs(actual["h_fingerprint"]["betti_0"] - pred["predicted_h_fingerprint"]["betti_0"])
        coupling_error = abs(actual["coupling_strength"] - pred["coupling_score"])

        results.append({
            "geometry": params,
            "shape": shape,
            "e_betti0_actual": actual["e_fingerprint"]["betti_0"],
            "e_betti0_predicted": pred["predicted_e_fingerprint"]["betti_0"],
            "e_error": e_error,
            "h_error": h_error,
            "coupling_error": coupling_error,
            "god_distance": pred["god_distance_e"],
        })

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
