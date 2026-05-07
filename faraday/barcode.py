# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.barcode — Field to Persistent Homology

Converts E and H field distributions into persistent homology barcodes.
The barcode is the topological signature of the field's structure —
it captures connected components (H0), loops/holes (H1), and voids (H2).

E and H fields of the same cavity mode share the same topology,
but their SIGNATURES are different — the God Tensor learns the mapping.
"""

from __future__ import annotations

import numpy as np

from faraday.logging import get_logger

log = get_logger(__name__)


def field_to_pointcloud(
    field: np.ndarray, threshold: float = 0.1, add_phase: bool = True
) -> np.ndarray:
    """
    Convert field magnitude to a point cloud for topological analysis.

    Points = grid positions where |field| > threshold * max(|field|).
    If add_phase=True, use (x, y, phase) as third coordinate instead of magnitude.

    Args:
        field: 2D complex field array
        threshold: fraction of max field to include
        add_phase: if True, use (x, y, phase) as 3rd coord; else use (x, y, |E|)

    Returns:
        points: (N, 3) numpy array of points
    """
    mag = np.abs(field)
    mask = mag > threshold * mag.max()
    if not mask.any():
        mask = mag > threshold * mag.mean()

    ny, nx = field.shape
    y_coords, x_coords = np.where(mask)

    if add_phase:
        phase = np.angle(field)
        points = np.column_stack(
            [
                x_coords / nx,
                y_coords / ny,
                (phase[mask] + np.pi) / (2 * np.pi),  # normalize phase to [0,1]
            ]
        )
    else:
        points = np.column_stack(
            [
                x_coords / nx,
                y_coords / ny,
                mag[mask] / (mag.max() + 1e-10),
            ]
        )
    return points


def compute_barcodes(
    data: np.ndarray, filtration: str = "rips", max_dim: int = 1, metric: str = "euclidean"
) -> dict:
    """
    Compute persistent homology barcodes.
    
    Args:
        data: (N, d) point cloud if filtration="rips", or (ny, nx) 2D array if filtration="cubical".
        filtration: 'rips' or 'cubical'.
        max_dim: maximum homology dimension to compute.
        metric: distance metric for rips.

    Returns:
        dict with betti_0, betti_1, num_h0_bars, num_h1_bars, diagrams
    """
    diagrams = []
    
    if filtration == "rips":
        try:
            from ripser import ripser
        except ImportError:
            return {"error": "ripser not installed — run: pip install ripser"}

        result = ripser(data, maxdim=max_dim, metric=metric)
        diagrams = result["dgms"]
        
    elif filtration == "cubical":
        try:
            import gudhi
        except ImportError:
            return {"error": "gudhi not installed — run: pip install gudhi"}
            
        # Cubical complex requires a grid. We compute superlevel sets by negating the data.
        # This highlights the peaks of the EM field.
        complex = gudhi.CubicalComplex(top_dimensional_cells=-data)
        complex.compute_persistence()
        
        # Extract diagrams for each dimension
        diagrams = []
        for dim in range(max_dim + 1):
            dgm = complex.persistence_intervals_in_dimension(dim)
            if len(dgm) > 0:
                diagrams.append(np.array(dgm))
            else:
                diagrams.append(np.empty((0, 2)))
    else:
        return {"error": f"Unknown filtration {filtration}"}

    betti = []
    for _d, dgm in enumerate(diagrams):
        if len(dgm) > 0:
            lifetimes = dgm[:, 1] - dgm[:, 0]
            # Handle infinities (for cubical, gudhi uses inf)
            lifetimes = lifetimes[np.isfinite(lifetimes)]
            if len(lifetimes) > 0:
                # Persistent features: bars significantly longer than median gap
                threshold = np.median(lifetimes) if len(lifetimes) > 1 else 0
                betti.append(len(lifetimes[lifetimes > threshold]))
            else:
                betti.append(0)
        else:
            betti.append(0)

    while len(betti) < 2:
        betti.append(0)

    return {
        "betti_0": betti[0],  # connected components
        "betti_1": betti[1],  # loops / holes
        "num_h0_bars": len(diagrams[0]),
        "num_h1_bars": len(diagrams[1]) if len(diagrams) > 1 else 0,
        "h0_lifetimes": [
            float(x) for x in (diagrams[0][:, 1] - diagrams[0][:, 0]) if not np.isinf(x)
        ] if len(diagrams) > 0 else [],
        "h1_lifetimes": [
            float(x) for x in (diagrams[1][:, 1] - diagrams[1][:, 0]) if not np.isinf(x)
        ]
        if len(diagrams) > 1
        else [],
        "diagrams": [dgm.tolist() for dgm in diagrams],
    }


def topological_fingerprint(
    field: np.ndarray, threshold: float = 0.1, filtration: str = "cubical"
) -> dict:
    """
    Full topological analysis of an EM field distribution.
    Returns Betti numbers, barcode statistics, and field metrics.

    Args:
        field: 2D complex field array
        threshold: fraction of max for point cloud extraction (used only for rips)
        filtration: 'rips' or 'cubical'

    Returns:
        dict with betti_0, betti_1, h0/h1 bars, field statistics, topological score
    """
    if filtration == "cubical":
        barcodes = compute_barcodes(np.abs(field), filtration="cubical")
    else:
        points = field_to_pointcloud(field, threshold)
        if len(points) < 10:
            return {"error": "Too few points for topological analysis"}
        barcodes = compute_barcodes(points, filtration="rips")

    mag = np.abs(field)

    total_energy = float(np.sum(mag**2))
    peak_energy = float(np.sum(mag[mag > threshold * mag.max()] ** 2))
    confinement_ratio = peak_energy / (total_energy + 1e-10)

    return {
        "betti_0": barcodes.get("betti_0", 0),
        "betti_1": barcodes.get("betti_1", 0),
        "h0_bars": barcodes.get("num_h0_bars", 0),
        "h1_bars": barcodes.get("num_h1_bars", 0),
        "h0_lifetimes": barcodes.get("h0_lifetimes", []),
        "h1_lifetimes": barcodes.get("h1_lifetimes", []),
        "field_max": float(mag.max()),
        "field_mean": float(mag.mean()),
        "field_std": float(mag.std()),
        "confinement_ratio": confinement_ratio,
        "num_grid_points": int(np.sum(mag > threshold * mag.max())),
        "topological_score": float(
            barcodes.get("betti_1", 0) * (np.mean(barcodes["h1_lifetimes"]) if barcodes.get("h1_lifetimes") else 0.0)
            + barcodes.get("betti_0", 0) * (np.mean(barcodes["h0_lifetimes"]) if barcodes.get("h0_lifetimes") else 0.0)
        ),
        "diagrams": barcodes.get("diagrams", []),
    }


def coupled_fingerprint(
    e_field: np.ndarray, h_field: np.ndarray, threshold: float = 0.1, filtration: str = "cubical"
) -> dict:
    """
    Compute coupled topological fingerprints for E and H fields together.

    Uses the Poynting vector approach: E field and |S| = |E| times |H| energy flux
    are the two scalar fields for topological comparison.

    Returns both individual fingerprints PLUS the cross-field coupling metrics:
      - emd_S: Earth Mover's Distance between |E| and |H| point clouds
              (0 = identical topology, higher = more decoupling)
      - coupling_strength: 1 / (1 + emd_S) -- bounded [0, 1]
    """
    e_fp = topological_fingerprint(e_field, threshold, filtration=filtration)
    h_fp = topological_fingerprint(h_field, threshold, filtration=filtration)

    e_pts = field_to_pointcloud(e_field, threshold, add_phase=False)
    h_pts = field_to_pointcloud(h_field, threshold, add_phase=False)

    emd_S = 1.0
    if len(e_pts) >= 5 and len(h_pts) >= 5:
        try:
            from scipy.stats import wasserstein_distance
            # Use simplified 1D wasserstein on magnitudes for performance
            emd_S = float(wasserstein_distance(e_pts[:, 2], h_pts[:, 2]))
        except ImportError:
            pass

    coupling_strength = 1.0 / (1.0 + emd_S)
    
    return {
        "e_fingerprint": e_fp,
        "h_fingerprint": h_fp,
        "emd_S": emd_S,
        "coupling_strength": coupling_strength,
    }
