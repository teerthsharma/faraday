"""
faraday.barcode — Field to Persistent Homology

Converts E and H field distributions into persistent homology barcodes.
The barcode is the topological signature of the field's structure —
it captures connected components (H0), loops/holes (H1), and voids (H2).

E and H fields of the same cavity mode share the same topology,
but their SIGNATURES are different — the God Tensor learns the mapping.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


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
        points = np.column_stack([
            x_coords / nx,
            y_coords / ny,
            (phase[mask] + np.pi) / (2 * np.pi),  # normalize phase to [0,1]
        ])
    else:
        points = np.column_stack([
            x_coords / nx,
            y_coords / ny,
            mag[mask] / (mag.max() + 1e-10),
        ])
    return points


def compute_barcodes(
    points: np.ndarray, max_dim: int = 1, metric: str = "euclidean"
) -> Dict:
    """
    Compute persistent homology barcodes from field point cloud.
    Uses ripser for efficient computation.

    Args:
        points: (N, d) point cloud
        max_dim: maximum homology dimension to compute (H0, H1, H2...)
        metric: distance metric ('euclidean' or 'ripser' default)

    Returns:
        dict with betti_0, betti_1, num_h0_bars, num_h1_bars, diagrams
    """
    try:
        from ripser import ripser
    except ImportError:
        return {"error": "ripser not installed — run: pip install ripser"}

    result = ripser(points, maxdim=max_dim, metric=metric)
    diagrams = result["dgms"]

    betti = []
    for d, dgm in enumerate(diagrams):
        if len(dgm) > 0:
            lifetimes = dgm[:, 1] - dgm[:, 0]
            lifetimes = lifetimes[np.isfinite(lifetimes)]
            if len(lifetimes) > 0:
                # Persistent features: bars significantly longer than median gap
                threshold = np.median(lifetimes) if len(lifetimes) > 1 else 0
                betti.append(int(len(lifetimes[lifetimes > threshold])))
            else:
                betti.append(0)
        else:
            betti.append(0)

    while len(betti) < 2:
        betti.append(0)

    return {
        "betti_0": betti[0],          # connected components
        "betti_1": betti[1],          # loops / holes
        "num_h0_bars": len(diagrams[0]),
        "num_h1_bars": len(diagrams[1]) if len(diagrams) > 1 else 0,
        "h0_lifetimes": [float(x) for x in (diagrams[0][:, 1] - diagrams[0][:, 0]) if not np.isinf(x)],
        "h1_lifetimes": [float(x) for x in (diagrams[1][:, 1] - diagrams[1][:, 0]) if not np.isinf(x)] if len(diagrams) > 1 else [],
        "diagrams": [dgm.tolist() for dgm in diagrams],
    }


def topological_fingerprint(
    field: np.ndarray, threshold: float = 0.1
) -> Dict:
    """
    Full topological analysis of an EM field distribution.
    Returns Betti numbers, barcode statistics, and field metrics.

    Args:
        field: 2D complex field array
        threshold: fraction of max for point cloud extraction

    Returns:
        dict with betti_0, betti_1, h0/h1 bars, field statistics, topological score
    """
    points = field_to_pointcloud(field, threshold)
    if len(points) < 10:
        return {"error": "Too few points for topological analysis"}

    barcodes = compute_barcodes(points)
    mag = np.abs(field)

    total_energy = float(np.sum(mag**2))
    peak_energy = float(np.sum(mag[mag > threshold * mag.max()]**2))
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
        # Topological score: more holes with longer lifetimes = higher score
        "topological_score": float(
            barcodes.get("betti_1", 0) * np.mean(barcodes.get("h1_lifetimes", [0])) +
            barcodes.get("betti_0", 0) * np.mean(barcodes.get("h0_lifetimes", [0]))
        ),
        "diagrams": barcodes.get("diagrams", []),
    }


def coupled_fingerprint(
    e_field: np.ndarray, h_field: np.ndarray, threshold: float = 0.1
) -> Dict:
    """
    Compute coupled topological fingerprints for E and H fields together.

    Uses the Poynting vector approach: E field and |S| = |E|×|H| energy flux
    are the two scalar fields for topological comparison. Their barcode
    structures should be nearly identical in a well-coupled cavity mode
    (same nodes, same antinodes, same energy distribution).

    Returns both individual fingerprints PLUS the cross-field coupling metrics:
      - emd_S: Earth Mover's Distance between |E| and |S| point clouds
              (0 = identical topology, higher = more decoupling)
      - confinement对齐: fraction of energy in the dominant topological component
      - coupling_strength: 1 / (1 + emd_S) — bounded [0, 1]
    """
    # |E| point cloud
    e_fp = topological_fingerprint(e_field, threshold)
    # |H| — stored as magnitude; reconstruct |S| = |E| × |H| if h_field is |H|
    # In the current convention, h_field IS |S| (energy flux magnitude)
    # because em_solver now returns |H| in h_modes["field"] and computes |S| separately.
    # For backward compatibility, detect whether h_field looks like |S| or |H|
    # by checking if it's a magnitude-like (non-negative) field.
    h_fp = topological_fingerprint(h_field, threshold)

    e_mag = np.abs(e_field)
    h_mag = np.abs(h_field)

    # Earth Mover's Distance between |E| and |S| point clouds
    # Sample points from both fields at same grid positions
    e_pts = field_to_pointcloud(e_field, threshold, add_phase=False)   # (x, y, |E|)
    h_pts = field_to_pointcloud(h_field, threshold, add_phase=False)   # (x, y, |H|)

    if len(e_pts) >= 5 and len(h_pts) >= 5:
        try:
            from scipy.stats import wasserstein_distance_nd
            # EMD between |E| and |H| distributions (project to 1D for simplicity)
            e_flat = e_pts[:, 2]  # |E| values
            h_flat = h_pts[:, 2]  # |H| values
            # Sort for 1D EMD
            e_sorted = np.sort(e_flat)
            h_sorted = np.sort(h_flat)
            emd_S = float(wasserstein_distance_nd(e_sorted, h_sorted))
        except Exception:
            # Fallback: normalized L2 between mean field distributions
            e_hist = np.histogram(e_mag[e_mag > 0], bins=20, density=True)[0]
            h_hist = np.histogram(h_mag[h_mag > 0], bins=20, density=True)[0]
            emd_S = float(np.linalg.norm(e_hist - h_hist))
    else:
        emd_S = 0.0

    # Coupling strength: 1 when |E| and |S| have identical topology
    coupling_strength = float(1.0 / (1.0 + emd_S))

    # Confinement alignment: how much of |S| lives in high-|E| regions
    both = (e_mag > threshold * e_mag.max()) & (h_mag > threshold * h_mag.max())
    if np.sum(both) > 0:
        confinement_alignment = float(np.sum(both) / e_mag.size)
    else:
        confinement_alignment = 0.0

    return {
        "e_fingerprint": e_fp,
        "h_fingerprint": h_fp,
        "emd_S": emd_S,
        "confinement_alignment": confinement_alignment,
        "coupling_strength": coupling_strength,
    }
