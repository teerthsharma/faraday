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

    Returns both individual fingerprints PLUS the cross-field metrics:
      - phase_alignment: correlation between E and H phase distributions
      - magnitude_ratio: |E|/|H| statistics
      - coupling_strength: how tightly coupled the two fields are topologically
    """
    e_fp = topological_fingerprint(e_field, threshold)
    h_fp = topological_fingerprint(h_field, threshold)

    # Cross-field coupling metrics
    e_mag = np.abs(e_field)
    h_mag = np.abs(h_field)
    e_phase = np.angle(e_field)
    h_phase = np.angle(h_field)

    # Mask where both fields are significant
    both = (e_mag > threshold * e_mag.max()) & (h_mag > threshold * h_mag.max())
    if np.sum(both) > 0:
        phase_diff = e_phase[both] - h_phase[both]
        phase_alignment = float(np.abs(np.mean(np.exp(1j * phase_diff))))
    else:
        phase_alignment = 0.0

    ratio = e_mag / (h_mag + 1e-10)
    magnitude_ratio = float(np.median(ratio[both])) if np.sum(both) > 0 else 1.0

    # Coupling strength: how similar are the barcode structures?
    e_sig = [e_fp.get("betti_0", 0), e_fp.get("betti_1", 0), e_fp.get("h0_bars", 0), e_fp.get("h1_bars", 0)]
    h_sig = [h_fp.get("betti_0", 0), h_fp.get("betti_1", 0), h_fp.get("h0_bars", 0), h_fp.get("h1_bars", 0)]

    # Euclidean distance between topological signatures
    coupling_strength = float(1.0 / (1.0 + np.linalg.norm(np.array(e_sig) - np.array(h_sig))))

    return {
        "e_fingerprint": e_fp,
        "h_fingerprint": h_fp,
        "phase_alignment": phase_alignment,
        "magnitude_ratio": magnitude_ratio,
        "coupling_strength": coupling_strength,
    }
