# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.barcode — Persistent Homology of Electromagnetic Fields.

Given a 2-D scalar EM field :math:`f : \\Omega \\to \\mathbb{R}_{\\ge 0}`
(typically :math:`|E|`, :math:`|H|` or the Poynting magnitude
:math:`|S|=|E||H|`), we compute its persistent homology in two flavours:

* **Cubical filtration** — the natural filtration on the regular grid::

      X_t = \\{(x,y) : -f(x,y) \\le t\\} = \\{(x,y) : f(x,y) \\ge -t\\}

  This is the *superlevel-set* filtration of :math:`f`, which is what we
  want for "where is the field's energy concentrated as a function of
  threshold".  Implemented via :mod:`gudhi.CubicalComplex`.

* **Vietoris–Rips filtration** — useful when we want to compare
  point-cloud distributions, e.g. for the Earth-Mover's-Distance-based
  coupling metric.  Implemented via :mod:`ripser`.

A barcode is a multiset of birth–death pairs

.. math::

   \\mathrm{B}_d(f) = \\{(b_i, d_i)\\}_{i=1}^{n_d}, \\quad d_i > b_i,

per homology dimension :math:`d \\in \\{0, 1, 2\\}`.  The Betti numbers we
report are *significant* features:

.. math::

   \\beta_d^{\\tau}(f) = \\bigl|\\{ i : d_i - b_i > \\tau \\cdot
                          \\max_j(d_j - b_j)\\}\\bigr|,

where :math:`\\tau` is a relative-persistence threshold (default
:math:`\\tau = 0.1`).  This avoids counting numerical-noise bars and is the
convention used by Edelsbrunner & Harer (*Computational Topology*, 2010).

References
----------
* Edelsbrunner & Harer, *Computational Topology*, AMS 2010, ch. VII.
* Bauer, *Ripser*, J. Appl. & Comput. Topology, 2021.
* Cohen-Steiner, Edelsbrunner, Harer, *Stability of persistence
  diagrams*, Discrete Comp. Geom. 37 (2007).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from faraday.exceptions import TopologyError
from faraday.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Field → point cloud
# ---------------------------------------------------------------------------


def field_to_pointcloud(
    field: np.ndarray, threshold: float = 0.1, add_phase: bool = True
) -> np.ndarray:
    """Convert a 2-D field to a (N, 3) point cloud above a relative threshold.

    Points consist of the (normalised) grid coordinates of pixels whose
    magnitude exceeds ``threshold * max(|field|)``.  The third column
    encodes either the local phase (default) or the magnitude.

    Parameters
    ----------
    field : np.ndarray
        2-D array of complex or real field values.
    threshold : float, optional
        Magnitude cut-off as a fraction of the peak.  If no point passes
        the cut-off, falls back to ``threshold * mean``.
    add_phase : bool, optional
        If True (default), use the field phase as the third coordinate;
        else use the normalised magnitude.

    Returns
    -------
    np.ndarray
        Array of shape ``(N, 3)``.
    """
    mag = np.abs(field)
    peak = float(mag.max())
    cutoff = threshold * peak if peak > 0 else threshold * float(mag.mean())
    mask = mag > cutoff
    if not mask.any():
        # Fallback: include the brightest 10% of pixels
        flat = mag.ravel()
        if flat.size == 0:
            return np.empty((0, 3))
        cutoff = float(np.quantile(flat, 0.9))
        mask = mag > cutoff

    ny, nx = field.shape
    y_coords, x_coords = np.where(mask)

    if add_phase:
        phase = np.angle(field) if np.iscomplexobj(field) else np.zeros_like(mag)
        third = (phase[mask] + np.pi) / (2 * np.pi)
    else:
        third = mag[mask] / (peak + 1e-12)

    return np.column_stack(
        [
            x_coords / max(nx, 1),
            y_coords / max(ny, 1),
            third,
        ]
    )


# ---------------------------------------------------------------------------
# Persistent homology
# ---------------------------------------------------------------------------


def _significant_count(
    intervals: np.ndarray, rel_tau: float = 0.1
) -> int:
    """Count bars whose persistence exceeds ``rel_tau * max_persistence``.

    Treats infinite-death bars as always-significant.  This is the
    standard "stability threshold" approach for filtering out short-lived
    homological noise (Edelsbrunner & Harer 2010, §VII.1).
    """
    if intervals.size == 0:
        return 0
    persistence = intervals[:, 1] - intervals[:, 0]
    finite_mask = np.isfinite(persistence)
    n_inf = int((~finite_mask).sum())
    if not finite_mask.any():
        return n_inf
    finite_p = persistence[finite_mask]
    threshold = rel_tau * float(finite_p.max())
    return int((finite_p > threshold).sum() + n_inf)


def compute_barcodes(
    data: np.ndarray,
    filtration: str = "rips",
    max_dim: int = 1,
    metric: str = "euclidean",
    rel_persistence_tau: float = 0.1,
) -> dict[str, Any]:
    """Compute persistent homology barcodes.

    Parameters
    ----------
    data : np.ndarray
        ``(N, d)`` point cloud if ``filtration='rips'``, else ``(ny, nx)``
        scalar field for ``filtration='cubical'``.
    filtration : {'rips', 'cubical'}
    max_dim : int
        Maximum homology dimension to compute.
    metric : str
        Distance metric for Rips (default ``'euclidean'``).
    rel_persistence_tau : float
        Threshold (as fraction of the largest finite persistence) below
        which a bar is treated as numerical noise and not counted in the
        Betti numbers.

    Returns
    -------
    dict
        ``betti_0``, ``betti_1``, ``num_h0_bars``, ``num_h1_bars``,
        ``h0_lifetimes``, ``h1_lifetimes`` and the raw ``diagrams``.
    """
    if filtration == "rips":
        try:
            from ripser import ripser
        except ImportError:
            return {"error": "ripser not installed — pip install ripser"}
        if data.ndim != 2:
            raise TopologyError(
                "rips filtration requires a 2D point cloud", shape=data.shape
            )
        result = ripser(data, maxdim=max_dim, metric=metric)
        diagrams = list(result["dgms"])
    elif filtration == "cubical":
        try:
            import gudhi
        except ImportError:
            return {"error": "gudhi not installed — pip install gudhi"}
        # Superlevel-set filtration: take -f so that gudhi's default
        # sublevel-set semantics give us the right persistence pairs.
        if data.ndim != 2:
            raise TopologyError(
                "cubical filtration requires a 2D scalar field", shape=data.shape
            )
        complex_ = gudhi.CubicalComplex(top_dimensional_cells=-data)
        complex_.compute_persistence()
        diagrams = []
        for d in range(max_dim + 1):
            arr = complex_.persistence_intervals_in_dimension(d)
            diagrams.append(np.asarray(arr) if len(arr) else np.empty((0, 2)))
    else:
        raise TopologyError(f"unknown filtration {filtration!r}", filtration=filtration)

    # Pad up to max_dim+1 dimensions
    while len(diagrams) < max_dim + 1:
        diagrams.append(np.empty((0, 2)))

    betti = [_significant_count(dgm, rel_persistence_tau) for dgm in diagrams]

    def _finite_lifetimes(dgm: np.ndarray) -> list[float]:
        if dgm.size == 0:
            return []
        lt = dgm[:, 1] - dgm[:, 0]
        return [float(x) for x in lt[np.isfinite(lt)]]

    return {
        "betti_0": betti[0],
        "betti_1": betti[1] if len(betti) > 1 else 0,
        "num_h0_bars": int(diagrams[0].shape[0]),
        "num_h1_bars": int(diagrams[1].shape[0]) if len(diagrams) > 1 else 0,
        "h0_lifetimes": _finite_lifetimes(diagrams[0]),
        "h1_lifetimes": _finite_lifetimes(diagrams[1])
        if len(diagrams) > 1
        else [],
        "diagrams": [d.tolist() for d in diagrams],
    }


# ---------------------------------------------------------------------------
# Topological fingerprint
# ---------------------------------------------------------------------------


def topological_fingerprint(
    field: np.ndarray,
    threshold: float = 0.1,
    filtration: str = "cubical",
    rel_persistence_tau: float = 0.1,
) -> dict[str, Any]:
    """Full topological fingerprint of a 2-D EM field.

    Parameters
    ----------
    field : np.ndarray
        2-D field array (real or complex).
    threshold : float
        Used only for the Rips filtration's point-cloud cut-off.
    filtration : {'cubical', 'rips'}
    rel_persistence_tau : float
        Relative-persistence threshold for Betti counting.
    """
    mag = np.abs(field).astype(float)

    if filtration == "cubical":
        barcodes = compute_barcodes(
            mag, filtration="cubical", rel_persistence_tau=rel_persistence_tau
        )
    else:
        points = field_to_pointcloud(field, threshold)
        if len(points) < 10:
            return {"error": "Too few points for topological analysis"}
        barcodes = compute_barcodes(
            points, filtration="rips", rel_persistence_tau=rel_persistence_tau
        )

    if "error" in barcodes:
        return barcodes

    total_energy = float(np.sum(mag**2))
    peak_energy = float(np.sum(mag[mag > threshold * max(mag.max(), 1e-12)] ** 2))
    confinement_ratio = peak_energy / max(total_energy, 1e-12)

    h0_lt = barcodes.get("h0_lifetimes", [])
    h1_lt = barcodes.get("h1_lifetimes", [])
    h0_mean = float(np.mean(h0_lt)) if h0_lt else 0.0
    h1_mean = float(np.mean(h1_lt)) if h1_lt else 0.0

    return {
        "betti_0": barcodes["betti_0"],
        "betti_1": barcodes["betti_1"],
        "h0_bars": barcodes["num_h0_bars"],
        "h1_bars": barcodes["num_h1_bars"],
        "h0_lifetimes": h0_lt,
        "h1_lifetimes": h1_lt,
        "field_max": float(mag.max()),
        "field_mean": float(mag.mean()),
        "field_std": float(mag.std()),
        "confinement_ratio": confinement_ratio,
        "num_grid_points": int(np.sum(mag > threshold * max(mag.max(), 1e-12))),
        "topological_score": float(
            barcodes["betti_1"] * h1_mean + barcodes["betti_0"] * h0_mean
        ),
        "diagrams": barcodes.get("diagrams", []),
    }


# ---------------------------------------------------------------------------
# Coupled E/H fingerprint with EMD coupling metric
# ---------------------------------------------------------------------------


def _wasserstein_2d(
    a: np.ndarray, b: np.ndarray, max_pts: int = 800
) -> float:
    """Optimal-transport distance between two 2-D point clouds.

    Uses the *Sinkhorn-free* cost: for moderately sized clouds we compute
    the exact 1-D Wasserstein on the magnitude column (column index 2),
    which is fast, robust, and theoretically a *lower bound* on the full
    2-D earth-mover's distance.  We additionally regularise it by the
    Wasserstein on each spatial coordinate so the metric remains
    sensitive to spatial separation between distributions.
    """
    from scipy.stats import wasserstein_distance

    if len(a) == 0 or len(b) == 0:
        return 1.0

    # Sub-sample uniformly for speed; full O(N²) EMD is overkill here.
    if len(a) > max_pts:
        a = a[np.linspace(0, len(a) - 1, max_pts).astype(int)]
    if len(b) > max_pts:
        b = b[np.linspace(0, len(b) - 1, max_pts).astype(int)]

    w_x = wasserstein_distance(a[:, 0], b[:, 0])
    w_y = wasserstein_distance(a[:, 1], b[:, 1])
    w_m = wasserstein_distance(a[:, 2], b[:, 2])
    return float((w_x + w_y + w_m) / 3.0)


def coupled_fingerprint(
    e_field: np.ndarray,
    h_field: np.ndarray,
    threshold: float = 0.1,
    filtration: str = "cubical",
    rel_persistence_tau: float = 0.1,
) -> dict[str, Any]:
    """Coupled topological fingerprint of an :math:`(E, H)` mode pair.

    Returns the individual fingerprints **plus** an Earth-Mover's-Distance
    coupling metric between the :math:`|E|` and Poynting :math:`|S|=|E||H|`
    point clouds:

    .. math::

       \\mathrm{coupling\\_strength}
        = \\exp(-\\mathrm{EMD}(|E|, |S|)) \\in (0, 1].

    The exponential bound replaces the previous ``1/(1+EMD)`` so the
    coupling metric is consistent with the documented decay form.
    """
    e_fp = topological_fingerprint(
        e_field, threshold, filtration=filtration,
        rel_persistence_tau=rel_persistence_tau,
    )
    h_fp = topological_fingerprint(
        h_field, threshold, filtration=filtration,
        rel_persistence_tau=rel_persistence_tau,
    )

    # Build the Poynting-magnitude field for the EMD comparison.
    s_field = np.abs(e_field) * np.abs(h_field)
    e_pts = field_to_pointcloud(e_field, threshold, add_phase=False)
    s_pts = field_to_pointcloud(s_field, threshold, add_phase=False)

    if len(e_pts) >= 5 and len(s_pts) >= 5:
        emd_S = _wasserstein_2d(e_pts, s_pts)
    else:
        emd_S = 1.0

    coupling_strength = float(np.exp(-emd_S))

    return {
        "e_fingerprint": e_fp,
        "h_fingerprint": h_fp,
        "emd_S": emd_S,
        "coupling_strength": coupling_strength,
        # `confinement_alignment`: a measure of how concentrated the
        # Poynting energy is in the same support as the E-field. Used by
        # the public demo.
        "confinement_alignment": _confinement_alignment(e_field, s_field, threshold),
    }


def _confinement_alignment(
    e_field: np.ndarray, s_field: np.ndarray, threshold: float
) -> float:
    """Return ``|supp(E*) ∩ supp(S*)| / |supp(E*)|`` for thresholded supports.

    A scalar in :math:`[0, 1]` quantifying how much of the high-energy
    region of :math:`|E|` is also high-energy in :math:`|S|`.
    """
    em = np.abs(e_field)
    sm = np.abs(s_field)
    if em.max() == 0 or sm.max() == 0:
        return 0.0
    e_supp = em > threshold * em.max()
    s_supp = sm > threshold * sm.max()
    e_size = float(e_supp.sum())
    if e_size == 0:
        return 0.0
    return float((e_supp & s_supp).sum() / e_size)
