"""
gcguard.core — Gaussian containment geometry primitives.

Gaussian Well:     G_well(r)  = exp(-|r - r₀|² / 2σ²)
Gaussian Barrier: G_barrier  = 1 - exp(-|r - r₀|² / 2σ²)
Gaussian Channel: product of Gaussians along a path.
"""

from __future__ import annotations

import numpy as np
from typing import Union, Tuple, Optional


ArrayLike = Union[np.ndarray, tuple]


class Well:
    """
    Gaussian containment well — attracts field energy toward a region.

    The well creates a multiplicative envelope that is maximal at the centre
    and decays Gaussianly with distance:

        G_well(r) = exp(-|r - center|² / 2σ²) ^ strength

    Parameters
    ----------
    center : array-like, shape (ndim,) or (ndim, nwells)
        Centre of the well (or centres of multiple wells).
    sigma : float
        Standard deviation of the Gaussian envelope.
    strength : float, default 1.0
        Exponent multiplier. Higher values sharpen the well.
    """

    def __init__(
        self,
        center: ArrayLike,
        sigma: float = 0.1,
        strength: float = 1.0,
    ):
        self.center = np.atleast_2d(center)
        if self.center.ndim == 1:
            self.center = self.center.reshape(1, -1)
        self.sigma = sigma
        self.strength = strength

    def __repr__(self) -> str:
        c = self.center[0] if len(self.center) == 1 else self.center
        return (
            f"Well(center={c.tolist()}, sigma={self.sigma}, "
            f"strength={self.strength})"
        )

    def gaussian(self, r: np.ndarray) -> np.ndarray:
        """
        Evaluate the well envelope over a coordinate grid.

        Parameters
        ----------
        r : ndarray, shape (*shape, ndim)
            Cartesian coordinates. shape is the spatial grid shape
            (e.g. (nx, ny) for 2-D or (nx, ny, nz) for 3-D) and ndim
            is the number of spatial dimensions.

        Returns
        -------
        envelope : ndarray, shape shape
            Gaussian well mask in the range [0, 1].
        """
        shape = r.shape[:-1]
        r = r.reshape(-1, r.shape[-1])  # (N, ndim)

        # Squared distances from each well centre
        diff = r[:, np.newaxis, :] - self.center[np.newaxis, :, :]  # (N, nwells, ndim)
        d2 = np.sum(diff ** 2, axis=-1)  # (N, nwells)
        d2_min = np.min(d2, axis=1)     # (N,) — closest well

        envelope = np.exp(-d2_min / (2.0 * self.sigma ** 2)) ** self.strength
        return envelope.reshape(shape)


class Barrier:
    """
    Gaussian barrier — repels field energy away from a region.

    The barrier creates a multiplicative mask that suppresses energy
    near the centre and approaches 1 far away:

        G_barrier(r) = 1 - exp(-|r - center|² / 2σ²) ^ strength

    Parameters
    ----------
    center : array-like, shape (ndim,) or (ndim, nbarriers)
        Centre of the barrier (or centres of multiple barriers).
    sigma : float
        Standard deviation of the Gaussian falloff.
    strength : float, default 1.0
        Exponent multiplier. Higher values sharpen the barrier.
    """

    def __init__(
        self,
        center: ArrayLike,
        sigma: float = 0.1,
        strength: float = 1.0,
    ):
        self.center = np.atleast_2d(center)
        if self.center.ndim == 1:
            self.center = self.center.reshape(1, -1)
        self.sigma = sigma
        self.strength = strength

    def __repr__(self) -> str:
        c = self.center[0] if len(self.center) == 1 else self.center
        return (
            f"Barrier(center={c.tolist()}, sigma={self.sigma}, "
            f"strength={self.strength})"
        )

    def gaussian(self, r: np.ndarray) -> np.ndarray:
        """
        Evaluate the barrier mask over a coordinate grid.

        Parameters
        ----------
        r : ndarray, shape (*shape, ndim)
            Cartesian coordinates.

        Returns
        -------
        mask : ndarray, shape shape
            Barrier mask in the range [0, 1], where 0 = fully blocked,
            1 = fully open.
        """
        shape = r.shape[:-1]
        r = r.reshape(-1, r.shape[-1])

        diff = r[:, np.newaxis, :] - self.center[np.newaxis, :, :]
        d2 = np.sum(diff ** 2, axis=-1)
        d2_min = np.min(d2, axis=1)

        gaussian_part = np.exp(-d2_min / (2.0 * self.sigma ** 2)) ** self.strength
        return (1.0 - gaussian_part).reshape(shape)


class Channel:
    """
    Gaussian channel — guides field energy along a polyline path.

    The channel is the product of narrow Gaussian cross-sections
    evaluated at every point along the path:

        G_channel(r) = ∏_segments ∏_points exp(-d_seg(r)² / 2σ²)

    where d_seg is the perpendicular distance from r to each path segment.

    Parameters
    ----------
    path : array-like, shape (n_points, ndim)
        Vertices of the polyline path.
    sigma : float, default 0.05
        Cross-sectional width of the channel.
    n_points : int, optional
        Sub-sample density along the path (default 100 per unit length).
    """

    def __init__(
        self,
        path: ArrayLike,
        sigma: float = 0.05,
        n_points: Optional[int] = None,
    ):
        self.raw_path = np.atleast_2d(path)
        if self.raw_path.ndim == 1:
            self.raw_path = self.raw_path.reshape(-1, 1)
        self.sigma = sigma

        # Default sub-sampling: 100 points per unit length, min 2
        if n_points is None:
            seg_lengths = np.linalg.norm(
                np.diff(self.raw_path, axis=0), axis=1
            )
            total_length = seg_lengths.sum()
            n_points = max(2, int(100 * total_length))
        self.n_points = n_points

    def __repr__(self) -> str:
        return (
            f"Channel(path=<{len(self.raw_path)} points>, "
            f"sigma={self.sigma})"
        )

    def _sample_path(self) -> np.ndarray:
        """Return uniformly-sampled points along the polyline."""
        path = self.raw_path
        n_seg = len(path) - 1
        if n_seg == 0:
            return path

        seg_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        total_length = seg_lengths.sum()
        if total_length == 0:
            return np.tile(path[0], (self.n_points, 1))

        # Cumulative arc length
        cumseg = np.concatenate([[0], np.cumsum(seg_lengths)])

        # Uniform arc-length samples
        s = np.linspace(0, total_length, self.n_points)
        sampled = np.zeros((self.n_points, path.shape[1]))

        for i in range(n_seg):
            mask = (s >= cumseg[i]) & (s < cumseg[i + 1])
            if not np.any(mask):
                continue
            t = (s[mask] - cumseg[i]) / seg_lengths[i]
            sampled[mask] = path[i] + t[:, np.newaxis] * (path[i + 1] - path[i])

        # Last point
        sampled[-1] = path[-1]
        return sampled

    def gaussian(self, r: np.ndarray) -> np.ndarray:
        """
        Evaluate the channel mask over a coordinate grid.

        Parameters
        ----------
        r : ndarray, shape (*shape, ndim)
            Cartesian coordinates.

        Returns
        -------
        mask : ndarray, shape shape
            Channel mask in the range [0, 1], where 1 = on the channel,
            0 = far from it.
        """
        shape = r.shape[:-1]
        r = r.reshape(-1, r.shape[-1])  # (N, ndim)
        path_pts = self._sample_path()  # (M, ndim)

        # Perpendicular distance from each grid point to each path point
        # Expand r and path_pts for broadcasting
        # r: (N, 1, ndim), path_pts: (1, M, ndim)
        diff = r[:, np.newaxis, :] - path_pts[np.newaxis, :, :]  # (N, M, ndim)
        d2 = np.sum(diff ** 2, axis=-1)  # (N, M)

        # Minimum distance to any path point
        d2_min = np.min(d2, axis=1)  # (N,)

        envelope = np.exp(-d2_min / (2.0 * self.sigma ** 2))
        return envelope.reshape(shape)


# ── Coordinate grid helpers ──────────────────────────────────────────────────

def make_grid(
    shape: Tuple[int, ...],
    extent: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Build a coordinate grid suitable for use with Well/Barrier/Channel.

    Parameters
    ----------
    shape : tuple of int
        Shape of the spatial grid, e.g. (nx, ny) or (nx, ny, nz).
    extent : tuple (lo, hi), optional
        Physical extent of the domain. Defaults to [0, 1] per dimension.

    Returns
    -------
    r : ndarray, shape (*shape, ndim)
        Cartesian coordinate array.
    """
    ndim = len(shape)
    if extent is None:
        extent = (0.0, 1.0)
    if len(extent) == 1:
        extent = extent * ndim
    # Convert flat (x0,x1,y0,y1,...) to ((x0,x1),(y0,y1),...) format
    elif len(extent) == ndim * 2 and ndim > 1:
        extent = tuple((extent[2 * d], extent[2 * d + 1]) for d in range(ndim))
    elif len(extent) == 2 and isinstance(extent[0], (int, float)):
        # Single (lo,hi) pair applied to all dimensions
        extent = extent * ndim
    elif len(extent) != ndim:
        raise ValueError(
            f"extent must have {ndim} elements (one (lo,hi) per dimension) "
            f"or {2 * ndim} elements in flat (x0,x1,y0,y1,...) format; "
            f"got {extent!r}"
        )

    grids = [
        np.linspace(extent[d][0], extent[d][1], shape[d])
        for d in range(ndim)
    ]
    mesh = np.meshgrid(*grids, indexing="ij")
    r = np.stack(mesh, axis=-1)  # (nx, ny[, nz], ndim)
    return r
