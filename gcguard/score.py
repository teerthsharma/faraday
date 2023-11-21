"""
gcguard.score — quantify field energy confinement.
"""

from __future__ import annotations

import numpy as np
from typing import Union, Sequence

from .core import Well, Barrier, Channel, make_grid


def _flat_extent(ndim: int) -> tuple:
    return tuple(v for d in range(ndim) for v in (0.0, 1.0))


def _r_from_field(field: np.ndarray, extent: tuple = None) -> np.ndarray:
    """Build coordinate grid matching field shape."""
    if extent is None:
        extent = _flat_extent(field.ndim)
    return make_grid(field.shape, extent=extent)


def confinement_score(
    field: np.ndarray,
    geometry: Union[Well, Barrier, Channel],
    r: Union[None, np.ndarray] = None,
    extent: Union[None, tuple] = None,
    field_axis: int = -1,
) -> float:
    """
    Fraction of field energy contained within a geometry.

    This computes the ratio of energy in the region (where the geometry's
    Gaussian mask is near 1) to total field energy:

        score = Σ |field_inside|²  /  Σ |field|²

    Parameters
    ----------
    field : ndarray
        Field amplitude (2-D or 3-D). Complex values are accepted;
        energy is taken as |field|².
    geometry : Well, Barrier, or Channel
        Geometry defining the containment region.
    r : ndarray, optional
        Coordinate array of shape (*field.shape, ndim).
    extent : tuple, optional
        Physical domain extent. Defaults to (0,1) per dimension.
    field_axis : int, default -1
        Axis along which field components lie (for vector fields).
        Set to -1 for scalar fields, or e.g. 0 for stacked (Ex, Ey, Ez)
        arrays where first axis is components.

    Returns
    -------
    score : float
        Fraction of energy in the region, in [0.0, 1.0].
    """
    if extent is None:
        extent = _flat_extent(field.ndim)
    if r is None:
        r = make_grid(field.shape, extent=extent)

    mask = geometry.gaussian(r)  # (nx, ny[, nz])

    if field_axis != -1 and field.ndim > 1:
        # Vector field: sum energy密度 over component axis
        field_energy = np.sum(np.abs(field) ** 2, axis=field_axis)
    else:
        field_energy = np.abs(field) ** 2

    # Expand mask to broadcast with field_energy
    mask_shape = (1,) * (field_energy.ndim - mask.ndim) + mask.shape
    mask = mask.reshape(mask_shape)

    inside_energy = np.sum(field_energy * mask)
    total_energy = np.sum(field_energy)

    if total_energy == 0:
        return 0.0
    return float(inside_energy / total_energy)


def bulk_score(
    field: np.ndarray,
    geometries: Sequence[Union[Well, Barrier, Channel]],
    r: Union[None, np.ndarray] = None,
    extent: Union[None, tuple] = None,
    field_axis: int = -1,
) -> float:
    """
    Combined confinement score across multiple geometries.

    Uses the full combined envelope (product of all geometry masks) as
    the inside/outside boundary.

    Parameters
    ----------
    field : ndarray
    geometries : sequence of Well, Barrier, Channel
    r, extent, field_axis
        See :func:`confinement_score`.

    Returns
    -------
    score : float
    """
    from .envelope import compute_envelope

    if extent is None:
        extent = _flat_extent(field.ndim)
    if r is None:
        r = make_grid(field.shape, extent=extent)

    envelope = compute_envelope(field, geometries, r=r)

    if field_axis != -1 and field.ndim > 1:
        field_energy = np.sum(np.abs(field) ** 2, axis=field_axis)
    else:
        field_energy = np.abs(field) ** 2

    mask_shape = (1,) * (field_energy.ndim - envelope.ndim) + envelope.shape
    mask = envelope.reshape(mask_shape)

    inside_energy = np.sum(field_energy * mask)
    total_energy = np.sum(field_energy)

    if total_energy == 0:
        return 0.0
    return float(inside_energy / total_energy)


def mean_squared_displacement(
    field: np.ndarray,
    geometry: Well,
    r: Union[None, np.ndarray] = None,
    extent: Union[None, tuple] = None,
    field_axis: int = -1,
) -> float:
    """
    Mean squared distance of field energy from a well's centre,
    weighted by the well's Gaussian density.

    This gives a continuous measure of how well-confined the field is
    relative to a reference well.

    Parameters
    ----------
    field, geometry, r, extent, field_axis
        See :func:`confinement_score`.

    Returns
    -------
    msd : float
        Weighted mean squared distance.
    """
    if extent is None:
        extent = _flat_extent(field.ndim)
    if r is None:
        r = make_grid(field.shape, extent=extent)

    shape = r.shape[:-1]
    r_flat = r.reshape(-1, r.shape[-1])

    # Well centre
    center = geometry.center[0]  # (ndim,)
    diff = r_flat - center[np.newaxis, :]  # (N, ndim)
    d2 = np.sum(diff ** 2, axis=1)  # (N,)

    if field_axis != -1 and field.ndim > 1:
        field_energy = np.sum(np.abs(field) ** 2, axis=field_axis)
    else:
        field_energy = np.abs(field) ** 2

    field_flat = field_energy.ravel()
    total_weight = field_flat.sum()

    if total_weight == 0:
        return 0.0
    return float(np.sum(d2 * field_flat) / total_weight)
