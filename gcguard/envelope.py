"""
gcguard.envelope — compute combined Gaussian containment envelopes.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence, Union

from .core import Well, Barrier, Channel, make_grid


ArrayLike = Union[np.ndarray, tuple]


def _default_extent(ndim: int) -> tuple:
    """Return a flat (x0,x1,y0,y1,...) extent for the given dimensionality."""
    return tuple(v for d in range(ndim) for v in (0.0, 1.0))


def compute_envelope(
    field: np.ndarray,
    geometries: Sequence[Union[Well, Barrier, Channel]],
    r: Optional[np.ndarray] = None,
    extent: Optional[tuple] = None,
) -> np.ndarray:
    """
    Compute the combined Gaussian containment envelope over a field.

    The total envelope is the pointwise product of all geometry masks:

        E_total = ∏_wells G_well  *  ∏_barriers G_barrier  *  ∏_channels G_channel

    Parameters
    ----------
    field : ndarray
        The field array (2-D or 3-D). Shape is used to determine the grid.
    geometries : sequence of Well, Barrier, Channel
        Geometries whose masks are multiplied together.
    r : ndarray, optional
        Pre-computed coordinate array of shape (*field.shape, ndim).
        If omitted, a unit-domain grid is constructed.
    extent : tuple, optional
        Physical extent of each dimension.  Can be either:
          - (x0,x1,y0,y1,...) flat form, or
          - ((x0,x1), (y0,y1), ...) per-dimension form.
        Defaults to (0,1) per dimension if not given.

    Returns
    -------
    envelope : ndarray
        Combined envelope of the same shape as ``field``, values in [0, 1].
    """
    if extent is None:
        extent = _default_extent(field.ndim)
    if r is None:
        shape = field.shape
        r = make_grid(shape, extent=extent)

    # Start with all-ones envelope
    envelope = np.ones(field.shape, dtype=np.float64)

    for g in geometries:
        envelope *= g.gaussian(r)

    return envelope


def confined_field(
    field: np.ndarray,
    geometries: Sequence[Union[Well, Barrier, Channel]],
    r: Optional[np.ndarray] = None,
    extent: Optional[tuple] = None,
) -> np.ndarray:
    """
    Apply the Gaussian containment envelope to a field.

    This multiplies ``field`` by ``compute_envelope(...)``, returning a new
    array representing the field after confinement.

    Parameters
    ----------
    field : ndarray
        Input field (2-D or 3-D).
    geometries : sequence of Well, Barrier, Channel
    r, extent : optional
        Passed to :func:`compute_envelope`.

    Returns
    -------
    confined : ndarray
        Field multiplied by the containment envelope.
    """
    if extent is None:
        extent = _default_extent(field.ndim)
    envelope = compute_envelope(field, geometries, r=r, extent=extent)
    return field * envelope


# ── FDFD integration helpers ──────────────────────────────────────────────────

def fdfd_envelope(
    field: np.ndarray,
    geometries: Sequence[Union[Well, Barrier, Channel]],
    dx: float = 1.0,
    extent: tuple = None,
) -> np.ndarray:
    """
    Like :func:`compute_envelope` but with physical grid spacing for
    FDFD (Finite-Difference Frequency-Domain) integration.

    This is a thin wrapper that converts ``dx`` and ``extent`` into a
    coordinate array before calling :func:`compute_envelope`.

    Parameters
    ----------
    field : ndarray
        Field values on an equispaced grid.
    geometries : sequence of Well, Barrier, Channel
    dx : float or tuple of float
        Grid spacing per dimension. Scalar means uniform spacing.
    extent : tuple, optional
        Physical extent per dimension. Can be flat (x0,x1,y0,y1,...) or
        per-dim ((x0,x1), (y0,y1), ...). Defaults to (0,1) per dimension.

    Returns
    -------
    envelope : ndarray
    """
    if extent is None:
        extent = _default_extent(field.ndim)

    if np.isscalar(dx):
        dx = (dx,) * field.ndim
    else:
        dx = tuple(dx)

    shape = field.shape

    # Convert flat extent to per-dim form for make_grid
    if len(extent) == field.ndim * 2:
        extent_per_dim = tuple((extent[2*d], extent[2*d+1]) for d in range(field.ndim))
    else:
        extent_per_dim = extent

    r = make_grid(shape, extent=extent_per_dim)

    # Physical scaling: multiply by the actual physical size of each dimension
    scale = np.array([
        extent_per_dim[d][1] - extent_per_dim[d][0] for d in range(field.ndim)
    ])
    r = r * scale

    return compute_envelope(field, geometries, r=r)


def fdfd_confined_field(
    field: np.ndarray,
    geometries: Sequence[Union[Well, Barrier, Channel]],
    dx: float = 1.0,
    extent: tuple = None,
) -> np.ndarray:
    """
    Apply confinement envelope to an FDFD field with physical units.

    Parameters
    ----------
    field, geometries, dx, extent
        See :func:`fdfd_envelope`.

    Returns
    -------
    confined : ndarray
    """
    envelope = fdfd_envelope(field, geometries, dx=dx, extent=extent)
    return field * envelope
