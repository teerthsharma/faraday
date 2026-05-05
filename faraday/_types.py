# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday._types — Shared type aliases for the faraday package.

These aliases make type annotations more readable and consistent
across all modules.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# TypedDict stub for ModeData (avoids circular import at type-check time)
# ---------------------------------------------------------------------------

class ModeData(dict[str, Any]):
    """
    Return type of ``solve_cavity_modes``.

    Attributes
    ----------
    geometry : str
        Shape name e.g. ``"rectangular"``.
    dims : tuple[float, ...]
        Physical dimensions.
    nx, ny : int
        Grid resolution.
    num_modes_found : int
        Number of valid modes.
    k_values : list[float]
        Wave numbers for valid modes.
    e_modes : dict[str, Any]
        E-field mode data keyed by mode name.
    h_modes : dict[str, Any]
        H-field (magnitude) mode data keyed by mode name.
    s_modes : dict[str, Any]
        Poynting vector mode data keyed by mode name.
    X, Y : list[list[float]]
        Grid coordinates.
    interior : list[list[bool]]
        Interior mask.
    """
    pass


# ---------------------------------------------------------------------------
# Array types
# ---------------------------------------------------------------------------

#: Generic float array (any dimension)
NDArrayFloat = np.ndarray

#: 2D float array
NDArrayFloat2D = np.ndarray

#: 1D float array
NDArrayFloat1D = np.ndarray

#: 2D complex array
NDArrayComplex2D = np.ndarray

# ---------------------------------------------------------------------------
# Domain-specific types
# ---------------------------------------------------------------------------

#: Persistent homology barcode
Barcode = list[tuple[float, float]]

#: Topological fingerprint dict
Fingerprint = dict[str, Any]

#: Fixed-length barcode embedding vector
Embedding = np.ndarray

#: Geometry parameters
GeometryParams = tuple[float, ...]

# ---------------------------------------------------------------------------
# God Tensor
# ---------------------------------------------------------------------------

#: Training sample dict representation
SampleDict = dict[str, Any]
