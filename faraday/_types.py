"""
faraday._types — Shared type aliases for the faraday package.

These aliases make type annotations more readable and consistent
across all modules.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

# --------------------------------------------------------------------
# Array types
# --------------------------------------------------------------------
#: 2D float array (typical field grid)
NDArrayFloat2D: TypeAlias = np.ndarray

#: 1D float array
NDArrayFloat1D: TypeAlias = np.ndarray

#: 2D complex array (typical complex field)
NDArrayComplex2D: TypeAlias = np.ndarray

#: Generic float array (any dimension)
NDArrayFloat: TypeAlias = np.ndarray

# --------------------------------------------------------------------
# Domain-specific types
# --------------------------------------------------------------------
#: A persistent homology barcode: list of (birth, death) float pairs
Barcode: TypeAlias = list[tuple[float, float]]

#: A topological fingerprint dict (returned by barcode.topological_fingerprint)
Fingerprint: TypeAlias = dict[str, object]

#: A fixed-length barcode embedding vector (from manifold_projector)
Embedding: TypeAlias = np.ndarray

#: Geometry parameters: (w, h) for rectangular, (r,) for circular
GeometryParams: TypeAlias = tuple[float, ...]

# --------------------------------------------------------------------
# Solvers
# --------------------------------------------------------------------
#: Mode data dict returned by solve_cavity_modes
ModeData: TypeAlias = dict[str, object]

# --------------------------------------------------------------------
# God Tensor
# --------------------------------------------------------------------
#: Training sample dict representation (for serialization)
SampleDict: TypeAlias = dict[str, object]
