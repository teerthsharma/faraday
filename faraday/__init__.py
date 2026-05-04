"""
faraday — Computational Faraday Tensor

Discover the unified E x H field coupling via topology-fixed-point
projection. The God Tensor: the fixed point where electric and magnetic
field signatures co-determine each other.

Architecture
------------
1. em_solver  — FDFD cavity solver. Computes E and H eigenmodes together.
2. barcode    — Field -> point cloud -> persistent homology barcode.
3. manifold_projector  — Barcode -> Hilbert coefficients -> manifold embedding.
4. god_tensor — T(E_sig) <-> T(H_sig) fixed-point iteration.
5. predict    — Given new geometry, predict E and H topology via God Tensor.

Example
-------
    from faraday import GodTensor

    gt = GodTensor(n_geometries=50, nx=40, ny=40)
    gt.collect_training_data()      # run FDFD across varied geometries
    gt.find_fixed_point(iters=200)  # iterate until T(T(x)) = T(x)
    pred = gt.predict(w=2.0, h=1.5) # predict E and H barcode for new geometry
"""

from __future__ import annotations

# Re-export public types
from faraday._types import (
    Barcode,
    Embedding,
    Fingerprint,
    GeometryParams,
    ModeData,
    NDArrayFloat,
)
from faraday.barcode import (
    compute_barcodes,
    coupled_fingerprint,
    field_to_pointcloud,
    topological_fingerprint,
)
from faraday.benchmarking import (
    ValidationReport,
    run_validation_experiment,
)

# CLI entry point
from faraday.cli import cli_main

# Core API
from faraday.em_solver import (
    CavityGeometry,
    CavityShape,
    WaveSuperposer,
    solve_cavity_modes,
)

# Re-export exceptions
from faraday.exceptions import (
    ConfigError,
    ConvergenceError,
    FaradayError,
    GeometryError,
    SolverError,
    TopologyError,
)
from faraday.god_tensor import GodTensor

# Re-export logging
from faraday.logging import get_logger
from faraday.manifold_projector import (
    ManifoldProjector,
    embed_barcode,
    embed_fingerprint,
)
from faraday.predict import predict_eh_barcode

__all__ = sorted([
    # Exceptions
    "ConfigError",
    "ConvergenceError",
    "FaradayError",
    "GeometryError",
    "SolverError",
    "TopologyError",
    # Logging
    "get_logger",
    # Types
    "Barcode",
    "Embedding",
    "Fingerprint",
    "GeometryParams",
    "ModeData",
    "NDArrayFloat",
    # Core API
    "CavityGeometry",
    "CavityShape",
    "cli_main",
    "compute_barcodes",
    "coupled_fingerprint",
    "embed_barcode",
    "embed_fingerprint",
    "field_to_pointcloud",
    "GodTensor",
    "ManifoldProjector",
    "predict_eh_barcode",
    "run_validation_experiment",
    "solve_cavity_modes",
    "topological_fingerprint",
    "ValidationReport",
    "WaveSuperposer",
])

__version__ = "0.1.0"
