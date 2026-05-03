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

from .em_solver import CavityGeometry, CavityShape, solve_cavity_modes, WaveSuperposer
from .barcode import field_to_pointcloud, compute_barcodes, topological_fingerprint, coupled_fingerprint
from .manifold_projector import ManifoldProjector, embed_barcode, embed_fingerprint
from .god_tensor import GodTensor
from .predict import predict_eh_barcode

__all__ = [
    "CavityGeometry",
    "CavityShape",
    "solve_cavity_modes",
    "WaveSuperposer",
    "field_to_pointcloud",
    "compute_barcodes",
    "topological_fingerprint",
    "coupled_fingerprint",
    "ManifoldProjector",
    "embed_barcode",
    "embed_fingerprint",
    "GodTensor",
    "predict_eh_barcode",
]

__version__ = "0.1.0"
