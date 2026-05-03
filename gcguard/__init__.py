"""
gcguard — Gaussian Containment Guard for EM field confinement.

A mathematical framework for containing, repelling, and channelling
electromagnetic field energy using Gaussian envelope functions.

Quick start
-----------
    import gcguard as gcg

    well    = gcg.Well(center=(0.5, 0.5), sigma=0.1, strength=1.0)
    barrier = gcg.Barrier(center=(0.65, 0.65), sigma=0.08, strength=2.0)
    channel = gcg.Channel(path=[(0,0), (1,1)], sigma=0.05)

    envelope = gcg.compute_envelope(field_2d, [well, barrier, channel])
    confined = field_2d * envelope

    score = gcg.confinement_score(field_2d, well)  # 0.0 – 1.0
"""

from __future__ import annotations

from .core import Well, Barrier, Channel, make_grid

from .envelope import (
    compute_envelope,
    confined_field,
    fdfd_envelope,
    fdfd_confined_field,
)

from .score import (
    confinement_score,
    bulk_score,
    mean_squared_displacement,
)

from .visualize import (
    plot_envelope,
    plot_vector_field_heatmap,
    plot_geometry_overview,
)

__all__ = [
    # core
    "Well",
    "Barrier",
    "Channel",
    "make_grid",
    # envelope
    "compute_envelope",
    "confined_field",
    "fdfd_envelope",
    "fdfd_confined_field",
    # score
    "confinement_score",
    "bulk_score",
    "mean_squared_displacement",
    # visualize
    "plot_envelope",
    "plot_vector_field_heatmap",
    "plot_geometry_overview",
]

__version__ = "0.1.0"
