"""
gcguard.cli — command-line demo for Gaussian Containment Guard.

Usage:
    gcguard demo --output demo.png
    gcguard demo --output demo.png --shape 300 --well-sigma 0.12 --barrier-sigma 0.08
    gcguard geometry-plot --output geo.png
"""

from __future__ import annotations

import argparse
import sys
import warnings
from typing import Optional

import numpy as np

from .core import Well, Barrier, Channel, make_grid
from .envelope import compute_envelope, confined_field
from .score import confinement_score, bulk_score
from .visualize import plot_vector_field_heatmap, plot_geometry_overview


# ── Demo field: 2-D standing wave with Gaussian sources ─────────────────────

def _make_demo_field(shape=(200, 200), kind="standing_wave") -> np.ndarray:
    """
    Generate a synthetic 2-D field for demonstration.

    Parameters
    ----------
    shape : tuple
    kind : str
        "standing_wave" — superposition of two counter-propagating waves.
        "dipole"         — field from two point sources.
        "vortex"         — swirling field.

    Returns
    -------
    field : ndarray, shape shape, complex
    """
    nx, ny = shape
    r = make_grid(shape, extent=(0.0, 1.0, 0.0, 1.0))
    x, y = r[..., 0], r[..., 1]

    if kind == "standing_wave":
        # Two counter-propagating Gaussian beams
        k = 2 * np.pi * 8  # ~8 wavelengths in unit domain
        w = 0.15           # beam width
        beam1 = np.exp(-((x - 0.25) ** 2 + (y - 0.5) ** 2) / w ** 2)
        beam2 = np.exp(-((x - 0.75) ** 2 + (y - 0.5) ** 2) / w ** 2)
        field = (
            beam1 * np.exp(1j * k * x)
            + 0.7 * beam2 * np.exp(-1j * k * (x - 1.0))
        )
    elif kind == "dipole":
        r1 = np.sqrt((x - 0.3) ** 2 + (y - 0.5) ** 2) + 1e-6
        r2 = np.sqrt((x - 0.7) ** 2 + (y - 0.5) ** 2) + 1e-6
        field = np.exp(1j * k * r1) / r1 - np.exp(1j * k * r2) / r2
        field = field.real + 1j * field.real  # make complex-ish
    elif kind == "vortex":
        x0, y0 = 0.5, 0.5
        theta = np.arctan2(y - y0, x - x0)
        r_val = np.sqrt((x - x0) ** 2 + (y - y0) ** 2) + 1e-6
        field = np.exp(1j * 3 * theta) * np.exp(-r_val ** 2 / 0.05)
    else:
        raise ValueError(f"Unknown field kind: {kind!r}")

    return field.astype(np.complex128)


# ── CLI commands ──────────────────────────────────────────────────────────────

def cmd_demo(args: argparse.Namespace) -> int:
    """Run the main demo: show a field + containment envelope."""
    print(f"[gcguard] Generating demo (shape={args.shape})…")

    field = _make_demo_field(shape=args.shape, kind=args.field_kind)

    # Build geometries from args
    well = Well(
        center=(args.well_x, args.well_y),
        sigma=args.well_sigma,
        strength=args.well_strength,
    )
    barrier = Barrier(
        center=(args.barrier_x, args.barrier_y),
        sigma=args.barrier_sigma,
        strength=args.barrier_strength,
    )
    channel = Channel(
        path=[(0.0, 0.5), (0.5, 0.8), (1.0, 0.5)],
        sigma=args.channel_sigma,
    )

    geometries = [well]
    if not args.no_barrier:
        geometries.append(barrier)
    if not args.no_channel:
        geometries.append(channel)

    # Confinement scores
    well_score = confinement_score(field, well)
    combined_score = bulk_score(field, geometries)
    print(f"[gcguard] Well-only confinement score : {well_score:.4f}")
    print(f"[gcguard] Combined confinement score  : {combined_score:.4f}")

    extent = (0.0, 1.0, 0.0, 1.0)
    fig = plot_vector_field_heatmap(
        np.abs(field),  # magnitude
        geometries,
        extent=extent,
        figsize=(12, 9),
        title="gcguard — Gaussian Containment Demo",
        show_quiver=True,
        show_colorbar=True,
        envelope_levels=[0.1, 0.3, 0.5, 0.7, 0.9],
        streamplot=False,
    )
    fig.text(
        0.99, 0.01,
        f"Well score: {well_score:.3f}  |  Combined: {combined_score:.3f}",
        ha="right", va="bottom",
        fontsize=8, color="gray",
        transform=fig.transFigure,
    )

    output = args.output or "gcguard_demo.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"[gcguard] Saved → {output}")
    return 0


def cmd_geometry_plot(args: argparse.Namespace) -> int:
    """Plot all geometry types as individual heatmap panels."""
    print("[gcguard] Generating geometry overview…")

    geometries = [
        Well(center=(0.5, 0.5), sigma=0.12, strength=1.0),
        Barrier(center=(0.25, 0.75), sigma=0.10, strength=1.5),
        Channel(path=[(0.0, 0.2), (0.5, 0.5), (1.0, 0.8)], sigma=0.05),
    ]

    fig = plot_geometry_overview(
        geometries,
        shape=args.shape,
        extent=(0.0, 1.0, 0.0, 1.0),
        figsize=(12, 4),
    )

    output = args.output or "gcguard_geometries.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"[gcguard] Saved → {output}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Print confinement scores for a synthetic field."""
    field = _make_demo_field(shape=args.shape, kind="standing_wave")

    well = Well(center=(0.5, 0.5), sigma=args.sigma, strength=1.0)
    barrier = Barrier(center=(args.bx, args.by), sigma=args.sigma, strength=2.0)

    ws = confinement_score(field, well)
    bs = confinement_score(field, barrier)
    cs = bulk_score(field, [well, barrier])

    print(f"Well confinement score    : {ws:.6f}")
    print(f"Barrier confinement score  : {bs:.6f}")
    print(f"Combined bulk score       : {cs:.6f}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="gcguard",
        description="Gaussian Containment Guard — EM field confinement library.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── demo ──────────────────────────────────────────────────────────────────
    p_demo = sub.add_parser("demo", help="Run the main demo and save a figure.")
    p_demo.add_argument("--output", "-o", default=None,
                        help="Output PNG path (default: gcguard_demo.png)")
    p_demo.add_argument("--shape", type=int, default=200,
                        help="Grid resolution (default: 200)")
    p_demo.add_argument("--field-kind", default="standing_wave",
                        choices=["standing_wave", "dipole", "vortex"],
                        help="Demo field type")
    p_demo.add_argument("--well-x", type=float, default=0.5)
    p_demo.add_argument("--well-y", type=float, default=0.5)
    p_demo.add_argument("--well-sigma", type=float, default=0.12)
    p_demo.add_argument("--well-strength", type=float, default=1.0)
    p_demo.add_argument("--barrier-x", type=float, default=0.65)
    p_demo.add_argument("--barrier-y", type=float, default=0.65)
    p_demo.add_argument("--barrier-sigma", type=float, default=0.08)
    p_demo.add_argument("--barrier-strength", type=float, default=2.0)
    p_demo.add_argument("--channel-sigma", type=float, default=0.05)
    p_demo.add_argument("--no-barrier", action="store_true")
    p_demo.add_argument("--no-channel", action="store_true")
    p_demo.set_defaults(func=cmd_demo)

    # ── geometry-plot ─────────────────────────────────────────────────────────
    p_geo = sub.add_parser("geometry-plot",
                           help="Plot individual geometry heatmaps.")
    p_geo.add_argument("--output", "-o", default=None)
    p_geo.add_argument("--shape", type=int, default=200)
    p_geo.set_defaults(func=cmd_geometry_plot)

    # ── score ─────────────────────────────────────────────────────────────────
    p_score = sub.add_parser("score", help="Print confinement scores.")
    p_score.add_argument("--shape", type=int, default=200)
    p_score.add_argument("--sigma", type=float, default=0.1)
    p_score.add_argument("--bx", type=float, default=0.65)
    p_score.add_argument("--by", type=float, default=0.35)
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
