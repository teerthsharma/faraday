"""
gcguard.visualize — matplotlib visualizations for Gaussian containment.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from typing import Sequence, Union, Optional, Tuple

from .core import Well, Barrier, Channel, make_grid
from .envelope import compute_envelope


def _to_rgba(field: np.ndarray, cmap="turbo") -> np.ndarray:
    """Normalize a 2-D array to RGBA using a colormap."""
    vmin, vmax = field.min(), field.max()
    if vmax == vmin:
        return np.zeros((*field.shape, 4))
    normed = (field - vmin) / (vmax - vmin)
    return cm.get_cmap(cmap)(normed)


def plot_envelope(
    field_or_envelope: np.ndarray,
    geometries: Sequence[Union[Well, Barrier, Channel]],
    r: Optional[np.ndarray] = None,
    extent: tuple = (0.0, 1.0, 0.0, 1.0),
    ax=None,
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = "turbo",
    title: Optional[str] = None,
    show_colorbar: bool = True,
    alpha: float = 0.85,
) -> plt.Figure:
    """
    Plot a 2-D field overlaid with the Gaussian containment envelope.

    The field is shown as a colourmap and the envelope contours are
    drawn on top.

    Parameters
    ----------
    field_or_envelope : ndarray
        Field to visualise, or a pre-computed envelope (detected by shape
        mismatch with geometries).
    geometries : sequence of geometry objects
    r, extent : optional
        See :func:`gcguard.envelope.compute_envelope`.
    ax : matplotlib Axes, optional
    figsize : tuple
    cmap : str
    title : str, optional
    show_colorbar : bool
    alpha : float
        Opacity of the field colourmap.

    Returns
    -------
    fig : matplotlib Figure
    """
    envelope = compute_envelope(
        field_or_envelope, geometries, r=r, extent=extent
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    # Parse extent
    if len(extent) == 4:
        xlim = extent[0:2]
        ylim = extent[2:4]
    else:
        xlim = (0, 1)
        ylim = (0, 1)

    im = ax.imshow(
        field_or_envelope,
        extent=(*xlim, *ylim),
        cmap=cmap,
        origin="lower",
        alpha=alpha,
        aspect="auto",
    )
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Field amplitude", fontsize=10)

    # Contours of the envelope
    contour_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    cs = ax.contour(
        envelope,
        extent=(*xlim, *ylim),
        levels=contour_levels,
        colors="white",
        linewidths=0.8,
        alpha=0.7,
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)
    if title:
        ax.set_title(title, fontsize=12)
    else:
        ax.set_title("Gaussian Containment Envelope", fontsize=12)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return fig


def plot_vector_field_heatmap(
    field: np.ndarray,
    geometries: Sequence[Union[Well, Barrier, Channel]],
    r: Optional[np.ndarray] = None,
    extent: tuple = (0.0, 1.0, 0.0, 1.0),
    ax=None,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = "viridis",
    title: Optional[str] = None,
    show_colorbar: bool = True,
    alpha: float = 0.75,
    envelope_alpha: float = 0.25,
    envelope_levels: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
    streamplot: bool = False,
    show_quiver: bool = True,
) -> plt.Figure:
    """
    Plot a vector-field heatmap with Gaussian containment overlay.

    For scalar fields the gradient is used as the pseudo-vector field.
    For vector fields (shape [..., ndim]) each component is visualised
    as a subplot panel.

    Parameters
    ----------
    field : ndarray, shape (nx, ny) or (nx, ny, ndim)
        Scalar amplitude or vector components.
    geometries : sequence of geometry objects
    r, extent : optional
        Passed to :func:`compute_envelope`.
    ax : matplotlib Axes, optional
    figsize : tuple
    cmap : str
    title : str, optional
    show_colorbar : bool
    alpha : float
        Opacity of the field colourmap.
    envelope_alpha : float
        Opacity of the envelope contours.
    envelope_levels : sequence of float
        Contour levels for the envelope.
    streamplot : bool
        If True, draw streamlines instead of a heatmap (for vector fields).
    show_quiver : bool
        If True, overlay a quiver plot of the vector field.

    Returns
    -------
    fig : matplotlib Figure
    """
    envelope = compute_envelope(field, geometries, r=r, extent=extent)

    if len(extent) == 4:
        xlim = extent[0:2]
        ylim = extent[2:4]
    else:
        xlim = (0, 1)
        ylim = (0, 1)

    shape = field.shape
    nx, ny = shape[:2]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    # ── Scalar field ─────────────────────────────────────────────────────────
    if field.ndim == 2 or (field.ndim == 3 and field.shape[-1] in (2, 3)):
        # Gradient of scalar field -> pseudo-vector
        f = field.squeeze()
        if f.ndim == 2:
            dy, dx = np.gradient(f)
            u, v = dx, dy
        elif field.ndim == 3 and field.shape[-1] == 2:
            u, v = field[..., 0], field[..., 1]
        elif field.ndim == 3 and field.shape[-1] == 3:
            u, v = field[..., 0], field[..., 1]
        else:
            raise ValueError(f"Unsupported field shape {field.shape}")

        magnitude = np.sqrt(u ** 2 + v ** 2)

        # Heatmap
        im = ax.imshow(
            magnitude,
            extent=(*xlim, *ylim),
            cmap=cmap,
            origin="lower",
            alpha=alpha,
            aspect="auto",
        )
        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("|∇E| (field gradient magnitude)", fontsize=10)

        if show_quiver:
            step = max(1, min(nx, ny) // 20)
            xi = np.arange(0, ny, step)
            yi = np.arange(0, nx, step)
            X, Y = np.meshgrid(
                np.linspace(xlim[0], xlim[1], ny),
                np.linspace(ylim[0], ylim[1], nx),
            )
            ax.quiver(
                X[::step, ::step], Y[::step, ::step],
                u[::step, ::step], v[::step, ::step],
                magnitude[::step, ::step],
                cmap="plasma",
                scale=4.0,
                alpha=0.7,
                width=0.004,
            )

        if streamplot:
            ax.streamplot(
                np.linspace(xlim[0], xlim[1], ny),
                np.linspace(ylim[0], ylim[1], nx),
                u, v,
                color=magnitude,
                cmap="plasma",
                density=1.5,
                linewidth=0.8,
                arrowsize=1.2,
            )
    else:
        # Just a colourmap
        im = ax.imshow(
            field,
            extent=(*xlim, *ylim),
            cmap=cmap,
            origin="lower",
            alpha=alpha,
            aspect="auto",
        )
        if show_colorbar:
            fig.colorbar(im, ax=ax, shrink=0.8)

    # ── Envelope contours ─────────────────────────────────────────────────────
    cs = ax.contour(
        envelope,
        extent=(*xlim, *ylim),
        levels=envelope_levels,
        colors=["#00ffff", "#ff00ff", "#ffff00"],
        linewidths=1.0,
        alpha=envelope_alpha,
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")

    # ── Geometry annotations ─────────────────────────────────────────────────
    for g in geometries:
        if isinstance(g, Well):
            colour, marker = "#00ff88", "o"
        elif isinstance(g, Barrier):
            colour, marker = "#ff4444", "x"
        elif isinstance(g, Channel):
            colour, marker = "#4488ff", "-"
        else:
            colour, marker = "white", "."

        centers = g.center if isinstance(g, (Well, Barrier)) else None
        if centers is not None:
            for c in centers:
                ax.plot(c[0], c[1], marker, color=colour,
                        ms=8, mew=1.5, label=type(g).__name__)
        elif isinstance(g, Channel):
            path = g.raw_path
            ax.plot(path[:, 0], path[:, 1], "--", color=colour,
                    lw=1.5, label="Channel")

    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)
    ax.set_title(title or "Gaussian Containment — Vector Field Heatmap", fontsize=12)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if not ax.get_legend_handles_labels()[0]:
        pass  # legend empty
    else:
        ax.legend(fontsize=8, loc="upper right")

    return fig


def plot_geometry_overview(
    geometries: Sequence[Union[Well, Barrier, Channel]],
    shape: Tuple[int, int] = (200, 200),
    extent: tuple = (0.0, 1.0, 0.0, 1.0),
    figsize: Tuple[float, float] = (10, 4),
    titles: Optional[Sequence[str]] = None,
) -> plt.Figure:
    """
    Plot each geometry type individually as a heatmap panel.

    Parameters
    ----------
    geometries : sequence of geometry objects
    shape : tuple
        Grid resolution for evaluation.
    extent : tuple
    figsize : tuple
    titles : sequence of str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    r = make_grid(shape, extent=extent)

    wells  = [g for g in geometries if isinstance(g, Well)]
    barr   = [g for g in geometries if isinstance(g, Barrier)]
    chan   = [g for g in geometries if isinstance(g, Channel)]

    n_panels = sum([len(wells), len(barr), len(chan)])
    if n_panels == 0:
        raise ValueError("No geometries provided")

    fig, axes = plt.subplots(1, n_panels, figsize=figsize,
                              constrained_layout=True, squeeze=False)

    panel_idx = 0
    labels = []
    handlers = []

    for group, label, cmap, panels in [
        (wells,   "Well",   "Greens",  len(wells)),
        (barr,    "Barrier","Reds",    len(barr)),
        (chan,    "Channel","Blues",   len(chan)),
    ]:
        for g in group:
            ax = axes[0, panel_idx]
            mask = g.gaussian(r)

            im = ax.imshow(
                mask,
                extent=(*extent[:2], *extent[2:]),
                cmap=cmap,
                origin="lower",
                aspect="auto",
            )
            fig.colorbar(im, ax=ax, shrink=0.7)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            if isinstance(g, (Well, Barrier)):
                for c in g.center:
                    ax.plot(c[0], c[1], "w*", ms=10, mew=0.5)
            elif isinstance(g, Channel):
                ax.plot(g.raw_path[:, 0], g.raw_path[:, 1], "w--", lw=1.5)

            panel_idx += 1

    return fig
