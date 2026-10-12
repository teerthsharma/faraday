# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.quantum — The quantum sector of the cavity duality.

The TM equation -ΔE = k²E with E = 0 on the wall is the Schrödinger equation
of a 2-D quantum billiard, so every real cavity mode is a time-reversal
symmetric quantum eigenstate and :mod:`faraday.duality` applies verbatim.
Breaking time reversal — taking a complex state ψ = u + iv, e.g. a degenerate
pair carrying angular momentum — opens three further, decoupled layers of
topology, each an integer or Z₂ invariant:

1. Phase vortices (real space, Z). Zeros of ψ carry a phase winding V. By the
   argument principle V equals the winding of ψ along the wall, and a planar
   curve (∂ₙu, ∂ₙv) cannot wind more than half as often as either component
   changes sign, so

       |V| <= min(N_wall(u), N_wall(v)) / 2

   — the quantum vortex charge is bounded by the classical wall data. Angular
   momentum states of the disk saturate it: l = N_wall / 2.
2. Holonomy (real space, Z per hole). In a cavity with holes

       V_bulk = w_outer - Σ w_holes

   where w are phase windings along each wall. In the concentric annulus the
   charge leaves the bulk (V_bulk = 0) and is carried entirely as quantised
   circulation around the hole — an Aharonov-Bohm-type holonomy.
3. Geometric phase (shape space, Z₂). A real mode transported around a
   closed loop of cavity shapes returns with sign -1 iff the loop encloses a
   diabolical point — the true level crossing at which the classical vortex
   charge Q of :mod:`faraday.duality` jumps.

(1)-(2) are verified in tests on disk, square, ellipse, annuli and a two-hole
disk; (3) on the TM31/TM12 crossing of the rectangle. The geometric phase
around diabolical points in microwave billiards is known experimentally
(Lauber, Weidenhammer & Dubbers, PRL 72, 1994).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy import ndimage

from faraday.duality import vortex_charges


def phase_charges(psi: np.ndarray, interior: np.ndarray) -> np.ndarray:
    """Phase-vortex charge of a complex field on every grid plaquette."""
    return vortex_charges(psi.real, psi.imag, interior)


def wall_windings(psi: np.ndarray, interior: np.ndarray) -> list[int]:
    """Winding of arg ψ along each wall loop, outer wall first.

    Each boundary component is walked as its loop of interior pixels touching
    the wall, ordered by angle about the loop's centroid (exact for
    star-shaped walls) and traversed counter-clockwise.
    """
    edge = interior & ~ndimage.binary_erosion(interior, border_value=0)
    labels, n = ndimage.label(edge, structure=np.ones((3, 3)))
    loops = []
    for i in range(1, n + 1):
        yy, xx = np.nonzero(labels == i)
        order = np.argsort(np.arctan2(yy - yy.mean(), xx - xx.mean()))
        phase = np.angle(psi[yy[order], xx[order]])
        d = np.diff(np.r_[phase, phase[0]])
        d = (d + np.pi) % (2 * np.pi) - np.pi
        loops.append((len(yy), int(np.rint(d.sum() / (2 * np.pi)))))
    loops.sort(key=lambda t: -t[0])  # the outer wall is the longest loop
    return [w for _, w in loops]


def berry_sign(states: Iterable[np.ndarray]) -> tuple[int, float]:
    """Z₂ geometric phase of a real state transported around a closed loop.

    ``states`` are the same mode sampled along a closed loop in parameter
    space (first and last at the same parameters). Each is aligned to the
    previous one (parallel transport: positive overlap); the returned sign is
    the overlap of the transported end state with the start state.

    Returns:
        (sign, min_step_overlap). sign = -1 ⇔ geometric phase π. The minimum
        overlap between consecutive states should be close to 1, otherwise
        the loop is under-sampled and the sign is not trustworthy.
    """
    first = prev = None
    min_overlap = 1.0
    for v in states:
        v = v / np.linalg.norm(v)
        if prev is None:
            first = v
        else:
            ov = float(np.sum(v * prev))
            min_overlap = min(min_overlap, abs(ov))
            if ov < 0:
                v = -v
        prev = v
    if first is None or prev is None:
        raise ValueError("need at least one state")
    return (1 if float(np.sum(prev * first)) > 0 else -1), min_overlap
