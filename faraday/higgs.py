# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.higgs - Photons in a Higgs background: walls and traps from a condensate.

In the Higgs (broken) phase of an abelian gauge theory, the photon has mass
m = sqrt(2) q |psi|. For the TM component E = E_z of a mode at frequency w:

    (-Δ + m²(x)) E = w² E,        m²(x) = m0² |psi(x)|² / v²

(scalar TM model: the gradient couplings that a varying mass adds to the
full Proca equations are neglected). The condensate is a field generator: its
geometry generates the photon's cavities. Three consequences are tested in
``tests/test_higgs.py`` and ``experiments/higgs_study.py``:

1. PEC = infinite Higgs mass. A condensate surrounding a normal (psi = 0)
   region D acts as a soft wall with penetration depth 1/m (Meissner). As
   m grows, the trapped frequencies converge to the PEC cavity values.
2. The mass gap protects topology. Outside D a trapped mode decays,
   ∂ₙE ≈ -κE with κ ≈ sqrt(m² - w²) > 0: a Robin wall. Along the wall
   (∂ₙE, ∂ₜE) = (-κE, E') turns half a revolution in the same sense at every
   zero of E, exactly as for E = 0, so the bulk-boundary law
   Q(D) = χ(D) + N_wall/2 of faraday.duality holds for every trapped mode
   (w < m). Above the gap (w > m) the field propagates into the condensate,
   κ loses its sign and the law fails for most modes. A hole filled with
   condensate carries the critical point of E ~ I_l(m r) cos(l θ) at its
   centre, of index 1 - l.
3. Higgs vortices trap light. In a vortex core psi = 0, so the photon is
   massless there and each core is a potential well of strength g = m0·ξ.
   In units of ξ the problem depends on g alone: below g ≈ 2.2 every vortex
   binds exactly one photon mode (an s shell), so the number of trapped modes
   equals the condensate's winding number L. Larger g fills atom-like shells
   in the order 1s, 1p, 2s, 1d, 2p, ...

Related known physics: the Anderson-Higgs mechanism and the Meissner effect;
fermion states bound in vortex cores (Caroli, de Gennes & Matricon 1964);
bosonic modes bound to string cores (Witten 1985, superconducting strings).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from faraday._types import ModeData
from faraday.duality import vortex_charges


def higgs_wall_mass2(normal_region: np.ndarray, m: float) -> np.ndarray:
    """m² of a uniform condensate everywhere outside a normal region (psi = 0 inside)."""
    return np.where(normal_region, 0.0, m * m)


def vortex_condensate(
    X: np.ndarray, Y: np.ndarray, cores: Iterable[tuple[float, float]], xi: float
) -> np.ndarray:
    """|psi|²/v² of unit-winding vortices: prod_i tanh²(|x - x_i| / ξ).

    A frozen ansatz (no self-consistent Ginzburg-Landau solve): each vortex is
    a zero of psi with healing length ξ.
    """
    psi2 = np.ones_like(X, dtype=float)
    for cx, cy in cores:
        psi2 *= np.tanh(np.hypot(X - cx, Y - cy) / xi) ** 2
    return psi2


def trapped(mode_data: ModeData, gap: float) -> list[int]:
    """Indices of modes below the mass gap (bound, w < m)."""
    return [j for j, w in enumerate(mode_data["k_values"]) if w < gap]


def region_charge(mode_data: ModeData, mode_index: int, region: np.ndarray) -> int:
    """Net vortex charge of H inside ``region`` (exact discrete Stokes sum)."""
    key = f"mode_{mode_index}"
    hx = np.array(mode_data["h_modes"][key]["hx"])
    hy = np.array(mode_data["h_modes"][key]["hy"])
    interior = np.array(mode_data["interior"], dtype=bool)
    w = vortex_charges(hx, hy, interior)
    r = np.asarray(region, dtype=bool)
    inside = r[:-1, :-1] & r[:-1, 1:] & r[1:, 1:] & r[1:, :-1]
    return int(w[inside].sum())


def box_interior(X: np.ndarray) -> np.ndarray:
    """Whole grid except its edge: the domain for a field confined by m²(x) alone."""
    inside = np.ones(X.shape, dtype=bool)
    inside[[0, -1], :] = False
    inside[:, [0, -1]] = False
    return inside
