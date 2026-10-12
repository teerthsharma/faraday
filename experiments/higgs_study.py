#!/usr/bin/env python3
# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
Higgs coupling study - reproduces every number in docs/HIGGS.md.

    python experiments/higgs_study.py      # ~3 min

1. PEC wall = infinite Higgs mass (convergence of trapped frequencies).
2. Mass gap protects topology: Q(D) = χ + N/2 for trapped vs untrapped modes.
3. Higgs-filled hole carries index 1 - l.
4. Vortex light traps: #trapped modes vs winding L; shell staircase vs g = m0·ξ.
"""

from __future__ import annotations

import logging

import numpy as np
import structlog

from faraday.duality import euler_characteristic, wall_nodal_count
from faraday.em_solver import solve_interior_modes
from faraday.higgs import (
    box_interior,
    higgs_wall_mass2,
    region_charge,
    trapped,
    vortex_condensate,
)

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

SHAPES = {
    "disk": lambda X, Y: np.hypot(X, Y) < 1.0,
    "rectangle 2x1.2": lambda X, Y: (abs(X) < 1.0) & (abs(Y) < 0.6),
    "stadium": lambda X, Y: ((abs(X) < 0.6) & (abs(Y) < 0.6))
    | ((abs(X) - 0.6) ** 2 + Y**2 < 0.36),
    "annulus": lambda X, Y: (np.hypot(X, Y) < 1.0) & (np.hypot(X, Y) > 0.35),
}


def grid(R, n):
    c = np.linspace(-R, R, n)
    return np.meshgrid(c, c)


def higgs(X, Y, mass2, k):
    return solve_interior_modes(X, Y, box_interior(X), num_modes=k, seed=0, mass2=mass2)


def field(modes, j):
    return np.array(modes["e_modes"][f"mode_{j}"]["field"])


def study_limit():
    X, Y = grid(2.0, 201)
    for name in ("disk", "stadium"):
        D = SHAPES[name](X, Y)
        k = solve_interior_modes(X, Y, D, num_modes=1, seed=0)["k_values"][0]
        ws = [higgs(X, Y, higgs_wall_mass2(D, m), 1)["k_values"][0] for m in (2, 4, 8, 16)]
        print(f"  {name:8s} PEC k1 = {k:.3f}; Higgs wall m = 2,4,8,16 → ω1 = {np.round(ws, 3).tolist()}")


def study_gap():
    X, Y = grid(2.0, 201)
    tb = ob = tu = ou = 0
    for f in SHAPES.values():
        D = f(X, Y)
        chi = euler_characteristic(D)
        for m in (3.0, 5.0, 8.0):
            modes = higgs(X, Y, higgs_wall_mass2(D, m), 40)
            bound = set(trapped(modes, m))
            for j in range(40):
                ok = 2 * region_charge(modes, j, D) == 2 * chi + wall_nodal_count(field(modes, j), D)
                if j in bound:
                    tb, ob = tb + 1, ob + ok
                else:
                    tu, ou = tu + 1, ou + ok
    print(f"  trapped (ω < m):   law holds {ob}/{tb}")
    print(f"  untrapped (ω > m): law holds {ou}/{tu} = {ou / tu:.0%}")


def study_hole():
    X, Y = grid(2.0, 201)
    r = np.hypot(X, Y)
    r_hole = 0.35
    D = (r < 1.0) & (r > r_hole)
    stats = {True: [0, 0], False: [0, 0]}
    for m in (4.0, 6.0, 8.0):
        modes = higgs(X, Y, higgs_wall_mass2(D, m), 12)
        for j in trapped(modes, m):
            jump = region_charge(modes, j, r < 1.0) - region_charge(modes, j, D)
            ok = jump == 1 - wall_nodal_count(field(modes, j), D) // 4
            firm = np.sqrt(m * m - modes["k_values"][j] ** 2) * r_hole > 0.8
            stats[firm][0] += ok
            stats[firm][1] += 1
    print(f"  hole index = 1 - l, κ·r_hole > 0.8:  {stats[True][0]}/{stats[True][1]}")
    print(f"  hole index = 1 - l, κ·r_hole <= 0.8: {stats[False][0]}/{stats[False][1]}"
          "  (near-Neumann hole wall)")


def study_vortices():
    X, Y = grid(3.0, 241)
    rng = np.random.default_rng(3)
    xi = 0.35
    for g in (2.0, 3.5):
        m0 = g / xi
        counts = []
        for L in range(0, 7):
            cores: list = []
            while len(cores) < L:
                p = rng.uniform(-2.0, 2.0, 2)
                if all(np.hypot(*(p - q)) > 1.4 for q in cores):
                    cores.append(p)
            modes = higgs(X, Y, m0**2 * vortex_condensate(X, Y, cores, xi), 4 * L + 4)
            counts.append(len(trapped(modes, m0)))
        print(f"  g = {g}: trapped photon modes for L = 0..6 vortices: {counts}")


def study_shells():
    X, Y = grid(8.0, 241)  # units of ξ
    prof = vortex_condensate(X, Y, [(0.0, 0.0)], 1.0)

    def n_bound(g):
        return len(trapped(higgs(X, Y, g * g * prof, 16), g))

    print("  g:      " + "  ".join(f"{g:4.1f}" for g in (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6)))
    print("  bound:  " + "  ".join(f"{n_bound(g):4d}" for g in (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6)))
    lo, hi = 2.0, 2.5  # bisect the p-shell threshold (bound count 1 → 3)
    for _ in range(12):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if n_bound(mid) >= 3 else (mid, hi)
    print(f"  p shell appears at g = {hi:.3f} (bisection, ±{hi - lo:.0e})")


def main():
    print("[1] PEC wall = infinite Higgs mass")
    study_limit()
    print("\n[2] Mass gap protects the bulk-boundary law")
    study_gap()
    print("\n[3] Condensate-filled hole")
    study_hole()
    print("\n[4] Higgs vortices trap light")
    study_vortices()
    study_shells()


if __name__ == "__main__":
    main()
