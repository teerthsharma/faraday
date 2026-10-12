#!/usr/bin/env python3
# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
Topological E ⇄ H duality study — reproduces every number in docs/DUALITY.md.

    python experiments/duality_study.py            # all studies (~5 min)
    python experiments/duality_study.py --quick    # coarser grids (~1 min)

Studies
-------
1. Bulk-boundary law Q = χ(D) + N_wall/2 and domain law charge(Ω) = χ(Ω)
   on simply connected cavities.
2. Falsification on multiply connected cavities (χ(D) = 1 - #holes).
3. Geometry-only prediction of rectangle mode topology vs FDFD.
4. True vs avoided level crossing: how the integer charge changes.
5. discover_coupling: does least squares rediscover the law from data?
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import structlog

from faraday.duality import (
    discover_coupling,
    mode_topology,
    predict_rectangle_topology,
)
from faraday.em_solver import solve_interior_modes

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


def _polar(X, Y):
    return np.hypot(X, Y), np.arctan2(Y, X)


SIMPLY_CONNECTED = {
    "rectangle 2.0x1.2": lambda X, Y: (abs(X) < 1.0) & (abs(Y) < 0.6),
    "ellipse 1.2x0.8": lambda X, Y: (X / 1.2) ** 2 + (Y / 0.8) ** 2 < 1,
    "stadium": lambda X, Y: ((abs(X) < 0.6) & (abs(Y) < 0.6))
    | ((abs(X) - 0.6) ** 2 + Y**2 < 0.36),
    "star": lambda X, Y: _polar(X, Y)[0]
    < 0.95 * (1 + 0.15 * np.cos(3 * _polar(X, Y)[1]) + 0.1 * np.sin(5 * _polar(X, Y)[1])),
    "trapezoid": lambda X, Y: (abs(Y) < 0.6) & (abs(X) < 1.0 + 0.25 * Y),
    "D-shape": lambda X, Y: ((X < 0) & (X > -0.7) & (abs(Y) < 0.9))
    | ((X >= 0) & (X**2 + Y**2 < 0.81)),
}

MULTIPLY_CONNECTED = {
    "annulus (concentric)": lambda X, Y: (np.hypot(X, Y) < 1.0) & (np.hypot(X, Y) > 0.35),
    "annulus (eccentric)": lambda X, Y: (np.hypot(X, Y) < 1.0)
    & (np.hypot(X - 0.3, Y + 0.1) > 0.3),
    "rectangle with hole": lambda X, Y: (abs(X) < 1.1) & (abs(Y) < 0.7)
    & (np.hypot(X - 0.35, Y - 0.15) > 0.25),
    "disk with two holes": lambda X, Y: (np.hypot(X, Y) < 1.1)
    & (np.hypot(X - 0.45, Y) > 0.22) & (np.hypot(X + 0.4, Y + 0.3) > 0.18),
    "square with square hole": lambda X, Y: (abs(X) < 1.0) & (abs(Y) < 1.0)
    & ~((abs(X - 0.2) < 0.3) & (abs(Y + 0.1) < 0.3)),
}


def grid(R: float, n: int):
    c = np.linspace(-R, R, n)
    return np.meshgrid(c, c)


def study_laws(shapes: dict, R: float, n: int, num_modes: int) -> list:
    tops = []
    for name, f in shapes.items():
        X, Y = grid(R, n)
        modes = solve_interior_modes(X, Y, f(X, Y), num_modes=num_modes, seed=0)
        ts = [mode_topology(modes, j) for j in range(num_modes)]
        tops += ts
        bad = [j for j, t in enumerate(ts) if not (t.boundary_law_holds and t.domain_law_holds)]
        print(
            f"  {name:26s} χ(D)={ts[0].euler_cavity:+d}  "
            f"Q per mode={[t.charge for t in ts]}  "
            f"{'all ✓' if not bad else f'✗ modes {bad}'}"
        )
    ok = sum(t.boundary_law_holds and t.domain_law_holds for t in tops)
    print(f"  → both laws hold on {ok}/{len(tops)} modes")
    return tops


def study_prediction(n_aspects: int, n: int, num_modes: int) -> None:
    rng = np.random.default_rng(7)
    hits = total = 0
    misses = []
    for a in np.sort(rng.uniform(1.0, 3.0, n_aspects)):
        w, h = a, 1.0
        R = 0.51 * w
        X, Y = grid(R, n)
        modes = solve_interior_modes(
            X, Y, (abs(X) < w / 2) & (abs(Y) < h / 2), num_modes=num_modes, seed=0
        )
        for j, p in enumerate(predict_rectangle_topology(w, h, num_modes)):
            t = mode_topology(modes, j)
            ok = (t.n_domains, t.n_wall, t.charge) == (p["n_domains"], p["n_wall"], p["charge"])
            hits += ok
            total += 1
            if not ok:
                misses.append(p["gap"])
    print(f"  geometry-only predictions correct: {hits}/{total} = {hits / total:.1%}")
    if misses:
        print(f"  every miss within {max(misses):.2%} of a level crossing")


def study_crossing(n: int) -> None:
    X, Y = grid(1.0, n)
    for delta in (0.0, 0.15):
        kind = "true crossing" if delta == 0 else "avoided crossing (parity broken)"
        print(f"  δ={delta}: {kind}")
        print("     w/h    gap%   Q(j=2) N_wall(j=2)   Q(j=3) N_wall(j=3)")
        for w in np.linspace(1.50, 1.78, 8):
            mask = (abs(Y) < 0.5) & (abs(X) < w / 2 + delta * Y)
            modes = solve_interior_modes(X, Y, mask, num_modes=4, seed=0)
            t2, t3 = mode_topology(modes, 2), mode_topology(modes, 3)
            gap = 100 * (t3.k - t2.k) / t2.k
            print(
                f"     {w:.3f} {gap:6.2f}     {t2.charge}        {t2.n_wall}"
                f"            {t3.charge}        {t3.n_wall}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--quick", action="store_true", help="coarser grids")
    args = parser.parse_args()
    n = 121 if args.quick else 181

    print("[1] Simply connected cavities")
    tops = study_laws(SIMPLY_CONNECTED, 1.3, n, 12)
    print("\n[2] Multiply connected cavities (falsification of the χ(D) term)")
    tops += study_laws(MULTIPLY_CONNECTED, 1.2, int(n * 1.66) | 1, 12)
    print("\n[3] Topology from geometry alone (rectangles)")
    study_prediction(8 if args.quick else 20, n, 10)
    print("\n[4] How charge changes across a level crossing (TM31 ↔ TM12)")
    study_crossing(n)
    print("\n[5] Least squares on E-side invariants → H-side charge")
    fit = discover_coupling(tops)
    print("  " + ", ".join(f"{k}={v:+.4f}" for k, v in fit.items()))


if __name__ == "__main__":
    main()
