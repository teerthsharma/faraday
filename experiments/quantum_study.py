#!/usr/bin/env python3
# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
Quantum sector study — reproduces every number in docs/QUANTUM.md.

    python experiments/quantum_study.py      # ~1 min

Studies
-------
1. Phase-vortex bound |V| <= min(N_wall(u), N_wall(v)) / 2 for ψ = u + iv,
   with random U(1) rotations inside degenerate pairs (gauge invariance).
2. Holonomy: V_bulk = w_outer - Σ w_holes on cavities with 0, 1, 2 holes.
3. Geometric phase: Z₂ sign of a mode transported around loops in the
   (width, tilt) shape plane, enclosing or missing the diabolical point.
4. God Tensor on duality signatures: idempotency of T, multiplicity of λ = 1.
"""

from __future__ import annotations

import logging

import numpy as np
import structlog

from faraday.duality import mode_topology
from faraday.em_solver import solve_interior_modes
from faraday.god_tensor import GodTensor
from faraday.quantum import berry_sign, phase_charges, wall_windings

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


def grid(R: float, n: int):
    c = np.linspace(-R, R, n)
    return np.meshgrid(c, c)


def field(modes, j):
    return np.array(modes["e_modes"][f"mode_{j}"]["field"])


SHAPES = {
    "disk": (lambda X, Y: np.hypot(X, Y) < 1.0, 1.05, 181),
    "square": (lambda X, Y: (abs(X) < 1) & (abs(Y) < 1), 1.05, 161),
    "ellipse": (lambda X, Y: (X / 1.2) ** 2 + (Y / 0.8) ** 2 < 1, 1.25, 181),
    "annulus": (lambda X, Y: (np.hypot(X, Y) < 1) & (np.hypot(X, Y) > 0.35), 1.05, 181),
    "eccentric annulus": (
        lambda X, Y: (np.hypot(X, Y) < 1) & (np.hypot(X - 0.3, Y + 0.1) > 0.3), 1.05, 181
    ),
    "disk with two holes": (
        lambda X, Y: (np.hypot(X, Y) < 1.1)
        & (np.hypot(X - 0.45, Y) > 0.22)
        & (np.hypot(X + 0.4, Y + 0.3) > 0.18),
        1.15,
        241,
    ),
}


def study_bound_and_holonomy() -> None:
    rng = np.random.default_rng(0)
    b_ok = b_tot = h_ok = h_tot = saturated = 0
    for name, (f, R, n) in SHAPES.items():
        X, Y = grid(R, n)
        mask = f(X, Y)
        modes = solve_interior_modes(X, Y, mask, num_modes=12, seed=0)
        tops = [mode_topology(modes, j) for j in range(12)]
        Vs = []
        for j in range(11):
            psi = field(modes, j) + 1j * field(modes, j + 1)
            psi *= np.exp(1j * rng.uniform(0, 2 * np.pi))  # gauge
            V = int(phase_charges(psi, mask).sum())
            bound = min(tops[j].n_wall, tops[j + 1].n_wall) // 2
            w = wall_windings(psi, mask)
            b_ok += abs(V) <= bound
            b_tot += 1
            h_ok += w[0] - sum(w[1:]) == V
            h_tot += 1
            saturated += abs(V) == bound and bound > 0
            Vs.append(V)
        print(f"  {name:20s} V per neighbouring pair = {Vs}")
    print(f"  bound |V| <= min(N_wall)/2: {b_ok}/{b_tot}  (saturated with V != 0: {saturated})")
    print(f"  holonomy V_bulk = w_outer - Σ w_holes: {h_ok}/{h_tot}")


def study_berry() -> None:
    X, Y = grid(1.0, 121)

    def mode(w, d, j):
        mask = (abs(Y) < 0.5) & (abs(X) < w / 2 + d * Y)
        return field(solve_interior_modes(X, Y, mask, num_modes=4, seed=0), j)

    w_star = np.sqrt(8 / 3)
    loops = {
        "encloses (w*, 0)": (w_star, 0.0),
        "misses (shifted in w)": (w_star + 0.25, 0.0),
        "misses (shifted in tilt)": (w_star, 0.30),
    }
    ts = np.linspace(0, 2 * np.pi, 65)
    for label, (cw, cd) in loops.items():
        for j in (2, 3):
            sign, ov = berry_sign(
                mode(cw + 0.12 * np.cos(t), cd + 0.15 * np.sin(t), j) for t in ts
            )
            print(f"  loop {label:25s} mode {j}: sign {sign:+d}  (min step overlap {ov:.3f})")


def study_god_tensor() -> None:
    for nx in (15, 30):
        gt = GodTensor(n_geometries=20)
        gt.collect_training_data(nx=nx, ny=nx, num_modes=4, seed=42)
        gt.find_fixed_point()
        E = np.array([s.e_embedding for s in gt.samples])
        H = np.array([s.h_embedding for s in gt.samples])
        s = gt.summary()
        print(
            f"  nx={nx}: law-violating entries {int((np.abs(E - H) > 1e-9).sum())}/{E.size}, "
            f"||T²-T||/||T|| = {s['idempotency_defect']:.1e}, "
            f"multiplicity of λ=1: {s['fixed_point_multiplicity']} (rank E = {np.linalg.matrix_rank(E)}), "
            f"god_score = {s['god_score']:.4f}"
        )


def main() -> None:
    print("[1-2] Phase vortices and holonomy (ψ = u + iv of neighbouring modes)")
    study_bound_and_holonomy()
    print("\n[3] Geometric phase around the TM31/TM12 diabolical point")
    study_berry()
    print("\n[4] God Tensor on duality signatures")
    study_god_tensor()


if __name__ == "__main__":
    main()
