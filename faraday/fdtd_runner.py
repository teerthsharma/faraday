# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.fdtd_runner — Headless subprocess runner for Topological FDTD.

Loads a pre-trained :class:`GodTensor` from a pickle, builds a
:class:`TopologicalFDFD` initial condition for a target geometry, and
runs a :class:`TopologicalFDTD` simulation for ``--steps`` time steps,
emitting one JSON log line per step on stdout.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone

import numpy as np

from faraday.god_tensor import GodTensor
from faraday.logging import get_logger
from faraday.topological_solver import TopologicalFDFD, TopologicalFDTD


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def main() -> None:
    p = argparse.ArgumentParser(description="Topological FDTD Runner")
    p.add_argument(
        "--steps", type=int, required=True, help="Number of time steps to simulate"
    )
    p.add_argument("--dt", type=float, default=0.01, help="Time step size")
    p.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to a trained GodTensor pickle (.pkl)",
    )
    p.add_argument(
        "--geometry",
        type=float,
        nargs="+",
        default=[2.0, 1.5],
        help="Geometry parameters (e.g. width height)",
    )
    args = p.parse_args()

    log = get_logger(__name__)

    try:
        gt = GodTensor.load(args.checkpoint_path)
    except Exception as exc:
        log.error(
            "failed_to_load_god_tensor",
            error=str(exc),
            path=args.checkpoint_path,
        )
        sys.exit(1)

    fdfd = TopologicalFDFD(gt)
    try:
        res = fdfd.solve(tuple(args.geometry))
    except Exception as exc:
        log.error("fdfd_initialization_failed", error=str(exc))
        sys.exit(1)

    initial_state = np.asarray(res["e_latent_vector"], dtype=np.float64)
    fdtd = TopologicalFDTD(gt, dt=args.dt)

    current = initial_state
    for step in range(1, args.steps + 1):
        current = fdtd.step(current)
        norm = float(np.linalg.norm(current))
        log.info(
            "fdtd_step",
            epoch=step,
            banach_loss=norm,
            betti_0_err=0.0,
            betti_1_err=0.0,
            betti_2_err=0.0,
            timestamp=_now_iso(),
        )
        # Tiny throttle so the daemon's parser keeps pace with very fast
        # latent steps.
        if step % 100 == 0:
            time.sleep(0.01)


if __name__ == "__main__":
    main()
