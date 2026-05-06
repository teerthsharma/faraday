# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.fdtd_runner — Headless subprocess for running Topological FDTD

This module is designed to be invoked by the execution_daemon.
It loads a pre-trained God Tensor, initiates a TopologicalFDFD field,
and runs a TopologicalFDTD simulation for N steps, emitting JSON logs.
"""

import argparse
import sys
import time
from datetime import datetime, timezone
import numpy as np

from faraday.god_tensor import GodTensor
from faraday.topological_solver import TopologicalFDFD, TopologicalFDTD
from faraday.logging import get_logger

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

def main():
    p = argparse.ArgumentParser(description="Topological FDTD Runner")
    p.add_argument("--steps", type=int, required=True, help="Number of time steps to simulate")
    p.add_argument("--dt", type=float, default=0.01, help="Time step size")
    p.add_argument("--checkpoint-path", type=str, required=True, help="Path to trained God Tensor checkpoint")
    p.add_argument("--geometry", type=float, nargs="+", default=[2.0, 1.5], help="Geometry parameters (e.g. width height)")
    args = p.parse_args()

    log = get_logger(__name__)

    try:
        gt, _, _ = GodTensor.load_checkpoint(args.checkpoint_path)
    except Exception as e:
        log.error("failed_to_load_god_tensor", error=str(e), path=args.checkpoint_path)
        sys.exit(1)

    fdfd = TopologicalFDFD(gt)
    try:
        res = fdfd.solve(tuple(args.geometry))
    except Exception as e:
        log.error("fdfd_initialization_failed", error=str(e))
        sys.exit(1)

    initial_state = np.array(res["e_latent_vector"])
    fdtd = TopologicalFDTD(gt, dt=args.dt)
    
    current_state = initial_state
    for step in range(1, args.steps + 1):
        current_state = fdtd.step(current_state)
        norm = float(np.linalg.norm(current_state))
        
        # Emit the exact JSON format expected by the execution daemon
        log.info(
            "fdtd_step",
            event="fdtd_step",
            epoch=step,  # We map step -> epoch for daemon compatibility
            banach_loss=norm,  # We map norm -> banach_loss for daemon tracking
            betti_0_err=0.0,
            betti_1_err=0.0,
            betti_2_err=0.0,
            timestamp=_now_iso()
        )
        
        # Small sleep to prevent swamping the daemon parser for fast topological steps
        if step % 100 == 0:
            time.sleep(0.01)

if __name__ == "__main__":
    main()
