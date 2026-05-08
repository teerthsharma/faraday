#!/usr/bin/env python3
# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
Faraday Demo — The God Tensor in Action.

End-to-end pipeline:

1. Collect training data — varied cavity geometries + E/H fields.
2. Learn the coupling operator T via least squares on Hilbert-series
   barcode embeddings.
3. Find the spectral fixed point (the God Tensor) via the normalised
   power method.
4. Predict E/H topology for a new (held-out) geometry.
5. Compare against ground-truth FDFD.
6. Run the full 80/20 held-out generalisation experiment.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from faraday import (
    CavityGeometry,
    CavityShape,
    GodTensor,
    coupled_fingerprint,
    solve_cavity_modes,
)
from faraday.benchmarking import run_validation_experiment
from faraday.predict import predict_eh_barcode


def main() -> None:
    print("=" * 60)
    print(" FARADAY -- Computational Faraday Tensor")
    print("   The God Tensor: spectral fixed point of E <-> H coupling")
    print("=" * 60)

    # 1. Collect training data
    print("\n[1] Collecting training data...")
    print("    20 random cavity geometries -> E/H field fingerprints")
    gt = GodTensor(n_geometries=20)
    gt.collect_training_data(nx=30, ny=30, num_modes=4)
    print(f"     OK  {len(gt.samples)} samples collected")

    # 2. Learn T
    print("\n[2] Learning coupling operator T: E_embedding -> H_embedding...")
    T = gt.learn_T()
    print(f"     OK  T.shape = {T.shape}")

    # 3. Find the God Tensor
    print("\n[3] Finding spectral fixed point (God Tensor)...")
    god = gt.find_fixed_point(iters=500, tol=1e-12)
    summary = gt.summary()
    print(f"     OK  god.shape         = {god.shape}")
    print(f"     OK  converged         = {summary['converged']}")
    print(f"     OK  final_residual    = {summary['final_spectral_residual']:.3e}")
    print(f"     OK  dominant_eigval   = {summary['dominant_eigenvalue']!r}")
    print(f"     OK  spectral_gap      = {summary['spectral_gap_ratio']:.4f}")
    print(f"     OK  god_score         = {summary['god_score']:.4f}")

    # 4. Predict for new geometry
    test_geom = (2.0, 1.2)
    print(f"\n[4] Predicting E/H for new geometry: w={test_geom[0]}, h={test_geom[1]}")
    pred = predict_eh_barcode(gt, test_geom, "rect")
    print(f"     KNN E Betti-0:    {pred['knn_e_fingerprint']['betti_0']:.0f}")
    print(f"     KNN H Betti-0:    {pred['knn_h_fingerprint']['betti_0']:.0f}")
    print(f"     god_distance E:   {pred['god_distance_e']:.4f}")
    print(f"     god_distance H:   {pred['god_distance_h']:.4f}")
    print(f"     coupling score:   {pred['coupling_score']:.4f}")

    # 5. Verify with actual FDFD
    print(f"\n[5] Verifying with actual FDFD on {test_geom}...")
    geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=test_geom)
    mode_data = solve_cavity_modes(geom, nx=30, ny=30, num_modes=4)
    e_field = np.asarray(mode_data["e_modes"]["mode_0"]["field"])
    h_field = np.asarray(mode_data["h_modes"]["mode_0"]["field"])
    actual = coupled_fingerprint(e_field, h_field)
    print(f"     Actual E Betti-0:        {actual['e_fingerprint']['betti_0']}")
    print(f"     Actual H Betti-0:        {actual['h_fingerprint']['betti_0']}")
    print(f"     EMD(|E|, |S|):           {actual['emd_S']:.4f}")
    print(f"     coupling_strength:       {actual['coupling_strength']:.4f}")
    print(f"     confinement_alignment:   {actual['confinement_alignment']:.2%}")

    # 6. Held-out generalisation experiment
    print("\n[6] Held-out generalisation experiment (80/20 split)...")
    val_report = run_validation_experiment(
        n_total=30,
        train_fraction=0.8,
        nx=30,
        ny=30,
        num_modes=4,
        seed=42,
    )
    print("\n " + val_report.summary())

    print("\n" + "=" * 60)
    print(" Faraday demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
