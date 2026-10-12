# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed — God Tensor rebuilt on the E ⇄ H duality (breaking)
- `GodTensor` embeddings are now duality signatures. The E-view is
  χ(D) + N_wall/2 and the H-view is the vortex charge Q, for the first
  `signature_modes` (16) modes. The `ManifoldProjector` autoencoder is no
  longer used: its ReLU erased negative charges.
- When the law holds, the learned T is an orthogonal projector (T² = T). The
  God Tensor is its fixed subspace. New fields: `god_basis`,
  `idempotency_defect`; `summary()` reports `fixed_point_multiplicity`.
- Fixed point: a degenerate λ* = 1 is treated as one subspace. The basis
  comes from the near-null space of T − λ*I, and convergence also stops on
  the eigen-residual. Distances are measured to the subspace
  (`GodTensor.distance_to_god`).
- `god_tensor.make_sample` and `duality_views` are shared by
  `collect_training_data` and `run_validation_experiment`. Old pickled models
  (50-D barcode embeddings) are not compatible.

### Added — quantum sector
- `faraday.quantum`:
  - `phase_charges`, `wall_windings` and `berry_sign`.
  - Phase-vortex bound |V| ≤ min N_wall/2 (66/66).
  - Holonomy V_bulk = w_outer − Σ w_holes (66/66).
  - Z₂ geometric phase around the TM31/TM12 diabolical point.
- `experiments/quantum_study.py`, `docs/QUANTUM.md`, `tests/test_quantum.py`.

### Added: Higgs coupling (`faraday.higgs`)
- `solve_interior_modes(..., mass2=m²)` solves (−Δ + m²(x))E = ω²E, the
  photon in a Higgs or Proca background (scalar TM model).
- `higgs_wall_mass2`, `vortex_condensate`, `trapped`, `region_charge`,
  `box_interior`.
- Findings:
  - A PEC wall is the infinite-mass limit of the condensate.
  - The mass gap protects Q = χ + N/2: 91/91 trapped modes against 34% of
    untrapped ones.
  - A condensate-filled hole carries index 1 − ℓ (22/22).
  - Each vortex traps n_shell(g) photon modes, with exactly 1 below
    g = m₀ξ = 2.299.
- `experiments/higgs_study.py`, `docs/HIGGS.md`, `tests/test_higgs.py`.

### Added — topological E ⇄ H duality
- `faraday.duality`: exact vortex charges of H (discrete Stokes), nodal
  domains and Euler characteristics, wall nodal count, the `ModeTopology`
  signature, geometry-only `predict_rectangle_topology` and
  `discover_coupling`. The bulk–boundary law Q = χ(D) + N_wall/2 and the
  domain law charge(Ω) = χ(Ω) hold on 132/132 modes of 11 cavities (χ = +1,
  0, −1). See `docs/DUALITY.md`.
- `em_solver.solve_interior_modes`: TM modes of any cavity given as a grid
  mask (non-convex, with holes); `solve_cavity_modes` delegates to it.
- `faraday topology` CLI command and `experiments/duality_study.py`.

### Fixed
- `em_solver`: the eigensolver skipped the fundamental mode and returned modes
  in descending k (`mode_0` was the *highest* computed mode). The Laplacian is
  now the symmetric Dirichlet operator on interior nodes, solved with
  shift-invert `eigsh(sigma=0)`; modes are sorted ascending and match the
  analytic TM_mn / Bessel-zero wave numbers. Circular cavities no longer crash
  when `nx != ny`. H-field curl is vectorised and keeps the component signs.
- `barcode`: Betti numbers dropped the essential class (a single blob had
  Betti-0 = 0; |H| always scored 0). Betti counts now include infinite bars and
  use a lattice-spacing noise threshold, so Betti-0 of E equals the nodal-domain
  count. `topological_score` no longer becomes NaN without H1 bars.
- `manifold_projector.fit`: backprop missed the decoder ReLU mask; the
  gradient is now exact (finite-difference tested).
- `god_tensor`: the fixed-point search seeded power iteration *on* an
  eigenvector (a repelling fixed point) and reported convergence at iteration
  0. It now runs a genuine Banach contraction (shifted inverse iteration onto
  the eigenvalue nearest 1) from a generic start and records
  `fixed_point_eigenvalue` and `contraction_factor`. H → E uses the
  pseudo-inverse T⁺ instead of Tᵀ; God-score and distances are sign-invariant;
  `GodTensor.predict` read a non-existent key and always returned `[]`.
- `predict`: KNN compared `(w, h)` with `(r,)` via silent broadcasting and
  could divide by zero for far queries; neighbours are now same-shape and the
  kernel is underflow-safe.
- CLI: `faraday solve` crashed, `faraday train` re-collected the dataset once
  per geometry and could not save a model (`-o` added), `faraday-demo`
  imported a module that did not exist (demo moved into `faraday.demo`).
- `WaveSuperposer` added an H mode even when none was requested and ignored the
  index passed to `add_h_mode`.
- CI: the Generalization job's regex could never match the report's
  `mean_E_err=… | mean_H_err=…` format.

### Added
- Physics/maths tests: analytic eigenvalues and O(h²) convergence, Maxwell
  energy equipartition, nodal-domain Betti numbers, exact autoencoder gradient,
  Banach contraction certification, CLI/config/burn coverage (coverage 89%).

## [0.1.0] — 2026-05-03

### Added
- `faraday` package — a computational electromagnetic simulator based on
  the God Tensor theory: learning the fixed-point of the E ⇄ H
  co-determination operator via topological (persistent homology) methods.
- `faraday.em_solver` — FDFD cavity solver with 5-point finite-difference
  Laplacian. Supports rectangular and circular PEC cavities. Computes TM
  eigenmodes and derives H-field via Maxwell curl equations.
- `faraday.barcode` — Persistent homology analysis via Ripser. Converts
  2D field distributions to topological barcodes (Betti numbers, lifetimes,
  H0/H1 diagrams). Includes coupled fingerprint combining E and H topology.
- `faraday.manifold_projector` — Hilbert series barcode embedding (fixed-length
  vectors from variable-length barcodes) with an autoencoder for learning
  the topological manifold.
- `faraday.god_tensor` — T-matrix learning via least-squares and fixed-point
  iteration. `GodTensor` class with `collect_training_data`,
  `learn_T`, `find_fixed_point`, `god_score`, `predict`.
- `faraday.predict` — KNN-based topology prediction for new geometries
  (ground truth baseline) and God Tensor projection path.
- `faraday.logging` — Structured logging via structlog (console + JSON).
- `faraday.exceptions` — Typed exception hierarchy:
  `FaradayError`, `ConvergenceError`, `SolverError`, `GeometryError`,
  `TopologyError`, `ConfigError`.
- `faraday._types` — Shared type aliases (`NDArrayFloat`, `Barcode`,
  `Fingerprint`, `Embedding`, `ModeData`, etc.).
- `faraday.config` — YAML configuration system with dataclass
  `FaradayConfig` and env-var overrides.
- `faraday.benchmarking` — Reproducible benchmark suite with JSON/CSV reporters.
- `tests/` — 30 unit tests covering geometry, solver, barcode, manifold
  projector, God Tensor pipeline, and prediction.
- `.github/workflows/ci.yml` — Multi-platform CI (Ubuntu + Python 3.10–3.12)
  with pytest + coverage.
- `docs/` — Sphinx documentation with theory, API reference, and quickstart.
- `demo.py` — End-to-end demonstration of the full pipeline.

### Changed
- `solve_cavity_modes` now uses ``which="LM"`` eigenvalue selection
  (largest-magnitude of the negative-semidefinite Laplacian), correctly
  returning dominant structural modes with rich topological features.
- `god_score` now uses an exponential decay formula
  ``exp(-mean_distance / 2)`` for numerically stable [0, 1] bounds.
- `ManifoldProjector.fit` gradient corrected from ``(d_h.T @ batch).T``
  to ``d_h.T @ batch``.

### Fixed
- `ManifoldProjector.fit()` gradient shape mismatch — `(d_h.T @ batch).T`
  broadcasting error on weight update.
- `CavityGeometry.contains()` broadcasting documented — 1D vs 2D inputs
  produce consistent element-wise masks via NumPy broadcasting rules.
- `structlog` `PrintLoggerFactory` incompatibility with `add_logger_name`
  processor — removed (not needed for `PrintLogger`).
