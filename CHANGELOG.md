# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
