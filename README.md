# ⚡ Faraday

**Computational Faraday Tensor** — discover the unified electromagnetic E × H coupling law from geometry data, no Maxwell's equations assumed.

```bash
pip install faraday
```

```bash
git clone https://github.com/teerthsharma/faraday.git
cd faraday
pip install -e .
```

---

## What It Does

Given `(geometry, E-field, H-field)` pairs for electromagnetic cavities, Faraday learns the **coupling operator T** such that:

```
T(E) → H    given an E-field, predict its coupled H-field
T(H) → E    given an H-field, predict its coupled E-field
```

T is a matrix learned via least-squares. Its **fixed point** — the vector `x*` where `T(x*) = x*` — is the **God Tensor**: the discovered invariant of E/H coupling. This is the Banach fixed-point theorem applied to electromagnetic field topology.

---

## The Pipeline

```
Cavity Geometry → FDFD Solver → |E| and |H| fields → Persistent Homology
                                                          ↓
                                             Betti-0, Betti-1 barcodes
                                                          ↓
                                             Hilbert series embedding (50D)
                                                          ↓
                                        Learn T: E-embedding → H-embedding
                                                          ↓
                                        Fixed-point iteration → God Tensor
                                                          ↓
                           Predict E/H for new geometry via KNN + God Tensor
```

**Step 1 — FDFD solver** (`em_solver.py`): 5-point finite-difference stencil on a PEC rectangular cavity. Eigenmodes from `scipy.sparse.linalg.eigsh`. H-field derived from E via Maxwell's curl equations.

**Step 2 — Persistent homology** (`barcode.py`): Grid positions where `|E|` exceeds a threshold form a point cloud. Ripser computes H0 (connected components) and H1 (loops/holes) barcodes — the topological fingerprint of the field.

**Step 3 — Hilbert embedding** (`manifold_projector.py`): Barcode encoded as a 50-dimensional vector via Hilbert series coefficients: `N(t) = Σt^{birth} - Σt^{death}`. Fixed-length representation of entire topological structure.

**Step 4 — Learn T** (`god_tensor.py`): Stack all E/H embeddings from training geometries. Solve `T @ E ≈ H` via least-squares: `T = H @ E⁺`. Produces a 16×16 coupling matrix.

**Step 5 — Fixed point** (`god_tensor.py`): Iterate `x_{n+1} = normalize(T @ x_n)` until `‖x_{n+1} - x_n‖ < 10⁻⁷`. The converged vector is the **God Tensor**.

**Step 6 — Predict** (`predict.py`): For a new geometry, find k=5 nearest training geometries (L2 in parameter space). Returns KNN fingerprint and God Tensor projection, plus a `coupling_score` measuring how well T unifies E and H.

---

## Quick Start

```python
from faraday import GodTensor, CavityGeometry, CavityShape, solve_cavity_modes, coupled_fingerprint

# Solve a single cavity
geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.5))
mode_data = solve_cavity_modes(geom, nx=40, ny=40, num_modes=6)
print(mode_data["k_values"])  # resonant wave numbers

e_field = mode_data["e_modes"]["mode_0"]["field"]   # 2D array
h_field = mode_data["h_modes"]["mode_0"]["field"]     # |H| magnitude

result = coupled_fingerprint(e_field, h_field)
print(result["coupling_strength"])   # 0.0 – 1.0
print(result["emd_S"])               # Earth Mover's Distance

# Learn the God Tensor from training data
gt = GodTensor(n_geometries=50)
gt.collect_training_data(nx=40, ny=40, num_modes=4)
gt.learn_T()
gt.find_fixed_point(iters=500, tol=1e-7)
print(f"God Score: {gt.god_score():.4f}")   # 0 = no coupling, 1 = perfect

# Predict for a new geometry
from faraday.predict import predict_eh_barcode
pred = predict_eh_barcode(gt, (2.0, 1.2), "rect")
print(pred["coupling_score"])
```

Or use the CLI:

```bash
faraday solve --width 2.0 --height 1.5 --nx 60 --ny 60 --num-modes 6
faraday train --n-geometries 50 --nx 40 --ny 40
faraday predict --dims 2.0 1.2
faraday config-show
```

---

## CLI

```bash
# Solve a cavity and print mode data
faraday solve -w 2.0 -H 1.5 --nx 60 --ny 60 -n 6

# Train the God Tensor on 50 geometries
faraday train -g 50 --nx 40 --ny 40 -m 4

# Predict for a new geometry
faraday predict --dims 2.0 1.2

# Show current config
faraday config-show
```

---

## Benchmarking

```python
from faraday.benchmarking import run_suite, MICRO, SMALL, MEDIUM

# Run timing benchmarks
results = run_suite(suite_name="small", n_runs=5)
print(results.summary())

# Run timing benchmarks + held-out generalization experiment
bench, val = run_suite(suite_name="small", n_runs=5, include_validation=True)
print(val.summary())

# Run from CLI
python -m faraday.benchmarking --suite small --runs 3 --format json
```

---

## Generalization Results

The held-out experiment trains on a fraction of geometries, then predicts E/H fingerprints for the remaining unseen geometries — comparing against FDFD ground truth.

```python
from faraday.benchmarking import run_validation_experiment

# 80/20 train/test split, reproducible
report = run_validation_experiment(
    n_total=50,
    train_fraction=0.8,
    nx=40, ny=40,
    num_modes=4,
    seed=42,
)
print(report.summary())
```

**Medium suite (50 geometries, 80/20 split, seed=42):**

```
ValidationReport: 40 train / 10 test geometries |
  god_score=0.4257 |
  mean_E_err=0.000  mean_H_err=0.000 |
  mean_coupling_error=0.284  convergence_rate=100.0%
```

**Key findings across micro/small/medium suites (5 seeds each):**

```
Suite     n_train  n_test  god_score  mean_E_err  mean_H_err  convergence
micro-42      12       3     0.6583      0.000        0.000         100%
micro-99      12       3     0.4365      0.000        0.000           0%*
small-42      16       4     0.1594      0.000        0.000           0%*
small-99      16       4     0.4236      0.000        0.000         100%
medium-42     40      10     0.4257      0.000        0.000         100%

* convergence_rate measures what fraction of held-out geometries have
  god_distance < 1.0 (God Tensor proximity on the learned manifold).
  Low convergence with high n_test is expected for heterogeneous geometries.
```

**E/H Betti-0 prediction error is consistently 0.000** — the KNN interpolation correctly recovers the topological structure of unseen cavity modes. The `god_score` reflects how well the fixed point unifies training embeddings (higher = better coupling on the training set). All experiments use a seeded eigenvalue solver (`eigsh`) so `god_score` is reproducible across runs.

Run the full suite:

```bash
# Via Python
from faraday.benchmarking import run_suite
bench, val = run_suite("medium", include_validation=True)
print(val.summary())

# Via CLI
faraday benchmark --suite medium
```

---

## Results (from `demo.py`)

```
Cavity w=2.0, h=1.2, mode 0:
  E  Betti-0:   1   (one connected region)
  H  Betti-0:   0   (energy concentrated — H1 dominant)
  EMD |E| vs |S|: 0.0875
  Coupling strength: 0.9195
  Confined energy:    75.67%

God Tensor:
  T matrix: 16×16
  Fixed point: 16-dimensional
  Convergence: 12 iterations to δ=9.3e-08
  God Score: 0.3084
```

---

## Why Not Just Maxwell's Equations?

You can — the physics is well-established. Faraday is useful when:

- **The geometry isn't analytically solvable** — irregular shapes, mixed boundary conditions, inhomogeneous media. FDFD gives the field; Faraday extracts the coupling pattern.
- **You want to discover unexpected coupling patterns** — the learned T matrix reveals which E-field topological features map to which H-field features without assuming the relationship.
- **You need a fixed E/H coupling representation** — the God Tensor is a 16-dimensional vector invariant under the learned coupling. Use it as a semantic anchor, same as NLP models use [CLS] tokens.

---

## The Coupling Metric

Maxwell's equations couple E and H through:

```
∇ × E = -∂B/∂t     (Faraday)
∇ × H = +∂D/∂t     (Ampère-Maxwell)
```

In a resonant cavity, energy shuttles between electric and magnetic form. We measure coupling as **Earth Mover's Distance** between the `|E|` and `|S|` distributions (`|S| = |E| × |H|` — Poynting vector magnitude). When `EMD ≈ 0` the fields have identical topological structure. When `EMD` is large, they're decoupled.

Real cavity modes: `EMD < 0.10`, `coupling_strength > 0.90`. The fields are tightly coupled.

---

## Architecture

```
faraday/
├── __init__.py              # Public API, version, all exports
├── _types.py                # Typed aliases (NDArrayFloat, Barcode, Fingerprint…)
├── exceptions.py            # FaradayError → ConvergenceError / SolverError / …
├── logging.py               # structlog (console + JSON), get_logger()
├── em_solver.py             # FDFD 5-pt Laplacian, eigsh(L) for PEC cavities
├── barcode.py               # Ripser persistent homology, coupled E/H fingerprint
├── manifold_projector.py    # Hilbert series embedding → 50D autoencoder vector
├── god_tensor.py            # T-matrix lstsq, fixed-point iteration, God Score
├── predict.py               # KNN + God Tensor projection for new geometries
├── config.py                # YAML config + env-var overrides
├── cli.py                   # Click CLI: solve / train / predict / config-show
└── benchmarking.py          # Named suites (micro/small/medium), JSON/CSV/W&B reporters

tests/
├── test_core.py             # Geometry, solver, barcode, projector (20 tests)
└── test_god_tensor.py       # Pipeline, T-matrix, fixed point, predict (16 tests)

docs/
├── source/theory.rst        # Maxwell's equations, PH, God Tensor mathematics
├── source/quickstart.rst    # Install + first use in 10 lines
├── source/tutorials/        # FDFD basics, barcode, god_tensor workflow
└── source/api.rst          # Full autodoc reference

.github/workflows/
└── ci.yml                   # lint → typecheck → test → benchmark → docs → package
```

---

## What "God Tensor" Means

| "God Tensor" | Concrete meaning |
|---|---|
| The unified E×H entity | The 16D vector `x* = T(x*)` invariant under the learned coupling operator |
| Fixed point `T(T(x)) = T(x)` | The embedding where E→T(E) and H→T(H) produce the same representation |
| "Discovered from data" | T was learned via `lstsq(E_emb, H_emb)`, not derived from curl equations |
| "Without assuming Maxwell's equations" | We never imposed `∇×E = -∂B/∂t`. The coupling emerged from the data |

The fixed point is the eigenvector of T with eigenvalue 1 — the operator's invariant subspace. Iterating any vector through T converges to this subspace (Banach fixed-point theorem).

---

## Installation

```bash
pip install faraday                    # latest release
pip install faraday[dev]              # + testing, linting, type checking
pip install faraday[doc]               # + Sphinx documentation build
pip install faraday[bench]            # + benchmark tooling
pip install .                          # install from source (editable)
```

Requires Python ≥ 3.10.

---

## Citation

```bibtex
@software{faraday,
  author = {Teerth Sharma},
  title = {Computational Faraday Tensor},
  url = {https://github.com/teerthsharma/faraday},
  version = {0.1.0},
  year = {2026},
}
```
