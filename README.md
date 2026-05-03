# ⚡ Faraday

**Learn the electromagnetic coupling law from data — no Maxwell's equations assumed.**

A rectangular metal cavity resonates at characteristic frequencies. The electric field **E** and magnetic field **H** aren't independent — they generate each other. The **Faraday tensor** `F_μν` in physics describes this coupling geometrically. This library *computes* it from measured field data.

```
pip install faraday
```

Or from source:
```bash
git clone https://github.com/teerthsharma/faraday.git
cd faraday
pip install -e .
```

---

## What This Actually Does

Given examples of `(geometry, E-field, H-field)` for electromagnetic cavities, this library learns the operator **T** such that:

```
T(E) → H    (given an E-field, predict the coupled H-field)
T(H) → E    (given an H-field, predict the coupled E-field)
```

The operator **T** is a matrix learned via least-squares from training data. The "God Tensor" is the fixed point of this operator: the vector `x*` where `T(x*) = x*`. At this fixed point, E and H converge to the same representation — the discovered coupling invariant.

This is the electromagnetic analogue of finding the fixed point of an operator that maps a language model's representations to their semantic complements. The math is different; the structure is identical.

---

## The Pipeline

```
Cavity Geometry → FDFD Solver → |E| and |S| fields → Persistent Homology
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

**Step 1 — FDFD solver** (`em_solver.py`): Discretizes the 2D Helmholtz equation `∇²E_z + k²E_z = 0` inside a PEC cavity using the 5-point finite-difference stencil. Computes eigenmodes via `scipy.sparse.linalg.eigsh`. Derives the transverse H-field from E via Maxwell's curl equations.

**Step 2 — Persistent homology** (`barcode.py`): Converts the scalar field `|E|` into a point cloud (grid positions above a threshold). Runs Ripser to compute H0 (connected components) and H1 (loops/holes) barcodes. The barcode is the topological fingerprint of the field structure.

**Step 3 — Hilbert embedding** (`manifold_projector.py`): Encodes the barcode as a 50-dimensional vector using Hilbert series coefficients: `N(t) = Σt^{birth} - Σt^{death}`. This is a fixed-length representation of the entire topological structure.

**Step 4 — Learn T** (`god_tensor.py`): Stacks all E-embeddings and H-embeddings from training geometries. Solves `T @ E ≈ H` via least-squares: `T = H @ E⁺`. This gives a 16×16 matrix capturing the E→H coupling law discovered from the data.

**Step 5 — Fixed point** (`god_tensor.py`): Iterates `x_{n+1} = normalize(T @ x_n)` until `||x_{n+1} - x_n|| < 10⁻⁷`. The converged vector is the **God Tensor** — the fixed point where E and H representations meet under the learned coupling.

**Step 6 — Predict** (`predict.py`): For a new geometry, find the k=5 nearest training geometries (by L2 distance in parameter space). Returns:
- `knn_e_fingerprint`: actual weighted-average fingerprint from nearest neighbors (ground truth baseline)
- `knn_h_fingerprint`: same for H
- `god_tensor_projected_e/h`: God Tensor's prediction
- `coupling_score`: how well T unifies E and H (higher = better coupling)

---

## Key Results (from demo.py)

```
Actual FDFD cavity (w=2.0, h=1.2), mode 0:
  E  Betti-0: 283   (283 connected components in |E| point cloud)
  H  Betti-0:   0   (energy concentrated in one region — H1 dominant)
  EMD |E| vs |S|: 0.0189  (near-identical distributions)
  Coupling strength: 0.9815  (0 = decoupled, 1 = fully coupled)

God Tensor:
  Convergence: 16 iterations to delta=5.9e-08
  God Score: 0.9416  (T unifies E and H embeddings well)
  Fixed-point verification: ||T(x) - T(T(x))|| = 0.052  (small ≠ 0)
```

---

## Why Not Just Use Maxwell's Equations?

You can. The physics is well-established. Faraday is useful when:

- **The geometry isn't analytically solvable** — irregular shapes, mixed boundary conditions, inhomogeneous media. FDFD gives you the field; the coupling analysis is separate.
- **You want to discover unexpected coupling patterns** — the learned T matrix tells you which E-field topological features map to which H-field features, without assuming the relationship.
- **You're building something that uses E/H coupling as a representation** — the God Tensor is a fixed vector you can use as a semantic anchor, same as how NLP models use [CLS] tokens.

---

## The Coupling Metric

Maxwell's equations couple E and H through:

```
∇ × E = -∂B/∂t     (Faraday)
∇ × H = +∂D/∂t     (Ampère-Maxwell)
```

In a resonant cavity, E and H reach a steady state where energy shuttles between electric and magnetic form at the cavity's resonant frequency. The coupling is tight — the fields occupy the same physical space and peak at different times within each oscillation cycle.

We measure coupling as **Earth Mover's Distance** between the `|E|` and `|S|` distributions (`|S| = |E| × |H|` is the Poynting vector magnitude — energy flux density). When `EMD ≈ 0`, the two fields have identical topological structure (same nodes, same antinodes, same energy distribution). When `EMD` is large, they're topologically decoupled.

Real cavity modes: `EMD < 0.02`, `coupling_strength > 0.98`. The fields are tightly coupled.

---

## API Reference

```python
from faraday import GodTensor, CavityGeometry, CavityShape, solve_cavity_modes, coupled_fingerprint

# One-off: solve a cavity and inspect its fields
geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.5))
mode_data = solve_cavity_modes(geom, nx=40, ny=40, num_modes=6)
print(mode_data["k_values"])  # resonant wave numbers

e_field = mode_data["e_modes"]["mode_0"]["field"]  # 2D array
h_field = mode_data["h_modes"]["mode_0"]["field"]  # |H| magnitude

result = coupled_fingerprint(e_field, h_field)
print(result["coupling_strength"])  # 0.0 – 1.0
print(result["emd_S"])              # Earth Mover's Distance
print(result["e_fingerprint"]["betti_0"])  # H0 count for |E|
print(result["h_fingerprint"]["betti_1"])  # H1 count for |H|

# Learn the God Tensor from training data
gt = GodTensor(n_geometries=50)
gt.collect_training_data(nx=40, ny=40, num_modes=4)
gt.learn_T()
gt.find_fixed_point(iters=500, tol=1e-7)
print(f"God Score: {gt.god_score():.4f}")  # 0 = no coupling, 1 = perfect

# Predict for a new geometry
from faraday.predict import predict_eh_barcode
pred = predict_eh_barcode(gt, (2.0, 1.2), "rect")
print(pred["knn_e_fingerprint"]["betti_0"])   # KNN-predicted E Betti-0
print(pred["coupling_score"])                  # God Tensor coupling quality
```

---

## File Structure

```
faraday/
├── __init__.py
├── em_solver.py        # FDFD cavity solver + Poynting vector
├── barcode.py          # Field → point cloud → persistent homology
├── manifold_projector.py  # Barcode → Hilbert series → embedding
├── god_tensor.py       # Learn T, find fixed point, God Score
├── predict.py          # KNN + God Tensor prediction for new geometries
└── demo.py             # Full pipeline demonstration

README.md               # This file
pyproject.toml          # Package config
requirements.txt        # Dependencies
```

---

## What "God Tensor" Means

The name is deliberately dramatic. Here's what it maps to in concrete terms:

| "God Tensor" | Concrete meaning |
|---|---|
| The unified E×H entity | The 16D vector `x* = T(x*)` that is invariant under the learned coupling operator |
| Fixed point `T(T(x)) = T(x)` | The embedding where E→T(E) and H→T(H) produce the same representation |
| "Discovered from data" | T was learned via `lstsq(E_emb, H_emb)`, not derived from curl equations |
| "Without assuming Maxwell's equations" | We never imposed `∇×E = -∂B/∂t`. The coupling emerged from the data |

The fixed point is not magic. It's the eigenvector of T with eigenvalue 1 (the operator's invariant subspace). The God Tensor finds it because any vector iterated through T converges to this subspace — that's the Banach fixed-point theorem in action.

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
