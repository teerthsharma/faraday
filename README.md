# ⚡ Faraday

**Computational Faraday Tensor — discover the unified E × H field coupling via topology-fixed-point projection.**

The electromagnetic field is not two things. Electric field **E** and magnetic field **H** are two views of one geometric object — the **Faraday tensor** `F_μν`. This library learns that tensor from data.

```
E field ──┐              ┌── E fingerprint
          ├──► God Tensor ──► H fingerprint
H field ──┘              └── coupled field
```

## The God Tensor

The **God Tensor** is the fixed point of E ⇄ H co-determination:

```
T(E) → H     (electric field encodes magnetic field)
T(H) → E     (magnetic field encodes electric field)
T(T(x)) = T(x)   [fixed point — the invariant]
```

At convergence, `T(x)` is the **Faraday tensor** — the operator that IS the unified field. It was discovered from data, not assumed from Maxwell's equations.

## Architecture

| Module | Role |
|--------|------|
| `em_solver` | FDFD cavity solver. Computes coupled E and H eigenmodes. |
| `barcode` | Field → point cloud → persistent homology barcode. |
| `manifold_projector` | Barcode → Hilbert coefficients → autoencoder embedding. |
| `god_tensor` | T(E) ⇄ T(H) fixed-point iteration → God Tensor. |
| `predict` | Given new geometry → predict E and H topology via God Tensor. |

## Install

```bash
pip install faraday
```

Or from source:
```bash
git clone https://github.com/teerthsharma/faraday.git
cd faraday
pip install -e .
```

## Quick Start

```python
from faraday import GodTensor

# 1. Collect training data: varied cavity geometries with E and H fields
gt = GodTensor(n_geometries=50)
gt.collect_training_data(nx=40, ny=40)

# 2. Learn the coupling operator T
gt.learn_T()

# 3. Find the fixed point — the God Tensor
gt.find_fixed_point(iters=500, tol=1e-7)

# 4. Predict E and H topology for a new geometry (no FDFD needed)
pred = gt.predict(w=2.0, h=1.5)
print(f"Predicted E Betti-0: {pred['e_fingerprint']['betti_0']}")
print(f"God distance: {pred['god_distance_e']:.6f}")
print(f"Coupling score: {pred['coupling_score']:.4f}")

# 5. God Score — how well does T unify E and H?
print(f"God Score: {gt.god_score():.4f}")
```

## The Math

### FDFD Cavity Solver
For a hollow PEC cavity, TM modes satisfy:
```
∇²E_z + k²E = 0
∇²H_z + k²H = 0
```
Both share the same eigenvalue `k` — they are linked by Maxwell's equations.

### Persistent Homology
Convert field `|E(x,y)|` to a point cloud, compute H0/H1 barcodes:
```
Barcode: [(birth, death), ...]  — each bar = one topological feature
```

### Hilbert Series Coefficients
```
N(t) = Σ t^{birth} - Σ t^{death}
```
Polynomial encoding of the entire topological structure.

### God Tensor Fixed Point
```
T @ e_emb ≈ h_emb
x_{n+1} = normalize(T @ x_n)
x_* = lim_{n→∞} x_n
T(x_*) = x_*   ← God Tensor
```

## Physics

E and H generate each other (Faraday + Ampère-Maxwell):
```
∇ × E = -∂B/∂t
∇ × H = +∂D/∂t
```

The God Tensor learns the **coupling invariant** from data. It discovers what Maxwell's equations already encode — but it discovers it from E × H co-determination, not from assuming the equations.

## Why "Faraday"?

Michael Faraday introduced the field concept — E and H as space-filling entities that generate each other. The **Faraday tensor** `F_μν` (in relativity) unifies them geometrically. This library *computes* that tensor from data.

## Citation

If this helps your research:
```bibtex
@software{faraday,
  author = {Teerth Sharma},
  title = {Computational Faraday Tensor},
  url = {https://github.com/teerthsharma/faraday}
}
```
