# The Quantum Sector: How the Causal Chain Splits into Quantum Topology

> One cause (the cavity's shape) produces several consequences, and they
> separate into independent topological invariants. Each is an integer or a
> sign, and each is measured exactly.

Reproduce:

```bash
python experiments/quantum_study.py     # ~90 s
pytest tests/test_quantum.py            # 18 tests, ~18 s
```

## 1. Why "quantum"

The TM problem `−ΔE = k²E` with `E = 0` on the wall is, symbol for symbol,
the Schrödinger equation of a particle in a 2-D box (a quantum billiard):
E ↔ ψ, k² ↔ 2mE/ħ². So:

- A real cavity mode is a **time-reversal-symmetric** quantum eigenstate.
  Everything in `docs/DUALITY.md` applies to it unchanged; its vortices are
  the critical points of ψ.
- A **complex** state ψ = u + iv breaks time reversal: it carries current,
  for example angular momentum from a degenerate pair. Its zeros are phase
  vortices with an integer winding V.

## 2. How the consequences separate

```
                     cavity shape D
                          │
             ┌────────────┴─────────────┐
       spectrum of D               family of shapes
             │                           │
   real modes (time-reversal   complex states (time       loops in shape
   symmetric)                  reversal broken)           space
             │                           │                      │
   H vortices: Q = χ + N/2     phase vortices V           Z₂ geometric phase
   (DUALITY.md)                 |V| ≤ min N/2              −1 iff the loop
                               V_bulk = w_out − Σw_holes  encloses a crossing
```

| Layer | Invariant | Lives in | Law | Result |
|---|---|---|---|---|
| Classical, real | vortex charge Q of H | real space, Z | Q = χ(D) + N_wall/2 | 132/132 (DUALITY.md) |
| Quantum, complex | phase charge V of ψ = u + iv | real space, Z | \|V\| ≤ min(N_wall(u), N_wall(v))/2 | **66/66**, saturated 16 times |
| Quantum, holes | windings w on each wall | wall loops, Z per hole | V_bulk = w_outer − Σ w_holes | **66/66** |
| Shape space | sign of a mode transported around a loop | parameter space, Z₂ | −1 ⇔ the loop encloses a diabolical point | **6/6** loops |

### 2.1 Phase vortices are bounded by classical wall data
By the argument principle, V equals the winding of ψ along the wall. Next to
the wall ψ ≈ d·(∂ₙu + i∂ₙv), and a planar curve cannot wind more than half
as often as either coordinate changes sign. So

```
|V|  ≤  min(N_wall(u), N_wall(v)) / 2  =  min(Q_u, Q_v) − χ(D)
```

The quantum vortex charge is capped by the classical H charge of its real
parts. Angular-momentum states of the disk saturate the bound: ℓ = N_wall/2,
with V = ±1, ±2, ±3, ±4 for successive degenerate pairs. V is gauge-invariant
(unchanged by ψ → e^{iα}ψ), and the test checks this.

### 2.2 In cavities with holes, the charge leaves the bulk
```
V_bulk = w_outer − Σ_holes w_hole
```
- In the concentric annulus, every angular-momentum pair has V_bulk = 0,
  while both walls carry winding ±ℓ.
- The quantum charge has moved out of the bulk entirely and sits as quantised
  circulation around the hole, an Aharonov–Bohm-type holonomy.
- Compare the classical sector, where the hole only shifts the bulk charge
  through χ(D) = 1 − holes.

### 2.3 The geometric phase marks where the classical charge jumps
In the rectangle-with-tilt family (width w, tilt δ), modes TM31 and TM12
cross at exactly one point, (w*, δ) = (√(8/3), 0). That point is a
diabolical point.

| loop in (w, δ) | mode 2 | mode 3 |
|---|---|---|
| encloses (w*, 0) | **−1** | **−1** |
| misses it (shifted in w) | +1 | +1 |
| misses it (shifted in δ) | +1 | +1 |

The smallest overlap between consecutive steps is 0.990, so the transport is
resolved. The diabolical point is exactly where the classical charge Q of
modes 2 and 3 swaps (DUALITY.md §4). Two different topologies meet there: a
Z₂ monodromy in shape space and an integer jump in real space.

## 3. The God Tensor in this picture

Real modes satisfy E-view = H-view (χ + N/2 = Q). The learned T is therefore
the projector onto realisable signatures, and the God Tensor is its fixed
subspace:

| 20 geometries × 16 modes | law violations | ‖T² − T‖/‖T‖ | dim fixed subspace | god_score |
|---|---|---|---|---|
| grid 30 | 0 / 320 | 3.4 × 10⁻¹⁵ | 11 (= rank) | 1.0000 |
| grid 15 | 24 / 320 | 0.63 | 7 (< rank 12) | 0.89 |

## 4. What is not exact

- **Chirality from H alone.** Σ_vortices index × sign(curl H) = Σ_Ω sign(E)·χ(Ω)
  holds only on **118/132** modes. At nodal crossings E = 0, so the sense of
  rotation there cannot be read from H. This quantity is not used in the God
  Tensor.
- **Complex superpositions of non-degenerate modes** are not stationary. Their
  V is a snapshot in time, though the bound still holds for every snapshot.

## 5. Prior art

- Geometric phase around diabolical points in microwave billiards:
  Lauber, Weidenhammer & Dubbers, *PRL* 72, 1004 (1994).
- Phase singularities and nodal statistics in billiards: Berry & Dennis;
  Stöckmann and coworkers.
- The argument principle and holonomy are classical complex analysis.
- What appears new here, pending a full-text literature check, is tying them
  to the classical wall count:
  - the bound |V| ≤ min N_wall/2 = min Q − χ;
  - the God Tensor as the projector certifying the E ⇄ H law.
