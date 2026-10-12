# Topological E ⇄ H Duality of TM Cavity Modes

> The wall decides the bulk: the net vortex charge of the magnetic field inside
> a PEC cavity is fixed by where the electric field's nodal lines touch the wall.

Reproduce every number below:

```bash
python experiments/duality_study.py          # ~25 s
python experiments/duality_study.py --quick  # coarser grids, ~10 s
faraday topology -w 2 -h 1 -n 8              # per-mode table for one cavity
```

## 1. Setup

TM mode of a planar PEC cavity D: `-ΔE = k²E` in D, `E = 0` on ∂D, with
`E = E_z`. In natural units (c = μ = ε = 1) Faraday's law gives the in-plane
magnetic field

```
H = (∂_y E, -∂_x E) / k        (∇E rotated by 90°, tangent to the wall)
```

and Ampère–Maxwell closes the loop: `curl_z H = k E`. The mode is the fixed
point of E → H → E (verified pointwise to 0.02–1.2 %).

Zeros of H are critical points of E. Each carries an integer winding number,
its **vortex charge**: +1 at extrema of E, −1 at saddles, 1 − r where 2r
nodal lines cross.

## 2. The laws

| Law | Statement | Origin |
|---|---|---|
| Domain law | net H charge inside each nodal domain Ω of E = χ(Ω) | Poincaré–Hopf; ∇E is normal and one-signed on ∂Ω (Hopf lemma) |
| Bulk–boundary law | **Q = χ(D) + N_wall / 2** | Poincaré–Hopf with boundary: each wall nodal point is a boundary zero of index −½ |
| Chirality | circulation of H around Ω = k ∫_Ω E dA, so the vortex sense is the sign of E | Stokes + Helmholtz |

Here χ(D) = 1 − (number of holes) and N_wall counts the points where nodal
lines of E meet any wall, the walls of holes included.

## 3. Evidence

| Study | Cavities | Result |
|---|---|---|
| Both laws, simply connected | rectangle, ellipse, stadium, star, trapezoid, D-shape (12 modes each) | **72 / 72** |
| Both laws, multiply connected (falsification of χ(D)) | concentric and eccentric annulus, rectangle with a hole, square with a square hole (χ = 0), disk with two holes (χ = −1) | **60 / 60** |
| Two-hole fundamental mode | one nodal domain, χ(D) = −1 | Q = **−1**: the antivortex the law requires |
| Geometry-only prediction (rectangles, no field solve) | 20 aspect ratios × 10 modes | **192 / 200 = 96 %**; every miss within 0.45 % of a level crossing |
| Least squares, E-side invariants → H charge | all 132 modes | coefficients χ(D) **1.0000**, N_wall **0.5000**, intercept 0, n_domains 0, residual **0** |

Notes on measurement:

- Vortex charges use edge-shared angle differences (exact discrete Stokes).
  Per-plaquette wrapping miscounts a zero lying on a grid line.
- A nodal line that grazes a small hole within one grid cell is under-resolved
  (two-hole disk, mode 2, needs dx ≤ 0.008). This is a resolution limit of the
  wall count, not a failure of the law.

## 4. Causality: how the topology changes

Geometry → spectrum ordering → wall nodal points → bulk H charge.

- **Separable or symmetric geometry.** Shape space is cut into cells by the
  surfaces where eigenvalues cross. Inside a cell the topology of mode j is
  constant and predictable from geometry alone. At a crossing, modes swap and
  their charges are exchanged, so the pair's total is conserved.
- **Generic (symmetry-broken) geometry.** Crossings are avoided and no swap is
  possible. Charge still changes, by a **nodal line detaching from or
  attaching to the wall** (N_wall changes by 2, Q by 1). The two modes change
  at different shape parameters, so in between both have Q = 2 and the pair's
  total is not conserved:

| TM31 ↔ TM12, w/h ≈ 1.63 | gap | Q(mode 2), Q(mode 3) across the sweep |
|---|---|---|
| rectangle (δ = 0) | closes (true crossing) | (2, 3) → (3, 2) in one step |
| tilted wall (δ = 0.15) | ≥ 1.7 % (avoided) | (2, 3) → **(2, 2)** → (3, 2) |

## 5. What it means for Faraday

- The legacy barcode embedding (`embed_fingerprint`) is about 99.9 % pixel and
  bar counts. E and H embeddings are identical (cosine = 1), so the learned T
  is close to the identity on the data. It encodes no topology.
- `ModeTopology.signature()` is an embedding that does: integer,
  scale-invariant, with exact constraints between its E side and its H side.
- `GodTensor` is now built on this. Its E-view is χ(D) + N_wall/2 per mode
  and its H-view is Q per mode, so the learned T is an exact orthogonal
  projector (T² = T) whenever the law holds. See the README and
  `docs/QUANTUM.md`.
- Only the wall data N_wall (one-dimensional) has to be learned for a new
  geometry. The bulk H topology then follows from the law, and every
  prediction carries a built-in consistency check.

## 6. Prior art (literature check: abstracts and snippets only)

- The domain law and the chirality relation are classical: Poincaré–Hopf with
  boundary (Morse 1929; Pugh, *Topology* 7, 1968), plus Stokes.
- Closest analogues of the bulk–boundary law, for other boundary conditions:
  - Judge & Mondal, *Comm. PDE* 47 (2022), arXiv 2108.10386: index formula for
    Neumann eigenfunctions on polygons.
  - Battaglia, Pistoia & Provenzano, arXiv 2402.01190: Steklov eigenfunctions,
    interior critical points counted from χ and boundary sign changes.
- We found no statement of the law for Dirichlet / TM cavities, of the
  wall-detachment mechanism at avoided crossings, or of these invariants used
  as ML targets. Read Judge–Mondal §4 and Bérard–Helffer (arXiv 2202.06587)
  §2.2 in full before claiming priority.

## 7. Limits

- 2-D TM modes, finite-difference staircase walls. TE modes (Neumann) and 3-D
  cavities are untested.
- Degenerate modes are basis-dependent. Compare net charge, not raw vortex
  counts.
- `wall_nodal_count` orders each wall loop by angle about its centroid. That
  is exact for star-shaped walls; strongly non-convex walls need a contour
  walk.
