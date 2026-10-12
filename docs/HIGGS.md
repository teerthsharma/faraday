# Higgs Coupling: the Condensate as a Field Generator

> A Higgs condensate gives the photon a mass. Wherever the condensate is
> present, light below the mass gap is expelled; wherever it vanishes, light
> is trapped. So the geometry of the scalar field *generates* the cavities of
> the gauge field, and its topology (vortex winding) sets how many light modes
> are trapped.

Reproduce:

```bash
python experiments/higgs_study.py    # ~2.5 min
pytest tests/test_higgs.py           # 12 tests, ~10 s
```

## 1. Model

In the broken phase of an abelian Higgs model, the photon mass is
m = √2 q|ψ|. For the TM component E = E_z at frequency ω:

```
(−Δ + m²(x)) E = ω² E,      m²(x) = m₀² |ψ(x)|² / v²,      H = (∂_y E, −∂_x E)/ω
```

`solve_interior_modes(..., mass2=m²)` solves this on any grid. The model has
four limits:

- **Scalar TM.** The gradient couplings ∝ ∇m² that a varying mass adds to the
  full Proca equations are neglected.
- **Frozen condensate.** ψ is prescribed: either a uniform condensate outside
  a normal region D, or unit vortices |ψ|² = Π tanh²(r_i/ξ). It is not a
  self-consistent Ginzburg–Landau solution, and the photon does not act back
  on it.
- **Single frequency.** All results are for time-harmonic modes.
- **Box.** The computational box has Dirichlet edges far from the region of
  interest.

## 2. Results

### 2.1 A perfect conductor is a condensate with infinite mass
A condensate around a normal region D is a soft wall with penetration depth
1/m (the Meissner effect). As m → ∞ the trapped frequencies converge to the
PEC cavity values, at rate O(1/m):

| cavity | PEC k₁ | ω₁ at m = 2, 4, 8, 16 |
|---|---|---|
| disk | 2.391 | 1.562, 1.910, 2.138, 2.266 |
| stadium | 2.939 | 1.685, 2.168, 2.517, 2.728 |

So the hard wall E = 0 used throughout `docs/DUALITY.md` is the infinite-mass
limit of a Higgs condensate.

### 2.2 The mass gap protects the bulk–boundary law
The law Q(D) = χ(D) + N_wall/2 was derived for E = 0 on the wall. With a soft
wall, a trapped mode (ω < m) decays into the condensate, so effectively

```
∂ₙE ≈ −κE,     κ ≈ √(m² − ω²) > 0          (a Robin wall)
```

Along the wall, the vector (∂ₙE, ∂ₜE) = (−κE, E′) turns half a revolution,
always in the same sense, at every zero of E. That is exactly the count
for E = 0, so the law carries over unchanged. Above the gap the field
propagates into the condensate and κ no longer has a definite sign.

| modes (disk, rectangle, stadium, annulus; m = 3, 5, 8) | law holds |
|---|---|
| trapped (ω < m) | **91 / 91** |
| untrapped (ω > m) | 131 / 389 = 34 % |

The mass gap is the boundary of topological protection.

### 2.3 A hole filled with condensate carries index 1 − ℓ
Inside a filled hole the field grows from the centre like I_ℓ(κr) cos ℓθ.
The critical point at the hole's centre has index 1 − ℓ. Enlarging the
counting region to swallow the hole therefore changes Q by exactly 1 − ℓ:

| condition | holds |
|---|---|
| κ·r_hole > 0.8 | **22 / 22** |
| κ·r_hole ≤ 0.8 | 2 / 3 |

The limit is physical. On the hole's wall the effective Robin coefficient is
κ I′_ℓ/I_ℓ, which is only about κ²r_h/2 for ℓ = 0. Close to the gap edge a
small hole's wall is therefore nearly **Neumann**. A ring of critical points
then sits on it, and the protection is lost.

### 2.4 Higgs vortices trap light
In a vortex core ψ = 0, so the photon is massless there and each core is a
potential well. In units of the healing length ξ, the problem depends on one
number, g = m₀ξ.

| g | trapped photon modes for L = 0 … 6 vortices |
|---|---|
| 2.0 | **0, 1, 2, 3, 4, 5, 6** (= L) |
| 3.5 | **0, 4, 8, 12, 16, 20, 24** (= 4L) |

A single vortex fills atom-like shells as g grows:

| g | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 5.0 | 6.0 |
|---|---|---|---|---|---|---|---|---|---|
| trapped modes | 1 | 1 | 1 | 3 | 3 | 4 | 6 | 8 | 13 |

- The shells fill in the order 1s, 1p, 2s, 1d, 2p, 1f, 3s, 2d. Shells with
  ℓ ≠ 0 come as degenerate pairs (±ℓ).
- The 1p shell appears at **g_p = 2.299**, located by bisection (±10⁻⁴).
- Below g_p every vortex binds exactly one photon mode, so **the number of
  trapped modes equals the condensate's winding number**.

**Reading g as a superconductor constant.** This step depends on the ansatz.
- A Ginzburg–Landau vortex core is roughly tanh(r/√2ξ_GL), and the photon
  mass is m = 1/λ, so g ≈ √2 ξ_GL/λ = √2/κ.
- The threshold then becomes κ_p ≈ 0.615, which is below the type-II
  boundary 1/√2.
- So within this model, each vortex of any type-II condensate traps exactly
  one photon mode, one per flux quantum.

## 3. How this extends the causal chain

```
Higgs condensate ψ(x) ──► photon mass m²(x) = m₀²|ψ|² ──► where light can live
      │                                                       │
      ├─ normal region D  ──► soft (Robin) wall ──► cavity modes, with Q = χ + N/2 protected while ω < m
      ├─ filled hole      ──► critical point of index 1 − ℓ at its centre
      └─ vortices (winding L) ──► massless cores ──► trapped light, n_shell(g) per vortex
```

## 4. Related physics and prior art

| This study | Known counterpart |
|---|---|
| The photon gains a mass and is expelled from the condensate | The Anderson–Higgs mechanism and the Meissner effect (textbook) |
| A PEC wall as the infinite-mass limit | London / Meissner limit, penetration depth λ → 0 |
| States bound in vortex cores | Fermions: Caroli, de Gennes & Matricon (1964). Bosons on string cores: Witten (1985), superconducting strings |

What looks new here, pending a literature check:
- the mass gap as the exact boundary of the bulk–boundary law (91/91 against
  34%);
- the 1 − ℓ index of condensate-filled holes;
- the shell staircase n(g), with "one trapped photon mode per vortex" for g < 2.3.

## 5. Next steps

- Solve the Ginzburg–Landau equations self-consistently (ψ and A together,
  with applied field), instead of using the frozen tanh ansatz.
- Use the full Proca equations with the ∇m² couplings, or 3-D vortex lines,
  where trapped modes become guided waves along the line.
- Vortex lattices: tunnelling between cores should turn the trapped modes
  into a photonic band, with band topology set by the lattice of windings.
