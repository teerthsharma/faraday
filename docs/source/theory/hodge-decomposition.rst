Hodge Decomposition and Field Topology
=====================================

Hodge decomposition is a central result in algebraic topology that uniquely
decomposes a vector field (or, by analogy, an embedding vector) into three
orthogonal components: **gradient**, **curl**, and **harmonic**. This provides
a theoretical bridge between the topological features captured by persistent
homology and the geometric structure of the field manifold.

The Classical Hodge Theorem
--------------------------

For a smooth vector field :math:`\mathbf{v}` on a compact Riemannian manifold
:math:`M` with boundary:

.. math::

   \mathbf{v} = \underbrace{d\alpha}_{\text{gradient}} \;+\; \underbrace{\delta\beta}_{\text{curl}} \;+\; \underbrace{h}_{\text{harmonic}}

where:

- :math:`d` — exterior derivative (gradient operator)
- :math:`\delta` — codifferential (divergence of curl)
- :math:`h` — harmonic component (kernel of both :math:`d` and :math:`\delta`)

The three components are :math:`L^2`-orthogonal.

Harmonic Decomposition of Manifolds
-----------------------------------

The **Hodge Laplacian** is:

.. math::

   \Delta = d\delta + \delta d

A differential form :math:`\omega` is **harmonic** if :math:`\Delta\omega = 0`.
The space of harmonic :math:`k`-forms is isomorphic to the :math:`k`-th
**Betti number** :math:`\beta_k`:

.. math::

   \dim \mathcal{H}^k(M) = \beta_k

This connects Hodge theory directly to persistent homology — :math:`\beta_0`
counts connected components, :math:`\beta_1` counts holes.

Application to Faraday
---------------------

While Faraday does not explicitly perform Hodge decomposition, the framework
uses analogous ideas:

1. **Betti numbers** from persistent homology give :math:`(\beta_0, \beta_1)`
   — the harmonic component of the topological structure

2. **Hilbert embedding** maps barcode structure into :math:`\mathbb{R}^d`,
   creating a manifold in the embedding space

3. The **God Tensor** T-matrix is analogous to the Hodge Laplacian:
   it encodes the coupling between E and H as an operator on the
   embedding manifold

4. The **fixed point** of T is the harmonic component — the invariant
   subspace orthogonal to both the E→H and H→E coupling directions

Hodge Decomposition of the Coupling Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By analogy, the T-matrix can be "Hodge-decomposed" relative to the
training data manifold:

- **Gradient part** (rank-1): maps overall energy scale changes
- **Curl part** (rank-:math:`d-1`): maps the actual E/H topological coupling
- **Harmonic part** (rank 0 or 1): the God Tensor fixed point

This is not a literal computation — the T-matrix is learned, not derived.
But the Hodge-theoretic perspective explains why the God Tensor fixed point
is meaningful: it is the component of the coupling that is orthogonal to
both gradient-like and curl-like deformations of the field.

Physical Interpretation
~~~~~~~~~~~~~~~~~~~~~~

In an electromagnetic cavity:

- **Gradient component** — uniform energy scaling (overall amplitude changes)
- **Curl component** — topological mode shape changes (loop creation/annihilation)
- **Harmonic component** — the invariant E/H coupling pattern (God Tensor)

The harmonic component (God Tensor) survives both gradient and curl deformations,
just as harmonic forms are invariant under exact forms.

References
----------

- G. H.Whitney. *Geometric Integration Theory*. Princeton University Press, 1957.
- J. R. Munkres. *Elements of Algebraic Topology*. Addison-Wesley, 1984.
- D. Bachman. *A Geometric Approach to Differential Forms*, 2nd ed., Birkhäuser, 2012.
- M. Nakahara. *Geometry, Topology and Physics*, 2nd ed., IOP Publishing, 2003.
  Chapter 6: Homology and Chapter 7: de Rham Cohomology.
