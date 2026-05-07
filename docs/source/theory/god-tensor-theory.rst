God Tensor: The Unified E × H Invariant
=======================================

The **God Tensor** is the core theoretical construct of Faraday: the fixed point
of the E/H coupling operator, representing the invariant where electric and
magnetic field topologies co-determine each other.

Coupling Operator T
-------------------

Given a set of :math:`M` training geometries, each with E/H Hilbert embeddings:

.. math::

   \mathbf{Z}_E = \begin{bmatrix} \mathbf{z}_E^{(1)} & \cdots & \mathbf{z}_E^{(M)} \end{bmatrix}^\top \in \mathbb{R}^{M \times d}

   \mathbf{Z}_H = \begin{bmatrix} \mathbf{z}_H^{(1)} & \cdots & \mathbf{z}_H^{(M)} \end{bmatrix}^\top \in \mathbb{R}^{M \times d}

the **coupling operator** :math:`T` is the :math:`d \times d` matrix that maps
E-field embeddings to H-field embeddings:

.. math::

   T \; \mathbf{z}_E^{(i)} \approx \mathbf{z}_H^{(i)} \quad \forall i

Solved via least-squares (normal equations):

.. math::

   T = \mathbf{Z}_H \, \mathbf{Z}_E^+ = \mathbf{Z}_H \, (\mathbf{Z}_E^\top \mathbf{Z}_E)^{-1} \mathbf{Z}_E^\top

The pseudoinverse :math:`\mathbf{Z}_E^+` exists as long as :math:`\text{rank}(\mathbf{Z}_E) = d`
(full column rank), requiring :math:`M \geq d` training geometries.

Learned from Data
~~~~~~~~~~~~~~~~~

The operator :math:`T` is learned — not derived from Maxwell's curl equations.
It captures whatever coupling pattern exists in the training data. If the
fields are tightly coupled (low EMD), T will learn a near-isometry. If decoupled,
T will be far from orthogonal.

T-Matrix Properties
-------------------

For a full-rank coupling operator in :math:`\mathbb{R}^d`:

- :math:`T` has :math:`d` singular values
- The singular value distribution reveals the **degrees of freedom** in E/H coupling
- If coupling is tight, singular values cluster near 1
- If coupling is weak, singular values scatter widely

Fixed Point Condition
---------------------

The **God Tensor** :math:`\mathbf{g}` is defined as the fixed point of :math:`T`:

.. math::

   T \, \mathbf{g} = \mathbf{g}

This is the eigenvector of :math:`T` with eigenvalue :math:`\lambda = 1`,
or equivalently the invariant subspace of the coupling operator.

Interpretation
~~~~~~~~~~~~~

- :math:`\mathbf{g}` is the embedding where E and H are **topologically identical**
- Any vector projected through :math:`T` converges toward :math:`\mathbf{g}`
- :math:`\mathbf{g}` is the **semantic anchor** of E/H coupling, analogous to
  the [CLS] token in transformer language models

Spectral Fixed-Point Iteration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Faraday finds :math:`\mathbf{g}` via iterative refinement:

.. math::

   \mathbf{x}_{n+1} = \frac{T \, \mathbf{x}_n}{\|T \, \mathbf{x}_n\|} = \text{normalize}(T \, \mathbf{x}_n)

This is the **power iteration** method for finding the dominant eigenvector.
Normalization prevents magnitude blow-up. The iteration converges to the
eigenspace of the largest singular value of :math:`T`. See :doc:`perron-frobenius-theorem`
for convergence theory.

God Score
---------

After converging to :math:`\mathbf{g}`, Faraday measures **how well T unifies E and H**
with the **God Score**:

.. math::

   S_{\text{god}} = \exp\!\left(-\frac{1}{M}\sum_{i=1}^{M} \|\mathbf{z}_E^{(i)} - T^\top\mathbf{z}_H^{(i)}\|\right)

- :math:`S_{\text{god}} \approx 1` — T is a near-isometry; E and H are tightly coupled
- :math:`S_{\text{god}} \approx 0` — T is far from isometric; coupling is weak

The exponential provides numerical stability for large inter-embedding distances.

Prediction via KNN + God Tensor
-------------------------------

For a new geometry, Faraday:

1. Finds the :math:`k=5` nearest training geometries (L2 in parameter space)
2. Interpolates their E/H embeddings via weighted average
3. Projects through :math:`T` and compares to the God Tensor
4. Returns a **coupling score** based on distance to :math:`\mathbf{g}`

This is geometrically analogous to a nearest-centroid classifier in the
embedding manifold.

References
----------

- Perron-Frobenius theorem: C. MacCluer, *The many proofs and applications of Perron's theorem*, 2000
- Power iteration: G. H. Golub, C. F. Van Loan. *Matrix Computations*, 4th ed., 2013
- Coupling operators in physics: R. Jackiw, "Comments on 'The Unreasonable
  Effectiveness of Symmetry'." *International Journal of Modern Physics B*, 2003.
