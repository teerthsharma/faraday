Persistent Homology
==================

Persistent homology (PH) is a method from algebraic topology that captures the
multi-scale topological structure of a space. Faraday uses PH to convert
2D electromagnetic field maps into fixed-length topological fingerprints.

Point Cloud Construction
------------------------

Given an E-field magnitude map :math:`|E(x,y)|`, Faraday constructs a point cloud
by thresholding:

.. math::

   P_E = \{(x_i, y_i) \mid |E(x_i,y_i)| > \tau_E\}

where :math:`\tau_E` is a persistence threshold (typically the field mean).
The same process produces :math:`P_H` from the H-field.

The resulting point cloud encodes where field energy is concentrated — a purely
topological representation that discards amplitude information.

Simplicial Complexes
--------------------

From the point cloud, PH builds a **filtration** of simplicial complexes:

.. math::

   K^0 \subseteq K^1 \subseteq K^2 \subseteq \cdots \subseteq K^n

At scale :math:`\epsilon`:

- **0-simplices** (vertices): individual points
- **1-simplices** (edges): connect points with distance :math:`\leq \epsilon`
- **2-simplices** (triangles): fill triangles formed by edges

The **Rips complex** :math:`R_\epsilon` is the maximal complex containing all
edges of length :math:`\leq \epsilon`.

Birth and Death of Topological Features
---------------------------------------

As :math:`\epsilon` grows from 0 to :math:`\infty`:

- A **connected component** (Betti-0) **born** when a vertex appears
- A **loop/hole** (Betti-1) **born** when an edge cycle forms
- A component or loop **dies** when it merges into or fills a larger structure

This creates a **barcode** — a multiset of birth-death intervals:

.. math::

   B = \{(b_i, d_i) \mid b_i < d_i\}

**Persistence** of a feature:

.. math::

   \text{pers}(b,d) = d - b

High-persistence features are robust topological signals; low-persistence
features are often noise.

Betti Numbers
-------------

The **Betti numbers** count distinct topological features:

- :math:`\beta_0` — number of connected components (H₀)
- :math:`\beta_1` — number of 1-dimensional holes/loops (H₁)
- :math:`\beta_2` — number of 2-dimensional voids (H₂)

For a rectangular PEC cavity, the fundamental TE₁₀ mode has:

- :math:`\beta_0 = 1` — one connected high-energy region in E
- :math:`\beta_1 = 0` — no holes in the energy distribution

This "1, 0" pair is the topological fingerprint of the mode.

Ripser Computation
------------------

Faraday uses the **Ripser** algorithm (Bauer, Kerber, Reininghaus, 2017) to
compute persistent homology. Ripser uses **clearage** — the algorithm repeatedly
clears columns to keep only the essential persistence pairs, achieving
significant speedups over naive matrix reduction.

The ``ripser`` Python package wraps Ripser's C++ implementation:

.. code-block:: python

   from ripser import ripser
   from persim import plot_diagrams

   result = ripser(point_cloud, maxdim=1)
   diagrams = result["dgms"]       # list of persistence diagrams
   beta_0 = len([b for b, d in diagrams[0] if d < np.inf])

Hilbert Series Representation
-----------------------------

A barcode is a variable-length multiset — unsuitable as a machine learning input.
Faraday converts barcodes to a fixed-length vector via the **Hilbert series**:

.. math::

   H_B(t) = \sum_i (t^{b_i} - t^{d_i})

Evaluating :math:`H_B(t)` at :math:`N` Chebyshev nodes produces an
:math:`N`-dimensional embedding vector. See :doc:`hilbert-embedding` for details.

References
----------

- H. Edelsbrunner, D. Letscher, A. Zomorodian. "Topological Persistence and
  Simplification." *Discrete & Computational Geometry*, 28:511–533, 2002.
- U. Bauer. "Ripser: Efficient Computation of Vietoris–Rips Persistence Barcodes."
  *Journal of Applied and Computational Topology*, 2021.
- A. Zomorodian, G. Carlsson. "Computing Persistent Homology." *Discrete &
  Computational Geometry*, 33:249–274, 2005.
