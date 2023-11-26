Hilbert Series Embedding
=======================

The Hilbert series maps a variable-length barcode to a fixed-length
vector in :math:`\mathbb{R}^N` suitable for machine learning. This is the bridge
from algebraic topology to linear algebra.

Hilbert Series Definition
------------------------

Given a barcode :math:`B = \{(b_i, d_i)\}`, the **Hilbert series** is:

.. math::

   H_B(t) = \sum_i \left(t^{b_i} - t^{d_i}\right)

- :math:`b_i` = birth scale of feature :math:`i`
- :math:`d_i` = death scale of feature :math:`i`
- Features with :math:`d_i = \infty` (never die) contribute only :math:`+t^{b_i}`

The generating function captures both the count and the persistence of all
topological features in a single analytic form.

Evaluating at Chebyshev Nodes
----------------------------

Evaluating :math:`H_B(t)` at :math:`N` Chebyshev nodes :math:`t_1, \ldots, t_N`
on :math:`[0,1]` produces an :math:`N`-dimensional vector:

.. math::

   \mathbf{h}_B = \left[ H_B(t_1),\; H_B(t_2),\; \ldots,\; H_B(t_N) \right] \in \mathbb{R}^N

Faraday uses :math:`N = 50` Chebyshev nodes, yielding a 50-dimensional
**Hilbert embedding** for each field (E and H separately).

Chebyshev nodes of the second kind on :math:`[0,1]`:

.. math::

   t_j = \frac{1}{2}\left(1 - \cos\frac{(2j-1)\pi}{2N}\right), \qquad j=1,\ldots,N

Autoencoder Projection
----------------------

The raw 50D Hilbert vector may contain redundant or correlated features.
Faraday optionally projects it through a shallow autoencoder (manifold_projector.py)
to learn a compact :math:`d`-dimensional representation (:math:`d = 16` by default):

.. math::

   \mathbf{z}_E &= \text{encoder}(\mathbf{h}_E) \in \mathbb{R}^{16}

   \mathbf{z}_H &= \text{encoder}(\mathbf{h}_H) \in \mathbb{R}^{16}

The autoencoder is trained on the training set's Hilbert embeddings and frozen
during God Tensor learning.

Energy Distribution Invariance
-----------------------------

The Hilbert series representation is **amplitude-invariant** — it captures only
the birth-death structure of topological features, not the absolute field strength.
This is desirable for Faraday because:

1. Different geometries have different absolute field scales
2. Only the *relative* distribution of energy across the cavity matters
3. The coupling between E and H should be scale-independent

The EMD (Earth Mover's Distance) between |E| and |H| distributions (computed
before PH) provides an amplitude-sensitive coupling measurement, complementing
the topology-only Hilbert embedding.

References
----------

- A. Zomorodian. *Topology for Computing*. Cambridge University Press, 2005.
- G. Carlsson. "Topology and Data." *Bulletin of the American Mathematical
  Society*, 46:255–308, 2009.
- V. M. Mancullo. "Admissible Chebyshev Nodes." *Journal of Approximation
  Theory*, 163:1–30, 2011.
