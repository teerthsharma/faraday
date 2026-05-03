Theory
======

Overview
--------

Faraday implements a novel framework for discovering and predicting electromagnetic
field topology through the concept of the **God Tensor** — a fixed-point operator
that unifies electric (E) and magnetic (H) field signatures.

Core Concepts
--------------

God Tensor
~~~~~~~~~~

The God Tensor :math:`\mathcal{G}` is defined as the fixed point of the coupled
field transformation:

.. math::

   \mathcal{G} = \lim_{n \to \infty} T^n(E, H)

where :math:`T` is the topology operator that maps field signatures to barcodes
and back. At the fixed point:

.. math::

   \mathcal{G}(E_{sig}) = \mathcal{G}(H_{sig})

This means the electric and magnetic topological fingerprints become
indistinguishable — they co-determine each other.

Persistent Homology
~~~~~~~~~~~~~~~~~~~

Faraday uses persistent homology to capture the topological features of field
configurations. The **barcode** :math:`B` is a multiset of birth-death intervals
representing:

* **Birth** — Scale at which a topological feature appears
* **Death** — Scale at which the feature disappears
* **Persistence** — death - birth (importance measure)

Hilbert-Schmidt Theory
~~~~~~~~~~~~~~~~~~~~~~~

The manifold projector maps barcodes to Hilbert space via Hilbert-Schmidt
inner products:

.. math::

   \langle B_1, B_2 \rangle_{HS} = \int_0^\infty b_1(t) \cdot b_2(t) \, dt

where :math:`b_i(t)` are the barcode representation functions.

FDFD Method
~~~~~~~~~~~

The Finite-Difference Frequency-Domain (FDFD) method solves Maxwell's equations
on a Yee lattice:

.. math::

   \nabla \times \mathbf{E} = i\omega\mu\mathbf{H}
   \nabla \times \mathbf{H} = -i\omega\epsilon\mathbf{E}

Architecture
------------

The pipeline architecture:

.. code-block:: text

   Field Data → Point Cloud → Barcode → Hilbert Embedding → God Tensor
        ↑                                                        ↓
        └──────────────── Fixed Point ←─────────────────────────┘

1. **em_solver** — FDFD cavity solver for E/H eigenmodes
2. **barcode** — Field → point cloud → persistent homology
3. **manifold_projector** — Barcode → Hilbert coefficients
4. **god_tensor** — Fixed-point iteration operator
5. **predict** — Geometry → predicted topology

References
----------

* Persistent Homology: Edelsbrunner et al.
* God Tensor: (internal Faraday reference)
* FDFD Methods: Taflove & Hagness
