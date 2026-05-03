Theory
======

This section contains the mathematical and physical foundations of Faraday.
Each document is self-contained and can be read independently.

.. toctree::
   :maxdepth: 1
   :caption: Foundational Theory

   theory/maxwell-equations
   theory/persistent-homology
   theory/hilbert-embedding
   theory/god-tensor-theory
   theory/banach-fixed-point
   theory/hodge-decomposition

Overview
--------

Faraday synthesizes six theoretical pillars:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Pillar
     - Role in Faraday
     - Key Reference
   * - Maxwell's Equations
     - FDFD eigenmode solver: :math:`\nabla^2\mathbf{E} + k^2\mathbf{E} = 0`
     - Taflove & Hagness, *Computational Electrodynamics*
   * - Persistent Homology
     - Field → barcode: topological fingerprint via Ripser
     - Edelsbrunner et al. (2002)
   * - Hilbert Series
     - Barcode → 50D vector: fixed-length embedding for ML
     - Zomorodian & Carlsson (2005)
   * - God Tensor
     - T-matrix: :math:`T = Z_H Z_E^+`, coupling operator
     - Internal
   * - Banach Fixed-Point
     - :math:`x_{n+1} = \text{normalize}(Tx_n)` → God Tensor
     - Banach (1922)
   * - Hodge Decomposition
     - Theoretical framing: gradient/curl/harmonic components
     - Whitney (1957)

Core Pipeline
-------------

.. math::

   \text{Geometry}
   \xrightarrow{\text{FDFD}}
   (|E|, |H|)
   \xrightarrow{\text{PH}}
   (B_E, B_H)
   \xrightarrow{\text{Hilbert}}
   (\mathbf{z}_E, \mathbf{z}_H)
   \xrightarrow{T = Z_H Z_E^+}
   T
   \xrightarrow{\text{Banach}}
   \mathbf{g} = T(\mathbf{g})

External Reading
----------------

These external resources complement the theory documents above:

- `Ripser repository <https://github.com/scikit-tda/ripser>`_ — Bauer et al., fast PH computation
- `Persim documentation <https://scikit-tda.org/persim>`_ — bottleneck/EMD matching
- `Javaplex <https://appliedtopology.github.io/javaplex/>`_ — MATLAB PH (Adamaszek et al.)
- `Gudhi <https://gudhi.inria.fr/>`_ — C++ PH library with Python bindings
- `TdaPackage.jl <https://github.com/juliaai/TDA.jl>`_ — PH in Julia
- Allen Hatcher, `Algebraic Topology <https://pi.math.cornell.edu/~hatcher/AT/ATpage.html>`_ — foundational homology theory
- Robert Ghrist, `Elementary Applied Topology <https://www.math.upenn.edu/~ghrist/AT.html>`_ — applied topology textbook
- `Nakahara <https://www.iops.org/esa/journals/njp-2003/01/379.pdf>`_ — Geometry, Topology and Physics
- `Golub & Van Loan <https://matrixcomputations.org/>`_ — Matrix Computations (power iteration)
