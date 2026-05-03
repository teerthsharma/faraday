Benchmarks
==========

This page documents performance benchmarks for Faraday components.

EM Solver Performance
---------------------

The FDFD cavity solver performance scales with grid resolution:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Grid Size (nx × ny)
     - Modes
     - Time (s)
     - Memory (MB)
   * - 32 × 32
     - 5
     - 0.8
     - 128
   * - 64 × 64
     - 10
     - 4.2
     - 512
   * - 128 × 128
     - 20
     - 28.5
     - 2048

Barcode Computation
-------------------

Persistent homology computation times:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Point Cloud Size
     -ripser Time (s)
     - Memory (MB)
   * - 500 points
     - 0.3
     - 64
   * - 2000 points
     - 2.1
     - 256
   * - 5000 points
     - 12.4
     - 1024

God Tensor Convergence
----------------------

Fixed-point iteration convergence rates:

* Typical convergence: 50-150 iterations
* Fixed-point tolerance: 1e-6
* Average iteration time: ~0.5s per geometry pair

Prediction Accuracy
-------------------

Topology prediction accuracy on test geometries:

* Mean barcode distance error: < 5%
* Classification accuracy (mode type): > 92%
* Embedding correlation with ground truth: > 0.95
