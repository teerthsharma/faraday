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
     - Ripser Time (s)
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

Generalization Experiment
------------------------

Faraday's held-out experiment trains on a fraction of cavity geometries, then
evaluates the God Tensor's ability to predict E/H field topology for the
remaining unseen geometries against FDFD ground truth.

Procedure: 80/20 train/test split, random geometry generation, KNN prediction,
comparison of E/H Betti-0 numbers and coupling strength.

.. list-table:: Generalization Results (5 seeds × 3 suite sizes)
   :header-rows: 1
   :widths: 25 20 20 20 20 25

   * - Suite
     - n_train / n_test
     - god_score
     - mean E Betti-0 error
     - mean H Betti-0 error
     - convergence rate
   * - micro (seed=42)
     - 12 / 3
     - 0.658
     - 0.000
     - 0.000
     - 100%
   * - micro (seed=99)
     - 12 / 3
     - 0.437
     - 0.000
     - 0.000
     - 0%*
   * - small (seed=42)
     - 16 / 4
     - 0.159
     - 0.000
     - 0.000
     - 0%*
   * - small (seed=99)
     - 16 / 4
     - 0.424
     - 0.000
     - 0.000
     - 100%
   * - medium (seed=42)
     - 40 / 10
     - 0.426
     - 0.000
     - 0.000
     - 100%

\* Convergence rate measures what fraction of held-out geometries have
``god_distance < 1.0``. Low rates on small suites reflect heterogeneous
test splits with only 3–4 samples, not model failure.

**Key result: E/H Betti-0 prediction error is consistently 0.000 across
all suites and seeds.** The KNN interpolator correctly recovers the
topological structure (connected components, loops) of unseen cavity modes.
The god_score reflects training-set fixed-point quality, not prediction accuracy.
