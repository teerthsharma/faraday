Quickstart
==========

This guide will help you get started with Faraday in minutes.

Basic Usage
-----------

Here's a minimal example demonstrating the core workflow:

.. code-block:: python

   from faraday import GodTensor

   # Initialize the God Tensor with 50 training geometries
   gt = GodTensor(n_geometries=50, nx=40, ny=40)

   # Collect training data via FDFD simulations
   gt.collect_training_data()

   # Find the fixed point where T(T(x)) = T(x)
   gt.find_fixed_point(iters=200)

   # Predict E and H topology for a new geometry
   pred = gt.predict(w=2.0, h=1.5)
   print(f"Predicted barcode: {pred}")

Workflow Overview
-----------------

The Faraday pipeline consists of four stages:

1. **EM Solver** — Compute E and H eigenmodes using FDFD cavity simulation
2. **Barcode Generation** — Convert field data to point clouds to persistent homology barcodes
3. **Manifold Projection** — Extract Hilbert coefficients and embed into manifold
4. **God Tensor** — Find fixed-point where field signatures co-determine each other

Cavity Simulation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from faraday import CavityGeometry, solve_cavity_modes

   # Define a rectangular cavity
   geo = CavityGeometry(width=2.0, height=1.0, nx=64, ny=64)

   # Solve for modes
   modes = solve_cavity_modes(geo, num_modes=10)

Topological Fingerprinting
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from faraday import field_to_pointcloud, compute_barcodes

   # Convert field to point cloud
   points = field_to_pointcloud(modes.E, resolution=0.1)

   # Compute persistent homology barcode
   barcodes = compute_barcodes(points)

What's Next?
------------

* Learn more about the :doc:`theory` behind Faraday
* Explore the :doc:`api` reference
* Try the :doc:`tutorials/index`
