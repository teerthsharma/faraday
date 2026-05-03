Basic FDFD Cavity Simulation
============================

This tutorial shows how to run a finite-difference frequency-domain (FDFD)
cavity simulation to compute electromagnetic eigenmodes.

Setup
-----

Import the necessary modules:

.. code-block:: python

   import numpy as np
   from faraday import CavityGeometry, CavityShape, solve_cavity_modes

Defining the Geometry
---------------------

Create a rectangular cavity geometry:

.. code-block:: python

   # Create a 2D rectangular cavity
   geo = CavityGeometry(
       width=2.0,
       height=1.0,
       shape=CavityShape.RECTANGLE,
       nx=64,
       ny=64
   )

   print(f"Cavity area: {geo.area}")
   print(f"Grid resolution: {geo.dx} x {geo.dy}")

Solving for Modes
-----------------

Compute the eigenmodes of the cavity:

.. code-block:: python

   # Solve for the first 10 modes
   modes = solve_cavity_modes(geo, num_modes=10)

   print(f"Found {len(modes)} modes")
   print(f"Mode frequencies: {modes.frequencies}")

Inspecting the Results
----------------------

Access the electric and magnetic field data:

.. code-block:: python

   # E and H fields are 2D arrays
   E = modes.E  # Electric field (nx × ny)
   H = modes.H  # Magnetic field (nx × ny)

   # Visualize the first mode
   import matplotlib.pyplot as plt

   plt.figure(figsize=(10, 4))
   plt.subplot(1, 2, 1)
   plt.imshow(np.abs(E[:, :, 0]), cmap="viridis")
   plt.title("|E| - Mode 0")
   plt.colorbar()

   plt.subplot(1, 2, 2)
   plt.imshow(np.abs(H[:, :, 0]), cmap="plasma")
   plt.title("|H| - Mode 0")
   plt.colorbar()

   plt.tight_layout()
   plt.show()

Next Steps
----------

* Generate persistent homology barcodes from the field data: :doc:`barcode_generation`
* Learn about the God Tensor fixed point: :doc:`god_tensor_workflow`
