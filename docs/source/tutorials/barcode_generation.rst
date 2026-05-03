Barcode Generation Tutorial
===========================

This tutorial demonstrates how to convert electromagnetic field data
into persistent homology barcodes for topological analysis.

Overview
--------

The barcode captures the topological features of field data across scales:
birth/death intervals represent when features appear and disappear.

Setup
-----

.. code-block:: python

   import numpy as np
   from faraday import (
       field_to_pointcloud,
       compute_barcodes,
       topological_fingerprint,
       coupled_fingerprint,
   )

Converting Fields to Point Clouds
---------------------------------

First, convert the field data to a point cloud:

.. code-block:: python

   # Simulated field data (replace with actual solver output)
   E_field = np.random.randn(64, 64) + 1j * np.random.randn(64, 64)

   # Convert to point cloud with Gaussian noise
   points = field_to_pointcloud(
       E_field,
       resolution=0.1,
       noise_scale=0.01
   )

   print(f"Point cloud shape: {points.shape}")

Computing Persistent Homology
-----------------------------

Compute the barcode from the point cloud:

.. code-block:: python

   # Compute barcodes at multiple dimensions
   barcodes = compute_barcodes(
       points,
       max_dim=2,      # Compute H0, H1, H2
       max_edge=2.0   # Maximum edge length
   )

   print(f"Number of H0 features: {len(barcodes[0])}")
   print(f"Number of H1 features: {len(barcodes[1])}")

Topological Fingerprint
-----------------------

Extract a fixed-dimensional fingerprint:

.. code-block:: python

   # Compute topological fingerprint (vector of Hilbert coefficients)
   fp = topological_fingerprint(
       barcodes,
       dim=1,        # Focus on H1 (loops)
       n_bins=50     # Number of bins for Hilbert coefficients
   )

   print(f"Fingerprint shape: {fp.shape}")

Coupled Fingerprint
-------------------

Combine E and H field fingerprints:

.. code-block:: python

   H_field = np.random.randn(64, 64) + 1j * np.random.randn(64, 64)

   coupled_fp = coupled_fingerprint(
       E_field,
       H_field,
       resolution=0.1,
       dim=1
   )

   print(f"Coupled fingerprint shape: {coupled_fp.shape}")

Visualizing Barcodes
--------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   # Plot H0 barcode (connected components)
   if len(barcodes[0]) > 0:
       birth_0 = [b[0] for b in barcodes[0]]
       death_0 = [b[1] for b in barcodes[0]]

       plt.figure(figsize=(8, 3))
       for i, (b, d) in enumerate(zip(birth_0, death_0)):
           plt.plot([b, d], [i, i], "b-", linewidth=2)
       plt.xlabel("Filtration Value")
       plt.ylabel("Component")
       plt.title("H0 Barcode (Connected Components)")
       plt.tight_layout()
       plt.show()

Next Steps
----------

* Apply manifold projection to embed barcodes in Hilbert space: :doc:`god_tensor_workflow`
