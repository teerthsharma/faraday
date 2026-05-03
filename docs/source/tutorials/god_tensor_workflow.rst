God Tensor Workflow
===================

This tutorial demonstrates the complete God Tensor workflow: collecting training
data, finding the fixed point, and making predictions.

Overview
--------

The God Tensor :math:`\mathcal{G}` is the fixed point of the coupled field
transformation where electric and magnetic topological fingerprints become
indistinguishable.

Setup
-----

.. code-block:: python

   from faraday import GodTensor
   import numpy as np

Initializing the God Tensor
---------------------------

Create a God Tensor instance with training parameters:

.. code-block:: python

   gt = GodTensor(
       n_geometries=50,    # Number of training geometries
       nx=40,              # Grid resolution x
       ny=40,              # Grid resolution y
       seed=42             # Reproducibility seed
   )

   print(f"God Tensor initialized with {gt.n_geometries} geometries")
   print(f"Grid: {gt.nx} x {gt.ny}")

Collecting Training Data
------------------------

Run FDFD simulations across varied geometries to collect training data:

.. code-block:: python

   # This generates random cavity geometries and solves for E/H modes
   gt.collect_training_data()

   print(f"Training data collected: {len(gt.training_geometries)} geometries")
   print(f"Training barcodes shape: {gt.barcodes_E.shape}")

Finding the Fixed Point
-----------------------

Iterate until the topology operator reaches a fixed point:

.. code-block:: python

   # Find fixed point: T(T(x)) = T(x)
   result = gt.find_fixed_point(
       iters=200,           # Maximum iterations
       tol=1e-6,            # Convergence tolerance
       verbose=True         # Print progress
   )

   print(f"Converged in {result['n_iter']} iterations")
   print(f"Fixed point distance: {result['distance']:.2e}")

Making Predictions
------------------

Predict E and H topology for new geometries:

.. code-block:: python

   # Predict for a new geometry
   prediction = gt.predict(
       w=2.5,   # Width
       h=1.2    # Height
   )

   print(f"Predicted E barcode: {prediction['E_barcode'][:3]}...")
   print(f"Predicted H barcode: {prediction['H_barcode'][:3]}...")

   # Also get the embedded representation
   embedding = prediction['embedding']
   print(f"Embedding shape: {embedding.shape}")

Accessing the God Tensor
------------------------

The converged God Tensor is stored for analysis:

.. code-block:: python

   # Access the God Tensor matrix
   G = gt.god_tensor
   print(f"God Tensor shape: {G.shape}")

   # Eigenvalues reveal dominant topological modes
   eigenvalues = gt.eigenvalues
   print(f"Leading eigenvalues: {eigenvalues[:5]}")

Saving and Loading
-----------------

Save the trained God Tensor for later use:

.. code-block:: python

   # Save to disk
   gt.save("god_tensor_trained.pkl")

   # Load later
   gt_loaded = GodTensor.load("god_tensor_trained.pkl")

Advanced: Custom Geometry Sampling
----------------------------------

Use custom geometry distributions:

.. code-block:: python

   from faraday import CavityGeometry

   # Define custom geometry generator
   def custom_sampler(n):
       geometries = []
       for i in range(n):
           w = np.random.uniform(1.0, 5.0)
           h = np.random.uniform(0.5, 3.0)
           geo = CavityGeometry(width=w, height=h, nx=40, ny=40)
           geometries.append(geo)
       return geometries

   gt2 = GodTensor(n_geometries=30, nx=40, ny=40)
   gt2.collect_training_data(geometry_sampler=custom_sampler)

Next Steps
----------

* Explore the :doc:`../theory` behind the God Tensor
* Read the :doc:`../api` reference for all available methods
