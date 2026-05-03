Installation
============

Requirements
------------

* Python 3.10 or higher
* NumPy
* SciPy

Installing Faraday
------------------

Install the latest stable release via pip:

.. code-block:: bash

   pip install faraday

Or install from source:

.. code-block:: bash

   git clone https://github.com/yourusername/faraday.git
   cd faraday
   pip install -e .

Development Installation
-------------------------

To set up a development environment:

.. code-block:: bash

   git clone https://github.com/yourusername/faraday.git
   cd faraday
   pip install -e ".[dev]"
   pip install -r requirements-doc.txt

Verifying Installation
----------------------

To verify that Faraday is installed correctly:

.. code-block:: python

   import faraday
   print(faraday.__version__)

Dependencies
------------

Core dependencies:

* ``numpy>=1.24.0``
* ``scipy>=1.10.0``

Optional dependencies:

* ``matplotlib`` - For visualization
* ``persim`` - For persistent homology diagnostics
