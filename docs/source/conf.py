"""
Sphinx configuration for Faraday documentation.
"""

import os
import sys

# Add the project root to the path for autoapi
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
project = "Faraday"
copyright = "2026, Faraday Contributors"
author = "Faraday Contributors"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- AutoAPI configuration ---------------------------------------------------
autoapi_dirs = ["../../faraday"]
autoapi_ignore = ["*/tests/*", "*/build/*", "*/__pycache__/*"]
autoapi_member_order = "bysource"
autoapi_template_dir = "_templates/autoapi"

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Napoleon configuration ---------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
