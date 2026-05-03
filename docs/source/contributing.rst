Contributing
============

We welcome contributions to Faraday! This guide covers how to get involved.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally::

   git clone https://github.com/yourusername/faraday.git
   cd faraday
   pip install -e ".[dev]"

Development Workflow
--------------------

Code Style
~~~~~~~~~~

We use ``ruff`` for linting and formatting::

   ruff check .
   ruff format .

Pre-commit Hooks
~~~~~~~~~~~~~~~~

We use pre-commit hooks to ensure code quality::

   pip install pre-commit
   pre-commit install

Running Tests
~~~~~~~~~~~~~

Run the test suite with ``pytest``::

   pytest tests/

Run tests with coverage::

   pytest --cov=faraday tests/

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

Build the documentation locally::

   cd docs
   pip install -r requirements-doc.txt
   make html

Submitting Changes
------------------

1. Create a feature branch::

   git checkout -b feature/your-feature-name

2. Make your changes and commit them

3. Push to your fork and submit a pull request

4. Ensure CI passes (tests, linting, docs build)

Code of Conduct
----------------

Please be respectful and constructive in all interactions.

License
-------

By contributing, you agree that your contributions will be licensed under the MIT License.
