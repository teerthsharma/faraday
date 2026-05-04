# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""Nox sessions for faraday testing across multiple environments."""

import nox

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]
PYTHON_DEFAULT = "3.11"


@nox.session(python=PYTHON_DEFAULT)
def tests(session):
    """Run the test suite with coverage."""
    session.install(".[dev]")
    session.install("ripser>=0.6", "persim>=0.3")  # ensure available
    session.run("pytest", "tests/", "--cov=faraday", "--cov-fail-under=80", "-v")


@nox.session(python=PYTHON_DEFAULT)
def lint(session):
    """Run ruff linter."""
    session.install("ruff>=0.3")
    session.run("ruff", "check", "faraday/", "--fix")
    session.run("ruff", "format", "--check", "faraday/")


@nox.session(python=PYTHON_DEFAULT)
def typecheck(session):
    """Run pyright type checker."""
    session.install("pyright>=1.1")
    session.install("-e", ".")
    session.run("pyright", "faraday/")


@nox.session(python=PYTHON_VERSIONS)
def test_all_pythons(session):
    """Test across all supported Python versions."""
    session.install(".[dev]")
    session.install("ripser>=0.6", "persim>=0.3")
    session.run("pytest", "tests/", "-v", "--timeout=60")


@nox.session(python=PYTHON_DEFAULT)
def benchmark(session):
    """Run benchmark suite."""
    session.install(".[dev]")
    session.install("pytest-benchmark>=4.0")
    session.run("pytest", "tests/benchmarks/", "-v")


@nox.session(python=PYTHON_DEFAULT)
def precommit(session):
    """Run pre-commit hooks on all files."""
    session.install("pre-commit>=3.6")
    session.run("pre-commit", "run", "--all-files")
