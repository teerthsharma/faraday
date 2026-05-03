"""
faraday.exceptions — Typed exception hierarchy for the faraday package.

All exceptions inherit from FaradayError. Each has a docstring describing
when it is raised and what payload it carries.
"""

from __future__ import annotations

from typing import Any


class FaradayError(Exception):
    """
    Base exception for all faraday errors.

    All library-specific exceptions inherit from this class, making it
    easy to catch any faraday error with a single ``except`` clause.

    Attributes
    ----------
    msg : str
        Human-readable description of the error.
    context : dict[str, Any]
        Additional structured context about the error (geometry params,
        indices, etc.) useful for debugging.
    """

    def __init__(self, msg: str, **context: Any) -> None:
        super().__init__(msg)
        self.msg = msg
        self.context: dict[str, Any] = context

    def __str__(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.__class__.__name__}({self.msg!r}, {ctx_str})"
        return f"{self.__class__.__name__}({self.msg!r})"


class ConvergenceError(FaradayError):
    """
    Raised when a numerical fixed-point or iterative method fails to converge.

    This applies to God Tensor fixed-point iteration and similar procedures.

    Example
    -------
    >>> raise ConvergenceError("fixed point did not converge within 500 iters", iters=500, tol=1e-7)
    """

    pass


class SolverError(FaradayError):
    """
    Raised when the FDFD cavity solver fails.

    This can happen for degenerate geometries, numerical instabilities,
    or when the sparse eigensolver fails to find the requested modes.

    Example
    -------
    >>> raise SolverError("eigsh failed to converge", geometry=(2.0, 1.0), nx=60, ny=60)
    """

    pass


class GeometryError(FaradayError):
    """
    Raised when an invalid or unsupported geometry is provided.

    This includes degenerate cases (zero width/height), unsupported
    shapes, or malformed geometry parameter tuples.

    Example
    -------
    >>> raise GeometryError("width must be positive", width=0.0)
    """

    pass


class TopologyError(FaradayError):
    """
    Raised when topological (persistent homology) computation fails.

    This can occur when the point cloud has too few points, the barcode
    computation encounters a numerical issue, or the embedding dimension
    is invalid.

    Example
    -------
    >>> raise TopologyError("too few points for barcode computation", n_points=5, threshold=0.1)
    """

    pass


class ConfigError(FaradayError):
    """
    Raised when an invalid configuration value is provided.

    This applies to YAML config loading, CLI argument validation,
    and programmatic API calls with bad parameters.

    Example
    -------
    >>> raise ConfigError("nx must be a positive integer", nx=-5)
    """

    pass
