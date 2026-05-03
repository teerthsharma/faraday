"""
faraday.config — YAML configuration system with typed dataclass.

Loads, validates, and merges configuration from files and environment
variables. All user-facing parameters live in FaradayConfig.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from faraday.exceptions import ConfigError

_DEFAULT_CONFIG_PATHS = [
    Path("faraday.yaml"),
    Path("faraday.yml"),
    Path.home() / ".config" / "faraday" / "config.yaml",
]


@dataclass
class SolverConfig:
    """FDFD solver settings."""

    nx: int = 40
    ny: int = 40
    n_modes: int = 6
    frequency: float = 1.0
    boundary: str = "pec"


@dataclass
class TrainingConfig:
    """Training / data-collection settings."""

    n_geometries: int = 50
    width_min: float = 0.5
    width_max: float = 5.0
    height_min: float = 0.5
    height_max: float = 5.0
    iters: int = 200
    tol: float = 1e-7


@dataclass
class PredictConfig:
    """Prediction settings."""

    model_path: str | None = None
    batch_size: int = 32


@dataclass
class TopologyConfig:
    """Persistent homology / barcode settings."""

    n_points: int = 500
    max_edge: float = 2.0
    n_workers: int = 1


@dataclass
class LoggingConfig:
    """Logging / structlog settings."""

    level: str = "INFO"
    structured: bool = True


@dataclass
class FaradayConfig:
    """
    Root configuration object for the entire faraday package.

    Attributes
    ----------
    solver : SolverConfig
        FDFD cavity solver parameters.
    training : TrainingConfig
        Training data collection parameters.
    predict : PredictConfig
        Inference / prediction parameters.
    topology : TopologyConfig
        Persistent homology computation parameters.
    logging : LoggingConfig
        Logging configuration.
    config_file : Optional[Path]
        Path to the loaded config file, if any.
    """

    solver: SolverConfig = field(default_factory=SolverConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    config_file: Path | None = field(default=None, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FaradayConfig:
        """Construct a FaradayConfig from a flat dictionary."""
        kw = {}
        sub_fields = {"solver", "training", "predict", "topology", "logging"}
        for key in sub_fields:
            if key in data and isinstance(data[key], dict):
                kw[key] = _sub_config(key, data[key])
            elif key not in data:
                kw[key] = _default_for(key)
            else:
                raise ConfigError(
                    f"'{key}' must be a dict", key=key, got=type(data[key]).__name__
                )
        # Pass through top-level keys that are not sub-field names
        remaining = {k: v for k, v in data.items() if k not in sub_fields}
        if remaining:
            kw.update(remaining)
        return cls(**kw)

    @classmethod
    def from_yaml(cls, path: str | Path) -> FaradayConfig:
        """Load configuration from a YAML file."""
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}", path=str(p))
        try:
            raw = yaml.safe_load(p.read_text())
        except yaml.YAMLError as exc:
            raise ConfigError(f"failed to parse YAML: {exc}", path=str(p)) from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigError(
                "config file must contain a YAML dict", path=str(p), got=type(raw).__name__
            )
        cfg = cls.from_dict(raw)
        cfg.config_file = p.resolve()
        return cfg

    @classmethod
    def load(cls, paths: list[str | Path] | None = None) -> FaradayConfig:
        """
        Load the first found config file, falling back to defaults.

        Searches the following locations in order:
        1. Paths provided via ``paths`` argument.
        2. ``FARADAY_CONFIG`` environment variable.
        3. ``~/.config/faraday/config.yaml``
        4. ``faraday.yaml`` / ``faraday.yml`` in current directory.
        """
        candidates: list[Path] = []
        if paths:
            candidates.extend(Path(p) for p in paths)
        env_path = os.environ.get("FARADAY_CONFIG")
        if env_path:
            candidates.append(Path(env_path))
        candidates.extend(_DEFAULT_CONFIG_PATHS)

        for p in candidates:
            if p.exists() and p.is_file():
                return cls.from_yaml(p)

        # Return defaults if no file found
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation (YAML-compatible)."""
        return {
            "solver": _as_dict(self.solver),
            "training": _as_dict(self.training),
            "predict": _as_dict(self.predict),
            "topology": _as_dict(self.topology),
            "logging": _as_dict(self.logging),
        }

    def to_yaml(self, path: str | Path) -> None:
        """Serialize config to a YAML file."""
        p = Path(path)
        p.write_text(yaml.dump(self.to_dict(), sort_keys=False))
        self.config_file = p.resolve()


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _as_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        return {f: getattr(obj, f) for f in obj.__dataclass_fields__}
    return dict(obj) if isinstance(obj, dict) else obj  # pragma: no cover


def _sub_config(name: str, data: dict[str, Any]) -> Any:
    """Return a sub-config dataclass instance from a dict."""
    mapping = {
        "solver": SolverConfig,
        "training": TrainingConfig,
        "predict": PredictConfig,
        "topology": TopologyConfig,
        "logging": LoggingConfig,
    }
    cls = mapping.get(name)
    if cls is None:
        raise ConfigError(f"unknown config section: {name!r}")  # pragma: no cover
    valid = set(cls.__dataclass_fields__)
    unknown = set(data.keys()) - valid
    if unknown:
        raise ConfigError(
            f"unknown field(s) in '{name}': {', '.join(sorted(unknown))}",
            section=name,
            unknown=list(unknown),
        )
    return cls(**data)


def _default_for(name: str) -> Any:
    mapping = {
        "solver": SolverConfig,
        "training": TrainingConfig,
        "predict": PredictConfig,
        "topology": TopologyConfig,
        "logging": LoggingConfig,
    }
    return mapping[name]()
