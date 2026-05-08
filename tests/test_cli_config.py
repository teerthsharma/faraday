# Copyright (c) 2026 Teerth Sharma. All rights reserved.
"""Tests for faraday.cli and faraday.config — coverage-critical modules."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from faraday.config import (
    FaradayConfig,
    LoggingConfig,
    PredictConfig,
    SolverConfig,
    TopologyConfig,
    TrainingConfig,
    _as_dict,
    _default_for,
    _sub_config,
)
from faraday.exceptions import ConfigError

# =========================================================================
# faraday.config
# =========================================================================


class TestSolverConfig:
    def test_defaults(self) -> None:
        sc = SolverConfig()
        assert sc.nx == 40
        assert sc.ny == 40
        assert sc.n_modes == 6
        assert sc.boundary == "pec"

    def test_custom(self) -> None:
        sc = SolverConfig(nx=80, ny=80, n_modes=12, frequency=2.0)
        assert sc.nx == 80
        assert sc.frequency == 2.0


class TestTrainingConfig:
    def test_defaults(self) -> None:
        tc = TrainingConfig()
        assert tc.n_geometries == 50
        assert tc.iters == 200


class TestPredictConfig:
    def test_defaults(self) -> None:
        pc = PredictConfig()
        assert pc.model_path is None
        assert pc.batch_size == 32


class TestTopologyConfig:
    def test_defaults(self) -> None:
        tc = TopologyConfig()
        assert tc.n_points == 500
        assert tc.max_edge == 2.0


class TestLoggingConfig:
    def test_defaults(self) -> None:
        lc = LoggingConfig()
        assert lc.level == "INFO"
        assert lc.structured is True


class TestFaradayConfig:
    def test_defaults(self) -> None:
        cfg = FaradayConfig()
        assert isinstance(cfg.solver, SolverConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert cfg.config_file is None

    def test_from_dict_empty(self) -> None:
        cfg = FaradayConfig.from_dict({})
        assert cfg.solver.nx == 40  # defaults

    def test_from_dict_with_sections(self) -> None:
        cfg = FaradayConfig.from_dict({
            "solver": {"nx": 100, "ny": 100},
            "training": {"n_geometries": 200},
        })
        assert cfg.solver.nx == 100
        assert cfg.training.n_geometries == 200

    def test_from_dict_non_dict_section_raises(self) -> None:
        with pytest.raises(ConfigError, match="must be a dict"):
            FaradayConfig.from_dict({"solver": "bad"})

    def test_from_yaml_valid(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({
            "solver": {"nx": 15, "ny": 15},
            "training": {"n_geometries": 5},
        }))
        cfg = FaradayConfig.from_yaml(cfg_file)
        assert cfg.solver.nx == 15
        assert cfg.config_file == cfg_file.resolve()

    def test_from_yaml_missing_file(self) -> None:
        with pytest.raises(ConfigError, match="not found"):
            FaradayConfig.from_yaml("/tmp/definitely_not_a_file.yaml")

    def test_from_yaml_invalid_yaml(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("{{{{bad yaml")
        with pytest.raises(ConfigError, match="parse YAML"):
            FaradayConfig.from_yaml(bad)

    def test_from_yaml_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        cfg = FaradayConfig.from_yaml(empty)
        assert cfg.solver.nx == 40  # defaults

    def test_from_yaml_non_dict(self, tmp_path: Path) -> None:
        bad = tmp_path / "list.yaml"
        bad.write_text("- item1\n- item2\n")
        with pytest.raises(ConfigError, match="YAML dict"):
            FaradayConfig.from_yaml(bad)

    def test_to_dict(self) -> None:
        cfg = FaradayConfig()
        d = cfg.to_dict()
        assert "solver" in d
        assert "training" in d
        assert d["solver"]["nx"] == 40

    def test_to_yaml(self, tmp_path: Path) -> None:
        cfg = FaradayConfig()
        out = tmp_path / "out.yaml"
        cfg.to_yaml(out)
        assert out.exists()
        loaded = yaml.safe_load(out.read_text())
        assert loaded["solver"]["nx"] == 40
        assert cfg.config_file == out.resolve()

    def test_load_defaults(self) -> None:
        # No config files should be at random paths
        cfg = FaradayConfig.load(paths=["/tmp/nonexistent_faraday_cfg.yaml"])
        assert isinstance(cfg, FaradayConfig)

    def test_load_from_env(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "env.yaml"
        cfg_file.write_text(yaml.dump({"solver": {"nx": 77}}))
        old = os.environ.get("FARADAY_CONFIG")
        try:
            os.environ["FARADAY_CONFIG"] = str(cfg_file)
            cfg = FaradayConfig.load()
            assert cfg.solver.nx == 77
        finally:
            if old is None:
                os.environ.pop("FARADAY_CONFIG", None)
            else:
                os.environ["FARADAY_CONFIG"] = old

    def test_load_from_paths(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "custom.yaml"
        cfg_file.write_text(yaml.dump({"solver": {"nx": 33}}))
        cfg = FaradayConfig.load(paths=[str(cfg_file)])
        assert cfg.solver.nx == 33


class TestConfigHelpers:
    def test_sub_config_unknown_fields(self) -> None:
        with pytest.raises(ConfigError, match="unknown field"):
            _sub_config("solver", {"nx": 40, "bogus_field": True})

    def test_default_for(self) -> None:
        s = _default_for("solver")
        assert isinstance(s, SolverConfig)
        t = _default_for("training")
        assert isinstance(t, TrainingConfig)

    def test_as_dict_dataclass(self) -> None:
        sc = SolverConfig(nx=99)
        d = _as_dict(sc)
        assert d["nx"] == 99

    def test_as_dict_plain_dict(self) -> None:
        d = _as_dict({"a": 1})
        assert d == {"a": 1}


# =========================================================================
# faraday.cli
# =========================================================================


class TestCLI:
    @pytest.fixture(autouse=True)
    def _runner(self) -> None:
        self.runner = CliRunner()

    def test_main_help(self) -> None:
        from faraday.cli import main
        result = self.runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "topology-fixed-point" in result.output

    def test_solve_default(self) -> None:
        from faraday.cli import main
        result = self.runner.invoke(main, ["solve", "--width", "2.0", "--height", "1.0", "--nx", "15", "--ny", "15", "--n-modes", "2"])
        assert result.exit_code == 0
        assert "k =" in result.output

    def test_solve_custom_geometry(self) -> None:
        from faraday.cli import main
        result = self.runner.invoke(main, ["solve", "-w", "1.5", "-h", "0.8", "--nx", "15", "--ny", "15"])
        assert result.exit_code == 0

    def test_config_show_defaults(self) -> None:
        from faraday.cli import main
        result = self.runner.invoke(main, ["config-show"])
        assert result.exit_code == 0
        assert "solver" in result.output.lower()

    def test_config_show_yaml(self) -> None:
        from faraday.cli import main
        result = self.runner.invoke(main, ["config-show", "--yaml"])
        assert result.exit_code == 0
        parsed = yaml.safe_load(result.output)
        assert "solver" in parsed

    def test_config_show_with_file(self, tmp_path: Path) -> None:
        from faraday.cli import main
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml.dump({"solver": {"nx": 11}}))
        result = self.runner.invoke(main, ["config-show", "-c", str(cfg_file)])
        assert result.exit_code == 0

    def test_demo_command(self) -> None:
        from faraday.cli import main
        result = self.runner.invoke(main, ["demo", "--n-geometries", "5", "--seed", "42"])
        assert result.exit_code == 0
        assert "coupling_score" in result.output

    def test_train_command(self) -> None:
        from faraday.cli import main
        result = self.runner.invoke(main, [
            "train",
            "--n-geometries", "5",
            "--iters", "10",
            "--seed", "42",
        ])
        assert result.exit_code == 0
        assert "god_score" in result.output

    def test_train_with_save(self, tmp_path: Path) -> None:
        from faraday.cli import main
        save_path = str(tmp_path / "test_gt.pkl")
        result = self.runner.invoke(main, [
            "train",
            "--n-geometries", "5",
            "--iters", "10",
            "--seed", "42",
            "--save", save_path,
        ])
        assert result.exit_code == 0
        assert Path(save_path).exists()

    def test_predict_in_memory(self) -> None:
        from faraday.cli import main
        result = self.runner.invoke(main, [
            "predict", "-w", "2.0", "-h", "1.0",
        ])
        assert result.exit_code == 0
        assert "Betti-0" in result.output

    def test_predict_from_checkpoint(self, tmp_path: Path) -> None:
        from faraday import GodTensor
        from faraday.cli import main

        # Train and save a model first
        gt = GodTensor(n_geometries=5)
        gt.collect_training_data(nx=15, ny=15, num_modes=2, seed=42)
        gt.learn_T()
        gt.find_fixed_point(iters=10)
        model_path = str(tmp_path / "model.pkl")
        gt.save(model_path)

        result = self.runner.invoke(main, [
            "predict", "-w", "2.0", "-h", "1.0",
            "--model", model_path,
        ])
        assert result.exit_code == 0
        assert "Betti-0" in result.output
