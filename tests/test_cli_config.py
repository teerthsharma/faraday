# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
Tests for the CLI, configuration, exceptions, wave superposition and the
benchmark / burn drivers.
"""

import json

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

TINY_CONFIG = {
    "solver": {"nx": 14, "ny": 14, "n_modes": 2},
    "training": {"n_geometries": 6, "iters": 50},
}


@pytest.fixture
def tiny_config(tmp_path):
    path = tmp_path / "faraday.yaml"
    path.write_text(yaml.safe_dump(TINY_CONFIG))
    return str(path)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self, tmp_path, monkeypatch):
        from faraday.config import FaradayConfig

        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("FARADAY_CONFIG", raising=False)
        monkeypatch.setattr("faraday.config._DEFAULT_CONFIG_PATHS", [])
        cfg = FaradayConfig.load()
        assert cfg.solver.nx == 40
        assert cfg.config_file is None

    def test_yaml_roundtrip(self, tmp_path):
        from faraday.config import FaradayConfig

        cfg = FaradayConfig.from_dict({"solver": {"nx": 17}})
        path = tmp_path / "out.yaml"
        cfg.to_yaml(path)
        back = FaradayConfig.from_yaml(path)
        assert back.solver.nx == 17
        assert back.to_dict() == cfg.to_dict()
        assert back.config_file == path.resolve()

    def test_load_from_env(self, tiny_config, monkeypatch):
        from faraday.config import FaradayConfig

        monkeypatch.setenv("FARADAY_CONFIG", tiny_config)
        assert FaradayConfig.load().solver.nx == 14

    def test_load_from_explicit_paths(self, tiny_config, tmp_path):
        from faraday.config import FaradayConfig

        cfg = FaradayConfig.load([tmp_path / "missing.yaml", tiny_config])
        assert cfg.training.n_geometries == 6

    @pytest.mark.parametrize(
        ("content", "match"),
        [
            ("solver: [1, 2", "failed to parse YAML"),
            ("- 1\n- 2\n", "must contain a YAML dict"),
            ("solver:\n  bogus: 1\n", "unknown field"),
            ("solver: 3\n", "must be a dict"),
        ],
    )
    def test_invalid_files(self, tmp_path, content, match):
        from faraday import ConfigError
        from faraday.config import FaradayConfig

        path = tmp_path / "bad.yaml"
        path.write_text(content)
        with pytest.raises(ConfigError, match=match):
            FaradayConfig.from_yaml(path)

    def test_missing_file(self, tmp_path):
        from faraday import ConfigError
        from faraday.config import FaradayConfig

        with pytest.raises(ConfigError, match="not found"):
            FaradayConfig.from_yaml(tmp_path / "nope.yaml")

    def test_empty_file_gives_defaults(self, tmp_path):
        from faraday.config import FaradayConfig

        path = tmp_path / "empty.yaml"
        path.write_text("")
        assert FaradayConfig.from_yaml(path).solver.nx == 40


def test_exception_str_includes_context():
    from faraday import ConvergenceError, FaradayError

    err = ConvergenceError("no fixed point", iters=5)
    assert isinstance(err, FaradayError)
    assert str(err) == "ConvergenceError('no fixed point', iters=5)"
    assert str(FaradayError("plain")) == "FaradayError('plain')"


# ---------------------------------------------------------------------------
# Wave superposition
# ---------------------------------------------------------------------------


class TestWaveSuperposer:
    @pytest.fixture
    def setup(self):
        from faraday import CavityGeometry, CavityShape, WaveSuperposer, solve_cavity_modes

        geom = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(2.0, 1.0))
        modes = solve_cavity_modes(geom, nx=16, ny=16, num_modes=2, seed=0)
        X, Y = np.array(modes["X"]), np.array(modes["Y"])
        return WaveSuperposer(geom, modes), modes, X, Y

    def test_no_h_mode_means_zero_h_field(self, setup):
        ws, _, X, Y = setup
        ws.add_e_mode(0)
        assert np.all(ws.h_field_at(X, Y) == 0)

    def test_h_mode_index_is_respected(self, setup):
        ws, modes, X, Y = setup
        ws.add_e_mode(0).add_h_mode(1, amplitude=2.0)
        expected = 2.0 * np.array(modes["h_modes"]["mode_1"]["field"])
        np.testing.assert_allclose(ws.h_field_at(X, Y), expected)

    def test_plane_waves_and_poynting(self, setup):
        ws, _, X, Y = setup
        ws.add_e_wave(1.0, angle=0.0, wavelength=1.0)
        ws.add_h_wave(0.5, angle=np.pi / 2, wavelength=1.0)
        E, H = ws.coupled_field_at(X, Y, t=0.3)
        np.testing.assert_allclose(np.abs(E), 1.0)
        np.testing.assert_allclose(np.abs(H), 0.5)
        np.testing.assert_allclose(ws.poynting_vector(X, Y, t=0.3), 0.5)
        assert ws.e_waves[0].k == pytest.approx(2 * np.pi)

    def test_missing_modes_raise(self, setup):
        ws, *_ = setup
        with pytest.raises(KeyError):
            ws.add_e_mode(99)
        with pytest.raises(KeyError):
            ws.add_h_mode(99)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_solve_prints_fundamental_first(self, tiny_config):
        from faraday.cli import main

        out = CliRunner().invoke(
            main, ["solve", "-w", "2", "-h", "1", "--nx", "41", "--ny", "41", "-c", tiny_config]
        )
        assert out.exit_code == 0, out.output
        first = out.output.split("mode 1: k = ")[1].split(",")[0]
        assert float(first) == pytest.approx(np.pi * np.hypot(0.5, 1.0), rel=5e-3)

    def test_train_then_predict_with_saved_model(self, tiny_config, tmp_path):
        from faraday.cli import main, predict

        model = str(tmp_path / "gt.pkl")
        runner = CliRunner()
        out = runner.invoke(main, ["train", "-c", tiny_config, "-o", model])
        assert out.exit_code == 0, out.output
        assert "converged=True" in out.output

        out = runner.invoke(main, ["predict", "-w", "2", "-h", "1", "-m", model, "-c", tiny_config])
        assert out.exit_code == 0, out.output
        assert "persistence=inf" in out.output

        out = runner.invoke(predict, ["-w", "2", "-h", "1", "-c", tiny_config])
        assert out.exit_code == 0, out.output
        assert "untrained" in out.output

    def test_config_show(self, tiny_config):
        from faraday.cli import main

        runner = CliRunner()
        out = runner.invoke(main, ["config-show", "-c", tiny_config])
        assert out.exit_code == 0
        assert "nx: 14" in out.output
        out = runner.invoke(main, ["config-show", "-c", tiny_config, "--yaml"])
        assert yaml.safe_load(out.output)["solver"]["nx"] == 14

    def test_demo_entry_point_imports(self, tiny_config, monkeypatch):
        import faraday.demo
        from faraday.cli import demo

        called = []
        monkeypatch.setattr(faraday.demo, "main", lambda: called.append(True))
        out = CliRunner().invoke(demo, ["-c", tiny_config])
        assert out.exit_code == 0, out.output
        assert called == [True]


# ---------------------------------------------------------------------------
# Benchmark / burn drivers
# ---------------------------------------------------------------------------


class TestBenchmarking:
    def test_unknown_suite(self):
        from faraday.benchmarking import run_suite

        with pytest.raises(ValueError, match="Unknown suite"):
            run_suite("galactic")

    def test_save_report_json_and_csv(self, tmp_path):
        from faraday.benchmarking import BenchmarkReport, BenchmarkResult, save_report

        report = BenchmarkReport(
            suite="unit",
            timestamp="now",
            total_duration_s=1.0,
            results=[BenchmarkResult(name="x", duration_s=0.5)],
        )
        save_report(report, output_dir=str(tmp_path), formats=["json", "csv"])
        assert json.loads((tmp_path / "benchmark_unit.json").read_text())["suite"] == "unit"
        assert (tmp_path / "benchmark_unit.csv").read_text().splitlines()[1] == "x,0.500000"

    def test_cli_micro_suite(self, tmp_path, monkeypatch, capsys):
        from faraday import benchmarking

        monkeypatch.setattr(
            "sys.argv", ["bench", "--suite", "micro", "--output", str(tmp_path)]
        )
        benchmarking.cli_main()
        assert "god_tensor_full_pipeline" in capsys.readouterr().out
        assert (tmp_path / "benchmark_micro.json").exists()

    def test_burn_contracts_and_resumes(self, tmp_path, monkeypatch, capsys):
        from faraday.benchmarking import run_burn

        ckpt = str(tmp_path / "burn.npz")
        # Checkpoint every epoch for the test instead of every 10k
        monkeypatch.setattr("faraday.benchmarking.CHECKPOINT_EVERY", 1)
        run_burn(
            epochs=4, n_geometries=6, nx=14, ny=14, num_modes=2, checkpoint_path=ckpt
        )
        lines = [
            json.loads(line)
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("{")
        ]
        losses = [ln["banach_loss"] for ln in lines if ln.get("event") == "burn_epoch"]
        assert len(losses) == 4
        assert losses[0] > 1e-3  # generic start: a real first step
        assert losses[-1] < 1e-8  # contracted onto the fixed point

        run_burn(epochs=6, resume_from=4, checkpoint_path=ckpt)
        resumed = [
            json.loads(line)["epoch"]
            for line in capsys.readouterr().out.splitlines()
            if line.startswith('{"event": "burn_epoch"')
        ]
        assert resumed == [5, 6]
