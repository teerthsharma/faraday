# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.cli — Click-based command-line interface for faraday.

Commands
--------
solve   — Run FDFD cavity solve for a given geometry.
train   — Collect training data and train the God Tensor.
predict — Predict E/H topology for a new geometry using a trained model.
config-show — Display the active configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    pass

from faraday.config import FaradayConfig

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _resolve_config(ctx: click.Context, param: click.Option, value: str | None):
    cfg = FaradayConfig.load() if value is None else FaradayConfig.from_yaml(value)
    ctx.obj = getattr(ctx, "obj", {}) or {}
    ctx.obj["config"] = cfg
    return cfg


# ----------------------------------------------------------------------
# Group / main
# ----------------------------------------------------------------------


@click.group()
@click.version_option(package_name="faraday")
def main() -> None:
    """Computational Faraday Tensor — topology-fixed-point field predictor."""
    pass


# ----------------------------------------------------------------------
# solve
# ----------------------------------------------------------------------


@main.command()
@click.option(
    "-w", "--width", type=float, default=2.0, help="Cavity width (um / normalised units)."
)
@click.option(
    "-h", "--height", type=float, default=1.0, help="Cavity height."
)
@click.option(
    "-n", "--n-modes", "n_modes", type=int, default=None,
    help="Number of modes to compute. Defaults to config value.",
)
@click.option(
    "--nx", type=int, default=None, help="Grid points in x. Defaults to config value."
)
@click.option(
    "--ny", type=int, default=None, help="Grid points in y. Defaults to config value."
)
@click.option(
    "-c", "--config", "config_path", type=str, default=None,
    help="Path to YAML config file. See faraday.config for search paths.",
    callback=_resolve_config,
    is_eager=True,
)
@click.pass_context
def solve(
    ctx: click.Context,
    width: float,
    height: float,
    n_modes: int | None,
    nx: int | None,
    ny: int | None,
    config_path: str | None,
) -> None:
    """
    Run the FDFD cavity solver for a given geometry.

    Computes TE and TM eigenmodes and prints the first few resonant
    frequencies (eigenvalues k²).

    Example
    -------
    $ faraday solve --width 2.0 --height 1.0 --n-modes 4
    """
    # Lazy import to avoid circular reference during package init
    from faraday import CavityGeometry, CavityShape, solve_cavity_modes

    cfg: FaradayConfig = ctx.obj["config"]
    nx_val = nx if nx is not None else cfg.solver.nx
    ny_val = ny if ny is not None else cfg.solver.ny
    n_modes_val = n_modes if n_modes is not None else cfg.solver.n_modes

    geometry = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(width, height))

    click.echo(f"[faraday solve] geometry=({width} x {height}), nx={nx_val}, ny={ny_val}, modes={n_modes_val}")

    modes = solve_cavity_modes(geometry, nx=nx_val, ny=ny_val, num_modes=n_modes_val)
    mode_list = list(modes.values())[: int(n_modes_val)]

    click.echo(f"\nResonant frequencies (k²) for first {n_modes_val} modes:")
    for i, m in enumerate(mode_list, 1):
        click.echo(f"  mode {i}: k² = {m['eigenvalue']:.6f}")


# ----------------------------------------------------------------------
# train
# ----------------------------------------------------------------------


@main.command()
@click.option(
    "-n", "--n-geometries", type=int, default=None,
    help="Number of random geometries to generate. Defaults to config value.",
)
@click.option(
    "--iters", type=int, default=None,
    help="Fixed-point iterations. Defaults to config value.",
)
@click.option(
    "-c", "--config", "config_path", type=str, default=None,
    help="Path to YAML config file.",
    callback=_resolve_config,
    is_eager=True,
)
@click.pass_context
def train(
    ctx: click.Context,
    n_geometries: int | None,
    iters: int | None,
    config_path: str | None,
) -> None:
    """
    Collect training data and train the God Tensor.

    Runs FDFD across a sweep of random rectangular geometries,
    builds barcodes, projects to manifold, and finds the fixed point
    via God Tensor iteration.

    Example
    -------
    $ faraday train --n-geometries 30 --iters 100
    """
    # Lazy import to avoid circular reference during package init
    from faraday import GodTensor

    cfg: FaradayConfig = ctx.obj["config"]
    n_geo = n_geometries if n_geometries is not None else cfg.training.n_geometries
    iters_val = iters if iters is not None else cfg.training.iters

    click.echo(f"[faraday train] n_geometries={n_geo}, iters={iters_val}")

    gt = GodTensor(n_geometries=n_geo)

    def _step() -> None:
        pass

    with click.progressbar(length=n_geo, label="Collecting training data") as bar:
        def _progress() -> None:
            bar.update(1)

        # Collect data with default solver params
        for i in range(n_geo):
            try:
                gt.collect_training_data(
                    nx=cfg.solver.nx,
                    ny=cfg.solver.ny,
                    num_modes=cfg.solver.n_modes,
                )
            except Exception as e:
                click.echo(f"Warning: sample {i} failed: {e}", err=True)
            _progress()

    click.echo("Training data collected. Finding fixed point...")
    gt.find_fixed_point(iters=iters_val)
    click.echo("Training complete.")


# ----------------------------------------------------------------------
# predict
# ----------------------------------------------------------------------


@main.command()
@click.option(
    "-w", "--width", type=float, required=True, help="Cavity width."
)
@click.option(
    "-h", "--height", type=float, required=True, help="Cavity height."
)
@click.option(
    "-m", "--model", "model_path", type=str, default=None,
    help="Path to a trained GodTensor checkpoint. If omitted, uses in-memory model.",
)
@click.option(
    "-c", "--config", "config_path", type=str, default=None,
    help="Path to YAML config file.",
    callback=_resolve_config,
    is_eager=True,
)
@click.pass_context
def predict_cmd(
    ctx: click.Context,
    width: float,
    height: float,
    model_path: str | None,
    config_path: str | None,
) -> None:
    """
    Predict E and H barcodes for a new geometry using the trained God Tensor.

    Example
    -------
    $ faraday predict --width 2.5 --height 1.2 --model god_tensor_checkpoint.pkl
    """
    # Lazy import to avoid circular reference during package init
    from faraday import GodTensor

    cfg: FaradayConfig = ctx.obj["config"]

    if model_path is None:
        click.echo("[faraday predict] No model provided — creating an untrained GodTensor.")
        gt = GodTensor(n_geometries=cfg.training.n_geometries)
    else:
        click.echo(f"[faraday predict] Loading model from {model_path}")
        gt = GodTensor.load(model_path)  # type: ignore[attr-defined]

    barcode = gt.predict(w=width, h=height)  # type: ignore[attr-defined]

    click.echo(f"\nPredicted E-field barcode for geometry ({width} x {height}):")
    for birth, death in barcode:
        persistence = death - birth
        click.echo(f"  birth={birth:.4f}, death={death:.4f}, persistence={persistence:.4f}")


# ----------------------------------------------------------------------
# config-show
# ----------------------------------------------------------------------


@main.command("config-show")
@click.option(
    "-c", "--config", "config_path", type=str, default=None,
    help="Path to YAML config file. Defaults to first found in search path.",
    callback=_resolve_config,
    is_eager=True,
)
@click.option(
    "--yaml", "as_yaml", is_flag=True, default=False,
    help="Output in YAML format instead of human-readable.",
)
@click.pass_context
def config_show(
    ctx: click.Context,
    config_path: str | None,
    as_yaml: bool,
) -> None:
    """
    Display the active faraday configuration.

    Shows which file was loaded (if any) and the current values of all
    configuration sections.

    Example
    -------
    $ faraday config-show
    $ faraday config-show --yaml
    """
    cfg: FaradayConfig = ctx.obj["config"]

    if as_yaml:
        import yaml  # type: ignore[import]
        click.echo(yaml.dump(cfg.to_dict(), sort_keys=False))
        return

    path_str = str(cfg.config_file) if cfg.config_file else "<defaults>"

    click.echo(f"Configuration source: {path_str}\n")
    click.echo("Sections:")
    for section_name in ["solver", "training", "predict", "topology", "logging"]:
        sub = getattr(cfg, section_name)
        click.echo(f"  [{section_name}]")
        for field_name in sub.__dataclass_fields__:  # type: ignore[attr-defined]
            val = getattr(sub, field_name)
            click.echo(f"    {field_name}: {val}")
        click.echo()


# ----------------------------------------------------------------------
# Entry point (used by pyproject.toml[project.scripts])
# ----------------------------------------------------------------------

cli_main = main


# ----------------------------------------------------------------------
# Standalone entry points (faraday-predict, faraday-demo)
# These bypass the click group so they work as direct console_scripts.
# ----------------------------------------------------------------------


@click.command()
@click.option(
    "-w", "--width", type=float, required=True, help="Cavity width."
)
@click.option(
    "-h", "--height", type=float, required=True, help="Cavity height."
)
@click.option(
    "-m", "--model", "model_path", type=str, default=None,
    help="Path to a trained GodTensor checkpoint.",
)
@click.option(
    "-c", "--config", "config_path", type=str, default=None,
    help="Path to YAML config file.",
    callback=_resolve_config,
    is_eager=True,
)
@click.pass_context
def predict(
    ctx: click.Context,
    width: float,
    height: float,
    model_path: str | None,
    config_path: str | None,
) -> None:
    """
    Predict E and H barcodes for a new geometry using the trained God Tensor.

    Example
    -------
    $ faraday predict --width 2.5 --height 1.2 --model god_tensor_checkpoint.pkl
    $ faraday-predict --width 2.5 --height 1.2
    """
    # Lazy import to avoid circular reference during package init
    from faraday import GodTensor

    cfg: FaradayConfig = ctx.obj["config"]

    if model_path is None:
        click.echo("[faraday predict] No model provided — creating an untrained GodTensor.")
        gt = GodTensor(n_geometries=cfg.training.n_geometries)
    else:
        click.echo(f"[faraday predict] Loading model from {model_path}")
        gt = GodTensor.load(model_path)  # type: ignore[attr-defined]

    barcode = gt.predict(w=width, h=height)  # type: ignore[attr-defined]

    click.echo(f"\nPredicted E-field barcode for geometry ({width} x {height}):")
    for birth, death in barcode:
        persistence = death - birth
        click.echo(f"  birth={birth:.4f}, death={death:.4f}, persistence={persistence:.4f}")


@click.command()
@click.option(
    "-c", "--config", "config_path", type=str, default=None,
    help="Path to YAML config file.",
    callback=_resolve_config,
    is_eager=True,
)
def demo(config_path: str | None) -> None:
    """
    Run the full Faraday demo: collect training data, learn T,
    find the God Tensor, and verify predictions against FDFD.

    Example
    -------
    $ faraday demo
    $ faraday-demo
    """
    # Lazy import to avoid circular reference during package init
    from faraday import demo as _demo_mod  # type: ignore[attr-defined]

    _demo_mod.main()


if __name__ == "__main__":
    main()
