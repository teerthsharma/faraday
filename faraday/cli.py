# Copyright (c) 2026 Teerth Sharma. All rights reserved.
# Attribution: Computational Faraday Tensor by Teerth Sharma (https://github.com/teerthsharma)
#
"""
faraday.cli — Click-based command-line interface.

Sub-commands
------------
solve         Run FDFD cavity solve for a given geometry.
train         Collect training data and train the God Tensor.
predict       Predict E/H topology for a new geometry using a trained model.
demo          End-to-end demo: solve, train, find fixed point, predict, validate.
config-show   Display the active configuration.
"""

from __future__ import annotations

import click

from faraday.config import FaradayConfig

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _resolve_config(
    ctx: click.Context, _param: click.Option, value: str | None
) -> FaradayConfig:
    cfg = FaradayConfig.load() if value is None else FaradayConfig.from_yaml(value)
    ctx.obj = getattr(ctx, "obj", {}) or {}
    ctx.obj["config"] = cfg
    return cfg


# ----------------------------------------------------------------------
# Group / main
# ----------------------------------------------------------------------


@click.group()
@click.version_option(package_name="faradayFDTD")
def main() -> None:
    """Computational Faraday Tensor — topology-fixed-point field predictor."""


# ----------------------------------------------------------------------
# solve
# ----------------------------------------------------------------------


@main.command()
@click.option(
    "-w", "--width", type=float, default=2.0, help="Cavity width (normalised units)."
)
@click.option("-h", "--height", type=float, default=1.0, help="Cavity height.")
@click.option(
    "-n",
    "--n-modes",
    "n_modes",
    type=int,
    default=None,
    help="Number of modes to compute. Defaults to config value.",
)
@click.option(
    "--nx", type=int, default=None, help="Grid points in x. Defaults to config value."
)
@click.option(
    "--ny", type=int, default=None, help="Grid points in y. Defaults to config value."
)
@click.option(
    "-c",
    "--config",
    "config_path",
    type=str,
    default=None,
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
    """Run the FDFD cavity solver for a given geometry.

    Computes TM eigenmodes and prints the first few resonant wave-numbers.

    Example::

        $ faraday solve --width 2.0 --height 1.0 --n-modes 4
    """
    from faraday import CavityGeometry, CavityShape, solve_cavity_modes

    cfg: FaradayConfig = ctx.obj["config"]
    nx_val = nx if nx is not None else cfg.solver.nx
    ny_val = ny if ny is not None else cfg.solver.ny
    n_modes_val = n_modes if n_modes is not None else cfg.solver.n_modes

    geometry = CavityGeometry(shape=CavityShape.RECTANGULAR, dims=(width, height))
    click.echo(
        f"[faraday solve] geometry=({width} x {height}), "
        f"nx={nx_val}, ny={ny_val}, modes={n_modes_val}"
    )

    result = solve_cavity_modes(
        geometry, nx=nx_val, ny=ny_val, num_modes=n_modes_val
    )
    click.echo(f"\nResonant wave-numbers k for first {result['num_modes_found']} modes:")
    for i, kk in enumerate(result["k_values"][:n_modes_val], 1):
        click.echo(f"  mode {i}: k = {kk:.6f}, k^2 = {kk * kk:.6f}")


# ----------------------------------------------------------------------
# train
# ----------------------------------------------------------------------


@main.command()
@click.option(
    "-n",
    "--n-geometries",
    type=int,
    default=None,
    help="Number of random geometries to generate. Defaults to config value.",
)
@click.option(
    "--iters",
    type=int,
    default=None,
    help="Fixed-point iterations. Defaults to config value.",
)
@click.option(
    "--save",
    type=str,
    default=None,
    help="Save the trained GodTensor to this pickle path.",
)
@click.option(
    "--seed", type=int, default=42, help="RNG seed for reproducibility."
)
@click.option(
    "-c",
    "--config",
    "config_path",
    type=str,
    default=None,
    help="Path to YAML config file.",
    callback=_resolve_config,
    is_eager=True,
)
@click.pass_context
def train(
    ctx: click.Context,
    n_geometries: int | None,
    iters: int | None,
    save: str | None,
    seed: int,
    config_path: str | None,
) -> None:
    """Collect training data and train the God Tensor.

    Runs FDFD across a sweep of random rectangular geometries, builds
    barcodes, projects to the manifold, and finds the spectral fixed
    point.

    Example::

        $ faraday train --n-geometries 30 --iters 200 --save gt.pkl
    """
    from faraday import GodTensor

    cfg: FaradayConfig = ctx.obj["config"]
    n_geo = n_geometries if n_geometries is not None else cfg.training.n_geometries
    iters_val = iters if iters is not None else cfg.training.iters

    click.echo(f"[faraday train] n_geometries={n_geo}, iters={iters_val}, seed={seed}")

    gt = GodTensor(n_geometries=n_geo)
    click.echo("[1/3] Collecting training data...")
    gt.collect_training_data(
        nx=cfg.solver.nx,
        ny=cfg.solver.ny,
        num_modes=cfg.solver.n_modes,
        seed=seed,
    )
    click.echo(f"     ✓ {len(gt.samples)} samples collected")

    click.echo("[2/3] Learning coupling operator T...")
    gt.learn_T()

    click.echo(f"[3/3] Finding spectral fixed point ({iters_val} iters)...")
    gt.find_fixed_point(iters=iters_val, tol=cfg.training.tol)

    click.echo(
        f"\n     ✓ god_score = {gt.god_score():.4f} | "
        f"converged = {gt.fixed_point_converged}"
    )

    if save is not None:
        gt.save(save)
        click.echo(f"\nSaved trained GodTensor to: {save}")


# ----------------------------------------------------------------------
# predict
# ----------------------------------------------------------------------


@main.command()
@click.option("-w", "--width", type=float, required=True, help="Cavity width.")
@click.option("-h", "--height", type=float, required=True, help="Cavity height.")
@click.option(
    "-m",
    "--model",
    "model_path",
    type=str,
    default=None,
    help="Path to a trained GodTensor checkpoint. If omitted, an in-memory model is trained.",
)
@click.option(
    "-c",
    "--config",
    "config_path",
    type=str,
    default=None,
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
    """Predict E and H barcodes for a new geometry.

    Example::

        $ faraday predict --width 2.5 --height 1.2 --model gt.pkl
    """
    from faraday import GodTensor
    from faraday.predict import predict_eh_barcode

    cfg: FaradayConfig = ctx.obj["config"]

    if model_path is None:
        click.echo("[faraday predict] No model provided — training a small one in-memory.")
        gt = GodTensor(n_geometries=cfg.training.n_geometries)
        gt.collect_training_data(
            nx=cfg.solver.nx, ny=cfg.solver.ny, num_modes=cfg.solver.n_modes, seed=42
        )
        gt.learn_T()
        gt.find_fixed_point(iters=cfg.training.iters)
    else:
        click.echo(f"[faraday predict] Loading model from {model_path}")
        gt = GodTensor.load(model_path)

    pred = predict_eh_barcode(gt, (width, height), shape="rect")

    click.echo(
        f"\nPrediction for geometry ({width} x {height}):\n"
        f"  KNN E Betti-0:    {pred['knn_e_fingerprint']['betti_0']:.0f}\n"
        f"  KNN H Betti-0:    {pred['knn_h_fingerprint']['betti_0']:.0f}\n"
        f"  god_distance E:   {pred['god_distance_e']:.4f}\n"
        f"  god_distance H:   {pred['god_distance_h']:.4f}\n"
        f"  coupling_score:   {pred['coupling_score']:.4f}"
    )


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


@main.command("demo")
@click.option("-w", "--width", type=float, default=2.0, help="Test geometry width.")
@click.option("-h", "--height", type=float, default=1.0, help="Test geometry height.")
@click.option(
    "--n-geometries", type=int, default=10, help="Number of training geometries."
)
@click.option("--seed", type=int, default=42, help="RNG seed for reproducibility.")
def demo(width: float, height: float, n_geometries: int, seed: int) -> None:
    """Run the full Faraday demo.

    Trains a small GodTensor, finds the spectral fixed point, and
    predicts the topological signature of a held-out geometry.

    Example::

        $ faraday demo --width 2.5 --height 1.2 --n-geometries 20
    """
    from faraday import GodTensor
    from faraday.predict import predict_eh_barcode

    click.echo(
        f"[faraday demo] training {n_geometries} geometries (seed={seed}) ..."
    )
    gt = GodTensor(n_geometries=n_geometries)
    gt.collect_training_data(nx=20, ny=20, num_modes=3, seed=seed)
    gt.learn_T()
    gt.find_fixed_point(iters=200)

    pred = predict_eh_barcode(gt, (width, height), shape="rect")
    click.echo(
        f"\nPrediction for ({width} x {height}):\n"
        f"  knn_e_betti0    = {pred['knn_e_fingerprint']['betti_0']:.0f}\n"
        f"  knn_h_betti0    = {pred['knn_h_fingerprint']['betti_0']:.0f}\n"
        f"  god_distance_e  = {pred['god_distance_e']:.4f}\n"
        f"  coupling_score  = {pred['coupling_score']:.4f}"
    )


# ---------------------------------------------------------------------------
# config-show
# ---------------------------------------------------------------------------


@main.command("config-show")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=str,
    default=None,
    help="Path to YAML config file. Defaults to first found in search path.",
    callback=_resolve_config,
    is_eager=True,
)
@click.option(
    "--yaml",
    "as_yaml",
    is_flag=True,
    default=False,
    help="Output in YAML format instead of human-readable.",
)
@click.pass_context
def config_show(
    ctx: click.Context, config_path: str | None, as_yaml: bool
) -> None:
    """Display the active faraday configuration.

    Example::

        $ faraday config-show
        $ faraday config-show --yaml
    """
    cfg: FaradayConfig = ctx.obj["config"]

    if as_yaml:
        import yaml

        click.echo(yaml.dump(cfg.to_dict(), sort_keys=False))
        return

    path_str = str(cfg.config_file) if cfg.config_file else "<defaults>"
    click.echo(f"Configuration source: {path_str}\n")
    click.echo("Sections:")
    for section_name in ["solver", "training", "predict", "topology", "logging"]:
        sub = getattr(cfg, section_name)
        click.echo(f"  [{section_name}]")
        for field_name in sub.__dataclass_fields__:
            click.echo(f"    {field_name}: {getattr(sub, field_name)}")
        click.echo()


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

cli_main = main

if __name__ == "__main__":
    main()
