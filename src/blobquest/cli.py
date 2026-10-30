"""Command-line interface: ``blobquest train | evaluate | show-map``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from blobquest.config import ALGORITHMS, Config, ConfigError, load_config
from blobquest.env import Outcome
from blobquest.persistence import PLOTS_DIR, create_run_dir, load_run, save_run
from blobquest.plots import save_all_plots
from blobquest.render import LEGEND, policy_and_values, render_world
from blobquest.state import energy_bucket
from blobquest.trainer import Trainer
from blobquest.world import WorldGenerationError, build_world

DEFAULT_CONFIG = Path("configs/default.yaml")


def _load(path: Path | None) -> Config:
    """Load ``path`` or the default config, falling back to built-in defaults."""
    if path is not None:
        return load_config(path)
    return load_config(DEFAULT_CONFIG) if DEFAULT_CONFIG.is_file() else Config()


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Fold CLI flags into the config."""
    overrides: dict[str, dict[str, Any]] = {}

    def put(section: str, key: str, value: Any) -> None:
        if value is not None:
            overrides.setdefault(section, {})[key] = value

    put("training", "episodes", getattr(args, "episodes", None))
    put("training", "seed", getattr(args, "seed", None))
    put("world", "map_seed", getattr(args, "map_seed", None))
    put("agent", "algorithm", getattr(args, "algorithm", None))
    put("movement", "diagonals", getattr(args, "diagonals", None))
    put("output", "runs_dir", getattr(args, "runs_dir", None))
    put("output", "run_name", getattr(args, "name", None))
    if getattr(args, "no_plots", False):
        put("output", "save_plots", False)
    return config.with_overrides(**overrides) if overrides else config


def _fresh_bucket(config: Config) -> int:
    """Energy bucket of a blob that has just spawned."""
    return energy_bucket(config.energy.initial, config.energy.maximum, config.energy.num_buckets)


def cmd_train(args: argparse.Namespace) -> int:
    """Train an agent, save the run and print the result."""
    config = _apply_overrides(_load(args.config), args)
    trainer = Trainer(config, logger=print)
    print(f"Map ({trainer.world.rows}x{trainer.world.cols}, {trainer.world.num_boosters} boosters):")
    print(render_world(trainer.world))
    print(LEGEND)
    print(
        f"\nTraining {config.agent.algorithm} for {config.training.episodes} episodes "
        f"({trainer.encoder.num_states:,} states x {trainer.env.num_actions} actions)\n"
    )
    history = trainer.train()

    rollout = trainer.final_rollout()
    policy, values = policy_and_values(trainer.agent, trainer.encoder, trainer.world, _fresh_bucket(config))
    summary: dict[str, Any] = {
        **history.summary(config.output.moving_average_window),
        "final_greedy_outcome": rollout.outcome.value,
        "final_greedy_steps": rollout.steps,
        "final_greedy_reward": rollout.total_reward,
    }
    run_dir = create_run_dir(config.output.runs_dir, config.output.run_name)
    save_run(run_dir, config, trainer.world, trainer.agent, history, summary)
    if config.output.save_plots:
        save_all_plots(
            history,
            trainer.world,
            policy,
            values,
            rollout.path,
            config.movement.diagonals,
            config.output,
            run_dir / PLOTS_DIR,
        )

    print("\nGreedy policy (best-rested trained state per cell, no boosters used):")
    print(render_world(trainer.world, policy=policy))
    print(f"\nGreedy run: {rollout.outcome.value} in {rollout.steps} steps, reward {rollout.total_reward:.1f}")
    print(render_world(trainer.world, path=rollout.path))
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key:<28} {value:.3f}" if isinstance(value, float) else f"  {key:<28} {value}")
    print(f"\nRun saved to {run_dir}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Replay a saved run's greedy policy and print its path."""
    config, world, q = load_run(args.run)
    config = config.with_overrides(training={"seed": args.seed}) if args.seed is not None else config
    trainer = Trainer(config, world=world)
    if trainer.agent.q.shape != q.shape:
        raise ConfigError(f"saved Q-table shape {q.shape} does not match config {trainer.agent.q.shape}")
    trainer.agent.q[:] = q
    successes = 0
    for i in range(args.episodes):
        rollout = trainer.final_rollout()
        successes += rollout.outcome is Outcome.GOAL
        print(f"episode {i + 1}: {rollout.outcome.value:<9} steps {rollout.steps:>4} reward {rollout.total_reward:.1f}")
        if i == 0:
            print(render_world(world, path=rollout.path))
    print(f"\nGreedy success rate: {successes}/{args.episodes}")
    return 0


def cmd_show_map(args: argparse.Namespace) -> int:
    """Print the map a config would generate."""
    config = _apply_overrides(_load(args.config), args)
    world = build_world(config.world)
    print(render_world(world))
    print(LEGEND)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with ``train``, ``evaluate`` and ``show-map`` subcommands."""
    parser = argparse.ArgumentParser(prog="blobquest", description="Blob Quest: tabular RL in a foggy grid world.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train an agent and save a run")
    train.add_argument("--config", type=Path, help=f"YAML config (default: {DEFAULT_CONFIG})")
    train.add_argument("--episodes", type=int)
    train.add_argument("--seed", type=int, help="training seed")
    train.add_argument("--map-seed", type=int, help="seed for random map generation")
    train.add_argument("--algorithm", choices=ALGORITHMS)
    train.add_argument("--diagonals", action=argparse.BooleanOptionalAction, default=None)
    train.add_argument("--runs-dir", help="output root directory")
    train.add_argument("--name", help="run name prefix")
    train.add_argument("--no-plots", action="store_true", help="skip PNG plots")
    train.set_defaults(func=cmd_train)

    evaluate = sub.add_parser("evaluate", help="replay a saved run greedily")
    evaluate.add_argument("run", type=Path, help="run directory, e.g. runs/blobquest_20260101-120000")
    evaluate.add_argument("--episodes", type=int, default=1)
    evaluate.add_argument("--seed", type=int, help="seed for greedy tie-breaking")
    evaluate.set_defaults(func=cmd_evaluate)

    show = sub.add_parser("show-map", help="print the map for a config")
    show.add_argument("--config", type=Path)
    show.add_argument("--map-seed", type=int)
    show.set_defaults(func=cmd_show_map)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ConfigError, WorldGenerationError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
