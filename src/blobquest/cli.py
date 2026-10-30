"""Command-line interface: ``blobquest train | evaluate | replay | compare | show-map``."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from blobquest.compare import (
    COMPARE_ALGORITHMS,
    COMPARISON_CSV,
    COMPARISON_PLOT,
    format_results,
    run_comparison,
    save_comparison,
)
from blobquest.config import ALGORITHMS, Config, ConfigError, load_config
from blobquest.env import ACTION_NAMES, Outcome
from blobquest.persistence import PLOTS_DIR, create_run_dir, load_explored, load_run, save_run
from blobquest.plots import save_all_plots
from blobquest.render import (
    FOG_LEGEND,
    LEGEND,
    REPLAY_BLOB_CHAR,
    energy_bar,
    policy_and_values,
    render_fog,
    render_replay,
    render_world,
    side_by_side,
)
from blobquest.state import energy_bucket
from blobquest.trainer import GreedyStep, Trainer
from blobquest.world import WorldGenerationError, build_world

DEFAULT_CONFIG = Path("configs/default.yaml")
CLEAR_SCREEN = "\033[H\033[J"
REPLAY_LEGEND = (
    f"{REPLAY_BLOB_CHAR} blob  * path so far  G goal  # wall  T trap  X pit  D drain  E energy  S speed  R reset"
)


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
        f"\nTraining {config.agent.algorithm}{' with sensing' if config.agent.sensing else ''}"
        f" for {config.training.episodes} episodes "
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
        "failed_episodes": len(trainer.mistakes),
        "repeat_mistake_rate": trainer.mistakes.repeat_rate(),
    }
    run_dir = create_run_dir(config.output.runs_dir, config.output.run_name)
    save_run(run_dir, config, trainer.world, trainer.agent, history, summary, trainer.mistakes)
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
            trainer.mistakes,
        )

    print("\nGreedy policy (best-rested trained state per cell, no boosters used):")
    print(render_world(trainer.world, policy=policy))
    print(f"\nGreedy run: {rollout.outcome.value} in {rollout.steps} steps, reward {rollout.total_reward:.1f}")
    print(render_world(trainer.world, path=rollout.path))
    print()
    fog = render_fog(trainer.world, history.explored, rollout.path)
    print(side_by_side(fog, render_world(trainer.world), "What the blob saw", "True map"))
    print(FOG_LEGEND)
    print("\nDeadliest cells:")
    print(trainer.mistakes.format_deadliest(trainer.world))
    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key:<28} {value:.3f}" if isinstance(value, float) else f"  {key:<28} {value}")
    print(f"\nRun saved to {run_dir}")
    return 0


def _load_trained(run: Path, seed: int | None) -> Trainer:
    """Trainer holding a saved run's map, Q-table and explored cells, ready for greedy play."""
    config, world, q = load_run(run)
    config = config.with_overrides(training={"seed": seed}) if seed is not None else config
    trainer = Trainer(config, world=world)
    if trainer.agent.q.shape != q.shape:
        raise ConfigError(f"saved Q-table shape {q.shape} does not match config {trainer.agent.q.shape}")
    trainer.agent.q[:] = q
    explored = load_explored(run)
    if explored is not None:
        trainer.history.explored[:] = explored  # what the blob can sense, if sensing is on
    return trainer


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Replay a saved run's greedy policy and print its path."""
    trainer = _load_trained(args.run, args.seed)
    world = trainer.world
    successes = 0
    for i in range(args.episodes):
        rollout = trainer.final_rollout()
        successes += rollout.outcome is Outcome.GOAL
        print(f"episode {i + 1}: {rollout.outcome.value:<9} steps {rollout.steps:>4} reward {rollout.total_reward:.1f}")
        if i == 0:
            print(render_world(world, path=rollout.path))
    print(f"\nGreedy success rate: {successes}/{args.episodes}")
    return 0


def _effects(step: GreedyStep) -> str:
    """Active drain/speed effects after ``step``, e.g. ``drain (3 left), speed (1 left)``."""
    effects = (("drain", step.drain_left), ("speed", step.speed_left))
    active = [f"{name} ({left} left)" for name, left in effects if left]
    return ", ".join(active) or "none"


def _show_frame(text: str, delay: float) -> None:
    """Print one replay frame, redrawing in place on a terminal, then wait ``delay`` seconds."""
    if delay > 0 and sys.stdout.isatty():
        print(CLEAR_SCREEN, end="")
    print(text, flush=True)
    if delay > 0:
        time.sleep(delay)


def cmd_replay(args: argparse.Namespace) -> int:
    """Play one greedy episode of a saved run step by step in the terminal."""
    if args.delay < 0:
        raise ConfigError("--delay must be >= 0")
    trainer = _load_trained(args.run, args.seed)
    world, maximum = trainer.world, trainer.config.energy.maximum
    actions = ACTION_NAMES[: trainer.env.num_actions]
    limit = trainer.config.training.max_steps_per_episode
    trail = [world.start]
    total = 0.0
    start = trainer.config.energy.initial
    header = f"step {0:>3}/{limit}  start\nenergy {energy_bar(start, maximum)}  effects: none"
    _show_frame(f"{header}\n{render_replay(world, world.start)}\n", args.delay)
    for i, step in enumerate(trainer.replay_steps(), start=1):
        total += step.result.reward
        trail.extend(step.cells_entered)
        events = ", ".join(e.value for e in step.result.events) or "-"
        header = (
            f"step {i:>3}/{limit}  {actions[step.action]:<10} reward {step.result.reward:+7.1f}  total {total:+8.1f}\n"
            f"energy {energy_bar(step.energy, maximum)}  effects: {_effects(step)}  events: {events}"
        )
        _show_frame(f"{header}\n{render_replay(world, step.position, trail)}\n", args.delay)
    print(f"Outcome: {step.result.outcome.value} after {i} steps, total reward {total:.1f}")
    print(REPLAY_LEGEND)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Train random, Q-learning and SARSA on one map and seeds; print a table and save a plot."""
    if args.seeds < 1:
        raise ConfigError("--seeds must be >= 1")
    config = _apply_overrides(load_config(args.config), args)
    print(f"Comparing {', '.join(COMPARE_ALGORITHMS)} over {args.seeds} seed(s), {config.training.episodes} episodes")
    comparison = run_comparison(config, args.seeds, logger=print)
    print(f"\nMap ({comparison.world.rows}x{comparison.world.cols}), seeds {list(comparison.seeds)}:")
    print(render_world(comparison.world))
    window = config.output.moving_average_window
    print(f"\nResults (mean ± std over seeds; train metrics over the last {window} episodes):")
    print(format_results(comparison))
    out_dir = create_run_dir(config.output.runs_dir, f"{config.output.run_name}_compare")
    save_comparison(comparison, config, out_dir)
    print(f"\nComparison saved to {out_dir} ({COMPARISON_PLOT}, {COMPARISON_CSV})")
    return 0


def cmd_show_map(args: argparse.Namespace) -> int:
    """Print the map a config would generate."""
    config = _apply_overrides(_load(args.config), args)
    world = build_world(config.world)
    print(render_world(world))
    print(LEGEND)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with ``train``, ``evaluate``, ``replay``, ``compare`` and ``show-map`` subcommands."""
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

    replay = sub.add_parser("replay", help="watch one greedy episode of a saved run step by step")
    replay.add_argument("run", type=Path, help="run directory, e.g. runs/blobquest_20260101-120000")
    replay.add_argument("--delay", type=float, default=0.3, help="seconds between steps (default: 0.3)")
    replay.add_argument("--seed", type=int, help="seed for greedy tie-breaking")
    replay.set_defaults(func=cmd_replay)

    compare = sub.add_parser("compare", help="train random, q_learning and sarsa on the same map and seeds")
    compare.add_argument("--config", type=Path, required=True, help="YAML config shared by every run")
    compare.add_argument("--seeds", type=int, default=5, help="seeds per algorithm, from training.seed (default: 5)")
    compare.add_argument("--episodes", type=int)
    compare.add_argument("--runs-dir", help="output root directory")
    compare.set_defaults(func=cmd_compare)

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
