"""Train several algorithms on the same map and seeds and summarise the results."""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import numpy as np

from blobquest.config import Config, save_config
from blobquest.metrics import TrainingHistory
from blobquest.persistence import CONFIG_FILE, WORLD_FILE
from blobquest.plots import plot_comparison
from blobquest.trainer import Trainer
from blobquest.world import World, build_world

COMPARE_ALGORITHMS: tuple[str, ...] = ("random", "q_learning", "sarsa")
COMPARISON_CSV = "comparison.csv"
COMPARISON_PLOT = "comparison.png"


@dataclass(frozen=True)
class RunResult:
    """Headline numbers of one (algorithm, seed) training run."""

    algorithm: str
    seed: int
    greedy_success: float
    """Success rate of the greedy policy at the final evaluation."""
    train_success: float
    """Training success rate over the last ``moving_average_window`` episodes."""
    mean_reward: float
    mean_steps_on_success: float
    repeat_mistake_rate: float


RESULT_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(RunResult))
METRICS: tuple[tuple[str, str], ...] = (
    ("greedy_success", "greedy success"),
    ("train_success", "train success"),
    ("mean_reward", "reward"),
    ("mean_steps_on_success", "steps to goal"),
    ("repeat_mistake_rate", "repeat mistakes"),
)


@dataclass
class Comparison:
    """Everything a comparison produced: the shared map, per-run results and histories."""

    world: World
    seeds: tuple[int, ...]
    results: list[RunResult]
    histories: dict[str, list[TrainingHistory]]


def run_comparison(
    config: Config,
    num_seeds: int,
    algorithms: Sequence[str] = COMPARE_ALGORITHMS,
    logger: Callable[[str], None] | None = None,
) -> Comparison:
    """Train every algorithm with seeds ``training.seed .. training.seed + num_seeds - 1``.

    The map is built once, so all runs share it; each run gets a fresh agent.
    """
    if num_seeds < 1:
        raise ValueError("number of seeds must be >= 1")
    world = build_world(config.world)
    seeds = tuple(config.training.seed + i for i in range(num_seeds))
    window = config.output.moving_average_window
    results: list[RunResult] = []
    histories: dict[str, list[TrainingHistory]] = {}
    for algorithm in algorithms:
        for seed in seeds:
            run_config = config.with_overrides(agent={"algorithm": algorithm}, training={"seed": seed})
            trainer = Trainer(run_config, world=world)
            history = trainer.train()
            stats = history.summary(window)
            result = RunResult(
                algorithm=algorithm,
                seed=seed,
                greedy_success=history.evaluations[-1].success_rate,
                train_success=stats["success_rate"],
                mean_reward=stats["mean_reward"],
                mean_steps_on_success=stats["mean_steps_on_success"],
                repeat_mistake_rate=trainer.mistakes.repeat_rate(),
            )
            results.append(result)
            histories.setdefault(algorithm, []).append(history)
            if logger is not None:
                logger(
                    f"{algorithm:<10} seed {seed:>4}: greedy success {result.greedy_success:6.1%}"
                    f" | train success {result.train_success:6.1%} | reward {result.mean_reward:8.2f}"
                )
    return Comparison(world, seeds, results, histories)


def _mean_std(values: list[float], percent: bool) -> str:
    """``mean ± std`` over seeds, ignoring NaNs; ``-`` if nothing is defined."""
    arr = np.array(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return "-"
    mean, std = float(arr.mean()), float(arr.std())
    return f"{mean:.1%} ± {std:.1%}" if percent else f"{mean:.1f} ± {std:.1f}"


def format_results(comparison: Comparison) -> str:
    """Text table with one row per algorithm, each cell ``mean ± std`` across seeds."""
    percent = {"greedy_success", "train_success", "repeat_mistake_rate"}
    header = ["algorithm", *(label for _, label in METRICS)]
    rows = [header]
    for algorithm in comparison.histories:
        runs = [r for r in comparison.results if r.algorithm == algorithm]
        rows.append([algorithm, *(_mean_std([getattr(r, key) for r in runs], key in percent) for key, _ in METRICS)])
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]

    def line(row: list[str]) -> str:
        # Algorithm name left-aligned, numbers right-aligned.
        return "  ".join(
            [row[0].ljust(widths[0]), *(cell.rjust(w) for cell, w in zip(row[1:], widths[1:], strict=True))]
        )

    return "\n".join([line(rows[0]), "  ".join("-" * w for w in widths), *(line(row) for row in rows[1:])])


def save_comparison(comparison: Comparison, config: Config, out_dir: Path) -> list[Path]:
    """Write the config, map, per-run results CSV and the comparison plot into ``out_dir``."""
    save_config(config, out_dir / CONFIG_FILE)
    (out_dir / WORLD_FILE).write_text("\n".join(comparison.world.to_layout()) + "\n", encoding="utf-8")
    csv_path = out_dir / COMPARISON_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(RESULT_FIELDS)
        writer.writerows(astuple(r) for r in comparison.results)
    plot = plot_comparison(comparison.histories, config.output, out_dir / COMPARISON_PLOT)
    return [out_dir / CONFIG_FILE, out_dir / WORLD_FILE, csv_path, plot]
