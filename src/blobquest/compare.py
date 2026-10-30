"""Train several algorithms on the same map and seeds and summarise the results."""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from blobquest.config import Config, save_config
from blobquest.metrics import TrainingHistory
from blobquest.persistence import CONFIG_FILE, WORLD_FILE
from blobquest.plots import plot_comparison
from blobquest.trainer import Trainer
from blobquest.world import Position, World, build_world

COMPARE_ALGORITHMS: tuple[str, ...] = ("random", "q_learning", "sarsa")
COMPARISON_CSV = "comparison.csv"
COMPARISON_PLOT = "comparison.png"


@dataclass(frozen=True)
class RunResult:
    """Headline numbers of one (variant, seed) training run."""

    variant: str
    """Label of the configuration being compared (the algorithm name for ``compare``)."""
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
PERCENT_METRICS: frozenset[str] = frozenset({"greedy_success", "train_success", "repeat_mistake_rate"})

Overrides = Mapping[str, Mapping[str, Any]]
"""Config section overrides, e.g. ``{"agent": {"sensing": True}}``."""


@dataclass
class Comparison:
    """Everything a comparison produced: the shared map, per-run results and histories by variant."""

    world: World
    seeds: tuple[int, ...]
    results: list[RunResult]
    histories: dict[str, list[TrainingHistory]]


def variant_config(config: Config, overrides: Overrides, seed: int) -> Config:
    """``config`` with a variant's overrides applied and ``training.seed`` set to ``seed``."""
    sections = {name: dict(values) for name, values in overrides.items()}
    sections.setdefault("training", {})["seed"] = seed
    return config.with_overrides(**sections)


def _train_one(config: Config, world: World, variant: str) -> tuple[RunResult, TrainingHistory]:
    """Train one run on ``world`` and summarise it."""
    trainer = Trainer(config, world=world)
    history = trainer.train()
    stats = history.summary(config.output.moving_average_window)
    result = RunResult(
        variant=variant,
        algorithm=config.agent.algorithm,
        seed=config.training.seed,
        greedy_success=history.evaluations[-1].success_rate,
        train_success=stats["success_rate"],
        mean_reward=stats["mean_reward"],
        mean_steps_on_success=stats["mean_steps_on_success"],
        repeat_mistake_rate=trainer.mistakes.repeat_rate(),
    )
    return result, history


def _train_one_in_worker(
    config: Config, grid: np.ndarray, start: Position, variant: str
) -> tuple[RunResult, TrainingHistory]:
    """Worker-process entry point: ``World`` holds a mappingproxy and cannot be pickled, so rebuild it."""
    return _train_one(config, World.from_grid(grid, start, config.world.max_boosters), variant)


def run_variants(
    config: Config,
    variants: Mapping[str, Overrides],
    num_seeds: int,
    logger: Callable[[str], None] | None = None,
    jobs: int = 1,
) -> Comparison:
    """Train every variant with seeds ``training.seed .. training.seed + num_seeds - 1``.

    The map is built once from ``config``, so all runs share it; each run gets a
    fresh agent. With ``jobs > 1`` runs go to worker processes; every run is
    seeded on its own, so the results do not depend on ``jobs``.
    """
    if num_seeds < 1:
        raise ValueError("number of seeds must be >= 1")
    if jobs < 1:
        raise ValueError("jobs must be >= 1")
    world = build_world(config.world)
    seeds = tuple(config.training.seed + i for i in range(num_seeds))
    tasks = [
        (variant_config(config, overrides, seed), variant) for variant, overrides in variants.items() for seed in seeds
    ]
    if jobs == 1:
        outputs: Iterable[tuple[RunResult, TrainingHistory]] = (_train_one(c, world, v) for c, v in tasks)
        return _collect(world, seeds, outputs, logger)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_train_one_in_worker, c, world.grid, world.start, v) for c, v in tasks]
        return _collect(world, seeds, (f.result() for f in futures), logger)


def _collect(
    world: World,
    seeds: tuple[int, ...],
    outputs: Iterable[tuple[RunResult, TrainingHistory]],
    logger: Callable[[str], None] | None,
) -> Comparison:
    """Gather run outputs (in task order) into a :class:`Comparison`, logging each one."""
    results: list[RunResult] = []
    histories: dict[str, list[TrainingHistory]] = {}
    for result, history in outputs:
        results.append(result)
        histories.setdefault(result.variant, []).append(history)
        if logger is not None:
            logger(
                f"{result.variant:<18} seed {result.seed:>4}: greedy success {result.greedy_success:6.1%}"
                f" | train success {result.train_success:6.1%} | reward {result.mean_reward:8.2f}"
            )
    return Comparison(world, seeds, results, histories)


def run_comparison(
    config: Config,
    num_seeds: int,
    algorithms: Sequence[str] = COMPARE_ALGORITHMS,
    logger: Callable[[str], None] | None = None,
) -> Comparison:
    """Train every algorithm on the same map and seeds (see :func:`run_variants`)."""
    return run_variants(config, {a: {"agent": {"algorithm": a}} for a in algorithms}, num_seeds, logger)


def _mean_std(values: list[float], percent: bool) -> str:
    """``mean ± std`` over seeds, ignoring NaNs; ``-`` if nothing is defined."""
    arr = np.array(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return "-"
    mean, std = float(arr.mean()), float(arr.std())
    return f"{mean:.1%} ± {std:.1%}" if percent else f"{mean:.1f} ± {std:.1f}"


def summary_rows(comparison: Comparison, first_column: str = "algorithm") -> list[list[str]]:
    """Header plus one row per variant, each metric as ``mean ± std`` across seeds."""
    rows = [[first_column, *(label for _, label in METRICS)]]
    for variant in comparison.histories:
        runs = [r for r in comparison.results if r.variant == variant]
        rows.append([variant, *(_mean_std([getattr(r, k) for r in runs], k in PERCENT_METRICS) for k, _ in METRICS)])
    return rows


def format_results(comparison: Comparison, first_column: str = "algorithm") -> str:
    """Aligned text table of :func:`summary_rows`."""
    rows = summary_rows(comparison, first_column)
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

    def line(row: list[str]) -> str:
        # Variant name left-aligned, numbers right-aligned.
        return "  ".join(
            [row[0].ljust(widths[0]), *(cell.rjust(w) for cell, w in zip(row[1:], widths[1:], strict=True))]
        )

    return "\n".join([line(rows[0]), "  ".join("-" * w for w in widths), *(line(row) for row in rows[1:])])


def format_markdown(comparison: Comparison, first_column: str = "variant") -> str:
    """GitHub-flavoured markdown table of :func:`summary_rows`."""
    rows = summary_rows(comparison, first_column)
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "|".join([":---", *("---:" for _ in rows[0][1:])]) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join(lines)


def save_comparison(comparison: Comparison, config: Config, out_dir: Path) -> list[Path]:
    """Write the config, map, per-run results CSV and the comparison plot into ``out_dir``."""
    save_config(config, out_dir / CONFIG_FILE)
    (out_dir / WORLD_FILE).write_text("\n".join(comparison.world.to_layout()) + "\n", encoding="utf-8")
    csv_path = out_dir / COMPARISON_CSV
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(RESULT_FIELDS)
        writer.writerows(astuple(r) for r in comparison.results)
    plot = plot_comparison(comparison.histories, config.output, out_dir / COMPARISON_PLOT, "Algorithms on the same map")
    return [out_dir / CONFIG_FILE, out_dir / WORLD_FILE, csv_path, plot]
