"""Saving and loading runs: config, Q-table, map, metrics and summary."""

from __future__ import annotations

import csv
from dataclasses import astuple
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from blobquest.agents import TabularAgent
from blobquest.config import Config, load_config, save_config
from blobquest.metrics import EPISODE_FIELDS, EVAL_FIELDS, TrainingHistory
from blobquest.mistakes import MISTAKE_FIELDS, MistakeLog
from blobquest.world import World

CONFIG_FILE = "config.yaml"
QTABLE_FILE = "q_table.npz"
WORLD_FILE = "world.txt"
EPISODES_FILE = "episodes.csv"
EVALS_FILE = "evaluations.csv"
MISTAKES_FILE = "mistakes.csv"
SUMMARY_FILE = "summary.yaml"
PLOTS_DIR = "plots"


def create_run_dir(runs_dir: str | Path, run_name: str, now: datetime | None = None) -> Path:
    """Create ``runs_dir/<run_name>_<timestamp>`` (suffixing if it already exists)."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    base = Path(runs_dir) / f"{run_name}_{stamp}"
    candidate, suffix = base, 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{suffix}")
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def save_q_table(agent: TabularAgent, path: Path) -> None:
    """Store the Q-table plus the metadata needed to rebuild the agent."""
    np.savez_compressed(path, q=agent.q, algorithm=np.array(agent.name), gamma=np.array(agent.gamma))


def load_q_table(path: Path) -> np.ndarray:
    """Load the Q-table array saved by :func:`save_q_table`."""
    with np.load(path) as data:
        return data["q"].copy()


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    """Write ``rows`` to ``path`` with ``header``."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def save_run(
    run_dir: Path,
    config: Config,
    world: World,
    agent: TabularAgent,
    history: TrainingHistory,
    summary: dict[str, Any],
    mistakes: MistakeLog | None = None,
) -> None:
    """Write every artefact of a finished run into ``run_dir``."""
    save_config(config, run_dir / CONFIG_FILE)
    (run_dir / WORLD_FILE).write_text("\n".join(world.to_layout()) + "\n", encoding="utf-8")
    save_q_table(agent, run_dir / QTABLE_FILE)
    np.save(run_dir / "visits.npy", history.visits)
    _write_csv(run_dir / EPISODES_FILE, EPISODE_FIELDS, [astuple(r) for r in history.episodes])
    _write_csv(run_dir / EVALS_FILE, EVAL_FIELDS, [astuple(r) for r in history.evaluations])
    if mistakes is not None:
        _write_csv(run_dir / MISTAKES_FILE, MISTAKE_FIELDS, [astuple(m) for m in mistakes.mistakes])
    with (run_dir / SUMMARY_FILE).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)


def load_run(run_dir: str | Path) -> tuple[Config, World, np.ndarray]:
    """Load the config, exact map and Q-table of a saved run."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    config = load_config(run_dir / CONFIG_FILE)
    layout = (run_dir / WORLD_FILE).read_text(encoding="utf-8").splitlines()
    world = World.from_layout(layout, config.world.max_boosters)
    return config, world, load_q_table(run_dir / QTABLE_FILE)
