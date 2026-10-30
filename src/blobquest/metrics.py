"""Per-episode records and summary statistics."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np

from blobquest.env import Outcome


@dataclass(frozen=True)
class EpisodeRecord:
    """What happened in one training episode."""

    episode: int
    total_reward: float
    steps: int
    outcome: str
    final_energy: int
    epsilon: float
    alpha: float
    trap_hits: int
    boosters_collected: int
    resets: int
    wall_bumps: int
    mean_abs_td_error: float


@dataclass(frozen=True)
class EvalRecord:
    """Greedy (epsilon = 0) evaluation taken during training."""

    episode: int
    success_rate: float
    mean_reward: float
    mean_steps: float


EPISODE_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(EpisodeRecord))
EVAL_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(EvalRecord))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over ``window`` items; early items average what exists so far."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    cumsum = np.cumsum(np.insert(values, 0, 0.0))
    idx = np.arange(1, values.size + 1)
    lo = np.maximum(idx - window, 0)
    return (cumsum[idx] - cumsum[lo]) / (idx - lo)


@dataclass
class TrainingHistory:
    """Everything recorded during a training run."""

    rows: int
    cols: int
    episodes: list[EpisodeRecord] = field(default_factory=list)
    evaluations: list[EvalRecord] = field(default_factory=list)
    visits: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.visits = np.zeros((self.rows, self.cols), dtype=np.int64)

    def column(self, name: str) -> np.ndarray:
        """Values of one :class:`EpisodeRecord` field across all episodes."""
        if name not in EPISODE_FIELDS:
            raise KeyError(f"unknown episode field {name!r}")
        return np.array([getattr(rec, name) for rec in self.episodes])

    def outcome_fractions(self, window: int) -> dict[str, np.ndarray]:
        """Rolling share of each final outcome over the last ``window`` episodes."""
        outcomes = self.column("outcome") if self.episodes else np.array([], dtype=str)
        return {
            o.value: moving_average((outcomes == o.value).astype(np.float64), window)
            for o in Outcome
            if o is not Outcome.RUNNING
        }

    def summary(self, last_n: int) -> dict[str, float]:
        """Headline numbers over the final ``last_n`` episodes."""
        recent = self.episodes[-last_n:]
        if not recent:
            return {}
        rewards = np.array([r.total_reward for r in recent])
        steps = np.array([r.steps for r in recent])
        success = np.array([r.outcome == Outcome.GOAL.value for r in recent])
        result = {
            "episodes": float(len(self.episodes)),
            "window": float(len(recent)),
            "success_rate": float(success.mean()),
            "mean_reward": float(rewards.mean()),
            "mean_steps": float(steps.mean()),
            "mean_steps_on_success": float(steps[success].mean()) if success.any() else float("nan"),
        }
        if self.evaluations:
            result["final_greedy_success_rate"] = self.evaluations[-1].success_rate
        return result
