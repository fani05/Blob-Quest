"""Per-episode schedules for exploration rate (epsilon) and learning rate (alpha)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from blobquest.config import ScheduleConfig


class Schedule(Protocol):
    """Anything that maps an episode index to a value."""

    def __call__(self, episode: int) -> float:
        """Value to use during ``episode`` (0-based)."""
        ...


@dataclass(frozen=True)
class ConstantSchedule:
    """Always returns ``value``."""

    value: float

    def __call__(self, episode: int) -> float:
        """Return the constant value."""
        return self.value


@dataclass(frozen=True)
class LinearSchedule:
    """Moves linearly from ``start`` to ``end`` over ``duration`` episodes, then holds."""

    start: float
    end: float
    duration: int

    def __call__(self, episode: int) -> float:
        """Interpolated value for ``episode``."""
        fraction = min(max(episode, 0) / self.duration, 1.0)
        return self.start + fraction * (self.end - self.start)


@dataclass(frozen=True)
class ExponentialSchedule:
    """Decays geometrically from ``start`` toward ``end`` by ``rate`` each episode."""

    start: float
    end: float
    rate: float

    def __call__(self, episode: int) -> float:
        """Decayed value for ``episode``."""
        return self.end + (self.start - self.end) * self.rate ** max(episode, 0)


def make_schedule(config: ScheduleConfig, total_episodes: int) -> Schedule:
    """Build the schedule described by ``config`` for a run of ``total_episodes``."""
    if config.kind == "constant":
        return ConstantSchedule(config.start)
    if config.kind == "linear":
        duration = max(1, round(config.decay_fraction * total_episodes))
        return LinearSchedule(config.start, config.end, duration)
    if config.kind == "exponential":
        return ExponentialSchedule(config.start, config.end, config.decay_rate)
    raise ValueError(f"unknown schedule kind: {config.kind}")
