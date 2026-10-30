"""Transitions and a replay memory that revisits past mistakes more often."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NO_ACTION = -1
"""Placeholder ``next_action`` for terminal transitions."""


@dataclass(frozen=True, slots=True)
class Transition:
    """One step of experience, with states already encoded as integers.

    ``next_action`` is the action actually chosen in ``next_state`` (needed by
    SARSA); it is ``NO_ACTION`` when the episode terminated.
    """

    state: int
    action: int
    reward: float
    next_state: int
    next_action: int
    terminated: bool


class ReplayMemory:
    """Fixed-size ring buffer with weighted sampling.

    Transitions whose reward is at or below ``mistake_threshold`` (falling into a
    pit, hitting a trap, running out of energy) get ``mistake_weight`` times the
    sampling probability, so painful lessons are rehearsed more often.
    """

    def __init__(
        self,
        capacity: int,
        mistake_threshold: float,
        mistake_weight: float,
        rng: np.random.Generator,
    ) -> None:
        self.capacity = capacity
        self.mistake_threshold = mistake_threshold
        self.mistake_weight = mistake_weight
        self._rng = rng
        self._items: list[Transition] = []
        self._weights = np.zeros(capacity, dtype=np.float64)
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def num_mistakes(self) -> int:
        """How many stored transitions count as mistakes."""
        return int(np.count_nonzero(self._weights[: len(self)] > 1.0))

    def add(self, transition: Transition) -> None:
        """Store ``transition``, overwriting the oldest one when full."""
        weight = self.mistake_weight if transition.reward <= self.mistake_threshold else 1.0
        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._cursor] = transition
        self._weights[self._cursor] = weight
        self._cursor = (self._cursor + 1) % self.capacity

    def sample(self, count: int) -> list[Transition]:
        """Draw ``count`` transitions with replacement, weighted toward mistakes."""
        size = len(self)
        if size == 0 or count <= 0:
            return []
        weights = self._weights[:size]
        indices = self._rng.choice(size, size=count, replace=True, p=weights / weights.sum())
        return [self._items[i] for i in indices]
