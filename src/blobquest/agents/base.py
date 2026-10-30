"""Shared machinery for tabular temporal-difference agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from blobquest.memory import Transition


class TabularAgent(ABC):
    """Epsilon-greedy agent backed by a ``(num_states, num_actions)`` Q-table.

    Subclasses only decide how to estimate the value of the next state
    (:meth:`bootstrap`); the TD update itself is shared.
    """

    name: ClassVar[str] = "tabular"

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        gamma: float,
        initial_q: float,
        rng: np.random.Generator,
    ) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.initial_q = initial_q
        self.q = np.full((num_states, num_actions), initial_q, dtype=np.float64)
        self._rng = rng

    def greedy_action(self, state: int, rng: np.random.Generator | None = None) -> int:
        """Best action for ``state``; ties are broken uniformly at random."""
        rng = rng if rng is not None else self._rng
        values = self.q[state]
        best = np.flatnonzero(values == values.max())
        return int(best[0]) if best.size == 1 else int(rng.choice(best))

    def act(self, state: int, epsilon: float, rng: np.random.Generator | None = None) -> int:
        """Epsilon-greedy action: explore with probability ``epsilon``."""
        rng = rng if rng is not None else self._rng
        if epsilon > 0.0 and rng.random() < epsilon:
            return int(rng.integers(self.num_actions))
        return self.greedy_action(state, rng)

    @abstractmethod
    def bootstrap(self, transition: Transition) -> float:
        """Estimated value of the next state used in the TD target."""

    def update(self, transition: Transition, alpha: float) -> float:
        """Apply one TD update and return the TD error before the update."""
        target = transition.reward
        if not transition.terminated:
            target += self.gamma * self.bootstrap(transition)
        td_error = target - self.q[transition.state, transition.action]
        self.q[transition.state, transition.action] += alpha * td_error
        return float(td_error)
