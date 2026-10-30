"""A baseline that wanders at random and never learns."""

from __future__ import annotations

import numpy as np

from blobquest.agents.base import TabularAgent
from blobquest.memory import Transition


class RandomAgent(TabularAgent):
    """Picks a uniformly random action every step, whatever epsilon says.

    Its Q-table stays at ``initial_q`` forever, so greedy rollouts are random
    too (all actions tie). Useful as a floor to compare learning agents against.
    """

    name = "random"

    def act(self, state: int, epsilon: float, rng: np.random.Generator | None = None) -> int:
        """Uniformly random action; ``state`` and ``epsilon`` are ignored."""
        rng = rng if rng is not None else self._rng
        return int(rng.integers(self.num_actions))

    def bootstrap(self, transition: Transition) -> float:
        """Never used for learning; reports the untouched initial value."""
        return self.initial_q

    def update(self, transition: Transition, alpha: float) -> float:
        """Do nothing: the random agent never learns."""
        return 0.0
