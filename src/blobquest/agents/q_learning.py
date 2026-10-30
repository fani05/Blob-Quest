"""Off-policy Q-learning."""

from __future__ import annotations

from blobquest.agents.base import TabularAgent
from blobquest.memory import Transition


class QLearningAgent(TabularAgent):
    """Bootstraps from the best next action: ``max_a' Q(s', a')``.

    It learns the value of the greedy policy even while exploring, which makes
    it a good fit for replaying old transitions from memory.
    """

    name = "q_learning"

    def bootstrap(self, transition: Transition) -> float:
        """Value of the greedy action in the next state."""
        return float(self.q[transition.next_state].max())
