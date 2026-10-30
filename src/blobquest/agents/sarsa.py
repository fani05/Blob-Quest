"""On-policy SARSA."""

from __future__ import annotations

from blobquest.agents.base import TabularAgent
from blobquest.memory import NO_ACTION, Transition


class SarsaAgent(TabularAgent):
    """Bootstraps from the action actually taken next: ``Q(s', a')``.

    Because exploration is baked into its targets, SARSA tends to learn safer
    routes that keep a margin from pits while epsilon is still high.
    """

    name = "sarsa"

    def bootstrap(self, transition: Transition) -> float:
        """Value of the action that was actually chosen in the next state."""
        if transition.next_action == NO_ACTION:
            raise ValueError("SARSA needs next_action for non-terminal transitions")
        return float(self.q[transition.next_state, transition.next_action])
