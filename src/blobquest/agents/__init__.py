"""Tabular reinforcement-learning agents."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from blobquest.agents.base import TabularAgent
from blobquest.agents.q_learning import QLearningAgent
from blobquest.agents.sarsa import SarsaAgent
from blobquest.config import AgentConfig

AGENTS: Mapping[str, type[TabularAgent]] = MappingProxyType(
    {QLearningAgent.name: QLearningAgent, SarsaAgent.name: SarsaAgent}
)


def make_agent(config: AgentConfig, num_states: int, num_actions: int, rng: np.random.Generator) -> TabularAgent:
    """Instantiate the agent named by ``config.algorithm``."""
    try:
        cls = AGENTS[config.algorithm]
    except KeyError as exc:
        raise ValueError(f"unknown algorithm {config.algorithm!r}; choose from {sorted(AGENTS)}") from exc
    return cls(num_states, num_actions, config.gamma, config.initial_q, rng)


__all__ = ["AGENTS", "QLearningAgent", "SarsaAgent", "TabularAgent", "make_agent"]
