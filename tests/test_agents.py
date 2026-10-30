import numpy as np
import pytest

from blobquest.agents import AGENTS, QLearningAgent, RandomAgent, SarsaAgent, make_agent
from blobquest.config import AgentConfig
from blobquest.memory import NO_ACTION, Transition
from blobquest.trainer import Trainer
from tests.conftest import make_config


def _agent(cls: type, seed: int = 0) -> QLearningAgent | SarsaAgent | RandomAgent:
    return cls(num_states=3, num_actions=2, gamma=0.9, initial_q=0.0, rng=np.random.default_rng(seed))


def test_q_learning_update_uses_max_next_value() -> None:
    agent = _agent(QLearningAgent)
    agent.q[1] = [2.0, 5.0]
    td = agent.update(Transition(0, 0, 1.0, 1, 0, False), alpha=0.5)
    assert td == pytest.approx(1.0 + 0.9 * 5.0)
    assert agent.q[0, 0] == pytest.approx(0.5 * (1.0 + 0.9 * 5.0))


def test_sarsa_update_uses_chosen_next_action() -> None:
    agent = _agent(SarsaAgent)
    agent.q[1] = [2.0, 5.0]
    agent.update(Transition(0, 0, 1.0, 1, 0, False), alpha=0.5)
    assert agent.q[0, 0] == pytest.approx(0.5 * (1.0 + 0.9 * 2.0))


@pytest.mark.parametrize("cls", [QLearningAgent, SarsaAgent])
def test_terminal_transition_does_not_bootstrap(cls: type) -> None:
    agent = _agent(cls)
    agent.q[1] = [100.0, 100.0]
    agent.update(Transition(0, 1, -10.0, 1, NO_ACTION, True), alpha=1.0)
    assert agent.q[0, 1] == pytest.approx(-10.0)


def test_sarsa_requires_next_action() -> None:
    with pytest.raises(ValueError):
        _agent(SarsaAgent).update(Transition(0, 0, 0.0, 1, NO_ACTION, False), alpha=0.1)


def test_greedy_action_picks_best() -> None:
    agent = _agent(QLearningAgent)
    agent.q[2] = [-1.0, 3.0]
    assert agent.act(2, epsilon=0.0) == 1


def test_ties_are_broken_randomly() -> None:
    agent = _agent(QLearningAgent)
    choices = {agent.greedy_action(0) for _ in range(50)}
    assert choices == {0, 1}


def test_full_exploration_is_uniform() -> None:
    agent = _agent(QLearningAgent)
    agent.q[0] = [10.0, 0.0]
    picks = [agent.act(0, epsilon=1.0) for _ in range(2000)]
    assert 0.4 < np.mean(picks) < 0.6


def test_same_seed_same_actions() -> None:
    a, b = _agent(QLearningAgent, seed=5), _agent(QLearningAgent, seed=5)
    assert [a.act(0, 0.5) for _ in range(20)] == [b.act(0, 0.5) for _ in range(20)]


@pytest.mark.parametrize(("algorithm", "cls"), [("q_learning", QLearningAgent), ("sarsa", SarsaAgent)])
def test_factory(algorithm: str, cls: type) -> None:
    agent = make_agent(AgentConfig(algorithm=algorithm), 4, 4, np.random.default_rng(0))
    assert isinstance(agent, cls)
    assert agent.q.shape == (4, 4)


def test_random_agent_is_registered() -> None:
    assert AGENTS["random"] is RandomAgent
    agent = make_agent(AgentConfig(algorithm="random"), 4, 4, np.random.default_rng(0))
    assert isinstance(agent, RandomAgent)


def test_random_agent_never_learns() -> None:
    agent = _agent(RandomAgent)
    before = agent.q.copy()
    assert agent.update(Transition(0, 0, -100.0, 1, 1, False), alpha=1.0) == 0.0
    assert agent.update(Transition(0, 1, 50.0, 1, NO_ACTION, True), alpha=1.0) == 0.0
    assert np.array_equal(agent.q, before)


def test_random_agent_ignores_epsilon() -> None:
    agent = _agent(RandomAgent)
    agent.q[0] = [10.0, 0.0]
    picks = [agent.act(0, epsilon=0.0) for _ in range(2000)]
    assert 0.4 < np.mean(picks) < 0.6


def test_random_agent_q_table_untouched_by_training() -> None:
    config = make_config(
        ["@..", ".X.", "..G"],
        agent={"algorithm": "random", "initial_q": 0.5},
        training={"episodes": 50, "eval_every": 25, "eval_episodes": 1},
    )
    trainer = Trainer(config)
    history = trainer.train()
    assert len(history.episodes) == 50
    assert np.all(trainer.agent.q == 0.5)
