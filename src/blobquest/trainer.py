"""Training loop: run episodes, learn from them, evaluate, record metrics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from blobquest.agents import TabularAgent, make_agent
from blobquest.config import Config
from blobquest.env import BlobEnv, Event, Outcome, StepResult
from blobquest.memory import NO_ACTION, ReplayMemory, Transition
from blobquest.metrics import EpisodeRecord, EvalRecord, TrainingHistory
from blobquest.mistakes import MistakeLog
from blobquest.schedules import make_schedule
from blobquest.state import StateEncoder
from blobquest.world import Position, World, build_world

Logger = Callable[[str], None]


@dataclass(frozen=True)
class Rollout:
    """A single greedy episode: the path walked and how it ended."""

    path: tuple[Position, ...]
    total_reward: float
    steps: int
    outcome: Outcome
    final_energy: int


def make_encoder(world: World, config: Config) -> StateEncoder:
    """State encoder sized for ``world`` and the energy and sensing settings in ``config``."""
    sensing_grid = world.grid if config.agent.sensing else None
    return StateEncoder(world.rows, world.cols, config.energy.num_buckets, world.num_boosters, sensing_grid)


@dataclass(frozen=True)
class GreedyStep:
    """One step of a greedy episode, with the environment status right after it."""

    action: int
    result: StepResult
    position: Position
    cells_entered: tuple[Position, ...]
    energy: int
    drain_left: int
    speed_left: int


def greedy_steps(
    env: BlobEnv, agent: TabularAgent, encoder: StateEncoder, rng: np.random.Generator
) -> Iterator[GreedyStep]:
    """Reset ``env`` and yield each step of one episode played with epsilon = 0."""
    state = env.reset()
    while True:
        action = agent.greedy_action(encoder.encode(state), rng)
        result = env.step(action)
        yield GreedyStep(action, result, env.position, env.cells_entered, env.energy, env.drain_left, env.speed_left)
        state = result.state
        if result.terminated or result.truncated:
            return


def greedy_rollout(env: BlobEnv, agent: TabularAgent, encoder: StateEncoder, rng: np.random.Generator) -> Rollout:
    """Play one episode with epsilon = 0 and record the visited cells."""
    path = [env.world.start]
    total = 0.0
    for step in greedy_steps(env, agent, encoder, rng):
        total += step.result.reward
        path.append(step.position)
    return Rollout(tuple(path), total, env.steps, step.result.outcome, env.energy)


class Trainer:
    """Owns the world, environment, agent and memory for one training run.

    All randomness flows from ``config.training.seed`` (agent, memory and
    evaluation each get an independent child generator) plus
    ``config.world.map_seed`` for the map, so a run is fully reproducible.
    """

    def __init__(self, config: Config, logger: Logger | None = None, world: World | None = None) -> None:
        self.config = config
        self._log = logger
        self.world = world if world is not None else build_world(config.world)
        self.env = BlobEnv(self.world, config)
        self.encoder = make_encoder(self.world, config)

        agent_seed, memory_seed, eval_seed = np.random.SeedSequence(config.training.seed).spawn(3)
        self._eval_rng = np.random.default_rng(eval_seed)
        self.agent = make_agent(
            config.agent, self.encoder.num_states, self.env.num_actions, np.random.default_rng(agent_seed)
        )
        mem = config.memory
        self.memory: ReplayMemory | None = (
            ReplayMemory(mem.capacity, mem.mistake_threshold, mem.mistake_weight, np.random.default_rng(memory_seed))
            if mem.enabled
            else None
        )
        episodes = config.training.episodes
        self.epsilon = make_schedule(config.agent.epsilon, episodes)
        self.alpha = make_schedule(config.agent.alpha, episodes)
        self.history = TrainingHistory(self.world.rows, self.world.cols)
        self.mistakes = MistakeLog()

    def train(self) -> TrainingHistory:
        """Run all configured episodes and return the recorded history."""
        cfg = self.config.training
        for episode in range(cfg.episodes):
            record = self.run_episode(episode)
            self.history.episodes.append(record)
            is_last = episode == cfg.episodes - 1
            if (episode + 1) % cfg.eval_every == 0 or is_last:
                self.history.evaluations.append(self.evaluate(episode))
            if self._log is not None and ((episode + 1) % cfg.log_every == 0 or is_last):
                self._log(self._progress_line(episode))
        return self.history

    def run_episode(self, episode: int) -> EpisodeRecord:
        """Play and learn from one epsilon-greedy episode, then replay memory."""
        epsilon, alpha = self.epsilon(episode), self.alpha(episode)
        env, agent, enc = self.env, self.agent, self.encoder
        events: Counter[Event] = Counter()
        td_errors: list[float] = []
        total = 0.0

        # Sensing only knows cells from earlier episodes, so freeze the snapshot now.
        env.set_knowledge(self.history.explored)
        s = enc.encode(env.reset())
        self.history.explored[env.position] = True
        a = agent.act(s, epsilon)
        while True:
            result = env.step(a)
            s_next = enc.encode(result.state)
            # Choosing a' before the update lets the same loop serve SARSA and Q-learning.
            a_next = NO_ACTION if result.terminated else agent.act(s_next, epsilon)
            transition = Transition(s, a, result.reward, s_next, a_next, result.terminated)
            td_errors.append(abs(agent.update(transition, alpha)))
            if self.memory is not None:
                self.memory.add(transition)
            total += result.reward
            events.update(result.events)
            self.history.visits[env.position] += 1
            for cell in env.cells_entered:
                self.history.explored[cell] = True
            if result.terminated or result.truncated:
                break
            s, a = s_next, a_next

        if self.memory is not None:
            for old in self.memory.sample(self.config.memory.replay_updates):
                agent.update(old, alpha)

        self.mistakes.record(episode, result.outcome, env.position, env.energy)
        return EpisodeRecord(
            episode=episode,
            total_reward=total,
            steps=env.steps,
            outcome=result.outcome.value,
            final_energy=env.energy,
            epsilon=epsilon,
            alpha=alpha,
            trap_hits=events[Event.TRAP],
            boosters_collected=events[Event.ENERGY_BOOST] + events[Event.SPEED_BOOST],
            resets=events[Event.RESET],
            wall_bumps=events[Event.WALL_BUMP],
            mean_abs_td_error=float(np.mean(td_errors)),
        )

    def evaluate(self, episode: int) -> EvalRecord:
        """Greedy evaluation without learning, using its own random stream."""
        self.env.set_knowledge(self.history.explored)
        rollouts = [
            greedy_rollout(self.env, self.agent, self.encoder, self._eval_rng)
            for _ in range(self.config.training.eval_episodes)
        ]
        return EvalRecord(
            episode=episode,
            success_rate=float(np.mean([r.outcome is Outcome.GOAL for r in rollouts])),
            mean_reward=float(np.mean([r.total_reward for r in rollouts])),
            mean_steps=float(np.mean([r.steps for r in rollouts])),
        )

    def final_rollout(self) -> Rollout:
        """One greedy episode with the trained agent (for rendering and plots)."""
        self.env.set_knowledge(self.history.explored)
        return greedy_rollout(self.env, self.agent, self.encoder, self._eval_rng)

    def replay_steps(self) -> Iterator[GreedyStep]:
        """Step-by-step greedy episode with the trained agent (for ``blobquest replay``)."""
        self.env.set_knowledge(self.history.explored)
        return greedy_steps(self.env, self.agent, self.encoder, self._eval_rng)

    def _progress_line(self, episode: int) -> str:
        """One-line progress report for the terminal."""
        window = self.config.output.moving_average_window
        stats = self.history.summary(window)
        greedy = self.history.evaluations[-1].success_rate if self.history.evaluations else float("nan")
        return (
            f"ep {episode + 1:>6}/{self.config.training.episodes}"
            f" | eps {self.epsilon(episode):.3f}"
            f" | alpha {self.alpha(episode):.3f}"
            f" | reward(avg{window}) {stats['mean_reward']:8.2f}"
            f" | success(avg{window}) {stats['success_rate']:6.1%}"
            f" | greedy success {greedy:6.1%}"
        )
