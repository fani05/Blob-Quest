"""The Blob Quest environment: hidden map dynamics behind a tiny step() API."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from blobquest.config import Config
from blobquest.state import FOG, NEIGHBOUR_OFFSETS, State, energy_bucket
from blobquest.tiles import Tile
from blobquest.world import Position, World

ACTIONS_4: tuple[Position, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTIONS_8: tuple[Position, ...] = ACTIONS_4 + ((-1, 1), (1, 1), (1, -1), (-1, -1))
ACTION_NAMES: tuple[str, ...] = ("up", "right", "down", "left", "up-right", "down-right", "down-left", "up-left")


def action_deltas(diagonals: bool) -> tuple[Position, ...]:
    """Row/column offsets for each action index."""
    return ACTIONS_8 if diagonals else ACTIONS_4


class Outcome(StrEnum):
    """How an episode is going or how it ended."""

    RUNNING = "running"
    GOAL = "goal"
    PIT = "pit"
    EXHAUSTED = "exhausted"
    TIMEOUT = "timeout"


TERMINAL_OUTCOMES: frozenset[Outcome] = frozenset({Outcome.GOAL, Outcome.PIT, Outcome.EXHAUSTED})


class Event(StrEnum):
    """Things that can happen during a single step (for metrics and debugging)."""

    WALL_BUMP = "wall_bump"
    TRAP = "trap"
    DRAIN = "drain"
    ENERGY_BOOST = "energy_boost"
    SPEED_BOOST = "speed_boost"
    RESET = "reset"
    PIT = "pit"
    GOAL = "goal"


@dataclass(frozen=True)
class StepResult:
    """Everything returned by :meth:`BlobEnv.step`.

    ``terminated`` means a true end (goal, pit, no energy) and must not be
    bootstrapped; ``truncated`` means the step limit cut the episode short.
    """

    state: State
    reward: float
    terminated: bool
    truncated: bool
    outcome: Outcome
    events: tuple[Event, ...]


class BlobEnv:
    """Grid environment. The agent only ever sees :class:`State` objects.

    Step order: pay energy (drain-aware), then move one cell, or ``speed_cells``
    cells while a speed boost is active. Every cell entered triggers its tile;
    walls stop movement early, and goal/pit/reset end the move immediately.
    """

    def __init__(self, world: World, config: Config) -> None:
        self.world = world
        self._energy_cfg = config.energy
        self._movement = config.movement
        self._rewards = config.rewards
        self._max_steps = config.training.max_steps_per_episode
        self._deltas = action_deltas(config.movement.diagonals)
        self._pos: Position = world.start
        self._energy = 0
        self._used_mask = 0
        self._drain_left = 0
        self._speed_left = 0
        self._steps = 0
        self._done = True
        self._entered: list[Position] = []
        self._sensing = config.agent.sensing
        self._known = np.zeros((world.rows, world.cols), dtype=bool)

    @property
    def num_actions(self) -> int:
        """Size of the action space (4 or 8)."""
        return len(self._deltas)

    @property
    def position(self) -> Position:
        """Current blob position (for logging and plots, not for the agent)."""
        return self._pos

    @property
    def cells_entered(self) -> tuple[Position, ...]:
        """Every cell entered during the last step, in order (empty after a wall bump)."""
        return tuple(self._entered)

    @property
    def energy(self) -> int:
        """Current raw energy value."""
        return self._energy

    @property
    def drain_left(self) -> int:
        """Steps of energy drain still to pay (0 when no drain is active)."""
        return self._drain_left

    @property
    def speed_left(self) -> int:
        """Moves left on the current speed boost (0 when none is active)."""
        return self._speed_left

    @property
    def steps(self) -> int:
        """Steps taken in the current episode."""
        return self._steps

    def set_knowledge(self, known: np.ndarray) -> None:
        """Set which cells count as visited for sensing (copied; used until changed)."""
        if known.shape != self._known.shape:
            raise ValueError(f"knowledge mask shape {known.shape} does not match map {self._known.shape}")
        self._known = known.astype(bool, copy=True)

    def observe(self) -> State:
        """Return the blob's current state."""
        return State(
            row=self._pos[0],
            col=self._pos[1],
            energy_bucket=energy_bucket(self._energy, self._energy_cfg.maximum, self._energy_cfg.num_buckets),
            used_boosters_mask=self._used_mask,
            drain_active=self._drain_left > 0,
            speed_active=self._speed_left > 0,
            neighbours=self._sense() if self._sensing else (),
        )

    def _sense(self) -> tuple[int, ...]:
        """Tile of each neighbour the blob already knows, ``FOG`` for the rest."""
        r, c = self._pos
        seen = []
        for dr, dc in NEIGHBOUR_OFFSETS:
            cell = (r + dr, c + dc)
            known = self.world.is_passable(cell) and self._known[cell]
            seen.append(int(self.world.tile_at(cell)) if known else FOG)
        return tuple(seen)

    def reset(self) -> State:
        """Start a new episode at the start cell with full initial energy."""
        self._pos = self.world.start
        self._energy = self._energy_cfg.initial
        self._used_mask = 0
        self._drain_left = 0
        self._speed_left = 0
        self._steps = 0
        self._done = False
        self._entered = []
        return self.observe()

    def step(self, action: int) -> StepResult:
        """Apply ``action`` and return the new state, reward and episode status."""
        if self._done:
            raise RuntimeError("episode has finished; call reset() first")
        if not 0 <= action < self.num_actions:
            raise ValueError(f"action {action} out of range [0, {self.num_actions})")

        rewards = self._rewards
        events: list[Event] = []
        self._entered = []
        self._steps += 1
        reward = rewards.step

        if self._drain_left > 0:
            self._energy -= self._energy_cfg.drain_step_cost
            self._drain_left -= 1
            reward += rewards.drain_step
        else:
            self._energy -= self._energy_cfg.step_cost

        cells = 1
        if self._speed_left > 0:
            cells = self._movement.speed_cells
            self._speed_left -= 1

        dr, dc = self._deltas[action]
        outcome = Outcome.RUNNING
        for moved in range(cells):
            target = (self._pos[0] + dr, self._pos[1] + dc)
            if not self.world.is_passable(target):
                if moved == 0:
                    reward += rewards.wall_bump
                    events.append(Event.WALL_BUMP)
                break
            self._pos = target
            self._entered.append(target)
            tile_reward, outcome, stop = self._enter(target, events)
            reward += tile_reward
            if stop:
                break

        if outcome is Outcome.RUNNING and self._energy <= 0:
            outcome = Outcome.EXHAUSTED
            reward += rewards.exhausted

        terminated = outcome in TERMINAL_OUTCOMES
        truncated = not terminated and self._steps >= self._max_steps
        if truncated:
            outcome = Outcome.TIMEOUT
        self._done = terminated or truncated
        return StepResult(self.observe(), reward, terminated, truncated, outcome, tuple(events))

    def _enter(self, pos: Position, events: list[Event]) -> tuple[float, Outcome, bool]:
        """Apply the effect of the tile at ``pos``; return (reward, outcome, stop_moving)."""
        rewards = self._rewards
        tile = self.world.tile_at(pos)
        if tile is Tile.GOAL:
            events.append(Event.GOAL)
            return rewards.goal, Outcome.GOAL, True
        if tile is Tile.PIT:
            events.append(Event.PIT)
            return rewards.pit, Outcome.PIT, True
        if tile is Tile.TRAP:
            events.append(Event.TRAP)
            self._energy -= self._energy_cfg.trap_loss
            return rewards.trap, Outcome.RUNNING, False
        if tile is Tile.DRAIN:
            events.append(Event.DRAIN)
            self._drain_left = self._energy_cfg.drain_duration
            return 0.0, Outcome.RUNNING, False
        if tile is Tile.RESET:
            events.append(Event.RESET)
            self._pos = self.world.start
            return rewards.reset, Outcome.RUNNING, True
        if tile in (Tile.ENERGY, Tile.SPEED):
            bit = 1 << self.world.booster_index[pos]
            if self._used_mask & bit:
                return 0.0, Outcome.RUNNING, False
            self._used_mask |= bit
            if tile is Tile.ENERGY:
                events.append(Event.ENERGY_BOOST)
                self._energy = min(self._energy + self._energy_cfg.booster_gain, self._energy_cfg.maximum)
                return rewards.energy_booster, Outcome.RUNNING, False
            events.append(Event.SPEED_BOOST)
            self._speed_left = self._movement.speed_duration
            return rewards.speed_booster, Outcome.RUNNING, False
        return 0.0, Outcome.RUNNING, False
