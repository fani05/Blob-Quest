"""Typed, validated configuration for Blob Quest.

Every tunable number in the project lives in one of the frozen dataclasses
below. A YAML file only needs to list the values it wants to change; anything
missing falls back to the defaults defined here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

import yaml

Algorithm = Literal["q_learning", "sarsa", "random"]
ScheduleKind = Literal["constant", "linear", "exponential"]

ALGORITHMS: tuple[str, ...] = ("q_learning", "sarsa", "random")
SCHEDULE_KINDS: tuple[str, ...] = ("constant", "linear", "exponential")


class ConfigError(ValueError):
    """Raised when a configuration file or value is invalid."""


def _require(condition: bool, message: str) -> None:
    """Raise ``ConfigError`` with ``message`` unless ``condition`` holds."""
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class WorldConfig:
    """Map size, start/goal positions and how many of each tile to place.

    If ``layout`` is given, the map is parsed from it and the random-generation
    settings (densities and counts) are ignored.
    """

    rows: int = 15
    cols: int = 15
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] | None = None
    map_seed: int = 7
    layout: tuple[str, ...] | None = None
    wall_density: float = 0.15
    num_traps: int = 6
    num_pits: int = 4
    num_drains: int = 4
    num_energy_boosters: int = 2
    num_speed_boosters: int = 1
    num_resets: int = 2
    max_boosters: int = 6
    max_generation_attempts: int = 500

    def __post_init__(self) -> None:
        # YAML gives lists; normalise to tuples so the config stays hashable.
        object.__setattr__(self, "start", tuple(self.start))
        if self.goal is not None:
            object.__setattr__(self, "goal", tuple(self.goal))
        if self.layout is not None:
            object.__setattr__(self, "layout", tuple(self.layout))
            _require(len(self.layout) > 0, "world.layout must not be empty")
            return
        _require(self.rows >= 2 and self.cols >= 2, "world.rows and world.cols must be >= 2")
        _require(0.0 <= self.wall_density < 1.0, "world.wall_density must be in [0, 1)")
        counts = (
            self.num_traps,
            self.num_pits,
            self.num_drains,
            self.num_energy_boosters,
            self.num_speed_boosters,
            self.num_resets,
        )
        _require(all(c >= 0 for c in counts), "world tile counts must be >= 0")
        _require(
            self.num_energy_boosters + self.num_speed_boosters <= self.max_boosters,
            "world: number of boosters exceeds world.max_boosters",
        )
        _require(self.max_generation_attempts >= 1, "world.max_generation_attempts must be >= 1")

    @property
    def goal_position(self) -> tuple[int, int]:
        """Goal cell, defaulting to the bottom-right corner."""
        return self.goal if self.goal is not None else (self.rows - 1, self.cols - 1)


@dataclass(frozen=True)
class EnergyConfig:
    """Energy budget and every energy cost or gain in the world."""

    initial: int = 50
    maximum: int = 60
    step_cost: int = 1
    drain_step_cost: int = 2
    drain_duration: int = 5
    trap_loss: int = 10
    booster_gain: int = 15
    num_buckets: int = 5

    def __post_init__(self) -> None:
        _require(0 < self.initial <= self.maximum, "energy.initial must be in (0, maximum]")
        _require(self.step_cost > 0, "energy.step_cost must be > 0")
        _require(self.drain_step_cost >= self.step_cost, "energy.drain_step_cost must be >= step_cost")
        _require(self.drain_duration >= 0, "energy.drain_duration must be >= 0")
        _require(self.trap_loss >= 0 and self.booster_gain >= 0, "energy gains/losses must be >= 0")
        _require(1 <= self.num_buckets <= self.maximum, "energy.num_buckets must be in [1, maximum]")


@dataclass(frozen=True)
class MovementConfig:
    """Action set and speed-booster behaviour."""

    diagonals: bool = False
    speed_cells: int = 2
    speed_duration: int = 3

    def __post_init__(self) -> None:
        _require(self.speed_cells >= 1, "movement.speed_cells must be >= 1")
        _require(self.speed_duration >= 0, "movement.speed_duration must be >= 0")


@dataclass(frozen=True)
class RewardConfig:
    """Reward signal: what the blob is told after each step."""

    goal: float = 100.0
    pit: float = -100.0
    exhausted: float = -50.0
    step: float = -1.0
    drain_step: float = -1.0
    trap: float = -10.0
    wall_bump: float = -2.0
    reset: float = -5.0
    energy_booster: float = 2.0
    speed_booster: float = 1.0


@dataclass(frozen=True)
class ScheduleConfig:
    """A value that changes over episodes (exploration rate, learning rate).

    ``decay_fraction`` is used by ``linear`` (fraction of all episodes spent
    decaying); ``decay_rate`` is the per-episode factor used by ``exponential``.
    """

    kind: ScheduleKind = "linear"
    start: float = 1.0
    end: float = 0.05
    decay_fraction: float = 0.7
    decay_rate: float = 0.999

    def __post_init__(self) -> None:
        _require(self.kind in SCHEDULE_KINDS, f"schedule.kind must be one of {SCHEDULE_KINDS}")
        _require(0.0 < self.decay_fraction <= 1.0, "schedule.decay_fraction must be in (0, 1]")
        _require(0.0 < self.decay_rate <= 1.0, "schedule.decay_rate must be in (0, 1]")


@dataclass(frozen=True)
class AgentConfig:
    """Learning algorithm and its hyper-parameters."""

    algorithm: Algorithm = "q_learning"
    sensing: bool = False
    """Add the 4 neighbouring tiles to the state, for cells visited in earlier episodes (else FOG)."""
    gamma: float = 0.99
    initial_q: float = 0.0
    epsilon: ScheduleConfig = field(default_factory=ScheduleConfig)
    alpha: ScheduleConfig = field(
        default_factory=lambda: ScheduleConfig(kind="linear", start=0.3, end=0.05, decay_fraction=0.8)
    )

    def __post_init__(self) -> None:
        if isinstance(self.epsilon, dict):
            object.__setattr__(self, "epsilon", _build_section(ScheduleConfig, self.epsilon, "agent.epsilon"))
        if isinstance(self.alpha, dict):
            object.__setattr__(self, "alpha", _build_section(ScheduleConfig, self.alpha, "agent.alpha"))
        _require(self.algorithm in ALGORITHMS, f"agent.algorithm must be one of {ALGORITHMS}")
        _require(0.0 <= self.gamma <= 1.0, "agent.gamma must be in [0, 1]")
        _require(0.0 <= self.epsilon.end <= self.epsilon.start <= 1.0, "agent.epsilon must satisfy 0<=end<=start<=1")
        _require(0.0 < self.alpha.end <= self.alpha.start <= 1.0, "agent.alpha must satisfy 0<end<=start<=1")


@dataclass(frozen=True)
class MemoryConfig:
    """Replay memory that re-learns past transitions, favouring mistakes."""

    enabled: bool = True
    capacity: int = 20_000
    replay_updates: int = 64
    mistake_threshold: float = -10.0
    mistake_weight: float = 4.0

    def __post_init__(self) -> None:
        _require(self.capacity >= 1, "memory.capacity must be >= 1")
        _require(self.replay_updates >= 0, "memory.replay_updates must be >= 0")
        _require(self.mistake_weight >= 1.0, "memory.mistake_weight must be >= 1")


@dataclass(frozen=True)
class TrainingConfig:
    """Episode budget, seeding and evaluation cadence."""

    episodes: int = 10000
    max_steps_per_episode: int = 200
    seed: int = 42
    eval_every: int = 100
    eval_episodes: int = 5
    log_every: int = 1000

    def __post_init__(self) -> None:
        _require(self.episodes >= 1, "training.episodes must be >= 1")
        _require(self.max_steps_per_episode >= 1, "training.max_steps_per_episode must be >= 1")
        _require(self.eval_every >= 1 and self.eval_episodes >= 1, "training eval settings must be >= 1")
        _require(self.log_every >= 1, "training.log_every must be >= 1")


@dataclass(frozen=True)
class OutputConfig:
    """Where runs are written and how plots look."""

    runs_dir: str = "runs"
    run_name: str = "blobquest"
    save_plots: bool = True
    moving_average_window: int = 100
    outcome_window: int = 100
    plot_dpi: int = 120
    figure_width: float = 9.0
    figure_height: float = 5.0
    map_cell_inches: float = 0.45

    def __post_init__(self) -> None:
        _require(self.moving_average_window >= 1, "output.moving_average_window must be >= 1")
        _require(self.outcome_window >= 1, "output.outcome_window must be >= 1")
        _require(self.plot_dpi > 0, "output.plot_dpi must be > 0")


@dataclass(frozen=True)
class Config:
    """Root configuration object that bundles every section."""

    world: WorldConfig = field(default_factory=WorldConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    movement: MovementConfig = field(default_factory=MovementConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Config:
        """Build a config from a nested dict, rejecting unknown keys."""
        data = data or {}
        _require(isinstance(data, dict), "config root must be a mapping")
        sections = {f.name: f.type for f in fields(cls)}
        unknown = set(data) - set(sections)
        _require(not unknown, f"unknown config sections: {sorted(unknown)}")
        section_types: dict[str, type] = {
            "world": WorldConfig,
            "energy": EnergyConfig,
            "movement": MovementConfig,
            "rewards": RewardConfig,
            "agent": AgentConfig,
            "memory": MemoryConfig,
            "training": TrainingConfig,
            "output": OutputConfig,
        }
        built = {name: _build_section(section_types[name], data.get(name), name) for name in sections}
        return cls(**built)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain nested dict (lists instead of tuples) suitable for YAML."""
        return _to_plain(asdict(self))

    def with_overrides(self, **sections: dict[str, Any]) -> Config:
        """Return a copy with selected fields replaced, e.g. ``training={"seed": 1}``."""
        updated: dict[str, Any] = {}
        for name, values in sections.items():
            current = getattr(self, name)
            try:
                updated[name] = replace(current, **values)
            except TypeError as exc:
                raise ConfigError(f"invalid override for '{name}': {exc}") from exc
        return replace(self, **updated)


def _build_section(cls: type, data: Any, name: str) -> Any:
    """Instantiate dataclass ``cls`` from ``data``, with readable errors."""
    if data is None:
        return cls()
    _require(isinstance(data, dict), f"config section '{name}' must be a mapping")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    _require(not unknown, f"unknown keys in '{name}': {sorted(unknown)}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise ConfigError(f"invalid '{name}' section: {exc}") from exc


def _to_plain(value: Any) -> Any:
    """Recursively convert tuples to lists so ``yaml.safe_dump`` accepts them."""
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return Config.from_dict(data)


def save_config(config: Config, path: str | Path) -> None:
    """Write ``config`` to ``path`` as YAML."""
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
