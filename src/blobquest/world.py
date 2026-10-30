"""The hidden map: generation, parsing and queries."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from blobquest.config import ConfigError, WorldConfig
from blobquest.tiles import BLOCKING_TILES, BOOSTER_TILES, START_CHAR, TILE_CHARS, Tile, tile_from_char

Position = tuple[int, int]

_FOUR_NEIGHBOURS: tuple[Position, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))


class WorldGenerationError(RuntimeError):
    """Raised when no solvable random map could be generated."""


@dataclass(frozen=True, eq=False)
class World:
    """An immutable grid of tiles plus the start, goal and booster indices.

    ``booster_index`` maps each booster cell to its bit in the state's
    ``used_boosters_mask``.
    """

    grid: np.ndarray
    start: Position
    goal: Position
    booster_index: Mapping[Position, int]

    def __post_init__(self) -> None:
        self.grid.setflags(write=False)

    @property
    def rows(self) -> int:
        """Number of grid rows."""
        return int(self.grid.shape[0])

    @property
    def cols(self) -> int:
        """Number of grid columns."""
        return int(self.grid.shape[1])

    @property
    def num_boosters(self) -> int:
        """How many single-use boosters exist (bits in the booster mask)."""
        return len(self.booster_index)

    def in_bounds(self, pos: Position) -> bool:
        """True if ``pos`` lies on the grid."""
        return 0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols

    def tile_at(self, pos: Position) -> Tile:
        """Tile type at ``pos`` (must be in bounds)."""
        return Tile(int(self.grid[pos]))

    def is_passable(self, pos: Position) -> bool:
        """True if the blob may enter ``pos``."""
        return self.in_bounds(pos) and self.tile_at(pos) not in BLOCKING_TILES

    def to_layout(self) -> list[str]:
        """Serialise the map to layout strings (``@`` marks the start)."""
        lines = []
        for r in range(self.rows):
            chars = [TILE_CHARS[Tile(int(v))] for v in self.grid[r]]
            if r == self.start[0]:
                chars[self.start[1]] = START_CHAR
            lines.append("".join(chars))
        return lines

    @classmethod
    def from_grid(cls, grid: np.ndarray, start: Position, max_boosters: int) -> World:
        """Build a world from a tile grid, validating start, goal and boosters."""
        grid = np.asarray(grid, dtype=np.int8).copy()
        goals = list(zip(*np.nonzero(grid == Tile.GOAL), strict=True))
        if len(goals) != 1:
            raise ConfigError(f"map must contain exactly one goal, found {len(goals)}")
        if not (0 <= start[0] < grid.shape[0] and 0 <= start[1] < grid.shape[1]):
            raise ConfigError(f"start {start} is outside the map")
        if Tile(int(grid[start])) != Tile.EMPTY:
            raise ConfigError(f"start {start} must be an EMPTY cell")
        boosters = sorted(
            (int(r), int(c)) for r, c in zip(*np.nonzero(np.isin(grid, list(BOOSTER_TILES))), strict=True)
        )
        if len(boosters) > max_boosters:
            raise ConfigError(f"map has {len(boosters)} boosters, more than max_boosters={max_boosters}")
        goal = (int(goals[0][0]), int(goals[0][1]))
        index = MappingProxyType({pos: bit for bit, pos in enumerate(boosters)})
        return cls(grid=grid, start=(int(start[0]), int(start[1])), goal=goal, booster_index=index)

    @classmethod
    def from_layout(cls, lines: Iterable[str], max_boosters: int) -> World:
        """Parse a list of equal-length strings into a world."""
        rows = [line.strip() for line in lines if line.strip()]
        if not rows or len({len(r) for r in rows}) != 1:
            raise ConfigError("layout rows must be non-empty and of equal length")
        starts = [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == START_CHAR]
        if len(starts) != 1:
            raise ConfigError(f"layout must contain exactly one '{START_CHAR}', found {len(starts)}")
        try:
            grid = np.array([[tile_from_char(ch) for ch in row] for row in rows], dtype=np.int8)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        return cls.from_grid(grid, starts[0], max_boosters)


def is_solvable(grid: np.ndarray, start: Position, goal: Position) -> bool:
    """BFS check that ``goal`` is reachable from ``start`` with 4-way moves.

    Walls, pits and reset tiles count as blocked, so a solvable map always has
    a safe route that does not rely on luck or diagonal moves.
    """
    blocked = np.isin(grid, [Tile.WALL, Tile.PIT, Tile.RESET])
    rows, cols = grid.shape
    seen = np.zeros_like(blocked)
    seen[start] = True
    queue: deque[Position] = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in _FOUR_NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc] and not blocked[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return False


def generate_world(config: WorldConfig, rng: np.random.Generator) -> World:
    """Randomly place tiles until a solvable map is found."""
    start, goal = config.start, config.goal_position
    shape = (config.rows, config.cols)
    for pos, name in ((start, "start"), (goal, "goal")):
        if not (0 <= pos[0] < config.rows and 0 <= pos[1] < config.cols):
            raise ConfigError(f"world.{name} {pos} is outside the {config.rows}x{config.cols} map")
    if start == goal:
        raise ConfigError("world.start and world.goal must differ")

    free = [(r, c) for r in range(config.rows) for c in range(config.cols) if (r, c) not in (start, goal)]
    num_walls = int(round(config.wall_density * len(free)))
    placements: list[tuple[Tile, int]] = [
        (Tile.WALL, num_walls),
        (Tile.PIT, config.num_pits),
        (Tile.TRAP, config.num_traps),
        (Tile.DRAIN, config.num_drains),
        (Tile.RESET, config.num_resets),
        (Tile.ENERGY, config.num_energy_boosters),
        (Tile.SPEED, config.num_speed_boosters),
    ]
    if sum(count for _, count in placements) > len(free):
        raise ConfigError("world: too many tiles requested for the map size")

    for _ in range(config.max_generation_attempts):
        grid = np.full(shape, Tile.EMPTY, dtype=np.int8)
        grid[goal] = Tile.GOAL
        order = rng.permutation(len(free))
        cursor = 0
        for tile, count in placements:
            for idx in order[cursor : cursor + count]:
                grid[free[idx]] = tile
            cursor += count
        if is_solvable(grid, start, goal):
            return World.from_grid(grid, start, config.max_boosters)
    raise WorldGenerationError(
        f"no solvable map after {config.max_generation_attempts} attempts; lower wall_density or tile counts"
    )


def build_world(config: WorldConfig) -> World:
    """Create the world described by ``config`` (fixed layout or seeded random map)."""
    if config.layout is not None:
        return World.from_layout(config.layout, config.max_boosters)
    return generate_world(config, np.random.default_rng(config.map_seed))
