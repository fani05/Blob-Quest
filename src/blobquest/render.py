"""Text rendering of the map, greedy paths and learned policies."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from blobquest.agents import TabularAgent
from blobquest.state import State, StateEncoder
from blobquest.tiles import START_CHAR, TILE_CHARS, Tile
from blobquest.world import Position, World

ARROWS: tuple[str, ...] = ("↑", "→", "↓", "←", "↗", "↘", "↙", "↖")
PATH_CHAR = "*"
BLOB_CHAR = "O"
NO_POLICY = -1

LEGEND = (
    f"{START_CHAR} start  G goal  # wall  T trap  X pit  D drain  "
    f"E energy  S speed  R reset  {PATH_CHAR} path  {BLOB_CHAR} blob"
)


def policy_and_values(
    agent: TabularAgent, encoder: StateEncoder, world: World, max_bucket: int
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy action and state value for every cell, as a 2D slice of the Q-table.

    The full state has six dimensions, so a map can only show a slice. For each
    cell we use "no boosters used, no drain/speed active" and the highest energy
    bucket ``<= max_bucket`` whose Q-row has actually been trained - i.e. the
    best-rested blob the agent has ever been on that cell. Cells that are walls
    or were never trained get ``NO_POLICY`` and ``NaN``; ties pick the lowest
    action index.
    """
    actions = np.full((world.rows, world.cols), NO_POLICY, dtype=np.int64)
    values = np.full((world.rows, world.cols), np.nan)
    for r in range(world.rows):
        for c in range(world.cols):
            if world.tile_at((r, c)) is Tile.WALL:
                continue
            for bucket in range(max_bucket, -1, -1):
                q = agent.q[encoder.encode(State(r, c, bucket, 0, False, False))]
                if np.any(q != agent.initial_q):
                    actions[r, c] = int(np.argmax(q))
                    values[r, c] = float(q.max())
                    break
    return actions, values


def render_world(
    world: World,
    path: Sequence[Position] | None = None,
    policy: np.ndarray | None = None,
) -> str:
    """Render the map as text.

    With ``path``, visited cells show ``*`` and the final cell ``O``. With
    ``policy`` (from :func:`policy_and_values`), empty cells show an arrow for
    the greedy action instead.
    """
    chars = [[TILE_CHARS[world.tile_at((r, c))] for c in range(world.cols)] for r in range(world.rows)]
    if policy is not None:
        for r in range(world.rows):
            for c in range(world.cols):
                if world.tile_at((r, c)) is Tile.EMPTY and policy[r, c] != NO_POLICY:
                    chars[r][c] = ARROWS[policy[r, c]]
    if path:
        for r, c in path:
            if world.tile_at((r, c)) is Tile.EMPTY:
                chars[r][c] = PATH_CHAR
        end_r, end_c = path[-1]
        if (end_r, end_c) != world.goal:
            chars[end_r][end_c] = BLOB_CHAR
    sr, sc = world.start
    if not path or path[-1] != world.start:
        chars[sr][sc] = START_CHAR
    return "\n".join(" ".join(row) for row in chars)
