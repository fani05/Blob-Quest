"""Text rendering of the map, greedy paths and learned policies."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np

from blobquest.agents import TabularAgent
from blobquest.state import State, StateEncoder
from blobquest.tiles import START_CHAR, TILE_CHARS, Tile
from blobquest.world import Position, World

ARROWS: tuple[str, ...] = ("↑", "→", "↓", "←", "↗", "↘", "↙", "↖")
PATH_CHAR = "*"
BLOB_CHAR = "O"
FOG_CHAR = "?"
NO_POLICY = -1

LEGEND = (
    f"{START_CHAR} start  G goal  # wall  T trap  X pit  D drain  "
    f"E energy  S speed  R reset  {PATH_CHAR} path  {BLOB_CHAR} blob"
)
FOG_LEGEND = f"{FOG_CHAR} never visited during training"


def policy_and_values(
    agent: TabularAgent, encoder: StateEncoder, world: World, max_bucket: int
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy action and state value for every cell, as a 2D slice of the Q-table.

    The full state has six dimensions, so a map can only show a slice. For each
    cell we use "no boosters used, no drain/speed active" and the highest energy
    bucket ``<= max_bucket`` whose Q-row has actually been trained - i.e. the
    best-rested blob the agent has ever been on that cell. With sensing, the
    most-informed trained neighbour view wins within a bucket. Cells that are walls
    or were never trained get ``NO_POLICY`` and ``NaN``; ties pick the lowest
    action index.
    """
    actions = np.full((world.rows, world.cols), NO_POLICY, dtype=np.int64)
    values = np.full((world.rows, world.cols), np.nan)
    for r in range(world.rows):
        for c in range(world.cols):
            if world.tile_at((r, c)) is Tile.WALL:
                continue
            for bucket, view in product(range(max_bucket, -1, -1), encoder.views(r, c)):
                q = agent.q[encoder.encode(State(r, c, bucket, 0, False, False, view))]
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
    _mark_path(chars, world, path)
    return "\n".join(" ".join(row) for row in chars)


def render_fog(world: World, explored: np.ndarray, path: Sequence[Position] | None = None) -> str:
    """Render the map as the blob knows it: cells it never entered show ``?``.

    ``explored`` is a boolean ``(rows, cols)`` mask (``TrainingHistory.explored``).
    Explored cells show their real tile; ``path`` is marked as in :func:`render_world`.
    """
    if explored.shape != (world.rows, world.cols):
        raise ValueError(f"explored mask shape {explored.shape} does not match map {(world.rows, world.cols)}")
    chars = [
        [TILE_CHARS[world.tile_at((r, c))] if explored[r, c] else FOG_CHAR for c in range(world.cols)]
        for r in range(world.rows)
    ]
    _mark_path(chars, world, path)
    return "\n".join(" ".join(row) for row in chars)


def _mark_path(chars: list[list[str]], world: World, path: Sequence[Position] | None) -> None:
    """Overlay ``*`` on empty path cells, ``O`` where the blob stopped, and ``@`` at the start."""
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


def side_by_side(left: str, right: str, left_title: str = "", right_title: str = "", gap: int = 4) -> str:
    """Place two multi-line text blocks next to each other, each under an optional title."""
    titled = bool(left_title or right_title)
    left_lines = ([left_title] if titled else []) + left.splitlines()
    right_lines = ([right_title] if titled else []) + right.splitlines()
    width = max((len(line) for line in left_lines), default=0) + gap
    height = max(len(left_lines), len(right_lines))
    left_lines += [""] * (height - len(left_lines))
    right_lines += [""] * (height - len(right_lines))
    return "\n".join(f"{a:<{width}}{b}".rstrip() for a, b in zip(left_lines, right_lines, strict=True))


REPLAY_BLOB_CHAR = "@"
ENERGY_FULL = "#"
ENERGY_EMPTY = "."


def render_replay(world: World, position: Position, trail: Sequence[Position] = ()) -> str:
    """One replay frame: the true map, ``*`` on empty cells walked so far, the blob as ``@``."""
    chars = [[TILE_CHARS[world.tile_at((r, c))] for c in range(world.cols)] for r in range(world.rows)]
    for r, c in trail:
        if world.tile_at((r, c)) is Tile.EMPTY:
            chars[r][c] = PATH_CHAR
    chars[position[0]][position[1]] = REPLAY_BLOB_CHAR
    return "\n".join(" ".join(row) for row in chars)


def energy_bar(energy: int, maximum: int, width: int = 20) -> str:
    """Fixed-width bar such as ``[##########..........]  30/60``."""
    filled = round(width * min(max(energy, 0), maximum) / maximum)
    return f"[{ENERGY_FULL * filled}{ENERGY_EMPTY * (width - filled)}] {energy:>3}/{maximum}"
