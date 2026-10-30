"""What the blob can perceive, and how that maps to a Q-table row."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from blobquest.tiles import BLOCKING_TILES

FOG = -1
"""Sensed value for a neighbour the blob has not visited in an earlier episode."""

NEIGHBOUR_OFFSETS: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
"""Order of ``State.neighbours``: up, right, down, left (same as the first four actions)."""


class State(NamedTuple):
    """The blob's own state: position and internal status, never the map.

    With sensing enabled, ``neighbours`` holds the tile (as ``int(Tile)``) of the
    four adjacent cells the blob already knows, or ``FOG``; otherwise it is empty.
    """

    row: int
    col: int
    energy_bucket: int
    used_boosters_mask: int
    drain_active: bool
    speed_active: bool
    neighbours: tuple[int, ...] = ()


def energy_bucket(energy: int, maximum: int, num_buckets: int) -> int:
    """Discretise ``energy`` in ``[0, maximum]`` into ``num_buckets`` equal bins."""
    clipped = min(max(energy, 0), maximum)
    return min(clipped * num_buckets // (maximum + 1), num_buckets - 1)


class StateEncoder:
    """Bijection between ``State`` tuples and integer indices ``0..num_states-1``.

    Without sensing, the index is a mixed-radix number, the same layout
    ``numpy.ravel_multi_index`` would produce for ``shape``.

    With sensing (``sensing_grid`` given), the position already pins down which
    tile each neighbour *is*; the only new information is which neighbours the
    blob has discovered. So each cell gets a block of ``block * 2**k`` indices,
    where ``k`` is its number of enterable neighbours (walls and off-map cells
    can never be visited, so they are always ``FOG``), and the low ``k`` bits are
    the "known" flags. Tile types are restored from the grid on decode. That is
    at most 16x the plain table instead of the ``10**4``x a naive per-neighbour
    tile encoding would cost, with no unreachable rows.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        num_buckets: int,
        num_boosters: int,
        sensing_grid: np.ndarray | None = None,
    ) -> None:
        self.shape: tuple[int, ...] = (rows, cols, num_buckets, 2**num_boosters, 2, 2)
        self.num_states: int = 1
        for size in self.shape:
            self.num_states *= size
        self._block = self.num_states // (rows * cols)
        self.sensing = sensing_grid is not None
        if sensing_grid is None:
            return
        if sensing_grid.shape != (rows, cols):
            raise ValueError(f"sensing grid shape {sensing_grid.shape} does not match {(rows, cols)}")
        # For each cell: (direction, tile) of every neighbour that can ever be visited.
        self._sensable: list[tuple[tuple[int, int], ...]] = []
        for r in range(rows):
            for c in range(cols):
                cells = []
                for d, (dr, dc) in enumerate(NEIGHBOUR_OFFSETS):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and int(sensing_grid[nr, nc]) not in BLOCKING_TILES:
                        cells.append((d, int(sensing_grid[nr, nc])))
                self._sensable.append(tuple(cells))
        sizes = [self._block * 2 ** len(cells) for cells in self._sensable]
        self._offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
        self.num_states = int(self._offsets[-1])

    def encode(self, state: State) -> int:
        """Return the integer index of ``state``."""
        _, cols, buckets, masks, _, _ = self.shape
        index = state.row * cols + state.col
        index = index * buckets + state.energy_bucket
        index = index * masks + state.used_boosters_mask
        index = index * 2 + int(state.drain_active)
        index = index * 2 + int(state.speed_active)
        if not self.sensing:
            return index
        cell, inner = divmod(index, self._block)
        sensable = self._sensable[cell]
        return int(self._offsets[cell]) + (inner << len(sensable)) + self._known_bits(state, sensable)

    def decode(self, index: int) -> State:
        """Inverse of :meth:`encode`."""
        if not 0 <= index < self.num_states:
            raise ValueError(f"state index {index} out of range [0, {self.num_states})")
        neighbours: tuple[int, ...] = ()
        if self.sensing:
            cell = int(np.searchsorted(self._offsets, index, side="right")) - 1
            sensable = self._sensable[cell]
            inner, known = divmod(index - int(self._offsets[cell]), 2 ** len(sensable))
            index = cell * self._block + inner
            neighbours = _expand(sensable, known)
        values = []
        for size in reversed(self.shape):
            index, digit = divmod(index, size)
            values.append(digit)
        row, col, bucket, mask, drain, speed = reversed(values)
        return State(row, col, bucket, mask, bool(drain), bool(speed), neighbours)

    def views(self, row: int, col: int) -> list[tuple[int, ...]]:
        """Every valid ``neighbours`` value for a cell, most-informed first (``[()]`` without sensing)."""
        if not self.sensing:
            return [()]
        sensable = self._sensable[row * self.shape[1] + col]
        order = sorted(range(2 ** len(sensable)), key=lambda k: (-k.bit_count(), k))
        return [_expand(sensable, known) for known in order]

    def _known_bits(self, state: State, sensable: tuple[tuple[int, int], ...]) -> int:
        """Pack which sensable neighbours are known into bits, checking they match the map."""
        if len(state.neighbours) != len(NEIGHBOUR_OFFSETS):
            raise ValueError(f"sensing needs {len(NEIGHBOUR_OFFSETS)} neighbours, got {state.neighbours!r}")
        true_tiles = dict(sensable)
        bits = 0
        for d, seen in enumerate(state.neighbours):
            if seen == FOG:
                continue
            if true_tiles.get(d) != seen:
                raise ValueError(f"neighbour {d} of ({state.row}, {state.col}) cannot be sensed as {seen}")
            bits |= 1 << [s[0] for s in sensable].index(d)
        return bits


def _expand(sensable: tuple[tuple[int, int], ...], known: int) -> tuple[int, ...]:
    """Neighbour tiles for the ``known`` bit-set over ``sensable`` (others are ``FOG``)."""
    seen = [FOG] * len(NEIGHBOUR_OFFSETS)
    for bit, (d, tile) in enumerate(sensable):
        if known >> bit & 1:
            seen[d] = tile
    return tuple(seen)
