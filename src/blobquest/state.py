"""What the blob can perceive, and how that maps to a Q-table row."""

from __future__ import annotations

from typing import NamedTuple


class State(NamedTuple):
    """The blob's own state: position and internal status, never the map."""

    row: int
    col: int
    energy_bucket: int
    used_boosters_mask: int
    drain_active: bool
    speed_active: bool


def energy_bucket(energy: int, maximum: int, num_buckets: int) -> int:
    """Discretise ``energy`` in ``[0, maximum]`` into ``num_buckets`` equal bins."""
    clipped = min(max(energy, 0), maximum)
    return min(clipped * num_buckets // (maximum + 1), num_buckets - 1)


class StateEncoder:
    """Bijection between ``State`` tuples and integer indices ``0..num_states-1``.

    The index is a mixed-radix number, the same layout ``numpy.ravel_multi_index``
    would produce for ``shape``.
    """

    def __init__(self, rows: int, cols: int, num_buckets: int, num_boosters: int) -> None:
        self.shape: tuple[int, ...] = (rows, cols, num_buckets, 2**num_boosters, 2, 2)
        self.num_states: int = 1
        for size in self.shape:
            self.num_states *= size

    def encode(self, state: State) -> int:
        """Return the integer index of ``state``."""
        _, cols, buckets, masks, _, _ = self.shape
        index = state.row * cols + state.col
        index = index * buckets + state.energy_bucket
        index = index * masks + state.used_boosters_mask
        index = index * 2 + int(state.drain_active)
        return index * 2 + int(state.speed_active)

    def decode(self, index: int) -> State:
        """Inverse of :meth:`encode`."""
        if not 0 <= index < self.num_states:
            raise ValueError(f"state index {index} out of range [0, {self.num_states})")
        values = []
        for size in reversed(self.shape):
            index, digit = divmod(index, size)
            values.append(digit)
        row, col, bucket, mask, drain, speed = reversed(values)
        return State(row, col, bucket, mask, bool(drain), bool(speed))
