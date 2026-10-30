"""A log of failed episodes: where the blob died and whether it died there before."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields

import numpy as np

from blobquest.env import Outcome
from blobquest.tiles import TILE_CHARS
from blobquest.world import Position, World

DEATH_OUTCOMES: frozenset[Outcome] = frozenset({Outcome.PIT, Outcome.EXHAUSTED})
"""Failures that end the blob on a specific cell. Timeouts are logged but are not deaths."""


@dataclass(frozen=True)
class Mistake:
    """One failed episode."""

    episode: int
    outcome: str
    row: int
    col: int
    energy: int
    repeated: bool
    """True if this is a death on a cell where an earlier episode already died."""


@dataclass(frozen=True)
class WindowRate:
    """Repeated-mistake rate over episodes ``[start, end)``."""

    start: int
    end: int
    deaths: int
    repeats: int

    @property
    def rate(self) -> float:
        """Share of this window's deaths that happened on an already-deadly cell (NaN if none)."""
        return self.repeats / self.deaths if self.deaths else float("nan")


MISTAKE_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(Mistake))


class MistakeLog:
    """Records every failed episode and answers "does the blob keep dying in the same place?"."""

    def __init__(self) -> None:
        self.mistakes: list[Mistake] = []
        self._death_counts: Counter[Position] = Counter()

    def __len__(self) -> int:
        return len(self.mistakes)

    @property
    def deaths(self) -> list[Mistake]:
        """Failed episodes that ended in a death (pit or exhaustion)."""
        return [m for m in self.mistakes if m.outcome in DEATH_OUTCOMES]

    def record(self, episode: int, outcome: Outcome, position: Position, energy: int) -> Mistake | None:
        """Log a finished episode; successful (goal) episodes are ignored and return ``None``."""
        if outcome is Outcome.GOAL or outcome is Outcome.RUNNING:
            return None
        is_death = outcome in DEATH_OUTCOMES
        repeated = is_death and self._death_counts[position] > 0
        if is_death:
            self._death_counts[position] += 1
        mistake = Mistake(episode, outcome.value, position[0], position[1], energy, repeated)
        self.mistakes.append(mistake)
        return mistake

    def repeat_rate(self) -> float:
        """Repeated-mistake rate over the whole run (NaN if the blob never died)."""
        deaths = self.deaths
        return sum(m.repeated for m in deaths) / len(deaths) if deaths else float("nan")

    def repeat_rate_by_window(self, window: int, num_episodes: int) -> list[WindowRate]:
        """Repeated-mistake rate in consecutive, non-overlapping blocks of ``window`` episodes."""
        if window < 1:
            raise ValueError("window must be >= 1")
        deaths = np.zeros(max(num_episodes, 0), dtype=np.int64)
        repeats = np.zeros_like(deaths)
        for m in self.deaths:
            if m.episode < num_episodes:
                deaths[m.episode] += 1
                repeats[m.episode] += m.repeated
        rates = []
        for start in range(0, num_episodes, window):
            end = min(start + window, num_episodes)
            rates.append(WindowRate(start, end, int(deaths[start:end].sum()), int(repeats[start:end].sum())))
        return rates

    def deadliest_cells(self, n: int = 10) -> list[tuple[Position, int]]:
        """The ``n`` cells with the most deaths, ties broken by row then column."""
        return sorted(self._death_counts.items(), key=lambda item: (-item[1], item[0]))[:n]

    def format_deadliest(self, world: World | None = None, n: int = 10) -> str:
        """Text table of :meth:`deadliest_cells`, with the tile letter when ``world`` is given."""
        cells = self.deadliest_cells(n)
        if not cells:
            return "no deaths recorded"
        total = len(self.deaths)
        lines = [f"{'rank':>4}  {'cell':<9} {'tile':<4} {'deaths':>6}  {'share':>6}"]
        for rank, ((r, c), count) in enumerate(cells, start=1):
            tile = TILE_CHARS[world.tile_at((r, c))] if world is not None else "-"
            lines.append(f"{rank:>4}  {f'({r},{c})':<9} {tile:<4} {count:>6}  {count / total:>6.1%}")
        return "\n".join(lines)
