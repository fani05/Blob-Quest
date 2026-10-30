"""Tile types and their text representation."""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType


class Tile(IntEnum):
    """Every kind of cell that can appear on the map.

    The blob never observes these values; only the environment reads them.
    """

    EMPTY = 0
    WALL = 1
    GOAL = 2
    TRAP = 3
    PIT = 4
    DRAIN = 5
    ENERGY = 6
    SPEED = 7
    RESET = 8


START_CHAR = "@"
"""Layout character marking the start cell (the cell itself is EMPTY)."""

TILE_CHARS: Mapping[Tile, str] = MappingProxyType(
    {
        Tile.EMPTY: ".",
        Tile.WALL: "#",
        Tile.GOAL: "G",
        Tile.TRAP: "T",
        Tile.PIT: "X",
        Tile.DRAIN: "D",
        Tile.ENERGY: "E",
        Tile.SPEED: "S",
        Tile.RESET: "R",
    }
)

CHAR_TILES: Mapping[str, Tile] = MappingProxyType({char: tile for tile, char in TILE_CHARS.items()})

BOOSTER_TILES: frozenset[Tile] = frozenset({Tile.ENERGY, Tile.SPEED})
"""Tiles that are consumed on first use and tracked in the state's booster mask."""

BLOCKING_TILES: frozenset[Tile] = frozenset({Tile.WALL})
"""Tiles the blob cannot enter."""


def tile_from_char(char: str) -> Tile:
    """Return the tile for a layout character (``@`` counts as EMPTY)."""
    if char == START_CHAR:
        return Tile.EMPTY
    try:
        return CHAR_TILES[char]
    except KeyError as exc:
        valid = "".join(CHAR_TILES) + START_CHAR
        raise ValueError(f"unknown tile character {char!r}; valid characters: {valid}") from exc
