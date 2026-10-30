"""Shared test helpers."""

from __future__ import annotations

from typing import Any

import pytest

from blobquest.config import Config
from blobquest.env import BlobEnv
from blobquest.world import World, build_world


def make_config(layout: list[str] | None = None, **sections: dict[str, Any]) -> Config:
    """Default config, optionally with a fixed layout and section overrides."""
    config = Config()
    if layout is not None:
        sections = {**sections, "world": {**sections.get("world", {}), "layout": tuple(layout)}}
    return config.with_overrides(**sections) if sections else config


def make_env(layout: list[str], **sections: dict[str, Any]) -> tuple[BlobEnv, World, Config]:
    """Environment over a hand-written layout."""
    config = make_config(layout, **sections)
    world = build_world(config.world)
    return BlobEnv(world, config), world, config


@pytest.fixture
def corridor() -> list[str]:
    """A 1x5 corridor: start, empty cells, goal."""
    return ["@...G"]
