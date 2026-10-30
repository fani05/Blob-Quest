import numpy as np
import pytest

from blobquest.config import ConfigError, WorldConfig
from blobquest.tiles import Tile
from blobquest.world import World, WorldGenerationError, build_world, generate_world, is_solvable


def test_generated_world_is_deterministic() -> None:
    a = build_world(WorldConfig(map_seed=3))
    b = build_world(WorldConfig(map_seed=3))
    assert np.array_equal(a.grid, b.grid)


def test_different_seeds_give_different_maps() -> None:
    a = build_world(WorldConfig(map_seed=1))
    b = build_world(WorldConfig(map_seed=2))
    assert not np.array_equal(a.grid, b.grid)


@pytest.mark.parametrize("seed", range(10))
def test_generated_world_is_solvable_and_has_requested_tiles(seed: int) -> None:
    cfg = WorldConfig(map_seed=seed)
    world = build_world(cfg)
    assert is_solvable(world.grid, world.start, world.goal)
    assert world.goal == cfg.goal_position
    assert world.tile_at(world.start) is Tile.EMPTY
    assert np.count_nonzero(world.grid == Tile.PIT) == cfg.num_pits
    assert np.count_nonzero(world.grid == Tile.ENERGY) == cfg.num_energy_boosters
    assert world.num_boosters == cfg.num_energy_boosters + cfg.num_speed_boosters


def test_layout_roundtrip() -> None:
    layout = ["@.T#", ".XE.", "S.DG"]
    world = World.from_layout(layout, max_boosters=4)
    assert world.to_layout() == layout
    assert world.start == (0, 0)
    assert world.goal == (2, 3)
    assert world.booster_index == {(1, 2): 0, (2, 0): 1}


@pytest.mark.parametrize(
    "layout",
    [
        ["@..", "..."],  # no goal
        ["...", "..G"],  # no start
        ["@.G", "..G"],  # two goals
        ["@..", ".G"],  # ragged
        ["@.?", "..G"],  # unknown char
    ],
)
def test_bad_layouts_rejected(layout: list[str]) -> None:
    with pytest.raises(ConfigError):
        World.from_layout(layout, max_boosters=4)


def test_too_many_boosters_in_layout() -> None:
    with pytest.raises(ConfigError, match="max_boosters"):
        World.from_layout(["@EEG"], max_boosters=1)


def test_unsolvable_generation_raises() -> None:
    cfg = WorldConfig(
        rows=3,
        cols=3,
        wall_density=0.9,
        num_traps=0,
        num_pits=0,
        num_drains=0,
        num_energy_boosters=0,
        num_speed_boosters=0,
        num_resets=0,
        max_generation_attempts=5,
    )
    with pytest.raises(WorldGenerationError):
        generate_world(cfg, np.random.default_rng(0))


def test_solvability_ignores_paths_through_pits() -> None:
    world = World.from_layout(["@XG"], max_boosters=0)
    assert not is_solvable(world.grid, world.start, world.goal)


def test_grid_is_read_only() -> None:
    world = World.from_layout(["@.G"], max_boosters=0)
    with pytest.raises(ValueError):
        world.grid[0, 1] = Tile.WALL
