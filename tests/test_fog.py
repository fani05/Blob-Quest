import numpy as np
import pytest

from blobquest.render import FOG_CHAR, render_fog, render_world, side_by_side
from blobquest.trainer import Trainer
from blobquest.world import build_world
from tests.conftest import make_config, make_env

RIGHT, LEFT = 1, 3

MAP = [
    "@.T",
    ".#.",
    "..G",
]


def _world():
    return build_world(make_config(MAP).world)


def _grid(text: str) -> list[list[str]]:
    return [line.split(" ") for line in text.splitlines()]


def test_unexplored_cells_are_fog() -> None:
    world = _world()
    explored = np.zeros((3, 3), dtype=bool)
    explored[0, :] = True
    grid = _grid(render_fog(world, explored))
    assert grid[0] == ["@", ".", "T"]
    assert grid[1] == [FOG_CHAR] * 3
    assert grid[2] == [FOG_CHAR] * 3


def test_fully_explored_fog_matches_true_map() -> None:
    world = _world()
    assert render_fog(world, np.ones((3, 3), dtype=bool)) == render_world(world)


def test_fog_marks_greedy_route() -> None:
    world = _world()
    explored = np.zeros((3, 3), dtype=bool)
    path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    for cell in path:
        explored[cell] = True
    grid = _grid(render_fog(world, explored, path))
    assert grid[1][0] == "*" and grid[2][0] == "*" and grid[2][1] == "*"
    assert grid[2][2] == "G"
    assert grid[0][2] == FOG_CHAR  # trap off the route stays hidden
    assert grid[1][1] == FOG_CHAR  # walls are never entered


def test_fog_rejects_wrong_mask_shape() -> None:
    with pytest.raises(ValueError):
        render_fog(_world(), np.ones((2, 2), dtype=bool))


def test_env_reports_cells_entered() -> None:
    env, _, _ = make_env(["@S...G"])
    env.reset()
    assert env.cells_entered == ()
    env.step(RIGHT)
    assert env.cells_entered == ((0, 1),)
    env.step(RIGHT)  # speed boost active: passes through two cells
    assert env.cells_entered == ((0, 2), (0, 3))
    env.step(LEFT)
    env.step(LEFT)
    env.step(LEFT)
    env.step(LEFT)  # into the left edge
    assert env.cells_entered == ()


def test_trainer_tracks_explored_cells() -> None:
    config = make_config(MAP, training={"episodes": 100, "seed": 0, "eval_every": 50, "eval_episodes": 1})
    trainer = Trainer(config)
    history = trainer.train()
    assert history.explored[trainer.world.start]
    assert not history.explored[1, 1]  # wall
    assert np.array_equal(history.explored & (history.visits > 0), history.visits > 0)
    for cell in trainer.final_rollout().path:
        assert history.explored[cell]
    fog = render_fog(trainer.world, history.explored)
    hidden = sum(line.count(FOG_CHAR) for line in fog.splitlines())
    assert hidden == int((~history.explored).sum())


def test_side_by_side_aligns_columns() -> None:
    text = side_by_side("a b\nc d", "x\ny\nz", "left", "right", gap=2)
    lines = text.splitlines()
    assert lines[0] == "left  right"
    assert lines[1] == "a b   x"
    assert lines[2] == "c d   y"
    assert lines[3] == "      z"
