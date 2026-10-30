from pathlib import Path

import numpy as np
import pytest

from blobquest.cli import main
from blobquest.config import AgentConfig, Config, save_config
from blobquest.persistence import EXPLORED_FILE, load_explored
from blobquest.render import policy_and_values
from blobquest.state import FOG, State, StateEncoder
from blobquest.tiles import Tile
from blobquest.trainer import Trainer, make_encoder
from blobquest.world import build_world
from tests.conftest import make_config, make_env

RIGHT = 1

# Neighbours of (1, 1): up '.', right 'T', down '#', left 'X'.
CROSS = [
    "@..",
    "X.T",
    ".#G",
]

SMALL_MAP = [
    "@..#.",
    ".#.X.",
    ".T...",
    "X.#.E",
    "...DG",
]


def _grid(layout: list[str]) -> np.ndarray:
    return build_world(make_config(layout).world).grid


def test_sensing_is_off_by_default() -> None:
    assert AgentConfig().sensing is False
    env, _, _ = make_env(CROSS)
    assert env.reset().neighbours == ()


def test_plain_encoder_ignores_neighbours_field() -> None:
    enc = StateEncoder(3, 3, 2, 1)
    assert not enc.sensing
    assert enc.encode(State(1, 1, 1, 0, False, True)) == enc.encode(State(1, 1, 1, 0, False, True, ()))


def test_sensing_encoder_roundtrip_covers_every_index() -> None:
    grid = _grid(CROSS)
    enc = StateEncoder(3, 3, num_buckets=2, num_boosters=1, sensing_grid=grid)
    seen = set()
    for idx in range(enc.num_states):
        state = enc.decode(idx)
        assert len(state.neighbours) == 4
        assert enc.encode(state) == idx
        seen.add(state)
    assert len(seen) == enc.num_states


def test_sensing_encoder_size_counts_only_enterable_neighbours() -> None:
    grid = _grid(CROSS)
    plain = StateEncoder(3, 3, 2, 1)
    sensing = StateEncoder(3, 3, 2, 1, sensing_grid=grid)
    block = plain.num_states // 9
    enterable = 0
    for r in range(3):
        for c in range(3):
            k = sum(
                0 <= r + dr < 3 and 0 <= c + dc < 3 and grid[r + dr, c + dc] != Tile.WALL
                for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1))
            )
            enterable += 2**k
    assert sensing.num_states == block * enterable
    assert plain.num_states < sensing.num_states <= 16 * plain.num_states


def test_default_map_state_count_stays_manageable() -> None:
    config = Config().with_overrides(agent={"sensing": True})
    world = build_world(config.world)
    plain = make_encoder(world, Config())
    sensing = make_encoder(world, config)
    assert plain.num_states == 36_000
    # Lossless, and far below the dense 16x (576k) or naive 10**4x (360M) layouts.
    assert sensing.num_states < 16 * plain.num_states
    assert sensing.num_states * 4 * 8 < 20 * 2**20  # Q-table under 20 MiB


def test_encoder_rejects_impossible_neighbours() -> None:
    enc = StateEncoder(3, 3, 1, 0, sensing_grid=_grid(CROSS))
    ok = State(1, 1, 0, 0, False, False, (int(Tile.EMPTY), int(Tile.TRAP), FOG, int(Tile.PIT)))
    enc.encode(ok)
    with pytest.raises(ValueError):  # right neighbour is a trap, not a pit
        enc.encode(ok._replace(neighbours=(FOG, int(Tile.PIT), FOG, FOG)))
    with pytest.raises(ValueError):  # walls are never visited, so never sensed
        enc.encode(ok._replace(neighbours=(FOG, FOG, int(Tile.WALL), FOG)))
    with pytest.raises(ValueError):  # off-map neighbour
        enc.encode(State(0, 0, 0, 0, False, False, (int(Tile.EMPTY), FOG, FOG, FOG)))
    with pytest.raises(ValueError):
        enc.encode(ok._replace(neighbours=()))


def test_known_neighbours_get_their_own_rows() -> None:
    enc = StateEncoder(3, 3, 1, 0, sensing_grid=_grid(CROSS))
    fog = State(1, 1, 0, 0, False, False, (FOG,) * 4)
    known = fog._replace(neighbours=(FOG, int(Tile.TRAP), FOG, FOG))
    assert enc.encode(fog) != enc.encode(known)


def test_views_are_most_informed_first() -> None:
    enc = StateEncoder(3, 3, 1, 0, sensing_grid=_grid(CROSS))
    views = enc.views(1, 1)
    assert len(views) == 8  # three enterable neighbours, the wall below is never sensed
    assert views[0] == (int(Tile.EMPTY), int(Tile.TRAP), FOG, int(Tile.PIT))
    assert views[-1] == (FOG,) * 4
    assert StateEncoder(3, 3, 1, 0).views(1, 1) == [()]


def test_env_senses_only_known_enterable_neighbours() -> None:
    env, _, _ = make_env(CROSS, agent={"sensing": True})
    env.reset()
    env.step(RIGHT)  # now at (0, 1); neighbours: up off-map, right '.', down '.', left '@'
    assert env.observe().neighbours == (FOG,) * 4

    known = np.ones((3, 3), dtype=bool)
    env.set_knowledge(known)
    assert env.observe().neighbours == (FOG, int(Tile.EMPTY), int(Tile.EMPTY), int(Tile.EMPTY))
    known[:] = False  # set_knowledge copies, so later changes do not leak in
    assert env.observe().neighbours[1] == int(Tile.EMPTY)


def test_env_never_senses_walls() -> None:
    env, _, _ = make_env(["@#G"], agent={"sensing": True})
    env.set_knowledge(np.ones((1, 3), dtype=bool))
    assert env.reset().neighbours == (FOG, FOG, FOG, FOG)


def test_set_knowledge_rejects_wrong_shape() -> None:
    env, _, _ = make_env(CROSS, agent={"sensing": True})
    with pytest.raises(ValueError):
        env.set_knowledge(np.zeros((2, 2), dtype=bool))


def _sensing_config(seed: int = 0, episodes: int = 600) -> Config:
    return make_config(
        SMALL_MAP,
        agent={"sensing": True},
        training={"episodes": episodes, "seed": seed, "eval_every": 100, "eval_episodes": 2},
        energy={"initial": 20, "maximum": 30},
    )


def test_first_episode_senses_nothing_later_ones_do() -> None:
    trainer = Trainer(_sensing_config(episodes=2))
    trainer.run_episode(0)
    # Knowledge during episode 0 was empty, so only all-FOG rows were trained.
    rows = np.flatnonzero(np.any(trainer.agent.q != trainer.agent.initial_q, axis=1))
    assert all(trainer.encoder.decode(int(i)).neighbours == (FOG,) * 4 for i in rows)
    trainer.run_episode(1)
    rows = np.flatnonzero(np.any(trainer.agent.q != trainer.agent.initial_q, axis=1))
    assert any(trainer.encoder.decode(int(i)).neighbours != (FOG,) * 4 for i in rows)


def test_agent_learns_small_map_with_sensing() -> None:
    trainer = Trainer(_sensing_config())
    history = trainer.train()
    assert trainer.agent.q.shape == (trainer.encoder.num_states, trainer.env.num_actions)
    assert history.evaluations[-1].success_rate == 1.0
    policy, _ = policy_and_values(trainer.agent, trainer.encoder, trainer.world, max_bucket=4)
    assert (policy >= 0).any()


def test_sensing_training_is_reproducible() -> None:
    a = Trainer(_sensing_config(seed=4, episodes=200)).train()
    b = Trainer(_sensing_config(seed=4, episodes=200)).train()
    assert a.episodes == b.episodes


def test_cli_evaluate_restores_knowledge(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "cfg.yaml"
    save_config(_sensing_config().with_overrides(output={"runs_dir": str(tmp_path / "runs")}), config_path)
    assert main(["train", "--config", str(config_path), "--episodes", "300", "--no-plots"]) == 0
    run_dir = next((tmp_path / "runs").iterdir())
    assert (run_dir / EXPLORED_FILE).is_file()
    explored = load_explored(run_dir)
    assert explored is not None and explored.any()
    assert "with sensing" in capsys.readouterr().out
    assert main(["evaluate", str(run_dir), "--episodes", "2"]) == 0
    assert "Greedy success rate" in capsys.readouterr().out


def test_load_explored_missing_for_old_runs(tmp_path: Path) -> None:
    assert load_explored(tmp_path) is None
