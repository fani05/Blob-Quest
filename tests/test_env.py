import pytest

from blobquest.env import Event, Outcome
from tests.conftest import make_env

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DOWN_RIGHT = 5


def test_reset_observation(corridor: list[str]) -> None:
    env, world, config = make_env(corridor)
    state = env.reset()
    assert (state.row, state.col) == world.start
    assert env.energy == config.energy.initial
    assert state.used_boosters_mask == 0
    assert not state.drain_active and not state.speed_active


def test_plain_step_costs_energy(corridor: list[str]) -> None:
    env, _, config = make_env(corridor)
    env.reset()
    result = env.step(RIGHT)
    assert env.position == (0, 1)
    assert env.energy == config.energy.initial - config.energy.step_cost
    assert result.reward == config.rewards.step
    assert not result.terminated and not result.truncated
    assert result.outcome is Outcome.RUNNING


@pytest.mark.parametrize(("layout", "action"), [(["@#G"], RIGHT), (["@.G"], UP), (["@.G"], LEFT)])
def test_wall_and_edge_bumps(layout: list[str], action: int) -> None:
    env, world, config = make_env(layout)
    env.reset()
    result = env.step(action)
    assert env.position == world.start
    assert result.reward == config.rewards.step + config.rewards.wall_bump
    assert result.events == (Event.WALL_BUMP,)


def test_goal_terminates_with_reward() -> None:
    env, _, config = make_env(["@G"])
    env.reset()
    result = env.step(RIGHT)
    assert result.terminated and result.outcome is Outcome.GOAL
    assert result.reward == config.rewards.step + config.rewards.goal


def test_pit_kills() -> None:
    env, _, config = make_env(["@XG"])
    env.reset()
    result = env.step(RIGHT)
    assert result.terminated and result.outcome is Outcome.PIT
    assert result.reward == config.rewards.step + config.rewards.pit


def test_trap_drains_energy() -> None:
    env, _, config = make_env(["@T.G"])
    env.reset()
    result = env.step(RIGHT)
    e = config.energy
    assert env.energy == e.initial - e.step_cost - e.trap_loss
    assert result.reward == config.rewards.step + config.rewards.trap
    assert Event.TRAP in result.events


def test_drain_doubles_cost_for_configured_steps() -> None:
    env, _, config = make_env(["@D......G"])
    e = config.energy
    env.reset()
    env.step(RIGHT)
    assert env.observe().drain_active
    energy = env.energy
    for _ in range(e.drain_duration):
        result = env.step(RIGHT)
        assert env.energy == energy - e.drain_step_cost
        assert result.reward == config.rewards.step + config.rewards.drain_step
        energy = env.energy
    assert not env.observe().drain_active
    env.step(RIGHT)
    assert env.energy == energy - e.step_cost


def test_energy_booster_is_single_use_and_capped() -> None:
    env, world, config = make_env(["@E.G"], energy={"initial": 20, "maximum": 30})
    e = config.energy
    env.reset()
    result = env.step(RIGHT)
    assert env.energy == min(20 - e.step_cost + e.booster_gain, 30)
    assert result.reward == config.rewards.step + config.rewards.energy_booster
    assert env.observe().used_boosters_mask == 1 << world.booster_index[(0, 1)]
    env.step(RIGHT)
    energy = env.energy
    result = env.step(LEFT)  # back onto the used booster
    assert env.energy == energy - e.step_cost
    assert Event.ENERGY_BOOST not in result.events


def test_energy_booster_respects_maximum() -> None:
    env, _, config = make_env(["@E.G"], energy={"initial": 50, "maximum": 60})
    env.reset()
    env.step(RIGHT)
    assert env.energy == config.energy.maximum


def test_speed_booster_moves_two_cells_for_three_moves() -> None:
    env, _, _ = make_env(["@S......G"])
    env.reset()
    env.step(RIGHT)
    assert env.observe().speed_active
    positions = []
    for _ in range(3):
        env.step(RIGHT)
        positions.append(env.position[1])
    assert positions == [3, 5, 7]
    assert not env.observe().speed_active
    result = env.step(RIGHT)
    assert env.position == (0, 8) and result.outcome is Outcome.GOAL


def test_speed_move_stops_at_wall_without_bump_penalty() -> None:
    env, _, config = make_env(["@S.#G"])
    env.reset()
    env.step(RIGHT)
    result = env.step(RIGHT)
    assert env.position == (0, 2)
    assert Event.WALL_BUMP not in result.events
    assert result.reward == config.rewards.step


def test_speed_move_cannot_jump_over_pit() -> None:
    env, _, _ = make_env(["@SX..G"])
    env.reset()
    env.step(RIGHT)
    result = env.step(RIGHT)
    assert result.outcome is Outcome.PIT


def test_reset_tile_teleports_to_start() -> None:
    env, world, config = make_env(["@.RG"])
    env.reset()
    env.step(RIGHT)
    result = env.step(RIGHT)
    assert env.position == world.start
    assert result.reward == config.rewards.step + config.rewards.reset
    assert not result.terminated


def test_running_out_of_energy_fails() -> None:
    env, _, config = make_env(["@....G"], energy={"initial": 2, "maximum": 10})
    env.reset()
    assert not env.step(RIGHT).terminated
    result = env.step(RIGHT)
    assert result.terminated and result.outcome is Outcome.EXHAUSTED
    assert result.reward == config.rewards.step + config.rewards.exhausted


def test_reaching_goal_with_last_energy_succeeds() -> None:
    env, _, _ = make_env(["@G"], energy={"initial": 1, "maximum": 10})
    env.reset()
    assert env.step(RIGHT).outcome is Outcome.GOAL


def test_step_limit_truncates() -> None:
    env, _, _ = make_env(["@.....G"], training={"max_steps_per_episode": 2})
    env.reset()
    env.step(RIGHT)
    result = env.step(RIGHT)
    assert result.truncated and not result.terminated
    assert result.outcome is Outcome.TIMEOUT


def test_step_after_done_raises() -> None:
    env, _, _ = make_env(["@G"])
    env.reset()
    env.step(RIGHT)
    with pytest.raises(RuntimeError):
        env.step(RIGHT)


def test_invalid_action_raises(corridor: list[str]) -> None:
    env, _, _ = make_env(corridor)
    env.reset()
    with pytest.raises(ValueError):
        env.step(4)


def test_diagonal_moves() -> None:
    env, _, _ = make_env(["@.", ".G"], movement={"diagonals": True})
    assert env.num_actions == 8
    env.reset()
    assert env.step(DOWN_RIGHT).outcome is Outcome.GOAL


def test_energy_bucket_in_state_drops() -> None:
    env, _, _ = make_env(["@T.G"], energy={"initial": 30, "maximum": 30, "num_buckets": 3})
    start_bucket = env.reset().energy_bucket
    assert env.step(RIGHT).state.energy_bucket < start_bucket
