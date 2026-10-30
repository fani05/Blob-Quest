from pathlib import Path

import numpy as np
import pytest

from blobquest.cli import main
from blobquest.metrics import moving_average
from blobquest.persistence import EPISODES_FILE, PLOTS_DIR, QTABLE_FILE, load_run
from blobquest.render import policy_and_values, render_world
from blobquest.trainer import Trainer
from tests.conftest import make_config

SMALL_MAP = [
    "@..#.",
    ".#.X.",
    ".T...",
    "X.#.E",
    "...DG",
]


def _small_config(algorithm: str = "q_learning", seed: int = 0):
    return make_config(
        SMALL_MAP,
        agent={"algorithm": algorithm},
        training={"episodes": 600, "seed": seed, "eval_every": 100, "eval_episodes": 2},
        energy={"initial": 20, "maximum": 30},
    )


@pytest.mark.parametrize("algorithm", ["q_learning", "sarsa"])
def test_agent_learns_small_map(algorithm: str) -> None:
    trainer = Trainer(_small_config(algorithm))
    history = trainer.train()
    assert len(history.episodes) == 600
    assert history.evaluations[-1].success_rate == 1.0
    assert trainer.final_rollout().outcome.value == "goal"


def test_training_is_reproducible() -> None:
    a = Trainer(_small_config(seed=11)).train()
    b = Trainer(_small_config(seed=11)).train()
    assert a.episodes == b.episodes
    assert np.array_equal(a.visits, b.visits)


def test_different_seeds_differ() -> None:
    a = Trainer(_small_config(seed=1)).train()
    b = Trainer(_small_config(seed=2)).train()
    assert a.episodes != b.episodes


def test_moving_average() -> None:
    assert np.allclose(moving_average(np.array([1.0, 3.0, 5.0, 7.0]), 2), [1.0, 2.0, 4.0, 6.0])


def test_render_contains_path_and_policy() -> None:
    trainer = Trainer(_small_config())
    trainer.train()
    rollout = trainer.final_rollout()
    text = render_world(trainer.world, path=rollout.path)
    assert "*" in text and "G" in text
    policy, values = policy_and_values(trainer.agent, trainer.encoder, trainer.world, max_bucket=4)
    assert (policy >= 0).any()
    assert np.isnan(values[0, 3])  # wall


def test_cli_train_and_evaluate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "cfg.yaml"
    from blobquest.config import save_config

    save_config(_small_config().with_overrides(output={"runs_dir": str(tmp_path / "runs")}), config_path)
    assert main(["train", "--config", str(config_path), "--episodes", "300"]) == 0
    run_dir = next((tmp_path / "runs").iterdir())
    for name in (EPISODES_FILE, QTABLE_FILE, "config.yaml", "world.txt", "summary.yaml"):
        assert (run_dir / name).is_file()
    pngs = sorted(p.name for p in (run_dir / PLOTS_DIR).glob("*.png"))
    assert len(pngs) == 8

    config, world, q = load_run(run_dir)
    assert config.training.episodes == 300
    assert world.to_layout() == SMALL_MAP
    assert q.ndim == 2

    assert main(["evaluate", str(run_dir), "--episodes", "2"]) == 0
    assert "Greedy success rate" in capsys.readouterr().out


def test_cli_reports_bad_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("energy:\n  initial: -1\n", encoding="utf-8")
    assert main(["show-map", "--config", str(bad)]) == 2
    assert "error" in capsys.readouterr().err
