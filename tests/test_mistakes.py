import csv
from pathlib import Path

import numpy as np
import pytest

from blobquest.config import Config
from blobquest.env import Outcome
from blobquest.metrics import TrainingHistory
from blobquest.mistakes import MISTAKE_FIELDS, MistakeLog
from blobquest.persistence import MISTAKES_FILE, save_run
from blobquest.plots import plot_mistakes
from blobquest.tiles import Tile
from blobquest.trainer import Trainer
from blobquest.world import build_world
from tests.conftest import make_config

PIT_MAP = [
    "@.X",
    "..X",
    "X.G",
]


def test_goal_episodes_are_not_logged() -> None:
    log = MistakeLog()
    assert log.record(0, Outcome.GOAL, (2, 2), 10) is None
    assert len(log) == 0


def test_record_fields_and_repeat_flag() -> None:
    log = MistakeLog()
    first = log.record(3, Outcome.PIT, (0, 2), 7)
    second = log.record(5, Outcome.PIT, (0, 2), 4)
    other = log.record(6, Outcome.EXHAUSTED, (1, 1), 0)
    assert first is not None and (first.episode, first.outcome, first.row, first.col, first.energy) == (
        3,
        "pit",
        0,
        2,
        7,
    )
    assert not first.repeated
    assert second is not None and second.repeated
    assert other is not None and not other.repeated
    assert log.repeat_rate() == pytest.approx(1 / 3)


def test_timeouts_are_logged_but_are_not_deaths() -> None:
    log = MistakeLog()
    log.record(0, Outcome.TIMEOUT, (1, 1), 3)
    log.record(1, Outcome.TIMEOUT, (1, 1), 3)
    assert len(log) == 2
    assert log.deaths == []
    assert not any(m.repeated for m in log.mistakes)
    assert np.isnan(log.repeat_rate())
    assert log.deadliest_cells() == []


def test_repeat_rate_by_window() -> None:
    log = MistakeLog()
    # Window 0 (episodes 0-3): two deaths on new cells.
    log.record(0, Outcome.PIT, (0, 2), 5)
    log.record(2, Outcome.PIT, (1, 2), 5)
    # Window 1 (episodes 4-7): three deaths, two of them on known cells.
    log.record(4, Outcome.PIT, (0, 2), 5)
    log.record(5, Outcome.PIT, (2, 0), 5)
    log.record(7, Outcome.PIT, (1, 2), 5)
    # Window 2 (episodes 8-9, partial): no deaths.
    windows = log.repeat_rate_by_window(window=4, num_episodes=10)
    assert [(w.start, w.end, w.deaths, w.repeats) for w in windows] == [(0, 4, 2, 0), (4, 8, 3, 2), (8, 10, 0, 0)]
    assert windows[0].rate == 0.0
    assert windows[1].rate == pytest.approx(2 / 3)
    assert np.isnan(windows[2].rate)


def test_repeat_rate_by_window_rejects_bad_window() -> None:
    with pytest.raises(ValueError):
        MistakeLog().repeat_rate_by_window(0, 10)


def test_deadliest_cells_top_ten_sorted() -> None:
    log = MistakeLog()
    episode = 0
    for i in range(12):
        for _ in range(i + 1):
            log.record(episode, Outcome.PIT, (i, 0), 1)
            episode += 1
    log.record(episode, Outcome.PIT, (0, 5), 1)  # ties with (0, 0) at one death
    top = log.deadliest_cells()
    assert len(top) == 10
    assert top[0] == ((11, 0), 12)
    assert [count for _, count in top] == sorted((count for _, count in top), reverse=True)
    assert log.deadliest_cells(13)[-2:] == [((0, 0), 1), ((0, 5), 1)]  # ties broken by row, then column


def test_format_deadliest_shows_tile_letters() -> None:
    world = build_world(make_config(PIT_MAP).world)
    log = MistakeLog()
    log.record(0, Outcome.PIT, (0, 2), 3)
    log.record(1, Outcome.PIT, (0, 2), 3)
    log.record(2, Outcome.EXHAUSTED, (1, 1), 0)
    table = log.format_deadliest(world).splitlines()
    assert table[0].split() == ["rank", "cell", "tile", "deaths", "share"]
    assert table[1].split() == ["1", "(0,2)", "X", "2", "66.7%"]
    assert table[2].split() == ["2", "(1,1)", ".", "1", "33.3%"]
    assert MistakeLog().format_deadliest(world) == "no deaths recorded"


def _pit_trainer(algorithm: str = "q_learning") -> Trainer:
    config = make_config(
        PIT_MAP,
        agent={"algorithm": algorithm},
        training={"episodes": 200, "seed": 3, "eval_every": 50, "eval_episodes": 1},
    )
    return Trainer(config)


def test_trainer_logs_every_failed_episode() -> None:
    trainer = _pit_trainer()
    history = trainer.train()
    failed = [r.episode for r in history.episodes if r.outcome != Outcome.GOAL.value]
    assert failed, "a fresh agent should fall into a pit at least once"
    assert [m.episode for m in trainer.mistakes.mistakes] == failed
    for m in trainer.mistakes.mistakes:
        record = history.episodes[m.episode]
        assert m.outcome == record.outcome
        assert m.energy == record.final_energy
        if m.outcome == Outcome.PIT.value:
            assert trainer.world.tile_at((m.row, m.col)) is Tile.PIT


def test_mistakes_csv_is_saved(tmp_path: Path) -> None:
    trainer = _pit_trainer()
    history = trainer.train()
    save_run(tmp_path, trainer.config, trainer.world, trainer.agent, history, {}, trainer.mistakes)
    with (tmp_path / MISTAKES_FILE).open(encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == MISTAKE_FIELDS
    assert len(rows) - 1 == len(trainer.mistakes)


def test_save_run_without_mistakes_skips_csv(tmp_path: Path) -> None:
    trainer = _pit_trainer()
    save_run(tmp_path, trainer.config, trainer.world, trainer.agent, TrainingHistory(3, 3), {})
    assert not (tmp_path / MISTAKES_FILE).exists()


def test_plot_mistakes_writes_png(tmp_path: Path) -> None:
    trainer = _pit_trainer()
    history = trainer.train()
    out = plot_mistakes(trainer.mistakes, trainer.world, len(history.episodes), Config().output, tmp_path / "m.png")
    assert out.is_file() and out.stat().st_size > 0


def test_plot_mistakes_handles_empty_log(tmp_path: Path) -> None:
    world = build_world(make_config(PIT_MAP).world)
    out = plot_mistakes(MistakeLog(), world, 0, Config().output, tmp_path / "empty.png")
    assert out.is_file()
