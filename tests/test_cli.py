import csv
import re
from pathlib import Path

import pytest

from blobquest.cli import main
from blobquest.compare import COMPARE_ALGORITHMS, COMPARISON_CSV, COMPARISON_PLOT, RESULT_FIELDS
from blobquest.config import save_config
from blobquest.render import energy_bar
from tests.conftest import make_config

# Greedy route: speed boost, then through the drain to the goal.
TINY_MAP = [
    "@S.D",
    ".#T.",
    "...G",
]


def _write_config(tmp_path: Path, episodes: int = 300) -> Path:
    config = make_config(
        TINY_MAP,
        energy={"initial": 20, "maximum": 30},
        training={"episodes": episodes, "seed": 0, "eval_every": 50, "eval_episodes": 2, "log_every": episodes},
        output={"runs_dir": str(tmp_path / "runs"), "moving_average_window": 50},
    )
    path = tmp_path / "tiny.yaml"
    save_config(config, path)
    return path


@pytest.fixture
def trained_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    assert main(["train", "--config", str(_write_config(tmp_path)), "--no-plots"]) == 0
    capsys.readouterr()
    return next((tmp_path / "runs").iterdir())


def _frames(out: str) -> list[str]:
    return [block for block in out.split("\n\n") if block.startswith("step ")]


def test_replay_prints_every_step(trained_run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["replay", str(trained_run), "--delay", "0"]) == 0
    out = capsys.readouterr().out
    frames = _frames(out)
    match = re.search(r"Outcome: goal after (\d+) steps, total reward (-?\d+\.\d)", out)
    assert match, out
    assert len(frames) == int(match.group(1)) + 1  # start frame + one per step
    assert frames[0].startswith("step   0/")
    for frame in frames:
        lines = frame.splitlines()
        assert lines[1].startswith("energy [")
        grid = lines[-3:]
        assert sum(row.split(" ").count("@") for row in grid) == 1  # exactly one blob
    assert "effects: speed (3 left)" in out
    assert "drain (5 left)" in out
    assert re.search(r"reward\s+[+-]\d+\.\d\s+total", frames[1])
    last_total = re.findall(r"total\s+([+-]\d+\.\d)", frames[-1])[0]
    assert float(last_total) == pytest.approx(float(match.group(2)))


def test_replay_is_deterministic(trained_run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["replay", str(trained_run), "--delay", "0", "--seed", "3"])
    first = capsys.readouterr().out
    main(["replay", str(trained_run), "--delay", "0", "--seed", "3"])
    assert capsys.readouterr().out == first


def test_replay_rejects_negative_delay(trained_run: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["replay", str(trained_run), "--delay", "-1"]) == 2
    assert "--delay" in capsys.readouterr().err


def test_replay_missing_run(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["replay", str(tmp_path / "nope"), "--delay", "0"]) == 2
    assert "error" in capsys.readouterr().err


def test_replay_waits_between_steps(
    trained_run: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("blobquest.cli.time.sleep", sleeps.append)
    assert main(["replay", str(trained_run), "--delay", "0.25"]) == 0
    frames = _frames(capsys.readouterr().out)
    assert sleeps == [0.25] * len(frames)


def test_energy_bar() -> None:
    assert energy_bar(15, 30, width=10) == "[#####.....]  15/30"
    assert energy_bar(-3, 30, width=4) == "[....]  -3/30"
    assert energy_bar(30, 30, width=4) == "[####]  30/30"


def test_compare_prints_table_and_saves_plot(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _write_config(tmp_path, episodes=150)
    assert main(["compare", "--config", str(config), "--seeds", "2"]) == 0
    out = capsys.readouterr().out

    table = out[out.index("algorithm") :].splitlines()
    assert table[0].split()[0] == "algorithm"
    rows = {line.split()[0]: line for line in table[2:5]}
    assert set(rows) == set(COMPARE_ALGORITHMS)
    assert "100.0% ± 0.0%" in rows["q_learning"]  # tiny map: both learners always reach the goal greedily

    out_dir = next((tmp_path / "runs").glob("*_compare_*"))
    assert (out_dir / COMPARISON_PLOT).stat().st_size > 0
    assert (out_dir / "config.yaml").is_file() and (out_dir / "world.txt").is_file()
    with (out_dir / COMPARISON_CSV).open(encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    assert tuple(records[0]) == RESULT_FIELDS
    assert sorted((r["algorithm"], r["seed"]) for r in records) == sorted(
        (a, s) for a in COMPARE_ALGORITHMS for s in ("0", "1")
    )


def test_compare_is_reproducible(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _write_config(tmp_path, episodes=60)
    tables = []
    for _ in range(2):
        assert main(["compare", "--config", str(config), "--seeds", "2"]) == 0
        out = capsys.readouterr().out
        tables.append(out[out.index("Results") : out.index("Comparison saved")])
    assert tables[0] == tables[1]


def test_compare_episodes_override(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _write_config(tmp_path, episodes=500)
    assert main(["compare", "--config", str(config), "--seeds", "1", "--episodes", "20"]) == 0
    assert "20 episodes" in capsys.readouterr().out


def test_compare_requires_config() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["compare"])
    assert exc.value.code == 2


def test_compare_rejects_zero_seeds(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["compare", "--config", str(_write_config(tmp_path)), "--seeds", "0"]) == 2
    assert "--seeds" in capsys.readouterr().err
