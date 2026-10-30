import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from blobquest.compare import format_markdown, run_variants, variant_config
from blobquest.config import save_config
from tests.conftest import make_config

ROOT = Path(__file__).resolve().parent.parent
TINY_MAP = [
    "@S.D",
    ".#T.",
    "...G",
]


def _load_script() -> ModuleType:
    path = ROOT / "experiments" / "run_experiments.py"
    spec = importlib.util.spec_from_file_location("run_experiments", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # @dataclass looks the module up while the script loads
    spec.loader.exec_module(module)
    return module


def _config(episodes: int = 30):
    return make_config(
        TINY_MAP,
        energy={"initial": 20, "maximum": 30},
        training={"episodes": episodes, "seed": 0, "eval_every": 10, "eval_episodes": 1},
        output={"moving_average_window": 10},
    )


def test_variant_config_merges_overrides_and_seed() -> None:
    config = variant_config(_config(), {"agent": {"sensing": True}, "training": {"episodes": 7}}, seed=9)
    assert config.agent.sensing is True
    assert config.training.episodes == 7
    assert config.training.seed == 9


def test_parallel_runs_match_sequential() -> None:
    variants = {"q": {"agent": {"algorithm": "q_learning"}}, "s": {"agent": {"algorithm": "sarsa"}}}
    sequential = run_variants(_config(), variants, num_seeds=2)
    parallel = run_variants(_config(), variants, num_seeds=2, jobs=2)
    assert sequential.results == parallel.results
    assert [r.variant for r in sequential.results] == ["q", "q", "s", "s"]
    assert [h.episodes for h in sequential.histories["q"]] == [h.episodes for h in parallel.histories["q"]]


def test_run_variants_validates_arguments() -> None:
    with pytest.raises(ValueError):
        run_variants(_config(), {"q": {}}, num_seeds=0)
    with pytest.raises(ValueError):
        run_variants(_config(), {"q": {}}, num_seeds=1, jobs=0)


def test_markdown_table_shape() -> None:
    comparison = run_variants(_config(), {"q": {}, "r": {"agent": {"algorithm": "random"}}}, num_seeds=1)
    lines = format_markdown(comparison, "variant").splitlines()
    assert lines[0].startswith("| variant | greedy success |")
    assert lines[1] == "|:---|---:|---:|---:|---:|---:|"
    assert [line.split(" | ")[0] for line in lines[2:]] == ["| q", "| r"]


def test_experiment_matrix() -> None:
    script = _load_script()
    names = {e.name: e for e in script.EXPERIMENTS}
    assert set(names) == {"algorithms", "sensing", "epsilon"}
    assert set(names["algorithms"].variants) == {"random", "q_learning", "sarsa"}
    assert len(names["sensing"].variants) == 2
    kinds = {v["agent"]["epsilon"]["kind"] for v in names["epsilon"].variants.values()}
    assert kinds == {"constant", "linear", "exponential"}


def test_script_writes_results_and_fills_readme(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = _load_script()
    config_path = tmp_path / "tiny.yaml"
    save_config(_config(), config_path)
    readme = tmp_path / "README.md"
    blocks = "\n".join(
        f"{start}\npending\n{end}\n![plot](docs/images/{name}.png)"
        for name in ("settings", "algorithms", "sensing", "epsilon")
        for start, end in [script.markers(name)]
    )
    readme.write_text(f"# Title\n\n{blocks}\n\n## What I learned\n", encoding="utf-8")
    out = tmp_path / "docs"

    argv = ["--config", str(config_path), "--seeds", "1", "--out", str(out), "--update-readme", "--readme", str(readme)]
    assert script.main(argv) == 0
    capsys.readouterr()

    for name in ("algorithms", "sensing", "epsilon"):
        assert (out / "images" / f"{name}.png").stat().st_size > 0
    results = (out / "results.md").read_text(encoding="utf-8")
    assert results.count("| greedy success |") == 3
    assert "seeds [0]" in results and "30 episodes per run" in results

    text = readme.read_text(encoding="utf-8")
    assert "pending" not in text
    assert text.count("| greedy success |") == 3
    assert text.count("![plot](docs/images/") == 4  # images and surrounding text untouched
    assert text.endswith("## What I learned\n")
    for name in ("random", "q_learning", "sarsa", "constant 0.1", "| on |", "| off |"):
        assert name in text


def test_script_only_selected_experiment(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    script = _load_script()
    config_path = tmp_path / "tiny.yaml"
    save_config(_config(), config_path)
    out = tmp_path / "docs"
    assert script.main(["--config", str(config_path), "--seeds", "1", "--out", str(out), "--only", "sensing"]) == 0
    assert "2 runs of 30 episodes" in capsys.readouterr().out  # sensing off + on, one seed
    assert [p.name for p in (out / "images").iterdir()] == ["sensing.png"]
