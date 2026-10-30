"""Run the Blob Quest experiments and write a markdown results file plus PNG plots.

Three experiments, each variant trained with the same seeds on one map built from ``--config``:

* ``algorithms`` - random vs Q-learning vs SARSA
* ``sensing``    - Q-learning with neighbour sensing off vs on
* ``epsilon``    - Q-learning with a constant, linear and exponential epsilon schedule

Outputs (by default under ``docs/``): ``results.md`` with one table per experiment and
``images/<experiment>.png`` with the training success curves. With ``--update-readme`` the
settings line and each table are also written between the ``<!-- results:NAME:start -->`` /
``<!-- results:NAME:end -->`` markers in README.md (NAME = settings, algorithms, sensing, epsilon).

Usage::

    python experiments/run_experiments.py                        # 5 seeds, configs/default.yaml
    python experiments/run_experiments.py --jobs 4 --update-readme
    python experiments/run_experiments.py --config configs/hard.yaml --out docs/hard
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from blobquest.compare import Comparison, Overrides, format_markdown, run_variants
from blobquest.config import Config, load_config
from blobquest.plots import plot_comparison

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Experiment:
    """A named group of variants compared on the same map and seeds."""

    name: str
    title: str
    column: str
    variants: Mapping[str, Overrides]


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        "algorithms",
        "Q-learning vs SARSA vs random",
        "algorithm",
        {
            "random": {"agent": {"algorithm": "random"}},
            "q_learning": {"agent": {"algorithm": "q_learning"}},
            # On-policy SARSA trains without replay memory, as in configs/sarsa.yaml.
            "sarsa": {"agent": {"algorithm": "sarsa"}, "memory": {"enabled": False}},
        },
    ),
    Experiment(
        "sensing",
        "Neighbour sensing off vs on (Q-learning)",
        "sensing",
        {
            "off": {"agent": {"algorithm": "q_learning", "sensing": False}},
            "on": {"agent": {"algorithm": "q_learning", "sensing": True}},
        },
    ),
    Experiment(
        "epsilon",
        "Epsilon schedules (Q-learning)",
        "epsilon schedule",
        {
            "constant 0.1": {
                "agent": {"algorithm": "q_learning", "epsilon": {"kind": "constant", "start": 0.1, "end": 0.1}}
            },
            "linear 1→0.05": {
                "agent": {
                    "algorithm": "q_learning",
                    "epsilon": {"kind": "linear", "start": 1.0, "end": 0.05, "decay_fraction": 0.7},
                }
            },
            "exponential 1→0.05": {
                "agent": {
                    "algorithm": "q_learning",
                    "epsilon": {"kind": "exponential", "start": 1.0, "end": 0.05, "decay_rate": 0.999},
                }
            },
        },
    ),
)


def settings_line(config: Config, config_path: Path, comparison: Comparison) -> str:
    """One markdown line describing what the numbers were measured on."""
    world = comparison.world
    return (
        f"Config `{config_path.as_posix()}`, {world.rows}x{world.cols} map (map seed {config.world.map_seed}), "
        f"{config.training.episodes} episodes per run, seeds {list(comparison.seeds)}. "
        f"Cells are mean ± std across seeds; train metrics cover the last "
        f"{config.output.moving_average_window} episodes, greedy success is the final evaluation "
        f"({config.training.eval_episodes} greedy episodes)."
    )


def render_markdown(sections: Sequence[tuple[Experiment, str]], settings: str, image_prefix: str) -> str:
    """Markdown with one table and plot per experiment; images are linked under ``image_prefix``."""
    parts = [settings, ""]
    for experiment, table in sections:
        parts += [
            f"#### {experiment.title}",
            "",
            table,
            "",
            f"![{experiment.title}]({image_prefix}{experiment.name}.png)",
            "",
        ]
    return "\n".join(parts).rstrip() + "\n"


def markers(name: str) -> tuple[str, str]:
    """README comment markers around the generated block ``name``."""
    return f"<!-- results:{name}:start -->", f"<!-- results:{name}:end -->"


def update_readme(readme: Path, blocks: Mapping[str, str]) -> list[str]:
    """Replace the text between each block's markers in ``readme``; return the names that had no markers."""
    text = readme.read_text(encoding="utf-8")
    missing = []
    for name, body in blocks.items():
        start, end = markers(name)
        if start not in text or end not in text:
            missing.append(name)
            continue
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        text = f"{head}{start}\n{body.rstrip()}\n{end}{tail}"
    readme.write_text(text, encoding="utf-8", newline="\n")  # keep LF on Windows too
    return missing


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml", help="base YAML config")
    parser.add_argument("--seeds", type=int, default=5, help="seeds per variant (default: 5)")
    parser.add_argument("--episodes", type=int, help="override training.episodes (e.g. for a quick check)")
    parser.add_argument("--jobs", type=int, default=1, help="worker processes (results do not depend on it)")
    parser.add_argument("--out", type=Path, default=ROOT / "docs", help="writes results.md and images/ here")
    parser.add_argument(
        "--only", choices=[e.name for e in EXPERIMENTS], action="append", help="run only these experiments"
    )
    parser.add_argument("--update-readme", action="store_true", help="also fill the results markers in README.md")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected experiments and write the results."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.episodes is not None:
        config = config.with_overrides(training={"episodes": args.episodes})
    experiments = [e for e in EXPERIMENTS if args.only is None or e.name in args.only]
    images = args.out / "images"
    images.mkdir(parents=True, exist_ok=True)

    total = sum(len(e.variants) for e in experiments) * args.seeds
    print(f"{total} runs of {config.training.episodes} episodes ({args.jobs} job(s))", flush=True)
    sections: list[tuple[Experiment, str]] = []
    comparison: Comparison | None = None
    for experiment in experiments:
        print(f"\n== {experiment.title}", flush=True)
        started = time.perf_counter()
        comparison = run_variants(
            config, experiment.variants, args.seeds, logger=lambda line: print(line, flush=True), jobs=args.jobs
        )
        plot_comparison(comparison.histories, config.output, images / f"{experiment.name}.png", experiment.title)
        sections.append((experiment, format_markdown(comparison, experiment.column)))
        print(f"done in {time.perf_counter() - started:.0f}s", flush=True)
    if comparison is None:
        return 0

    config_label = (
        args.config.resolve().relative_to(ROOT) if args.config.resolve().is_relative_to(ROOT) else args.config
    )
    settings = settings_line(config, Path(config_label), comparison)
    results = args.out / "results.md"
    header = "# Blob Quest experiment results\n\nGenerated by `experiments/run_experiments.py`.\n\n"
    results.write_text(header + render_markdown(sections, settings, "images/"), encoding="utf-8", newline="\n")
    print(f"\nWrote {results} and {len(sections)} plot(s) in {images}")
    if args.update_readme:
        # Only the numbers: the README embeds the images itself.
        blocks = {"settings": settings, **{experiment.name: table for experiment, table in sections}}
        missing = update_readme(args.readme, blocks)
        print(f"Updated the results tables in {args.readme}")
        if missing:
            print(f"warning: no markers for {', '.join(missing)} in {args.readme}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
