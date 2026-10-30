"""PNG plots of training progress and the learned policy.

Uses matplotlib's object-oriented API with the Agg canvas directly (no
``pyplot``), so no global figure state is created and no GUI is ever needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, LogNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from blobquest.config import OutputConfig
from blobquest.env import action_deltas
from blobquest.metrics import TrainingHistory, moving_average
from blobquest.mistakes import MistakeLog
from blobquest.render import NO_POLICY
from blobquest.tiles import TILE_CHARS, Tile
from blobquest.world import Position, World

# Validated categorical palette (fixed order) and a one-hue sequential ramp.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SEQUENTIAL = ("#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
WALL_COLOR = "#3b3a37"

TILE_COLORS: dict[Tile, str] = {
    Tile.EMPTY: SURFACE,
    Tile.WALL: WALL_COLOR,
    Tile.GOAL: SERIES[2],
    Tile.TRAP: SERIES[1],
    Tile.PIT: SERIES[7],
    Tile.DRAIN: SERIES[6],
    Tile.ENERGY: SERIES[3],
    Tile.SPEED: SERIES[0],
    Tile.RESET: SERIES[4],
}
OUTCOME_COLORS: dict[str, str] = {"goal": SERIES[0], "pit": SERIES[1], "exhausted": SERIES[2], "timeout": SERIES[3]}

RAW_ALPHA = 0.25
LINE_WIDTH = 2.0
THIN_LINE = 1.0
ARROW_SCALE = 1.6
PATH_WIDTH = 2.5


def _figure(cfg: OutputConfig, size: tuple[float, float] | None = None) -> tuple[Figure, Axes]:
    """New figure with a single styled axis."""
    fig = Figure(figsize=size or (cfg.figure_width, cfg.figure_height), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    _style_axis(ax)
    return fig, ax


def _style_axis(ax: Axes) -> None:
    """Shared look: light surface, muted spines and grid behind the data."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)
    ax.grid(True, color=GRID, linewidth=THIN_LINE)
    ax.set_axisbelow(True)


def _finish(fig: Figure, ax: Axes, title: str, path: Path, dpi: int) -> Path:
    """Title, tight layout and save as PNG."""
    ax.set_title(title, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    return path


def plot_rewards(history: TrainingHistory, cfg: OutputConfig, path: Path) -> Path:
    """Episode reward (faint) with its moving average."""
    fig, ax = _figure(cfg)
    rewards = history.column("total_reward")
    episodes = np.arange(1, rewards.size + 1)
    ax.plot(episodes, rewards, color=SERIES[0], alpha=RAW_ALPHA, linewidth=THIN_LINE, label="per episode")
    ax.plot(
        episodes,
        moving_average(rewards, cfg.moving_average_window),
        color=SERIES[0],
        linewidth=LINE_WIDTH,
        label=f"moving average ({cfg.moving_average_window})",
    )
    ax.set_xlabel("episode")
    ax.set_ylabel("total reward")
    ax.legend(frameon=False, labelcolor=INK)
    return _finish(fig, ax, "Reward per episode", path, cfg.plot_dpi)


def plot_outcomes(history: TrainingHistory, cfg: OutputConfig, path: Path) -> Path:
    """Stacked rolling share of goal / pit / exhausted / timeout endings."""
    fig, ax = _figure(cfg)
    fractions = history.outcome_fractions(cfg.outcome_window)
    episodes = np.arange(1, len(history.episodes) + 1)
    names = list(OUTCOME_COLORS)
    ax.stackplot(
        episodes,
        [fractions[n] for n in names],
        labels=names,
        colors=[OUTCOME_COLORS[n] for n in names],
        edgecolor=SURFACE,
        linewidth=THIN_LINE,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("episode")
    ax.set_ylabel(f"share of last {cfg.outcome_window} episodes")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, labelcolor=INK)
    return _finish(fig, ax, "How episodes end", path, cfg.plot_dpi)


def plot_steps(history: TrainingHistory, cfg: OutputConfig, path: Path) -> Path:
    """Moving average of steps per episode."""
    fig, ax = _figure(cfg)
    steps = history.column("steps")
    episodes = np.arange(1, steps.size + 1)
    ax.plot(episodes, steps, color=SERIES[0], alpha=RAW_ALPHA, linewidth=THIN_LINE, label="per episode")
    ax.plot(
        episodes,
        moving_average(steps, cfg.moving_average_window),
        color=SERIES[0],
        linewidth=LINE_WIDTH,
        label=f"moving average ({cfg.moving_average_window})",
    )
    ax.set_xlabel("episode")
    ax.set_ylabel("steps")
    ax.legend(frameon=False, labelcolor=INK)
    return _finish(fig, ax, "Episode length", path, cfg.plot_dpi)


def plot_schedules(history: TrainingHistory, cfg: OutputConfig, path: Path) -> Path:
    """Exploration rate and learning rate over training (both in [0, 1])."""
    fig, ax = _figure(cfg)
    episodes = np.arange(1, len(history.episodes) + 1)
    ax.plot(episodes, history.column("epsilon"), color=SERIES[0], linewidth=LINE_WIDTH, label="epsilon (explore)")
    ax.plot(episodes, history.column("alpha"), color=SERIES[1], linewidth=LINE_WIDTH, label="alpha (learning rate)")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("episode")
    ax.set_ylabel("value")
    ax.legend(frameon=False, labelcolor=INK)
    return _finish(fig, ax, "Schedules", path, cfg.plot_dpi)


def plot_evaluation(history: TrainingHistory, cfg: OutputConfig, path: Path) -> Path:
    """Success rate of the greedy policy at each evaluation point."""
    fig, ax = _figure(cfg)
    x = [e.episode + 1 for e in history.evaluations]
    y = [e.success_rate for e in history.evaluations]
    ax.plot(x, y, color=SERIES[0], linewidth=LINE_WIDTH, marker="o", markersize=4)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("episode")
    ax.set_ylabel("greedy success rate")
    return _finish(fig, ax, "Greedy policy success rate", path, cfg.plot_dpi)


def _map_axes(world: World, cfg: OutputConfig) -> tuple[Figure, Axes]:
    """Square-celled figure sized to the map, without grid lines or ticks."""
    width = world.cols * cfg.map_cell_inches + cfg.map_cell_inches * 4
    height = world.rows * cfg.map_cell_inches + cfg.map_cell_inches
    fig, ax = _figure(cfg, (width, height))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


def _label_tiles(ax: Axes, world: World, skip_empty: bool = True) -> None:
    """Write each special tile's letter in its cell so identity is never color-only."""
    for r in range(world.rows):
        for c in range(world.cols):
            tile = world.tile_at((r, c))
            if skip_empty and tile in (Tile.EMPTY, Tile.WALL):
                continue
            ax.text(c, r, TILE_CHARS[tile], ha="center", va="center", color=INK, fontsize="small", weight="bold")
    ax.text(*world.start[::-1], "@", ha="center", va="center", color=INK, fontsize="small", weight="bold")


def plot_policy_map(
    world: World,
    policy: np.ndarray,
    path_cells: tuple[Position, ...],
    diagonals: bool,
    cfg: OutputConfig,
    path: Path,
) -> Path:
    """Tile map with greedy-action arrows and the greedy path drawn on top."""
    fig, ax = _map_axes(world, cfg)
    tiles = list(Tile)
    cmap = ListedColormap([TILE_COLORS[t] for t in tiles])
    norm = BoundaryNorm(np.arange(len(tiles) + 1) - 0.5, cmap.N)
    ax.imshow(world.grid, cmap=cmap, norm=norm)
    _label_tiles(ax, world)

    deltas = action_deltas(diagonals)
    rows, cols = np.nonzero((policy != NO_POLICY) & (world.grid == Tile.EMPTY))
    if rows.size:
        dr = np.array([deltas[a][0] for a in policy[rows, cols]])
        dc = np.array([deltas[a][1] for a in policy[rows, cols]])
        ax.quiver(cols, rows, dc, dr, angles="xy", scale_units="xy", scale=ARROW_SCALE, pivot="middle", color=INK_MUTED)

    if len(path_cells) > 1:
        pr, pc = zip(*path_cells, strict=True)
        ax.plot(pc, pr, color=SERIES[0], linewidth=PATH_WIDTH)
    handles = [
        Patch(facecolor=TILE_COLORS[t], edgecolor=GRID, label=f"{TILE_CHARS[t]} {t.name.lower()}") for t in tiles
    ]
    handles.append(Line2D([], [], color=SERIES[0], linewidth=PATH_WIDTH, label="greedy path"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, labelcolor=INK)
    return _finish(fig, ax, "Learned policy (most-rested blob, no boosters used) and greedy path", path, cfg.plot_dpi)


def plot_value_map(world: World, values: np.ndarray, cfg: OutputConfig, path: Path) -> Path:
    """Heatmap of max_a Q(s, a) per cell in the fresh-blob context."""
    fig, ax = _map_axes(world, cfg)
    cmap = LinearSegmentedColormap.from_list("blobquest_seq", SEQUENTIAL).with_extremes(bad=WALL_COLOR)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap)
    _label_tiles(ax, world)
    fig.colorbar(image, ax=ax, label="state value  max Q")
    return _finish(fig, ax, "Learned state values", path, cfg.plot_dpi)


def plot_visits(history: TrainingHistory, world: World, cfg: OutputConfig, path: Path) -> Path:
    """Log-scaled heatmap of how often each cell was visited during training."""
    fig, ax = _map_axes(world, cfg)
    cmap = LinearSegmentedColormap.from_list("blobquest_seq", SEQUENTIAL).with_extremes(bad=SURFACE)
    visits = np.ma.masked_less_equal(history.visits.astype(np.float64), 0.0)
    norm = LogNorm(vmin=1.0, vmax=max(float(history.visits.max()), 1.0) + 1.0)
    image = ax.imshow(visits, cmap=cmap, norm=norm)
    _label_tiles(ax, world)
    fig.colorbar(image, ax=ax, label="visits (log scale)")
    return _finish(fig, ax, "Where the blob went during training", path, cfg.plot_dpi)


def plot_mistakes(
    mistakes: MistakeLog, world: World, num_episodes: int, cfg: OutputConfig, path: Path, top_n: int = 10
) -> Path:
    """Repeated-mistake rate per window (left) and the deadliest cells (right)."""
    fig = Figure(figsize=(cfg.figure_width * 1.5, cfg.figure_height), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    ax, bars = fig.subplots(1, 2)
    _style_axis(ax)
    _style_axis(bars)

    windows = mistakes.repeat_rate_by_window(cfg.moving_average_window, num_episodes)
    x = [w.end for w in windows]
    ax.plot(x, [w.rate for w in windows], color=SERIES[1], linewidth=LINE_WIDTH, marker="o", markersize=4)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0, max(num_episodes, 1))
    ax.set_xlabel("episode")
    ax.set_ylabel(f"repeated deaths / deaths (per {cfg.moving_average_window})")
    ax.set_title("Dying where it died before", color=INK, loc="left")

    cells = mistakes.deadliest_cells(top_n)
    if cells:
        labels = [f"({r},{c}) {TILE_CHARS[world.tile_at((r, c))]}" for (r, c), _ in cells]
        counts = [count for _, count in cells]
        positions = np.arange(len(cells))[::-1]
        bars.barh(positions, counts, color=SERIES[1])
        bars.set_yticks(positions, labels)
        for y, count in zip(positions, counts, strict=True):
            bars.text(count, y, f" {count}", va="center", color=INK, fontsize="small")
        bars.grid(False, axis="y")
    else:
        bars.text(0.5, 0.5, "no deaths recorded", ha="center", va="center", color=INK_MUTED, transform=bars.transAxes)
        bars.set_yticks([])
    bars.set_xlabel("deaths")
    return _finish(fig, bars, f"Top {top_n} deadliest cells", path, cfg.plot_dpi)


def save_all_plots(
    history: TrainingHistory,
    world: World,
    policy: np.ndarray,
    values: np.ndarray,
    path_cells: tuple[Position, ...],
    diagonals: bool,
    cfg: OutputConfig,
    out_dir: Path,
    mistakes: MistakeLog | None = None,
) -> list[Path]:
    """Write every plot to ``out_dir`` and return the created file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_rewards(history, cfg, out_dir / "rewards.png"),
        plot_outcomes(history, cfg, out_dir / "outcomes.png"),
        plot_steps(history, cfg, out_dir / "steps.png"),
        plot_schedules(history, cfg, out_dir / "schedules.png"),
        plot_evaluation(history, cfg, out_dir / "evaluation.png"),
        plot_policy_map(world, policy, path_cells, diagonals, cfg, out_dir / "policy_map.png"),
        plot_value_map(world, values, cfg, out_dir / "value_map.png"),
        plot_visits(history, world, cfg, out_dir / "visits.png"),
    ]
    if mistakes is not None:
        paths.append(plot_mistakes(mistakes, world, len(history.episodes), cfg, out_dir / "mistakes.png"))
    return paths
