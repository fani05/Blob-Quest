# Blob Quest

A blob has to cross a foggy grid world to reach the goal. It can't see the map. All it knows is
its own state: where it is, roughly how much energy it has left, which boosters it has already
used, and whether a drain or speed effect is active. It learns which moves are good with
**tabular Q-learning or SARSA**, using only the rewards and penalties it collects over thousands
of episodes.

Everything runs in the terminal. Plots are saved as PNG files, and there is no GUI.

## Tiles

| Char | Tile   | Effect |
|------|--------|--------|
| `@`  | start  | spawn point |
| `G`  | goal   | success, episode ends |
| `#`  | wall   | blocks movement |
| `T`  | trap   | -10 energy |
| `X`  | pit    | death, episode ends |
| `D`  | drain  | the next 5 steps cost 2 energy |
| `E`  | energy | +15 energy, single use per episode |
| `S`  | speed  | the next 3 moves travel 2 cells, single use per episode |
| `R`  | reset  | teleport back to the start |

Every step costs energy. At 0 energy the episode fails.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -e ".[dev]"

blobquest show-map                                  # print the generated map
blobquest train                                     # Q-learning, configs/default.yaml
blobquest train --config configs/sarsa.yaml         # SARSA
blobquest train --config configs/handmade.yaml      # hand-drawn map with diagonals
blobquest train --episodes 20000 --seed 1 --map-seed 3
blobquest evaluate runs/<run-folder> --episodes 5   # replay a saved greedy policy

pytest
ruff check src tests
```

Each run writes `runs/<name>_<timestamp>/` containing `config.yaml`, `world.txt`,
`q_table.npz`, `episodes.csv`, `evaluations.csv`, `summary.yaml`, `visits.npy` and `plots/`.
The plots are rewards, outcomes, steps, schedules, evaluation, policy map, value map and visits.

## Layout

```
src/blobquest/
  config.py       frozen, validated dataclasses (every tunable number)
  tiles.py        Tile enum and text characters
  world.py        hidden map: seeded generation, layout parsing, solvability check
  state.py        State tuple and integer encoder for the Q-table
  env.py          dynamics: energy, tiles, speed, drain, termination
  agents/         TabularAgent base, Q-learning, SARSA
  schedules.py    epsilon / alpha schedules
  memory.py       replay memory that rehearses mistakes more often
  trainer.py      training loop, evaluation, greedy rollouts
  metrics.py      per-episode records and summaries
  plots.py        PNG plots (matplotlib Agg, no pyplot)
  render.py       ASCII map, path and policy rendering
  persistence.py  save and load runs
  cli.py          `blobquest train | evaluate | show-map`
configs/          default.yaml, sarsa.yaml, handmade.yaml
tests/            pytest suite
```
