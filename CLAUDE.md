# CLAUDE.md — Crossplay v13.1 "GitHub Ready"

## What is this project?

A Crossplay (Scrabble variant) engine with Monte Carlo AI evaluation. It plays
assisted games where the AI analyzes positions and recommends moves, and can
also run AI-vs-AI simulations.

**Key differences from Scrabble:** 3 blanks (not 2), 40-point bingo bonus
(not 50), different tile values, different bonus square layout, no center
star bonus, very different endgame rules.

## Endgame rules (NYT Crossplay)

When the tile bag is empty and all tiles are played, or no more moves can
be made, the game ends. **Both players get one final turn** after the bag
empties. **Leftover tiles do not count against the player** — there is no
tile penalty or transfer bonus. This differs significantly from Scrabble
where going out earns opponent's tile values and remaining tiles are
subtracted.

### Endgame heuristic for AI evaluation

Because of the "both get one final turn, no tile penalty" rule, the
endgame equity calculation is:

1. **When bag = 0 and it's your turn (your final turn):**
   - You will play one more move. Opponent will play one more move.
   - `equity = your_move_score - opp_best_response_score`
   - Leave value is irrelevant (leftover tiles don't penalize).
   - Maximize your move score; opponent maximizes theirs.

2. **When bag = 0 and it's opponent's turn (their final turn):**
   - Opponent plays one more move. You play one more move after.
   - `equity = -opp_move_score + your_best_response_score`

3. **Key implications for the engine:**
   - **Leave evaluation is meaningless** once both players have their
     final racks — there is no future draw and no tile penalty.
   - **Risk analysis changes**: the opponent has exactly 1 response
     move, not an infinite horizon. Threat = their single best move.
   - **3-ply simplifies to 2-ply**: your move + opponent's response
     (or opponent's move + your response). No ply 3 needed because
     the game ends after both final turns.
   - **MC simulation is unnecessary** when bag = 0 — opponent's rack
     can be inferred exactly (unseen tiles = their rack), so evaluate
     deterministically instead of sampling.
   - **When bag has 1-6 tiles**: the draw that empties the bag triggers
     final turns. Factor in that after this draw, each player gets
     exactly one more turn.

## Architecture overview

```
game_manager.py          ← Entry point, game loop, AI orchestration
  ├── mc_eval.py         ← Monte Carlo 2-ply evaluation (core AI)
  ├── mc_calibrate.py    ← Auto-benchmarks hardware, computes N×K per turn
  ├── move_finder_c.py   ← C-accelerated move finder + post-validation
  ├── move_finder_opt.py ← Optimized Python move finder (fallback)
  ├── gaddag_accel.so    ← Compiled Cython extension (GADDAG traversal)
  ├── gaddag_accel.pyx   ← Cython source for .so
  ├── gaddag.py          ← GADDAG trie, auto-build/cache from dictionary
  ├── gaddag_compact.py  ← Compact binary GADDAG format
  ├── board.py           ← Board representation (15x15 grid)
  ├── scoring.py         ← Move scoring with bonuses and crosswords
  ├── config.py          ← ALL constants: tile values, distribution, bonuses
  ├── dictionary.py      ← Word validation (196K words)
  ├── leave_eval.py      ← Rack leave evaluation + bingo probability DB
  ├── real_risk.py       ← Opponent threat analysis
  ├── lookahead.py       ← 1-ply lookahead
  ├── lookahead_3ply.py  ← 3-ply exhaustive endgame
  ├── parallel_eval.py   ← Multi-worker parallel evaluation
  ├── tile_tracker.py    ← Tile bag and tracking
  └── play_game.py       ← Game state management
```

## How to run

```bash
cd crossplay_v9
python3 game_manager.py
```

First run builds GADDAG (~48s) and runs MC calibration (~3s). Both are cached.

## Key design decisions

- **MIN_N = 25** for MC evaluation (based on Maven/Quackle research — enough
  candidates to catch moves that look weak in static eval but have strong
  leave synergy)
- **K ∈ [100, 2000]** — below 100, MC variance too high; above 2000,
  diminishing returns
- **Adaptive N×K** — auto-calibrated per hardware, equity-spread-based N,
  K fills remaining time budget
- **30-second turn budget** — split: 3s risk analysis, 21-27s MC, 0-6s 3-ply
- **Post-validation** in `move_finder_c.py` — all C-generated moves have
  main-axis words and cross-words re-validated in Python as a safety net
- **Saved games** use a generic registry pattern in `_init_default_games()` —
  factory functions named `_create_saved_game_N()`, identity comes from
  `GameState.name` and `opponent_name`

## Data files (committed, do not regenerate)

| File | Size | Purpose |
|------|------|---------|
| `crossplay_dict.pkl` | 2.3MB | Word dictionary (196K words) |
| `leave_bingo_prod.pkl` | 4.8MB | Bingo probability database |
| `pattern_index.pkl` | 11MB | Pattern matching index |
| `gaddag_accel.cpython-312-x86_64-linux-gnu.so` | 769KB | Pre-built C extension |

## Auto-generated files (gitignored)

| File | Purpose |
|------|---------|
| `gaddag_compact.bin` | 28MB compact GADDAG, built on first run |
| `.mc_calibration.json` | MC throughput cache, valid 48h |

## Cython C extension (gaddag_accel)

The pre-built `.so` is for CPython 3.12 x86_64 Linux (Claude Chat sandbox).
To rebuild for other platforms:

```bash
pip install cython
python3 setup_accel.py build_ext --inplace
```

**Windows / Claude Code note:** On a native Win11 host, the Cython extension
is *not* a speedup for Monte Carlo evaluation. The MC hot loop uses
`GADDAGMoveFinder` (pure Python) which runs at ~2.6M sims/s on 8 workers.
The C extension adds Python↔C call overhead per simulation that drops MC
throughput to ~3K sims/s — a ~900x regression. The C extension is only ~1.5x
faster for single-call move generation (8.4ms vs 12.4ms per rack), which
does not justify the MC penalty. The `.pyd` has been renamed to
`.pyd.disabled` and the engine runs pure Python.

## Coding conventions

- All board coordinates are **1-indexed** in the UI and `GameState`, but
  **0-indexed** internally in `board._grid` and move finders
- Blanks are tracked as `(row, col, letter)` tuples in `blank_positions`
- Move dicts use `'direction': 'H'` or `'V'`, plus `'row'`/`'col'` (1-indexed)
- The `_tiles_used()` method computes which rack tiles a move consumes

## Common tasks

**Add a new saved game:** Add a `_create_saved_game_N()` function and register
it in the `_SAVED_GAME_REGISTRY` dict inside `_init_default_games()`.

**Adjust MC parameters:** Edit constants in `mc_calibrate.py` (MIN_K, MAX_K,
MIN_N, EQUITY_SPREAD). Force recalibration with `calibrate(force=True)`.

**Performance benchmarking:** `python3 -c "from mc_calibrate import show_calibration; show_calibration()"`
