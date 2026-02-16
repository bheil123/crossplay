# CLAUDE.md -- Crossplay V15

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
empties. **Leftover tiles do not count against the player** -- there is no
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
     final racks -- there is no future draw and no tile penalty.
   - **Risk analysis changes**: the opponent has exactly 1 response
     move, not an infinite horizon. Threat = their single best move.
   - **3-ply simplifies to 2-ply**: your move + opponent's response
     (or opponent's move + your response). No ply 3 needed because
     the game ends after both final turns.
   - **MC simulation is unnecessary** when bag = 0 -- opponent's rack
     can be inferred exactly (unseen tiles = their rack), so evaluate
     deterministically instead of sampling.
   - **When bag has 1-6 tiles**: the draw that empties the bag triggers
     final turns. Factor in that after this draw, each player gets
     exactly one more turn.

## Architecture overview

```
game_manager.py          <- Entry point, game loop, AI orchestration
  |-- mc_eval.py         <- Monte Carlo 2-ply evaluation (core AI)
  |-- mc_calibrate.py    <- Auto-benchmarks hardware, computes NxK per turn
  |-- move_finder_c.py   <- C-accelerated move finder + post-validation
  |-- move_finder_opt.py <- Optimized Python move finder (fallback)
  |-- gaddag_accel.pyd   <- Compiled Cython extension (Windows)
  |-- gaddag_accel.so    <- Compiled Cython extension (Linux)
  |-- gaddag_accel.pyx   <- Cython source
  |-- gaddag.py          <- GADDAG trie, auto-build/cache from dictionary
  |-- gaddag_compact.py  <- Compact binary GADDAG format
  |-- board.py           <- Board representation (15x15 grid)
  |-- scoring.py         <- Move scoring with bonuses and crosswords
  |-- config.py          <- ALL constants: tile values, distribution, bonuses
  |-- dictionary.py      <- Word validation (196K words)
  |-- leave_eval.py      <- Rack leave evaluation + bingo probability DB
  |-- real_risk.py       <- Opponent threat analysis
  |-- lookahead.py       <- 1-ply lookahead
  |-- lookahead_3ply.py  <- 3-ply exhaustive endgame
  |-- parallel_eval.py   <- Multi-worker parallel evaluation
  |-- tile_tracker.py    <- Tile bag and tracking
  |-- play_game.py       <- Game state management
  |-- game_library.py    <- Game library persistence (CRUD, archive, search)
  |-- game_archive.py    <- Enriched move history + CSV export
  |-- games/             <- Persistent game library (git-tracked)
  |   |-- index.json     <- Slot assignments + per-opponent counters
  |   |-- active/        <- In-progress games (individual JSON)
  |   +-- archive.jsonl  <- Completed games (append-only JSONL)
  +-- superleaves/       <- SuperLeaves training pipeline
      |-- leave_table.py <- LeaveTable class (sorted tuple -> float)
      |-- fast_bot.py    <- Greedy bot for self-play
      |-- self_play.py   <- Single-game self-play loop
      |-- trainer.py     <- Parallel training orchestrator
      +-- validate.py    <- Head-to-head validation
```

## How to run

```bash
python run.py
```

Or as a module: `python -m crossplay.game_manager`

First run builds GADDAG (~48s) and runs MC calibration (~3s). Both are cached.

## Game library & persistence (V15)

Games are stored in a two-tier persistent library under `crossplay/games/`:

**Active games** (`games/active/{opponent}_{NNN}.json`):
- Individual JSON files for in-progress games
- Auto-saved on every `opp`, `play`, `rack` command
- Clean git diffs (one file per game)
- Loaded into slots 1-4 on startup via `games/index.json`

**Completed games** (`games/archive.jsonl`):
- Append-only JSONL for finished games
- One line per game, immutable after write
- Searchable by opponent name or word played

**Index** (`games/index.json`):
- Maps slots 1-4 to active game IDs
- Tracks per-opponent game counters (for sequential IDs)
- Optional opponent notes

**Game identification:** Per-opponent sequential IDs: `{opponent}_{NNN}`
(e.g., `canjam_002`, `sophie_001`). Counter tracked in `index.json`.

**Enriched move format:** Each move records rack (before play), tiles drawn,
timestamp, and engine recommendation (MC top 3 with equity and risk-adjusted
equity, plus whether the engine's top pick was followed).

**Commands:**
- `new N opponent` -- creates new game in library, assigns to slot N
- Games auto-save on every state change (no manual save needed)
- On game completion: archived to JSONL, active JSON deleted, slot freed
- First run auto-migrates from legacy factory functions

**Key module:** `game_library.py` -- `save_active()`, `load_active()`,
`list_active()`, `archive_completed()`, `search_archive()`,
`get_opponent_stats()`, `ensure_library_initialized()`

## Key design decisions

- **N=40 flat** for MC evaluation (early stopping makes per-candidate cost
  self-limiting at ~150-530 sims, so N=40 only costs 3-7s more than N=25
  in most cases while providing better equity spread coverage).
- **K=2000 ceiling** with convergence-based early stopping (SE < 1.0, check
  every 10 sims, min 100). Actual sims per candidate: 150-530 avg.
- **MC phase: 2-7s** (was 21-27s before early stopping). Turn time is now
  dominated by risk analysis (3s) and 3-ply (0-6s), not MC.
- **Risk-adjusted equity** (`risk_adj_equity`): `total_equity - expected_risk`.
  Shows conservative estimate with full (undampened) threat analysis weight.
  Displayed as `RiskEq` column in MC output when any candidate has nonzero risk.
- **Post-validation** in `move_finder_c.py` -- all C-generated moves have
  main-axis words and cross-words re-validated in Python as a safety net
- **Game library** persists all games as JSON/JSONL in `crossplay/games/`.
  Slots 1-4 are a cache view into the active library. Auto-saved on every
  state change. Completed games archived to JSONL. Git-tracked.
- **Accuracy analysis** in `ACCURACY.md` -- documents all factors impacting
  move evaluation accuracy, improvement roadmap, and equity formula breakdown
- **Baseline risk** -- board-wide threat analysis (`analyze_existing_threats()`)
  is cached at the start of `analyze()` and added to each move dict as
  `baseline_risk`. Surfaces pre-existing board vulnerabilities (open bonus
  squares from earlier moves) that per-move risk analysis doesn't scan.
  Constant across all moves, so informational only -- doesn't change rankings.

## Data files (committed, do not regenerate)

| File | Size | Purpose |
|------|------|---------|
| `crossplay_dict.pkl` | 2.3MB | Word dictionary (196K words) |
| `leave_bingo_prod.pkl` | 4.8MB | Bingo probability database |
| `pattern_index.pkl` | 11MB | Pattern matching index |
| `gaddag_accel.cpython-312-x86_64-linux-gnu.so` | 769KB | Pre-built C extension (Linux) |
| `gaddag_accel.cp312-win_amd64.pyd` | ~200KB | Pre-built C extension (Windows) |

## Auto-generated files (gitignored)

| File | Purpose |
|------|---------|
| `gaddag_compact.bin` | 28MB compact GADDAG, built on first run |
| `.mc_calibration.json` | MC throughput cache, valid 48h |

## Cython C extension (gaddag_accel)

Platform-specific compiled extensions for GADDAG traversal acceleration:
- **Linux:** `.so` for CPython 3.12 x86_64 (Claude Chat sandbox)
- **Windows:** `.pyd` for CPython 3.12 Win64 (native Win11 host)

To rebuild for your platform:

```bash
pip install cython
python setup_accel.py build_ext --inplace
```

The C extension accelerates single-call move generation (~1.5x faster per
rack). For MC evaluation, the pure-Python `GADDAGMoveFinder` is used in
parallel workers to avoid Python-C call overhead across process boundaries.

## Coding conventions

- All board coordinates are **1-indexed** in the UI and `GameState`, but
  **0-indexed** internally in `board._grid` and move finders
- Blanks are tracked as `(row, col, letter)` tuples in `blank_positions`
- Move dicts use `'direction': 'H'` or `'V'`, plus `'row'`/`'col'` (1-indexed)
- The `_tiles_used()` method computes which rack tiles a move consumes
- `GameState.final_turns_remaining` tracks endgame final turns in assisted
  mode: `None` = mid-game (bag not empty), `1` = one final turn left for
  whoever's turn it is, `0` = game over. When `== 1` and it's your turn,
  `analyze()` uses 1-ply (maximize score) instead of 2-ply, and threats
  are skipped since the opponent has no response

## IMPORTANT: No Unicode/emoji in output

**NEVER use emoji or non-ASCII characters in print statements, f-strings,
or any string that may be printed/logged.** Windows cp1252 encoding cannot
handle characters outside the cp1252 codepage, causing `UnicodeEncodeError`
crashes. This includes: emojis (any), arrows, check marks, box-drawing,
Greek letters, bullets, and similar.

Use ASCII alternatives: `->`, `[OK]`, `[X]`, `#`, `-`, `*`, `!`, etc.

A pre-commit hook enforces this rule.

## MC performance profile and optimization ideas

Current: 10 workers (ProcessPoolExecutor), Cython engine, ~2,824 sims/s
dense, ~789 sims/s very dense. Near-perfect linear scaling (no GIL --
separate processes). CPUs are at ~100% utilization during MC phase.

**Tail-call optimization (v13.2):** `_ctx_extend_right()` uses a `while`
loop to follow existing board tiles without recursion. When the next
square already has a tile, the function updates its local variables
(offset, wlen, row/col) and loops back instead of making a recursive
call. Only empty squares (which require branching over rack letters and
blanks) still recurse. This eliminates 40-60% of recursive calls on
dense boards. Measured improvement: +21% dense, +31% sparse, +13% vdense.
A full explicit-stack conversion was attempted but regressed performance
due to struct overhead exceeding Cython's efficient C-to-C calling.

**Time breakdown per simulation (dense board, ~2.1ms):**

| Component              | Time     | %    | Language |
|------------------------|----------|------|----------|
| GADDAG traversal       | 1,900 us | 90%  | C        |
| Word validation        | 100 us   | 5%   | Python   |
| Cross-check lookups    | 85 us    | 4%   | C        |
| Rack parsing           | 5 us     | 0.2% | C        |
| random.sample + join   | 2 us     | 0.1% | Python   |
| Result recording       | 4 us     | 0.2% | Python   |

The Cython fast path is already 96% C code. A pure C/Rust rewrite of the
move finder would only eliminate the 5% Python word-validation callbacks,
yielding ~5-8% speedup -- not worth the effort.

**Future software optimization ideas (each could yield 20-50%):**

1. **Faster `_ctx_get_child()` lookup** -- currently a linear scan through
   sorted children (up to 27 entries per node). This is the innermost hot
   function called ~600-800x per simulation. **Tested and rejected:**
   (a) Level 3 cache (`level3[27][27][27]`, 77KB) -- regressed 15-20% due
   to increased BoardContext size hurting cache locality and init cost.
   (b) Binary search for nodes with 4+ children -- regressed 2-5% due to
   branch prediction penalties outweighing saved comparisons (86% of nodes
   have 0-3 children, so the threshold check adds overhead to the common
   case). The linear scan with early exit is already optimal for this data
   distribution. Every production Scrabble engine (Quackle, Maven, Macondo,
   MAGPIE, wolges) uses the same linear scan approach.

2. **Anchor pre-filtering** -- **Analyzed and rejected.** Skipping "low
   productivity" anchors in the MC opponent-response search risks missing
   the opponent's actual best move. An anchor's value is rack-dependent --
   an anchor that's usually dead could be the best play for a specific
   random rack. Underestimating opponent threats inflates equity, making
   dangerous moves look safe. This defeats the purpose of MC evaluation.

3. **MC early stopping** -- **Implemented.** Each candidate's simulation
   loop tracks running sum/sum-of-squares for O(1) SE computation. After
   100 minimum sims, checks every 10 sims if SE < 1.0 (95% CI +/-2 pts).
   Self-calibrating: converged candidates stop early, close ones run full K.
   Density analysis (K=1000, 15 candidates, check_every=10):

   | Game phase | Unseen tiles | Sim reduction | Avg K per candidate |
   |------------|-------------|---------------|---------------------|
   | Early      | 82          | 69%           | 305                 |
   | Early-mid  | 65          | 72%           | 281                 |
   | Mid        | 52          | 73%           | 273                 |
   | Mid-late   | 39          | 84%           | 157                 |
   | Late       | 24          | 85%           | 149                 |
   | Very late  | 18          | 83%           | 175                 |

   99% confidence (SE < 0.76) was tested and rejected: costs ~75% more
   sims across all densities with zero ranking changes vs 95%. The freed
   sim budget is better spent on more candidates (higher N).
   Implementation: `mc_eval.py` in `_mc_eval_single_candidate` and
   `_mc_eval_exchange_candidate`.

4. **SIMD batch rack processing** -- process 4-8 opponent racks through the
   GADDAG simultaneously using vectorized operations. Could give 3-4x on
   traversal but requires complete architectural redesign.

**Hardware upgrade path (estimated 3-4x total MC throughput):**

Current dev hardware: i7-8700 (6C/12T, 12MB L3, DDR4-2133, ~2,824
dense sims/s). The 28MB GADDAG data does not fit in the 12MB L3 cache,
causing frequent DRAM fetches (~75ns) on `_ctx_get_child()` lookups.

Recommended upgrade: AMD Ryzen 9 9950X3D (~$699) + AM5 motherboard
(~$270) + 64GB DDR5-6000 (~$170) + cooler (~$35) = ~$1,175 total.

| Factor          | i7-8700          | 9950X3D            | Impact              |
|-----------------|------------------|--------------------|---------------------|
| L3 cache        | 12MB             | 128MB (3D V-Cache) | GADDAG fits in L3   |
| Cores/threads   | 6C/12T (10w MC)  | 16C/32T (20w+ MC)  | 2x parallel workers |
| IPC + clock     | Coffee Lake 4.6G | Zen 5 5.7G         | ~80% faster/core    |
| Memory          | DDR4-2133 34GB/s | DDR5-6000 90GB/s   | 2.5x bandwidth      |

The L3 cache is the biggest single factor: fitting the full GADDAG in
cache eliminates the main bottleneck and could ~2x per-worker throughput
alone. Combined with more cores and higher IPC, estimated ~9,000-12,000
dense sims/s (vs current ~2,824). Budget alternative: Ryzen 7 9800X3D
($449, 8C/16T, 96MB L3) saves $250 but fewer MC workers.

GPU/CUDA would not help -- GADDAG traversal is serial, branch-heavy
pointer-chasing through a trie, fundamentally incompatible with GPU
SIMD architecture.

## SuperLeaves training

On session start, check `crossplay/superleaves/status.json` for training
status. If status is "running" or "paused", report progress and suggest
resuming with `python -m crossplay.superleaves.trainer --resume --workers N`.

Quick start:
```bash
# Smoke test (100K games, ~2-3 hours)
python -m crossplay.superleaves.trainer --smoke-test --workers 4

# Full generation (1M games)
python -m crossplay.superleaves.trainer --workers 6

# Resume interrupted training
python -m crossplay.superleaves.trainer --resume --workers 6

# Validate trained table vs formula
python -m crossplay.superleaves.validate --table superleaves/gen1_100000.pkl --games 1000
```

Training runs in background and does not interfere with game play.
Uses `Board()` directly (no `Game` class), so it does not trigger game
library auto-save. Use `--workers N` to control CPU usage.

## Common tasks

**Start a new game:** `new N opponent` (e.g., `new 1 canjam`) -- creates
a new game in the library and assigns it to slot N. Auto-saved from then on.

**Adjust MC parameters:** Edit constants in `mc_calibrate.py` (MIN_K, MAX_K,
MIN_N, EQUITY_SPREAD). Force recalibration with `calibrate(force=True)`.

**Performance benchmarking:** `python3 -c "from mc_calibrate import show_calibration; show_calibration()"`
