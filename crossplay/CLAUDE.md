# CLAUDE.md -- Crossplay V17

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
   - **When bag has 1-8 tiles**: near-endgame hybrid evaluation.
     Bag-emptying moves get exact 3-ply. Non-emptying moves get
     parity-adjusted 1-ply (penalized for letting opponent empty bag).
     The player who empties the bag gains ~10 equity structural advantage.

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
  |-- board.py           <- Board representation + tiles_used() utility
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
  |-- nyt_filter.py      <- NYT curated word filter ([NYT?] warnings)
  |-- nyt_curated_words.txt <- Flagged words list (slurs, obscenities, trademarks)
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

## NYT curated word filter

NYT Crossplay uses a curated NASPA Word List 2023 that removes trademarks,
obscenities, and common slurs. Our engine uses the full NASPA dictionary,
so it may recommend words that Crossplay rejects.

**Files:**
- `nyt_curated_words.txt` -- plain text list of ~158 flagged words, editable
- `nyt_filter.py` -- loads the list at startup, provides O(1) lookups

**How it works:** When the engine displays move recommendations (1-ply table,
MC table, detailed top-3, final recommendation), any word on the curated list
gets a `[NYT?]` tag appended. This is a soft warning -- the word is valid in
NASPA but probably not in Crossplay.

**Categories:** slurs (38), obscenities (88), trademarks (32). Only HIGH and
MEDIUM confidence words are included. Mild/common words like DAMN, HELL, CRAP
are NOT flagged (NYT almost certainly keeps them).

**Maintenance:** If Crossplay rejects a word not on the list, add it to
`nyt_curated_words.txt`. If a flagged word is actually accepted, remove it.

## V17 changes: near-endgame evaluator + gen2 training

**Near-endgame hybrid evaluator (`evaluate_near_endgame()` in `lookahead_3ply.py`):**
When bag has 1-8 tiles, bag-emptying moves get exhaustive 3-ply evaluation over
all C(unseen, 7) opponent rack combinations (8-6435 combos, all tractable). Non-emptying
moves get parity-adjusted 1-ply equity (see bag parity below). This captures the
structural advantage of knowing the opponent's exact rack when you empty the bag.
Two-pass approach: instant parity-adjusted 1-ply candidates first, then exhaust
candidates under time budget. Routing in `game_manager.py._show_2ply_analysis()`
triggers when `1 <= bag_size <= 8`.

**Bag parity penalty (non-emptying moves at bag 1-8):**
Non-emptying moves are penalized based on the probability that the opponent will
empty the bag on their next turn, handing them the structural advantage of knowing
your exact rack + getting the last move (~10 equity points).

Formula: `parity_penalty = -P(opp_empties) * STRUCTURAL_ADVANTAGE`
where `P(opp_empties)` depends on `bag_after` (tiles left in bag after your draw):

| bag_after | P(opp empties) | penalty |
|:---------:|:--------------:|:-------:|
| 1         | 0.97           | -9.7    |
| 2         | 0.94           | -9.4    |
| 3         | 0.88           | -8.8    |
| 4         | 0.78           | -7.8    |
| 5         | 0.62           | -6.2    |
| 6         | 0.40           | -4.0    |
| 7         | 0.18           | -1.8    |

Constants in `lookahead_3ply.py`: `_PARITY_P_OPP_EMPTIES` (lookup table),
`_PARITY_STRUCTURAL_ADV = 10.0`. Display uses eval_type `'parity'` (vs
`'exhaust'` for bag-emptying moves and `'1ply'` for moves outside the range).

**tiles_used field name bug fix:**
`game_manager.py` stored rack tiles consumed as `'used'`, but `lookahead_3ply.py`
and `mc_eval.py` looked for `'tiles_used'`, falling back to `move['word']` which
overcounts (full word length vs actual rack tiles). Fixed at source (added
`'tiles_used': used` alongside `'used'`) and at all 5 consumer sites with
belt-and-suspenders fallback: `move.get('tiles_used', move.get('used', move['word']))`.

**Gen2 training signal improvements (`self_play.py`):**
- Equity-based signal: `signal = move_equity - top_score_equity` (not game-mean)
- Outcome-weighted: `outcome_mult = 1.0 + 0.3 * clamp(spread / 150, -1, 1)`
- Expanded `top_k=30` in `fast_bot.py` (was 15)

**Cross-generation training (`trainer.py --init-from`):**
`--init-from gen1_350000.pkl` loads a previous generation's table as starting point
for the next generation (starts at 0 games, not resume).

**Gen1 results:** 350K games, 921K leave entries. Validated at parity with formula
(498-501 over 1000 games, +2.9 avg spread). Deployed as `deployed_leaves.pkl`.

**Gen2 status:** Training 700K games with 4 workers, initialized from gen1.
Check `superleaves/status.json` for progress.

## V18 roadmap: bingo -> sweep terminology

NYT Crossplay calls a 7-tile play a "sweep" (40 pts), not a "bingo" (50 pts).
The codebase currently uses "bingo" throughout. Scope: 221 references across
21 files, 14 variable/constant names to rename, Cython rebuild required.
Estimated 2-3 hours. Deferred to V17 as a breaking terminology change.

Key renames: `BINGO_BONUS` -> `SWEEP_BONUS`, `bingo_prob` -> `sweep_prob`,
`is_bingo` -> `is_sweep`, `[!B]` marker -> `[!S]`, `[DB]` -> `[DS]`, etc.

## V16.1 changes: end command for game completion

**`end` command:**
`end YOUR_SCORE OPP_SCORE [win/loss/tie]` -- completes the current game slot.
Updates final scores, auto-detects result from spread (or uses explicit result),
archives to `archive.jsonl` with proper fields, deletes active JSON, and clears
the slot in `index.json`. This is now the enforced path for completing games --
no more manual JSON editing or stale active files after games end.

`GameManager.end_game(slot, result, your_score, opp_score)` is the programmatic
API. Slot defaults to `current_slot`, result auto-detected if omitted.

## V16 changes: move finder fix + auto-analyze

**Move finder bug fix (all 3 implementations):**
The GADDAG `_gen_left_part()` function had a guard `if partial_word == 0 and
limit == 0:` that prevented trying to place the first letter AT an anchor
square when there was empty space before it (limit > 0). This caused the
engine to miss valid moves where the word starts at the anchor rather than
extending left/up first. Example: ENRICHER at R3C14 V (57 pts, blank E) was
missed in favor of RICH at R3C7 H (16 pts) -- a 41-point miss.

Fixed in all three move finder implementations:
- `move_finder_opt.py` -- Python fast path (CompactGADDAG), two occurrences
- `move_finder_gaddag.py` -- Python slow path (tree GADDAG)
- `gaddag_accel.pyx` -- Cython C extension, two occurrences (rebuilt .pyd)

The fix removes `and limit == 0` from the guard condition. When limit > 0,
both strategies are now tried: (1) extend left first, (2) start at anchor.

**Auto-analyze after opponent moves:**
`record_opponent_move()` now automatically runs `analyze()` after recording
an opponent move (when a rack is set). This ensures the engine recommendation
is always available before any move is played, preventing situations where a
suboptimal move is recommended without engine backing. If no rack is set yet,
threats are shown with a tip to set the rack.

**Play command with post-draw rack:**
`play WORD R C H/V [NEW_RACK]` -- optional 6th argument is the post-draw
rack. When provided, `play_move()` skips tile validation against the current
rack and sets the rack directly. This handles the assisted-play workflow
where the user already played on the real board and reports the move with
their new rack. Drawn tiles are inferred by comparing old rack (minus tiles
used) to the new rack. Without the 6th arg, the old behavior (validate
against current rack, simulate draw) is preserved.

**Rack tracking (IMPORTANT):**
Always capture the player's rack with every move. When recording moves:
- Set `game.state.your_rack` BEFORE calling `play_move()` so the rack-before
  is saved in the move record
- After the move, either provide the new rack (post-draw) or let the engine
  simulate the draw
- For opponent moves, rack is unknown (null) — that's expected
- For NYT Crossplay games: NYT reports full move-by-move detail including
  racks after game completion. Backfill rack data post-game for analysis.
- Without rack data, post-game analysis comparing engine vs NYT recommendations
  is impossible

**Game management — use standard functions ONLY:**
Never edit game JSON files directly. Always use GameManager methods:
- `gm.new_game(slot, opponent_name)` — create a new game
- `game.play_move(word, row, col, horizontal)` — record your move
- `game.record_opponent_move(word, row, col, horizontal, score)` — record opponent
- `gm.end_game(slot, result, your_score, opp_score)` — archive completed game
- `game.state.your_rack = "LETTERS"` — set rack before analysis/play
These functions handle scoring, blank detection, board placement, bag tracking,
endgame detection, auto-save, and crash-recovery archiving.

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
- `board.tiles_used(board, word, row, col, horizontal)` is the canonical
  utility for computing which rack tiles a move consumes (returns `list`).
  `game_manager._tiles_used()` and `play_game._tiles_used()` are thin
  wrappers that return `str` via `''.join()`
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

**Benchmarked hardware:**

| Metric           | i7-8700 (desktop)       | i7-1065G7 (laptop)      |
|------------------|-------------------------|-------------------------|
| CPU              | 6C/12T, 4.6GHz, 12MB L3 | 4C/8T, 1.3-3.9GHz, 8MB L3 |
| RAM              | 34GB DDR4-2133          | 16GB LPDDR4             |
| MC workers       | 10                      | 6                       |
| Sparse (1w)      | 476 sims/s              | 476 sims/s              |
| Dense (1w)       | 282 sims/s              | 186 sims/s              |
| VDense (1w)      | 79 sims/s               | 60 sims/s               |
| Sparse (Nw)      | 2,856 sims/s            | 2,856 sims/s            |
| Dense (Nw)       | 2,824 sims/s            | 1,116 sims/s            |
| VDense (Nw)      | 789 sims/s              | 361 sims/s              |

Sparse boards perform identically per-worker (Ice Lake IPC compensates for
lower clock). Dense/VDense hit cache harder -- laptop's 8MB L3 vs 12MB L3
plus fewer workers yields ~40-46% of desktop multi-worker throughput.
Near-perfect linear scaling (no GIL -- separate processes).

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

Current dev hardware: i7-8700 desktop (6C/12T, 12MB L3, DDR4-2133,
~2,824 dense sims/s) and i7-1065G7 laptop (4C/8T, 8MB L3, LPDDR4,
~1,116 dense sims/s). The 28MB GADDAG data does not fit in either
machine's L3 cache, causing frequent DRAM fetches (~75ns) on
`_ctx_get_child()` lookups.

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
resuming.

**Restarting training (e.g. after context compaction):**
The easiest way is to double-click `start_training.bat` in the project root.
This uses the full Python path to avoid Windows Store alias issues and
auto-resumes from the latest checkpoint.

Alternatively, from a **native Windows terminal** (not Git Bash — multiprocessing
deadlocks in Git Bash):
```bash
C:\Users\billh\AppData\Local\Programs\Python\Python312\python.exe -m crossplay.superleaves.trainer --resume --generation 2 --games 700000
```

Or launch via PowerShell from Claude Code:
```bash
powershell.exe -Command "Start-Process 'C:\Users\billh\crossplay\start_training.bat'"
```

Worker count defaults to `cpu_count - 3` (reserves 3 cores for OS, Claude Code,
and game analysis). Override with `--workers N`. On the 8-core dev machine this
gives 5 workers.

**IMPORTANT:** After launching, workers take 30-90 seconds to load the GADDAG
before any progress appears. Don't kill the process during this startup phase.

Quick start:
```bash
# Smoke test (100K games, ~2-3 hours)
python -m crossplay.superleaves.trainer --smoke-test

# Full gen1 (350K games, ~22 hours)
python -m crossplay.superleaves.trainer --games 350000

# Gen2 from gen1 (700K games, equity-based signal)
python -m crossplay.superleaves.trainer --generation 2 --games 700000 --init-from gen1_350000.pkl

# Resume interrupted training (auto-finds latest checkpoint)
python -m crossplay.superleaves.trainer --resume --generation 2 --games 700000

# Validate trained table vs formula
python -m crossplay.superleaves.validate --table superleaves/gen1_350000.pkl --games 1000
```

**Deployment:** Copy a checkpoint to `deployed_leaves.pkl` in the superleaves
directory. `leave_eval.py` lazy-loads this file and uses trained values before
falling back to formula. Gen1 is currently deployed.

Training runs in background and does not interfere with game play.
Uses `Board()` directly (no `Game` class), so it does not trigger game
library auto-save. Use `--workers N` to control CPU usage.

**Training data in Git (LFS):** Two pkl files are tracked via Git LFS and
pushed to GitHub so the engine works immediately after cloning:
- `deployed_leaves.pkl` -- the live leave table used by the engine
- `gen1_350000.pkl` -- final gen1 output (seed for gen2 training)

All other pkl files (intermediate checkpoints, worker tables) are gitignored.
After training a new generation, update deployed_leaves.pkl and commit:
```bash
cp superleaves/gen2_700000.pkl superleaves/deployed_leaves.pkl
git add crossplay/superleaves/deployed_leaves.pkl
git commit -m "Deploy gen2 SuperLeaves"
git push
```

**Cleanup after training:** Intermediate checkpoints (gen2_10000.pkl through
gen2_690000.pkl) accumulate at ~21 MB each during training. Delete all except
the final checkpoint after training completes:
```bash
# Keep only gen2_700000.pkl (the final), delete intermediates
ls superleaves/gen2_*.pkl | grep -v gen2_700000 | xargs rm
```

**Cloning on a new machine:** `git clone` + `git lfs pull` fetches all code
and the deployed leave table. The engine is immediately playable. The GADDAG
(gaddag_compact.bin) auto-builds on first run (~48 seconds).

## Common tasks

**Start a new game:** `new N opponent` (e.g., `new 1 canjam`) -- creates
a new game in the library and assigns it to slot N. Auto-saved from then on.

**Adjust MC parameters:** Edit constants in `mc_calibrate.py` (MIN_K, MAX_K,
MIN_N, EQUITY_SPREAD). Force recalibration with `calibrate(force=True)`.

**Performance benchmarking:** `python3 -c "from mc_calibrate import show_calibration; show_calibration()"`
