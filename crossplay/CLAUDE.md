# CLAUDE.md -- Crossplay V21

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
game_manager.py          <- Game core (init, display, persistence, bag) + GameManager
  |-- game_analysis.py   <- GameAnalysisMixin: analyze, threats, MC display, endgame
  |-- game_moves.py      <- GameMovesMixin: play_move, record_opponent_move, exchange
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
  |-- config.py          <- ALL constants: tile values, distribution, bonuses, tuning
  |-- dictionary.py      <- Word validation (196K words)
  |-- leave_eval.py      <- Rack leave evaluation + bingo probability DB
  |                         (real_risk.py removed in V21.1)
  |-- lookahead.py       <- 1-ply lookahead
  |-- lookahead_3ply.py  <- 3-ply exhaustive endgame
  |-- parallel_eval.py   <- Multi-worker parallel evaluation
  |-- tile_tracker.py    <- Tile bag and tracking
  |-- log.py             <- Structured logging (get_logger, configure)
  |-- play_game.py       <- Game state management
  |-- game_library.py    <- Game library persistence (CRUD, archive, search)
  |-- game_archive.py    <- Enriched move history + CSV export
  |-- nyt_filter.py      <- NYT curated word filter ([NYT?] warnings)
  |-- nyt_curated_words.txt <- Flagged words list (slurs, obscenities, trademarks)
  |-- games/             <- Persistent game library (git-tracked)
  |   |-- index.json     <- Slot assignments + per-opponent counters
  |   |-- active/        <- In-progress games (individual JSON)
  |   +-- archive.jsonl  <- Completed games (append-only JSONL)
  |-- NYT/               <- NYT Crossplay analysis (screenshots + notes)
  |   |-- games/         <- Screenshots organized by date/opponent (gitignored)
  |   +-- analysis/      <- Comparison notes, findings
  +-- superleaves/       <- SuperLeaves training pipeline
      |-- leave_table.py <- LeaveTable class (sorted tuple -> float)
      |-- fast_bot.py    <- Greedy bot for self-play
      |-- self_play.py   <- Single-game self-play loop
      |-- trainer.py     <- Parallel training orchestrator
      +-- validate.py    <- Multi-worker head-to-head validation
tests/                   <- pytest test suite (91 tests)
  |-- conftest.py        <- Shared fixtures (board, board_with_hike, etc.)
  |-- test_config.py     <- Config constants, tile distribution, bonus squares
  |-- test_board.py      <- Board placement, queries, tiles_used
  |-- test_scoring.py    <- Tile values, word/move scoring, crosswords
  |-- test_tile_tracker.py <- Bag tracking, sync_with_board, blanks
  +-- test_game.py       <- Game class: mixins, DI, _get_tile_context, play_move
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
- Loaded into slots 1-8 on startup via `games/index.json`
- **No bag array stored** -- bag is reconstructed on load from tile
  distribution minus board tiles minus your rack minus blanks (via
  `Game._reconstruct_bag()`). This prevents stale bag state after
  manual edits or multi-computer git syncs.

**Completed games** (`games/archive.jsonl`):
- Append-only JSONL for finished games
- One line per game, immutable after write
- Searchable by opponent name or word played

**Index** (`games/index.json`):
- Maps slots 1-8 to active game IDs
- Tracks per-opponent game counters (for sequential IDs)
- Optional opponent notes

**Game identification:** Per-opponent sequential IDs: `{opponent}_{NNN}`
(e.g., `canjam_002`, `sophie_001`). Counter tracked in `index.json`.

**Enriched move format:** Each move records rack (before play), tiles drawn,
timestamp, and engine recommendation (MC top 3 with equity and risk-adjusted
equity, plus whether the engine's top pick was followed).

**Commands:**
- `new N opponent` -- creates new game in library, assigns to slot N
  (refuses if slot has an in-progress game; use `reset N` first)
- `load GAME_ID` -- loads an active game into the current slot (for
  resuming orphaned or unlinked games after session restart)
- `games` -- lists all active games with slot assignments; shows
  orphaned games not currently in any slot
- Games auto-save on every state change (no manual save needed)
- On game completion: archived to JSONL, active JSON deleted, slot freed
- First run auto-migrates from legacy factory functions

**Resuming after session restart:** Games in slots auto-load from
`index.json` on startup. If a game was orphaned (exists in
`games/active/` but not in any slot), startup prints a warning.
Use `games` to see all active games, then `load GAME_ID` to bring
an orphaned game back into a slot.

**Key module:** `game_library.py` -- `save_active()`, `load_active()`,
`list_active()`, `archive_completed()`, `search_archive()`,
`get_opponent_stats()`, `ensure_library_initialized()`

## Multi-computer workflow

Games are git-tracked and can be played from multiple computers. The
workflow to switch between machines:

1. **Before leaving a computer:** Commit and push game state
   (`git add crossplay/games/ && git commit -m "..." && git push`)
2. **When starting on a new computer:** `git pull` to get latest game files
3. **After pulling:** Use `reload` command to refresh in-memory game state
   from the updated JSON files on disk. Without this, the engine uses
   stale in-memory state from when it first loaded.

**`reload` command:**
- `reload` -- reloads all game slots from disk
- `reload N` -- reloads only slot N

**Why this is needed:** GameManager loads all games into memory at startup.
If `git pull` overwrites the JSON files afterward, the in-memory Game
objects still hold the old state. `reload` re-reads the JSON files and
reconstructs the board, scores, rack, and bag from the current disk state.

**For Claude Code sessions:** When the user says they pulled from another
computer, always call `gm.reload_games()` before recording moves or
analyzing. This prevents stale board state causing incorrect analysis.

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
  Slots 1-8 are a cache view into the active library. Auto-saved on every
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

## NYT move analysis (screenshots)

Game screenshots from NYT Crossplay are stored in `NYT/games/` organized
by date or opponent (e.g., `20260220_session1/`, `20260222_vs_sophie/`).
Screenshots are gitignored (large PNGs) but the directory structure and
README are tracked.

**Purpose:** Compare NYT's move recommendations against our engine to:
- Identify where NYT plays differently (and whether it's better or worse)
- Discover words NYT accepts/rejects that we don't expect
- Understand NYT's strategic preferences (defensive play, tile management)
- Find engine improvement opportunities

**Move record fields for NYT tracking:**
Each move in `board_moves` has these NYT-related fields:
- `engine` -- our engine's top recommendation at the time (dict or null)
- `nyt` -- NYT's suggested move if captured from screenshot (dict or null)
- `win_pct` -- NYT's displayed win probability (float or null)
- `engine_version` -- which engine version analyzed the position at play time
- `nyt_rating` -- NYT's rating for the move ("Best", "Excellent", "Great",
  "Good", "Fair", "Weak", "Chance to learn", or null)
- `nyt_strategy` -- NYT's per-move strategy score (0-99, or null)

**Game-level fields for NYT analysis:**
- `engine_version` -- engine version at archival time (auto-set by
  `archive_completed()`). Note: per-move `engine_version` may differ
  if the engine was updated mid-game.
- `nyt_analysis_version` -- engine version when post-game NYT comparison
  was performed. Important because the engine's move finder improves
  over time -- a "missed" recommendation may not have existed in the
  engine version used during play.
- `nyt_strategy_you` -- NYT's overall strategy score for you (0-99)
- `nyt_strategy_opp` -- NYT's overall strategy score for opponent (0-99)
- `nyt_luck_you` -- NYT's luck rating for you (0-100)
- `nyt_luck_opp` -- NYT's luck rating for opponent (0-100)

**Version tracking rationale:** The engine improves over time (e.g., the
V16 move finder fix added moves that V15 missed entirely). When comparing
"player vs engine recommendation," the relevant engine version is the one
running at play time (`engine_version` on each move), not the current
version. When doing post-game NYT comparison, `nyt_analysis_version`
records which engine was available for that analysis. This prevents
attributing "player ignored engine" when the engine hadn't found the move.

**Workflow:** After a completed game, take screenshots of NYT's move-by-move
review (NYT shows its recommended move for each turn). Store in
`NYT/games/YYYYMMDD_vs_OPPONENT/`. Use Claude to read the screenshots and
compare NYT vs engine recommendations move-by-move. Record the current
engine version as `nyt_analysis_version` on the archive record.

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
(498-501 over 1000 games, +2.9 avg spread).

**Gen2 results:** 1M games, 921K leave entries, initialized from gen1 with
equity-based signal. Validation (1000 games each):
- Gen2 vs formula: 488-507 (-1.1 spread) -- near parity
- Gen2 vs gen1: 507-488 (+1.5 spread) -- marginal improvement
Gen2 is deployed as `deployed_leaves.pkl`.

**C-accelerated move scoring (`score_moves_c` in `gaddag_accel.pyx`):**
Move scoring during training was moved from Python (in `move_finder_c.py`)
to Cython. The new `score_moves_c()` function converts the board grid, bonus
squares, and tile values to C arrays once per call, then scores all candidate
moves (main word + crosswords + bingo bonus) using C-level loops. This is
called by `find_all_moves_c()` after GADDAG traversal returns raw move tuples.
Output format is unchanged -- same move dicts with all fields preserved.
Result: **70% training speedup** (2.7 -> 4.6 games/sec at 5 workers).

**Orphaned worker cleanup (`_cleanup_stale_workers` in `trainer.py`):**
On Windows, orphaned multiprocessing workers from crashed/killed training runs
can hold file locks on `gaddag_accel.pyd` and compete for CPU. The trainer now
scans for orphaned Python multiprocessing workers (parent PID no longer alive)
at startup and terminates them before spawning new workers.

## V18 changes: TD-learning gen3 + Cython word_set optimization

**TD(0) learning for SuperLeaves gen3 (`self_play.py`):**
Replaces gen2's outcome-weighted equity signal with temporal difference
learning. Per-player trajectories tracked through each game. At game end,
a backward pass computes TD targets:

`td_target_t = advantage_t + gamma * V(next_leave_t+1)`

Where `advantage_t` = equity differential (same as gen2: `score + leave_val -
top_raw_score`), and `V(next_leave)` bootstraps from the current leave table.
Terminal moves get `V_next = 0` (Crossplay has no tile penalty). Outcome
weighting removed -- TD naturally propagates trajectory info through the
bootstrap chain. Gamma defaults to 0.97 (`--td-gamma` CLI arg).

Result: ~17 observations per game (vs gen2's ~0.9), giving ~19x more training
signal per game. Each leave gets credit/blame for what actually happened on
the next draw, not just whether this specific move was the best right now.

**Cython `word_set` optimization (`gaddag_accel.pyx`):**
`find_moves_c()` previously accepted a `dictionary` object and called
`st.dictionary.is_valid(word)` for every candidate move and cross-check --
4.2M Python method calls per 50 games. Each call dispatched a Python method,
called `.upper()` (redundant -- words already uppercase from C), then did
`word in self._words`.

Changed to accept `word_set` (the raw `set` from `dictionary._words`) and do
`word in st.word_set` directly in Cython. This eliminates:
- 4.2M Python method dispatch calls
- 4.4M redundant `.upper()` calls (0.76s per 50 games)
- Python/C boundary crossing overhead (~3.5s per 50 games)

The MC fast path (`BoardContext`) already used this pattern. Now the training
path matches.

**Profiled impact (50 games single-worker):**
- Before: `is_valid()` callbacks = 5.48s / 18.8s total = 29% of runtime
- After: `word in word_set` = ~1.2s (dict `__contains__` only, no overhead)
- Net: ~23% per-worker speedup

**Training throughput improvement:**
- Before (gen2 signal, old Cython): 15.3 g/s (5 workers, old machine)
- Before (TD signal, old Cython): 13.0 g/s (9 workers, new machine)
- After (TD signal, new Cython): **21.8 g/s** (9 workers) -- **68% faster**
- Batch steady-state: 24.5-25.6 g/s

**Projected training times at 21.8 g/s:**

| Games | Time |
|-------|------|
| 1M | ~12.7h |
| 2M | ~25.5h |
| 3M | ~38.2h (~1.6 days) |

**Files changed:**
- `gaddag_accel.pyx`: `_SearchState.dictionary` -> `.word_set`, `find_moves_c()`
  signature, `_cross_check()`, `_record_move()` -- all use `word in st.word_set`
- `gaddag_accel.cp312-win_amd64.pyd`: rebuilt
- `move_finder_c.py`: passes `dictionary._words` instead of `dictionary`
- `self_play.py`: TD trajectories + backward pass, `td_gamma` parameter
- `trainer.py`: `--td-gamma` CLI arg, threaded to workers

**Gen3 training results (1M games, TD-learning):**
- Completed at 27.6 g/s (9 workers), ~10.1 hours
- 921K leave entries, final checkpoint: `gen3_1000000.pkl`
- **Deployed** as `deployed_leaves.pkl` (V18.0.0)
- Validation: 513-484 vs formula (+2.1 spread), 510-486 vs gen2 (+1.0 spread)

**`reload` command (multi-computer game sync):**
`reload` / `reload N` -- re-reads game JSON files from disk after `git pull`
or external edits. Reconstructs in-memory Game objects with correct board
state, scores, rack, and bag. Essential when working from multiple computers
since GameManager caches game state in memory at startup.

`GameManager.reload_games(slot=None)` is the programmatic API. Reloads all
slots if `slot` is None, or a single slot if specified.

## V21.2 changes: DadBot v6 A/B tests (SuperLeaves + 3-ply)

**Two independent A/B tests for DadBot v6, run via tournament:**

1. **SuperLeaves gen4 vs formula** -- does TD-trained leave evaluation beat
   the hand-tuned formula?
2. **3-ply override for bag 9-21** -- does considering your counter-response
   (ply 3) improve play vs MC 2-ply only?

**Environment variable feature flags (independent of BOT_TIER):**
```
DADBOT_LEAVES=formula|superleaves   (default: formula -- V21 baseline)
DADBOT_3PLY=off|on                  (default: off -- V21 baseline)
```

**4 test configurations:**
- `baseline`: formula + no-3ply (should match v5 behavior)
- `superleaves`: superleaves + no-3ply (trained leaves only)
- `3ply`: formula + 3ply (3-ply override only)
- `both`: superleaves + 3ply (both features combined)

**3-ply integration design:**
- Bag 9-21 only: below 9, `evaluate_near_endgame()` handles with exact
  enumeration. Above 21, opponent gets too many unseen tiles for meaningful
  3-ply analysis.
- 15s time budget per move (acceptable for standard/deep tiers).
- Full override, not blend: when 3-ply produces a result, it replaces the
  MC pick. 3-ply's ply-3 counter-response gives it a structural advantage.
- Fallback to MC if 3-ply fails or times out.

**SuperLeaves integration:**
- `_leave_value()` checks `DADBOT_LEAVES` flag:
  - `formula`: always uses `_formula_leave()` (V21 behavior)
  - `superleaves`: trained table -> formula fallback -> bingo bonus

**3-ply bug fix (`lookahead_3ply.py`):**
`evaluate_3ply()` passed ALL unseen tiles as opponent's "rack" to the move
finder, allowing generation of physically impossible moves (e.g., 14-letter
words needing 13 tiles from rack when max rack is 7). Fixed by adding
`_tiles_from_rack()` helper that counts empty board positions in a move's
path, then filtering opponent moves to <= 7 tiles from rack.

**Files changed:**
- `crossplay-tournament/bots/dadbot_v6.py` -- new file (copied from v5),
  A/B flags, 3-ply import/integration, SuperLeaves toggle in `_leave_value()`
- `crossplay-tournament/run_tourney.py` -- rewritten for A/B test matrix
  with 4 configs, shared seed generation, env var injection per config
- `crossplay/lookahead_3ply.py` -- `_tiles_from_rack()` filter + bug fix

**Running the tournament:**
```bash
# Full A/B matrix (20 games x 4 configs = 80 games)
python run_tourney.py

# Quick smoke test (5 games per config)
python run_tourney.py --games 5

# Single config only
python run_tourney.py --config baseline

# With timing diagnostics
python run_tourney.py --timing
```

## V21 changes: heuristic-validated equity formula

**Context: DadBot v5 tournament validation.**
Multi-tier A/B testing (blitz/fast/standard/deep, 20 games each) proved
that a simple equity formula dominates all hand-tuned heuristics. Stripped
DadBot achieved monotonic scaling: 9-11 (blitz) -> 11-9 (fast) -> 16-4
(standard) -> 16-4 (deep). The pre-V21 engine with all heuristics showed
INVERSE scaling at higher compute (more candidates = worse play).

**Equity formula simplified:**
- 1-ply: `equity = score + leave_value` (was: + blocking - risk + DLS + DD + turnover + HVT + TW/DW)
- MC 2-ply: `total = mc_equity + leave_value` (was: + positional_adj * 0.5)
- Leave evaluation: formula only (was: SuperLeaves table -> formula fallback -> bingo bonus)

**Heuristics removed from ranking and code:**

V21.0 disabled heuristics from ranking but kept computation for display.
V21.1 removed ALL heuristic computation code for speed (~2050 lines deleted).

**Removed code:**
1. Risk analysis (`_calculate_probabilistic_risk`, `_calculate_exhaustive_opp_risk`)
2. Blocking bonus (`_calculate_blocking_bonus`)
3. Opening heuristics (`opening_heuristics.py` deleted entirely)
4. Threat analysis (`real_risk.py` deleted entirely, `show_existing_threats` removed)
5. Bingo blocking analysis (`_show_bingo_blocking_analysis`, `_find_opponent_bingo`)
6. Exchange evaluation (`_generate_exchange_candidates`, `_mc_eval_exchange_candidate`)
7. Blank correction (`_blank_correction_factor`)
8. Positional adjustment (`MC_POSITIONAL_DAMPEN`, `positional_adj` field)
9. ~40 dead config constants (risk thresholds, threat limits, blocking bonuses, etc.)

**Kept for A/B testing:**
- `leave_eval.py`: `USE_FORMULA_ONLY = True` flag + full SuperLeaves/bingo legacy path

**1-ply table simplified:** `# Word Position Pts Leave Equity`
**MC table simplified:** `# Word Pos Pts AvgOpp MaxOpp Std %Beats Leave MC Eq`

**Speed gain:** ~3s per turn saved (risk analysis was the bottleneck).
Target flow: move gen (0.5s) -> leave eval -> MC 2-ply (2-7s) = 3-8s total.

**Files changed (V21.0 + V21.1 combined):**

| File | Changes |
|------|---------|
| `leave_eval.py` | USE_FORMULA_ONLY flag, bypasses SuperLeaves + bingo |
| `mc_eval.py` | Removed exchange eval, blank correction, positional adj (~305 lines) |
| `game_analysis.py` | Removed risk/blocking/heuristics/exchange (~890 lines) |
| `game_moves.py` | Removed show_existing_threats calls |
| `game_manager.py` | Removed threat cache initialization |
| `opening_heuristics.py` | DELETED (~538 lines) |
| `real_risk.py` | DELETED (~400+ lines) |
| `config.py` | Removed ~40 dead constants |
| `__init__.py` | Version bump to 21.1.0 |

## V20.4 changes: threat analyzer (REMOVED in V21.1)

*V20.4 added cross-valid pattern injection and tiered wildcard limits to
`real_risk.py`. All removed in V21.1 along with the entire threat analysis
module -- DadBot v5 tournament proved threat heuristics degrade play quality.*

## V20 changes: architectural cleanup + test suite

**Deep architectural review** implementing 13 items for maintainability and
testability. No behavior changes -- same move recommendations, same output.

**Item 1: TileTracker consolidation.**
Added `Game._get_tile_context(rack=None)` returning `(tracker, unseen_dict,
bag_size)`. Replaced 14 copy-pasted 5-line TileTracker instantiation blocks
across `game_manager.py`, `game_analysis.py`, and `game_moves.py`. Removed
unused `TileTracker` imports from mixin modules.

**Item 2: Config centralization.**
Moved ~50 magic numbers from `game_analysis.py` and `game_manager.py` into
`config.py` as named constants. Categories: MC display, risk analysis,
analysis thresholds, scoring adjustments, display formatting, endgame, and
exchange evaluation. All constants are documented with inline comments.

**Item 3: DI for Game class.**
`Game.__init__` accepts optional `gaddag` and `dictionary` parameters for
dependency injection. When both are provided, `get_resources()` is skipped.
Enables fast unit testing without loading the 28MB GADDAG.

**Item 4: Formal test suite (pytest).**
91 tests across 6 files in `tests/`. Covers: config constants and tile
distribution (17), board placement and queries (25), tile tracker and bag
management (8), scoring with bonuses and crosswords (14), Game class with
mixin composition, DI, and play_move (27). Run: `pytest tests/`.

**Item 5: analysis_lock.py hardening.**
Lock file operations wrapped in try/except for robustness against filesystem
errors. Stale lock auto-cleared on timeout.

**Item 11: Structured logging.**
New `log.py` module with `get_logger()` and `configure()`. Internal diagnostics
(timing, fallback warnings, MC stats) use `logger.debug()`/`.warning()`.
User-facing output stays as `print()`. Configure via `CROSSPLAY_LOG_LEVEL`
env var (default: WARNING).

**Items 6,7,9,12,13:** Minor cleanups -- consistent method ordering, reduced
code duplication, improved error messages, cleaner import structure.

**Items 8,10:** Already implemented (cross-check cache in Cython BoardContext)
or deferred (MC pool lifecycle -- already persistent between analyze() calls).

**V19 changes: Game class mixin refactor.**
Split 2,875-line `Game` class into mixin modules:
- `game_analysis.py` (GameAnalysisMixin, ~1,500 lines): analyze, threats,
  MC display, endgame solvers, risk calculators, strategic analysis
- `game_moves.py` (GameMovesMixin, ~670 lines): play_move,
  record_opponent_move, record_exchange, validation
- `game_manager.py` (Game core + GameManager, ~1,400 lines): init, display,
  persistence, bag management, shared utilities

All `self.*` references work unchanged -- mixins share the same instance.
No changes to any calling code. `class Game(GameAnalysisMixin, GameMovesMixin):`

**Deferred: bingo -> sweep terminology rename.**
NYT calls a 7-tile play a "sweep" (40 pts), not a "bingo" (50 pts). Scope:
221 references across 21 files, Cython rebuild required. Deferred.

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

**Opponent move validation (V17.2):**
`record_opponent_move()` validates moves BEFORE placing them on the board.
Checks: board bounds, tile conflicts, connectivity to existing tiles (or
center coverage on first move), main word validity (GADDAG dictionary), and
cross-word validity. Invalid moves print specific errors and suggest `opp!`
to force-accept. The `opp!` override handles cases where WWF accepts a word
not in our NASPA dictionary. Validation is read-only and cannot corrupt board
state.

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

The C extension provides three acceleration paths:
- `find_moves_c()` -- GADDAG traversal for move generation
- `score_moves_c()` -- C-level move scoring (main word + crosswords + bingo)
- `find_best_score_c()` + `BoardContext` -- pre-computed MC simulation path

For MC evaluation, the pure-Python `GADDAGMoveFinder` is used in parallel
workers to avoid Python-C call overhead across process boundaries.

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
| Word validation        | 100 us   | 5%   | C (V18: word_set) |
| Cross-check lookups    | 85 us    | 4%   | C        |
| Rack parsing           | 5 us     | 0.2% | C        |
| random.sample + join   | 2 us     | 0.1% | Python   |
| Result recording       | 4 us     | 0.2% | Python   |

The Cython fast path is 97% C code (V18: word validation moved from Python
method dispatch to direct C-level set lookup -- see V18 changes).

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
C:\Users\billh\AppData\Local\Programs\Python\Python312\python.exe -m crossplay.superleaves.trainer --resume --generation N --games NNNNNN
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

# Gen2 from gen1 (1M games, equity-based signal)
python -m crossplay.superleaves.trainer --generation 2 --games 1000000 --init-from gen1_350000.pkl

# Gen3 from gen2 (TD-learning, 1M+ games, ~12.7h at 21.8 g/s)
python -m crossplay.superleaves.trainer --generation 3 --games 1000000 --init-from gen2_1000000.pkl --td-gamma 0.97

# Resume interrupted training (auto-finds latest checkpoint)
python -m crossplay.superleaves.trainer --resume --generation N --games NNNNNN

# Validate trained table vs formula (multi-worker, ~80s for 1000 games)
python -m crossplay.superleaves.validate --table superleaves/gen2_1000000.pkl --games 1000
python -m crossplay.superleaves.validate --table superleaves/gen2_1000000.pkl --games 1000 --workers 5
```

**Deployment:** Copy a checkpoint to `deployed_leaves.pkl` in the superleaves
directory. `leave_eval.py` lazy-loads this file and uses trained values before
falling back to formula. **Gen3 is currently deployed** (1M games, TD-learning, 921K entries).

**Validation (`validate.py`):** Multi-worker head-to-head bot matches.
Uses `ProcessPoolExecutor` with `cpu_count - 3` default workers (same as
trainer). Progress printed every 50 games with win/loss, spread, games/sec,
and ETA. C-accelerated move finder used when available. 1000 games completes
in ~80s with 5 workers (vs ~22min single-threaded).

Training runs in background and does not interfere with game play.
Uses `Board()` directly (no `Game` class), so it does not trigger game
library auto-save. Use `--workers N` to control CPU usage.

**Graceful pause/resume for game analysis** (`analysis_lock.py`):
The trainer and game analysis now coordinate CPU usage via a shared lock file.
When you call `game.analyze()`, it creates `.analysis_lock` which signals the
trainer to pause at the next checkpoint. The trainer resumes when analysis
completes and the lock is cleared. Lock has a 5-minute timeout for crash
recovery (stale lock auto-clears on trainer startup). This prevents the
11-12 thread thrashing that occurs when both run simultaneously.

**Recalibration handler:** To recalibrate MC speed (recommended if throughput drops):
1. Create signal file: `touch crossplay/.recalibrate_request`
2. Trainer checks for flag at each batch boundary and exits gracefully
3. Run calibration: `python -m crossplay.mc_calibrate calibrate --force`
4. Resume training: `python -m crossplay.superleaves.trainer --resume --generation N --games NNNNNN`

This allows MC speed tuning without killing workers mid-batch.

**Training data in Git (LFS):** Two pkl files are tracked via Git LFS and
pushed to GitHub so the engine works immediately after cloning:
- `deployed_leaves.pkl` -- the live leave table used by the engine (gen3)
- `gen3_1000000.pkl` -- final gen3 output (1M games, TD-learning)

All other pkl files (intermediate checkpoints, worker tables) are gitignored.
After training a new generation, update deployed_leaves.pkl and commit:
```bash
cp superleaves/gen3_NNNNNN.pkl superleaves/deployed_leaves.pkl
git add crossplay/superleaves/deployed_leaves.pkl
git commit -m "Deploy gen3 SuperLeaves"
git push
```

**Cleanup after training:** Intermediate checkpoints accumulate at ~21 MB
each during training. Delete all except the final checkpoint after training
completes:
```bash
# Keep only final checkpoint, delete intermediates
ls superleaves/gen3_*.pkl | grep -v gen3_FINAL | xargs rm
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
