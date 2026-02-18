# Crossplay V17.0

A Crossplay (Scrabble variant) engine with Monte Carlo AI evaluation,
trained SuperLeaves, and near-endgame hybrid evaluation.
See `VERSIONING.md` for version numbering rules.

## What's New in V17.0

- **Near-endgame hybrid evaluator** (bag 1-7): exhaustive 3-ply for bag-emptying
  moves, 1-ply equity for non-emptying moves
- **SuperLeaves gen1 deployed** (350K games, 921K leave patterns)
- **Gen2 training** with equity-based signal, outcome weighting, cross-generation bootstrapping
- **Parallel self-play training** with checkpointing and live status

## Quick Start

```bash
python run.py
```

On first run, the GADDAG (move generation data structure) will be built from
the dictionary. This takes ~48 seconds and is cached to disk for future runs
(subsequent loads take <0.1s).

## Requirements

- Python 3.12+ (x86_64 Linux for C acceleration)
- No pip dependencies required for gameplay

### Optional (for rebuilding C extension)

```bash
pip install cython
python3 setup_accel.py build_ext --inplace
```

## Files

### Core engine
| File | Purpose |
|------|---------|
| `game_manager.py` | Main entry point, game loop, AI orchestration |
| `mc_eval.py` | Monte Carlo 2-ply evaluation with fast path |
| `move_finder_c.py` | C-accelerated move finder wrapper |
| `move_finder_gaddag.py` | Python GADDAG move finder (fallback) |
| `move_finder_opt.py` | Optimized Python move finder |
| `gaddag_accel.so` | Compiled C extension for GADDAG traversal |
| `gaddag_accel.pyx` | Cython source (rebuild with setup_accel.py) |
| `setup_accel.py` | Build script for C extension |

### Game logic
| File | Purpose |
|------|---------|
| `board.py` | Board representation and tile placement |
| `scoring.py` | Move scoring with bonuses and crosswords |
| `config.py` | Tile values, distribution, bonus squares |
| `dictionary.py` | Word validation and lookup |
| `play_game.py` | Game state management |
| `tile_tracker.py` | Tile bag and tracking |

### AI components
| File | Purpose |
|------|---------|
| `leave_eval.py` | Leave (remaining rack) evaluation |
| `real_risk.py` | Opponent threat analysis |
| `lookahead.py` | 1-ply lookahead evaluation |
| `lookahead_3ply.py` | 3-ply exhaustive endgame + near-endgame hybrid evaluator |
| `parallel_eval.py` | Multi-worker parallel evaluation |
| `opening_heuristics.py` | Opening move strategy |
| `power_tiles.py` | High-value tile tracking |
| `blocked_cache.py` | Blocked square caching for risk |
| `exchange_eval.py` | Tile exchange evaluation |

### Data structures
| File | Purpose |
|------|---------|
| `gaddag.py` | GADDAG trie + auto-build/cache logic |
| `gaddag_compact.py` | Compact binary GADDAG format |
| `pattern_index.py` | Dictionary pattern matching |

### Data files
| File | Size | Purpose |
|------|------|---------|
| `crossplay_dict.pkl` | 2.3MB | Word dictionary (196K words) |
| `leave_bingo_prod.pkl` | 4.8MB | Bingo probability database |
| `pattern_index.pkl` | 11MB | Pattern matching index |
| `gaddag_compact.bin` | 28MB | Auto-generated on first run |

### Training data (Git LFS)
| File | Size | Purpose |
|------|------|---------|
| `superleaves/deployed_leaves.pkl` | 21MB | Live leave table (921K patterns) |
| `superleaves/gen1_350000.pkl` | 21MB | Gen1 final checkpoint (gen2 seed) |

These files are tracked via Git LFS and downloaded automatically on clone.
Intermediate training checkpoints are gitignored.

### Documentation
| File | Purpose |
|------|---------|
| `timing_results.md` | Performance benchmarks and N×K calibration |
| `game_records.md` | Complete game records with analysis |

## Architecture

### Move generation
GADDAG traversal via compiled C extension → Python scoring wrapper.
Falls back to pure Python if .so unavailable.

**Windows / Claude Code note:** On a native Win11 host, the Cython C extension
is *not* a speedup for MC. The Python `GADDAGMoveFinder` path achieves ~2.6M
sims/s (8 workers), while the C extension drops MC to ~3K sims/s due to
Python↔C call overhead in the tight simulation loop. The `.pyd` is disabled;
the engine runs pure Python on Windows.

### AI evaluation pipeline (per turn, 30s budget)
1. **Risk analysis** (3s): Full threat analysis on top candidates, prelim_equity fallback
2. **Monte Carlo 2-ply** (21-27s): Adaptive N×K — N based on candidate equity spread, K fills remaining budget
3. **3-ply endgame** (0-6s, bag ≤ 12): Exhaustive lookahead with adaptive timing

### MC auto-calibration
On first run, a 3-second benchmark measures MC throughput on the current
hardware and caches results for 24 hours. N×K parameters are computed
dynamically each turn based on calibrated throughput and candidate quality
spread, replacing the previous hardcoded phase-based table.

### Opponent blank detection
When recording opponent moves, if the reported score doesn't match the
calculated score, the engine automatically tries every possible blank
assignment to find which tile(s) must be blank:
- **Unique match**: auto-records the blank (e.g. "AUTO-DETECTED BLANK: Z")
- **Ambiguous**: lists all possibilities, asks user to confirm with `blanks=['X']`
- **No match**: flags the mismatch for manual review

### MC fast path
Workers use `_mc_find_best_score()` with:
- Cached `bytes(gdata)` (computed once per worker, not per sim)
- `SetDict` with `__contains__` (bypasses method dispatch)
- Inline scoring (no dict construction, tracks only best score)

## Rebuilding the C Extension

The `gaddag_accel.so` is pre-built for CPython 3.12 on x86_64 Linux.
To rebuild for a different platform:

```bash
pip install cython
python3 setup_accel.py build_ext --inplace
```

The .pyx source implements the same GADDAG traversal algorithm as
`move_finder_opt.py` but with Cython-accelerated node lookup.

**Important:** On native Win11 (Claude Code), the compiled `.pyd` *degrades*
MC performance (~3K vs ~2.6M sims/s) due to per-call Python↔C overhead in
the tight MC loop. The `.pyd` has been disabled. Only rebuild if you are on
Linux or have confirmed it helps on your platform.
