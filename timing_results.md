# Crossplay Performance & Optimization Results
# Updated: 2026-02-10 (v12.1 "C Buffers")
# Environment: 4 CPU cores, 4 parallel MC workers

## Current throughput (v12.1)

### Move generation (100 racks, dense board)

| Version | ms/rack | vs v9 baseline |
|---------|---------|----------------|
| v12.1 C Buffers | **8.7ms** | 1.7x faster |
| v11.2 Cdef Fast | 11.3ms | 1.3x faster |
| v9 orphaned .so | 14.1ms | baseline |
| Pure Python | ~60ms | 4x slower |

### MC simulation throughput (single worker)

| Board density    | sims/s (1 worker) | sims/s (4 workers) |
|------------------|-------------------|---------------------|
| Sparse (opening) | ~553              | ~2,200              |
| Dense (mid-late) | ~325              | ~1,300              |

### N×K allocation (adaptive, calibrated)

N and K are computed dynamically per turn based on:
1. **Calibrated throughput**: 3s benchmark on first run, cached 24h
2. **Candidate equity spread**: N includes all moves within 15 equity of best
3. **Game phase**: determines MC time budget (21-27s of 30s total)

Constraints: K ≥ 100 (variance floor), K ≤ 2,000 (diminishing returns),
N ≥ 15 (display minimum).

Example on reference hardware (4-worker, v12.1):

| Scenario | N | K | N×K | Reason |
|----------|---|---|-----|--------|
| Opening, wide spread | 120 | 574 | 68,880 | 120 within 15eq |
| Opening, tight spread | 15 | 2,000 | 30,000 | 10 within 15eq, K capped |
| Mid-game, wide spread | 120 | 370 | 44,400 | 120 within 15eq |
| Endgame, wide spread | 120 | 535 | 64,200 | 120 within 15eq |

### Budget allocation

| Phase                  | Risk | MC   | 3-ply | Total |
|------------------------|------|------|-------|-------|
| Opening (bag>70)       | 3s   | 27s  | -     | 30s   |
| Early-Mid (bag 50-70)  | 3s   | 27s  | -     | 30s   |
| Mid Game (bag 30-49)   | 3s   | 27s  | -     | 30s   |
| Late (bag 20-29)       | 3s   | 27s  | -     | 30s   |
| Late+ (bag 13-19)      | 3s   | 24s  | 3s    | 30s   |
| Pre-Endgame (bag 6-12) | 3s   | 21s  | 6s    | 30s   |
| Endgame (bag 0-5)      | 3s   | 26s  | 1s    | 30s   |

Dynamic 3-ply budget: `max(2.0, 30.0 - t_mc)` shares time with MC.

## Risk analysis

### Per-move risk cost (after optimization)

| Phase                  | Cost/move | Moves in 3s budget |
|------------------------|-----------|---------------------|
| Opening (bag>70)       | 245ms     | 12                  |
| Early-Mid (bag 50-70)  | 106ms     | 28                  |
| Mid Game (bag 30-49)   | 94ms      | 31                  |
| Late (bag 20-29)       | 25ms      | 120 (all)           |
| Late+ (bag 13-19)      | 36ms      | 83                  |
| Pre-Endgame (bag 6-12) | 37ms      | 81                  |
| Endgame (bag 0-5)      | 15ms      | 200 (all)           |

Two-tier: full risk for top candidates within 3s, prelim_equity fallback.

### Exhaustive opponent analysis (bag ≤ 2)

| Bag | Unseen | Unique racks | ms/candidate |
|-----|--------|--------------|--------------|
| 0   | ≤7     | 1            | ~1ms         |
| 1   | ≤8     | ≤8           | ~6ms         |
| 2   | ≤9     | ≤36          | ~15ms        |

## Native Win11 / Claude Code performance (2026-02-10)

**Environment:** Win11, 12-core CPU, Python 3.12, 8 MC workers

### Cython C extension: NOT a speedup on Win11

The Cython `.pyd` was compiled successfully but **degrades MC performance**:

| Config | MC sims/s | Move gen (ms/rack) |
|--------|-----------|-------------------|
| Pure Python (GADDAGMoveFinder) | **2,640,000** (8 workers) | 12.4ms |
| Cython `.pyd` enabled | **3,000** (8 workers) | 8.4ms |

The C extension is 1.5x faster for single-call move generation, but the
per-call Python↔C overhead in the tight MC simulation loop causes a ~900x
regression. The `.pyd` has been renamed to `.pyd.disabled`.

### MC worker scaling (pure Python)

| Workers | sims/s | Scaling |
|---------|--------|---------|
| 4 | 1,310,000 | baseline |
| 8 | 2,640,000 | 2.0x (linear) |
| 12 | 4,040,000 | 3.1x (linear) |

Worker cap set to 8 for balance.

### Endgame optimizations

| Bag size | Unique opp racks | Exhaustive risk time | MC skipped? |
|----------|-----------------|---------------------|-------------|
| 0 | 1 | ~1ms | Yes |
| 1 | 6 | ~6ms | Yes |
| 2 | 23 | ~15ms | Yes |
| 3 | 70 | ~900ms | Yes |
| 4 | 183 | ~2.4s | Yes |
| 5 | 428 | ~5.6s | Yes |

For bag ≤ 5, exhaustive risk + 3-ply replace MC entirely (exact results,
no sampling needed). Time budget for exhaustive risk: 90s.

## Performance history

| Version | Feature | Move gen | MC sims/s (4w) | N×K (mid) |
|---------|---------|----------|----------------|-----------|
| v9.0    | Leave Bingo | ~14ms (orphan .so) | ~650* | 35,934 |
| v10.2   | MC fast path | ~14ms | ~1,350* | 204,368* |
| v11.1   | Closures .so (rebuilt) | 14.6ms | ~480 | ~17,000 |
| v11.2   | Cdef functions | 11.3ms | ~650 | ~20,000 |
| **v12.1** | **C buffers + C rack** | **8.7ms** | **~1,300** | **~40,000** |

*v10.2 numbers used a since-deleted orphan .so with unknown source. v10.2 N×K
was miscalibrated based on that .so. v11.1+ numbers reflect rebuilt .so with
full source available.

### What each optimization did

| Optimization | Where | Impact |
|---|---|---|
| Cython .so (original, v9) | find_moves_c | 14x vs pure Python |
| MC fast path (v10.2) | mc_eval.py | Cached bytes(gdata) + SetDict + inline scoring |
| Python closures → cdef (v11.2) | gaddag_accel.pyx | 1.2x move gen (skip Python frame creation) |
| **C word buffer (v12.1)** | **gaddag_accel.pyx** | **char[15] replaces list append/pop** |
| **C rack array (v12.1)** | **gaddag_accel.pyx** | **int[27] replaces dict get/set** |
| **C blanks buffer (v12.1)** | **gaddag_accel.pyx** | **int[7] replaces list allocation** |
| **Cross-check bitmask (v12.1)** | **gaddag_accel.pyx** | **uint32 bitmask replaces set membership** |
| **Adaptive N×K calibration (v12.1)** | **mc_calibrate.py** | **Auto-benchmark, equity-spread-based N, K fills budget** |

### Profiling breakdown (v12.1, dense board)

- GADDAG traversal (C nodes): ~0.04ms/rack (<1%)
- Python overhead (cross-check, record_move, dict/is_valid): ~8.7ms/rack (99%)
- Inline scoring in MC: <0.1ms/rack (<1%)
- Cross-check calls: ~58/rack (cached, not a bottleneck)

## Roadmap

| Item | Priority | Impact | Status |
|------|----------|--------|--------|
| Grid as C array (char[225]) | Medium | ~1.2x move gen | Not started |
| Leave DB Phase 2 (scoring power) | Medium | Evaluation quality | Not started |
| DEFENSIVE strategy in ai_select_move | Low | Simulation mode only | Stub (TODO) |

## Version history

| Version | Name | Key Changes |
|---------|------|-------------|
| 9.0 | Leave Bingo | Bingo probability DB, composite leave eval, MC exchange |
| 10.1 | Cython Hot | Fixed Cython MC imports (was running Python fallback) |
| 10.2 | MC Calibrated | Fast path (2.4x MC), N×K for 30s budget |
| 11.1 | MC Calibrated Slim | Auto-build GADDAG, dead code cleanup, .pyx source rebuilt |
| 11.2 | Cdef Fast | Python closures → cdef functions (1.2x move gen) |
| **12.1** | **C Buffers** | **C word/rack/blanks buffers, cross-check bitmask (1.85x MC), adaptive N×K calibration, blank auto-detection** |
| **13.1** | **GitHub Ready** | **Generic game registry, post-validation for C move finder, bug fixes, GitHub/Claude Code ready** |
