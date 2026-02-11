"""
CROSSPLAY V13.1 "GitHub Ready" - C-accelerated MC with native data structures.

V13.1 changes (from v12.1):
    - Generic saved game registry (no hardcoded game names)
    - Post-validation layer for C move finder cross-words
    - Bug fixes: top_n display default, float format in blocking analysis
    - GitHub-ready: .gitignore, CLAUDE.md, setup guide

V12.1 highlights:
    - C word buffer: char[15] replaces Python list append/pop in traversal
    - C rack array: int[27] replaces Python dict for rack counts
    - C blanks buffer: int[7] replaces Python list allocation
    - Cross-check bitmask: uint32 replaces Python set membership
    - Adaptive N×K: auto-calibrates throughput, N based on equity spread
    - 1.85x MC throughput vs v11.2 (715 sims/sec 4-worker, dense board)
    - Auto-build GADDAG: first run builds + caches (~48s), then 0.08s loads
    - Bingo probability database: precomputed P(bingo) for all 1-6 tile leaves
    - MC-integrated exchange evaluation
    - Opponent blank auto-detection on score mismatch
    - 3-ply exhaustive endgame (bag ≤ 12)
    - Parallel 2-ply lookahead via ProcessPoolExecutor

Entry point: game_manager.GameManager
"""

__version__ = "13.1.0"
__author__ = "Claude"

# Import main classes for convenience
from .board import Board
from .config import TILE_VALUES, TILE_DISTRIBUTION, VALID_TWO_LETTER
from .tile_tracker import TileTracker
from .gaddag import GADDAG, get_gaddag
from .move_finder_gaddag import GADDAGMoveFinder, find_moves_gaddag
