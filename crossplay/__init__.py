"""
CROSSPLAY V21 - Heuristic-validated equity formula.

V21: Strip harmful heuristics from ranking pipeline based on DadBot v5
tournament results. Equity simplified to score + leave_value (1-ply) and
mc_equity + leave_value (2-ply MC). SuperLeaves, bingo bonus, positional
adjustment, blank correction, and exchange evaluation all disabled.
Heuristics retained for informational display only.

See VERSIONING.md for version numbering rules.
See CLAUDE.md for architecture overview and coding conventions.

Entry point: game_manager.GameManager
"""

__version__ = "21.4.0"
__author__ = "Claude"

# Import main classes for convenience
from .board import Board
from .config import TILE_VALUES, TILE_DISTRIBUTION, VALID_TWO_LETTER
from .tile_tracker import TileTracker
from .gaddag import GADDAG, get_gaddag
from .move_finder_gaddag import GADDAGMoveFinder, find_moves_gaddag
