"""
CROSSPLAY V17 - C-accelerated MC engine with trained SuperLeaves and near-endgame evaluation.

See VERSIONING.md for version numbering rules.
See CLAUDE.md for architecture overview and coding conventions.

Entry point: game_manager.GameManager
"""

__version__ = "17.1.0"
__author__ = "Claude"

# Import main classes for convenience
from .board import Board
from .config import TILE_VALUES, TILE_DISTRIBUTION, VALID_TWO_LETTER
from .tile_tracker import TileTracker
from .gaddag import GADDAG, get_gaddag
from .move_finder_gaddag import GADDAGMoveFinder, find_moves_gaddag
