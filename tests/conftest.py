"""Shared fixtures for crossplay tests."""

import pytest
from datetime import datetime
from crossplay.board import Board
from crossplay.config import TILE_DISTRIBUTION, TILE_VALUES, BOARD_SIZE


@pytest.fixture
def board():
    """Empty 15x15 board."""
    return Board()


@pytest.fixture
def board_with_hike(board):
    """Board with HIKE placed horizontally at R8 C6.
    H at (8,6)=2L, I(8,7), K(8,8), E(8,9). Score = H*2+I+K+E = 6+1+6+1 = 14.
    """
    board.place_word("HIKE", 8, 6, True)
    return board


@pytest.fixture
def board_crossword(board):
    """Board with HIKE (H) and FIT (V) crossing at shared I at (8,7)."""
    board.place_word("HIKE", 8, 6, True)
    board.place_word("FIT", 7, 7, False)  # F(7,7) I(8,7-shared) T(9,7)
    return board


def make_timestamp():
    return datetime.now().isoformat()
