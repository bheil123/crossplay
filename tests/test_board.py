"""Tests for board.py -- Board class and tiles_used utility."""

import pytest
from crossplay.board import Board, tiles_used
from crossplay.config import BOARD_SIZE, BONUS_SQUARES


class TestBoardBasics:
    def test_new_board_is_empty(self, board):
        assert board.is_board_empty()
        assert board.count_tiles() == 0

    def test_repr_empty(self, board):
        assert repr(board) == "Board(0 tiles)"

    def test_set_and_get_tile(self, board):
        board.set_tile(1, 1, 'A')
        assert board.get_tile(1, 1) == 'A'
        assert not board.is_empty(1, 1)
        assert board.is_occupied(1, 1)

    def test_set_tile_uppercases(self, board):
        board.set_tile(5, 5, 'z')
        assert board.get_tile(5, 5) == 'Z'

    def test_clear_tile(self, board):
        board.set_tile(3, 3, 'X')
        board.set_tile(3, 3, None)
        assert board.is_empty(3, 3)

    def test_invalid_position_raises(self, board):
        with pytest.raises(ValueError):
            board.get_tile(0, 1)
        with pytest.raises(ValueError):
            board.get_tile(16, 1)
        with pytest.raises(ValueError):
            board.set_tile(1, 0, 'A')

    def test_empty_position_returns_none(self, board):
        assert board.get_tile(8, 8) is None


class TestBoardBonuses:
    def test_bonus_at_known_position(self, board):
        assert board.get_bonus(1, 4) == '3W'
        assert board.is_triple_word(1, 4)

    def test_no_bonus_at_center(self, board):
        assert board.get_bonus(8, 8) is None

    def test_double_letter_at_known_position(self, board):
        assert board.get_bonus(1, 8) == '2L'
        assert board.is_double_letter(1, 8)

    def test_no_bonus_at_random_empty(self, board):
        assert board.get_bonus(3, 3) is None


class TestWordPlacement:
    def test_place_horizontal_word(self, board):
        new = board.place_word("HIKE", 8, 6, True)
        assert len(new) == 4
        assert board.get_tile(8, 6) == 'H'
        assert board.get_tile(8, 7) == 'I'
        assert board.get_tile(8, 8) == 'K'
        assert board.get_tile(8, 9) == 'E'

    def test_place_vertical_word(self, board):
        new = board.place_word("HIKE", 6, 8, False)
        assert board.get_tile(6, 8) == 'H'
        assert board.get_tile(7, 8) == 'I'
        assert board.get_tile(8, 8) == 'K'
        assert board.get_tile(9, 8) == 'E'

    def test_place_word_counts_tiles(self, board):
        board.place_word("HIKE", 8, 6, True)
        assert board.count_tiles() == 4
        assert repr(board) == "Board(4 tiles)"

    def test_shared_letter_not_double_counted(self, board_crossword):
        # HIKE(H) + FIT(V) share I at (8,7) -> 4 + 3 - 1 = 6 tiles
        assert board_crossword.count_tiles() == 6

    def test_conflict_raises(self, board_with_hike):
        with pytest.raises(ValueError, match="Conflict"):
            board_with_hike.place_word("HAZE", 8, 6, True)  # H ok, A != I

    def test_word_off_board_raises(self, board):
        with pytest.raises(ValueError, match="off board"):
            board.place_word("LONGWORD", 8, 12, True)  # extends past col 15

    def test_place_word_returns_new_positions(self, board_with_hike):
        # Place FIT vertically sharing I at (8,7)
        new = board_with_hike.place_word("FIT", 7, 7, False)
        # I at (8,7) already exists, so only F(7,7) and T(9,7) are new
        assert (7, 7) in new
        assert (9, 7) in new
        assert (8, 7) not in new
        assert len(new) == 2


class TestBoardQueries:
    def test_has_adjacent_tile(self, board_with_hike):
        # Above H at (8,6) -> (7,6) has adjacent
        assert board_with_hike.has_adjacent_tile(7, 6)
        # Far away has no adjacent
        assert not board_with_hike.has_adjacent_tile(1, 1)

    def test_get_all_tiles(self, board_with_hike):
        tiles = board_with_hike.get_all_tiles()
        assert len(tiles) == 4
        letters = {t[2] for t in tiles}
        assert letters == {'H', 'I', 'K', 'E'}

    def test_get_word_at_horizontal(self, board_with_hike):
        word, sr, sc = board_with_hike.get_word_at(8, 7, True)
        assert word == "HIKE"
        assert sr == 8
        assert sc == 6

    def test_get_word_at_empty(self, board):
        word, _, _ = board.get_word_at(5, 5, True)
        assert word == ''


class TestTilesUsed:
    def test_all_new_tiles(self, board):
        used = tiles_used(board, "HIKE", 8, 6, True)
        assert used == ['H', 'I', 'K', 'E']

    def test_shared_tile_not_counted(self, board_with_hike):
        # Place FIT vertically at (7,7) V -- I at (8,7) is already on board
        used = tiles_used(board_with_hike, "FIT", 7, 7, False)
        assert used == ['F', 'T']  # I is shared, not from rack

    def test_extending_word(self, board_with_hike):
        # Extend HIKE with S at (8,10) -> HIKES
        used = tiles_used(board_with_hike, "HIKES", 8, 6, True)
        assert used == ['S']  # Only S is new
