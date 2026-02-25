"""Tests for scoring.py -- word scoring, crosswords, move scoring."""

import pytest
from crossplay.board import Board
from crossplay.scoring import (
    get_tile_value, calculate_word_score, find_crosswords,
    calculate_move_score,
)
from crossplay.config import TILE_VALUES, BINGO_BONUS, BONUS_SQUARES


class TestTileValues:
    def test_known_values(self):
        assert get_tile_value('A') == 1
        assert get_tile_value('Z') == 10
        assert get_tile_value('E') == 1
        assert get_tile_value('Q') == 10
        assert get_tile_value('K') == 6

    def test_case_insensitive(self):
        assert get_tile_value('a') == get_tile_value('A')
        assert get_tile_value('z') == get_tile_value('Z')

    def test_unknown_returns_zero(self):
        assert get_tile_value('1') == 0
        assert get_tile_value('?') == 0


class TestWordScore:
    def test_hike_at_r8c6(self):
        """HIKE at R8C6 H: H at 2L -> H(3)*2=6, I(1), K(6), E(1) = 14."""
        board = Board()
        assert BONUS_SQUARES.get((8, 6)) == '2L'
        score = calculate_word_score(board, "HIKE", 8, 6, True)
        assert score == 14  # H(3*2) + I(1) + K(6) + E(1)

    def test_word_on_triple_letter(self):
        board = Board()
        assert BONUS_SQUARES.get((1, 1)) == '3L'
        # AB at R1C1 H: A at 3L -> A(1)*3=3, B(4) = 7
        score = calculate_word_score(board, "AB", 1, 1, True)
        assert score == 7

    def test_word_on_triple_word(self):
        board = Board()
        assert BONUS_SQUARES.get((1, 4)) == '3W'
        # AT at R1C4 H: 3W on A -> (A(1)+T(1))*3 = 6
        score = calculate_word_score(board, "AT", 1, 4, True)
        assert score == 6

    def test_word_on_double_letter(self):
        board = Board()
        assert BONUS_SQUARES.get((1, 8)) == '2L'
        score = calculate_word_score(board, "A", 1, 8, True)
        assert score == 2  # A(1)*2

    def test_blank_scores_zero(self):
        board = Board()
        # HIKE at R8C6 but K is blank (index 2): H(3*2)+I(1)+K(0)+E(1) = 8
        score = calculate_word_score(board, "HIKE", 8, 6, True,
                                     blanks_used=[2])
        assert score == 8

    def test_existing_tiles_no_bonus(self):
        """Bonuses only apply to newly placed tiles."""
        board = Board()
        board.set_tile(1, 1, 'A')  # Place A at 3L square
        # Score word: A(existing, no bonus) + B(new, no bonus at (1,2))
        score = calculate_word_score(board, "AB", 1, 1, True,
                                     new_tile_positions=[(1, 2)])
        assert score == 5  # A(1) + B(4)

    def test_no_bonus_at_center(self):
        """Center square has no bonus in Crossplay."""
        board = Board()
        assert BONUS_SQUARES.get((8, 8)) is None
        score = calculate_word_score(board, "A", 8, 8, True)
        assert score == 1  # Just A, no multiplier


class TestFindCrosswords:
    def test_no_crosswords_on_empty_board(self):
        board = Board()
        new_pos = [(8, 6), (8, 7), (8, 8), (8, 9)]
        cw = find_crosswords(board, "HIKE", 8, 6, True, new_pos)
        assert cw == []

    def test_crossword_with_adjacent(self):
        board = Board()
        board.place_word("HIKE", 8, 6, True)
        # Place O at (9,6) -> forms vertical crossword HO (H above, O new)
        new_pos = [(9, 6)]
        cw = find_crosswords(board, "O", 9, 6, True, new_pos)
        assert len(cw) == 1
        assert cw[0]['word'] == 'HO'


class TestCalculateMoveScore:
    def test_hike_move_score(self):
        board = Board()
        score, blanks = calculate_move_score(
            board, "HIKE", 8, 6, True, board_blanks=[]
        )
        assert score == 14  # H at 2L
        assert blanks == []

    def test_bingo_bonus_applied(self):
        """7 new tiles gets 40 bonus."""
        board = Board()
        # R1C1 has 3L and R1C4 has 3W -- compute expected accordingly
        # A(1,1)=3L->1*3=3, B(4), C(3), D(1,4)=3W->D(2), E(1), F(4), G(4)
        # word_mult=3 from 3W: (3+4+3+2+1+4+4)*3 = 21*3 = 63 + 40 bingo = 103
        score, _ = calculate_move_score(
            board, "ABCDEFG", 1, 1, True, board_blanks=[]
        )
        assert score == 103
