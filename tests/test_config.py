"""Tests for config.py -- tile distribution, bonus squares, constants."""

from crossplay.config import (
    TILE_VALUES, TILE_DISTRIBUTION, TOTAL_TILES, BOARD_SIZE,
    BONUS_SQUARES, TRIPLE_WORD_SQUARES, DOUBLE_WORD_SQUARES,
    TRIPLE_LETTER_SQUARES, DOUBLE_LETTER_SQUARES,
    VALID_TWO_LETTER, VALID_FRONT_HOOKS, VALID_BACK_HOOKS,
    BINGO_BONUS, RACK_SIZE, CENTER_ROW, CENTER_COL,
)


class TestTileDistribution:
    def test_total_tiles_is_100(self):
        assert TOTAL_TILES == 100

    def test_distribution_sums_to_total(self):
        assert sum(TILE_DISTRIBUTION.values()) == TOTAL_TILES

    def test_three_blanks(self):
        """Crossplay has 3 blanks, not 2 like Scrabble."""
        assert TILE_DISTRIBUTION['?'] == 3

    def test_all_letters_have_values(self):
        for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            assert ch in TILE_VALUES
            assert isinstance(TILE_VALUES[ch], int)
            assert TILE_VALUES[ch] > 0

    def test_blank_value_is_zero(self):
        assert TILE_VALUES['?'] == 0

    def test_high_value_tiles(self):
        assert TILE_VALUES['J'] == 10
        assert TILE_VALUES['Q'] == 10
        assert TILE_VALUES['Z'] == 10
        assert TILE_VALUES['X'] == 8

    def test_all_letters_have_distribution(self):
        for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ?':
            assert ch in TILE_DISTRIBUTION
            assert TILE_DISTRIBUTION[ch] >= 1


class TestBonusSquares:
    def test_no_center_bonus(self):
        """Crossplay has no center star bonus."""
        assert (CENTER_ROW, CENTER_COL) not in BONUS_SQUARES

    def test_bonus_square_symmetry(self):
        """All bonus squares should be 4-way symmetric."""
        center = (BOARD_SIZE + 1) / 2  # 8.0
        for (r, c), bonus in BONUS_SQUARES.items():
            # Reflect across both axes
            mr, mc = int(2 * center - r), int(2 * center - c)
            assert (mr, mc) in BONUS_SQUARES, f"Missing mirror of ({r},{c})"
            assert BONUS_SQUARES[(mr, mc)] == bonus

    def test_triple_word_count(self):
        assert len(TRIPLE_WORD_SQUARES) == 8

    def test_double_word_count(self):
        assert len(DOUBLE_WORD_SQUARES) == 8

    def test_triple_letter_count(self):
        assert len(TRIPLE_LETTER_SQUARES) == 20

    def test_double_letter_count(self):
        assert len(DOUBLE_LETTER_SQUARES) == 20

    def test_all_squares_1_indexed(self):
        for r, c in BONUS_SQUARES:
            assert 1 <= r <= BOARD_SIZE
            assert 1 <= c <= BOARD_SIZE


class TestBoardConstants:
    def test_board_size(self):
        assert BOARD_SIZE == 15

    def test_center(self):
        assert CENTER_ROW == 8
        assert CENTER_COL == 8

    def test_bingo_bonus(self):
        """Crossplay uses 40, not Scrabble's 50."""
        assert BINGO_BONUS == 40

    def test_rack_size(self):
        assert RACK_SIZE == 7


class TestTwoLetterWords:
    def test_common_two_letter_words(self):
        for word in ['AA', 'AB', 'AD', 'IS', 'IT', 'TO', 'AT', 'QI', 'ZA']:
            assert word in VALID_TWO_LETTER

    def test_no_invalid_entries(self):
        for word in VALID_TWO_LETTER:
            assert len(word) == 2
            assert word.isupper()

    def test_hooks_populated(self):
        # 'A' should have front hooks (letters before A forming valid 2-letter words)
        assert len(VALID_FRONT_HOOKS) > 0
        assert len(VALID_BACK_HOOKS) > 0
        # QI means Q can come before I
        assert 'Q' in VALID_FRONT_HOOKS.get('I', [])
