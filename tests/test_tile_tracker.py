"""Tests for tile_tracker.py -- TileTracker bag and unseen tracking."""

import pytest
from crossplay.tile_tracker import TileTracker
from crossplay.board import Board
from crossplay.config import TILE_DISTRIBUTION, TOTAL_TILES, RACK_SIZE


class TestTileTrackerInit:
    def test_initial_totals(self):
        t = TileTracker()
        total = sum(t.get_remaining(ch) for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ?')
        assert total == TOTAL_TILES

    def test_initial_remaining_matches_distribution(self):
        t = TileTracker()
        for ch, count in TILE_DISTRIBUTION.items():
            assert t.get_remaining(ch) == count


class TestSyncWithBoard:
    def test_empty_board_no_rack(self):
        t = TileTracker()
        board = Board()
        t.sync_with_board(board, your_rack="", blanks_in_rack=0,
                          blank_positions=[])
        # Nothing played, all tiles unseen
        assert t.get_unseen_count() == TOTAL_TILES
        assert t.get_bag_count() == TOTAL_TILES - RACK_SIZE  # opp has 7

    def test_board_with_word(self):
        t = TileTracker()
        board = Board()
        board.place_word("HIKE", 8, 6, True)
        t.sync_with_board(board, your_rack="ABCDEFG", blanks_in_rack=0,
                          blank_positions=[])
        # 4 on board + 7 in rack = 11 accounted for
        # Unseen = 100 - 4 - 7 = 89
        assert t.get_unseen_count() == TOTAL_TILES - 4 - 7

    def test_blank_on_board(self):
        t = TileTracker()
        board = Board()
        board.place_word("HIKE", 8, 6, True)
        # Suppose I at (8,7) is a blank
        t.sync_with_board(board, your_rack="ABCDEFG", blanks_in_rack=0,
                          blank_positions=[(8, 7, 'I')])
        # Blank used for I: remaining blanks should be 2 (started with 3)
        assert t.get_remaining('?') == 2

    def test_blank_in_rack(self):
        t = TileTracker()
        board = Board()
        # Note: blank in rack is tracked via blanks_in_rack param,
        # NOT via '?' in the your_rack string (which would double-count).
        # your_rack should contain letter tiles only.
        t.sync_with_board(board, your_rack="ABCDEG", blanks_in_rack=1,
                          blank_positions=[])
        # 1 blank in rack -> 3 - 1 = 2 remaining
        assert t.get_remaining('?') == 2

    def test_bag_count_with_rack(self):
        t = TileTracker()
        board = Board()
        board.place_word("HIKE", 8, 6, True)
        t.sync_with_board(board, your_rack="ABCDEFG", blanks_in_rack=0,
                          blank_positions=[])
        # bag = unseen - opp_rack(7)
        unseen = t.get_unseen_count()
        bag = t.get_bag_count()
        assert bag == unseen - RACK_SIZE


class TestGetTileContext:
    """Test Game._get_tile_context() via its components."""

    def test_returns_consistent_triple(self):
        """Simulate _get_tile_context pattern."""
        board = Board()
        board.place_word("HIKE", 8, 6, True)
        rack = "ABCDEFG"

        tracker = TileTracker()
        tracker.sync_with_board(board, your_rack=rack,
                                blanks_in_rack=rack.count('?'),
                                blank_positions=[])
        unseen = {}
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ?':
            remaining = tracker.get_remaining(letter)
            if remaining > 0:
                unseen[letter] = remaining
        bag_size = tracker.get_bag_count()

        assert sum(unseen.values()) == tracker.get_unseen_count()
        assert bag_size >= 0
        assert bag_size == tracker.get_unseen_count() - RACK_SIZE
