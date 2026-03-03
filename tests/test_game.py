"""Tests for Game class -- mixin composition, DI, _get_tile_context, play_move."""

import pytest
from datetime import datetime
from crossplay.game_manager import Game, GameState
from crossplay.board import Board
from crossplay.config import TOTAL_TILES, RACK_SIZE, ENDGAME_FINAL_TURNS


def _make_state(**overrides):
    """Create a GameState with required fields, allowing overrides."""
    now = datetime.now().isoformat()
    defaults = dict(
        name="test_game",
        board_moves=[],
        blank_positions=[],
        your_score=0,
        opp_score=0,
        your_rack="ABCDEFG",
        is_your_turn=True,
        opponent_name="testbot",
        created_at=now,
        updated_at=now,
    )
    defaults.update(overrides)
    return GameState(**defaults)


@pytest.fixture
def game():
    """A fresh Game with default resources (loads GADDAG/dictionary)."""
    return Game(state=_make_state())


@pytest.fixture
def game_mid(game):
    """Game with HIKE on the board."""
    game.board.place_word("HIKE", 8, 6, True)
    game.state.your_score = 14  # H at 2L = 14
    game.state.your_rack = "RSTLNEA"
    return game


class TestGameConstruction:
    def test_default_state(self):
        g = Game()
        assert g.state is not None
        assert g.board is not None
        assert g.gaddag is not None
        assert g.dictionary is not None

    def test_custom_state(self, game):
        assert game.state.name == "test_game"
        assert game.state.opponent_name == "testbot"
        assert game.state.your_rack == "ABCDEFG"

    def test_repr(self, game):
        r = repr(game)
        assert "testbot" in r
        assert "0-0" in r
        assert "no-id" in r  # game_id not set


class TestMixinComposition:
    def test_has_analysis_methods(self, game):
        assert hasattr(game, 'analyze')
        assert hasattr(game, '_analyze_impl')

    def test_has_move_methods(self, game):
        assert hasattr(game, 'play_move')
        assert hasattr(game, 'record_opponent_move')
        assert hasattr(game, 'record_exchange')

    def test_has_core_methods(self, game):
        assert hasattr(game, 'show_board')
        assert hasattr(game, 'show_status')
        assert hasattr(game, 'is_complete')
        assert hasattr(game, '_get_tile_context')


class TestDependencyInjection:
    def test_default_resources_loaded(self, game):
        assert game.gaddag is not None
        assert game.dictionary is not None

    def test_injected_both(self):
        """DI works when both gaddag and dictionary are provided."""
        g_sentinel = object()
        d_sentinel = object()
        g = Game(gaddag=g_sentinel, dictionary=d_sentinel)
        assert g.gaddag is g_sentinel
        assert g.dictionary is d_sentinel

    def test_partial_injection_loads_defaults(self):
        """Providing only one falls back to loading both from get_resources."""
        g = Game(gaddag=object())
        # Should have loaded real resources since dictionary wasn't provided
        assert g.dictionary is not None


class TestGetTileContext:
    def test_empty_board(self, game):
        tracker, unseen, bag_size = game._get_tile_context()
        # 100 total - 7 (your rack) = 93 unseen
        assert sum(unseen.values()) == TOTAL_TILES - len(game.state.your_rack)
        # bag = unseen - 7 (opp rack)
        assert bag_size == TOTAL_TILES - len(game.state.your_rack) - RACK_SIZE

    def test_with_board_tiles(self, game_mid):
        tracker, unseen, bag_size = game_mid._get_tile_context()
        # HIKE on board (4 tiles) + RSTLNEA rack (7) = 11 accounted
        expected_unseen = TOTAL_TILES - 4 - 7
        assert sum(unseen.values()) == expected_unseen

    def test_custom_rack_override(self, game_mid):
        _, unseen1, _ = game_mid._get_tile_context()
        _, unseen2, _ = game_mid._get_tile_context(rack="XYZ")
        # Different rack sizes -> different unseen counts
        assert sum(unseen1.values()) != sum(unseen2.values())

    def test_unseen_dict_only_positive(self, game):
        _, unseen, _ = game._get_tile_context()
        for count in unseen.values():
            assert count > 0


class TestPlayMove:
    def test_play_simple_move(self, game):
        ok, score = game.play_move("HIKE", 8, 6, True)
        assert ok
        assert score == 14  # H at 2L
        assert game.state.your_score == 14
        assert game.board.get_tile(8, 6) == 'H'

    def test_play_move_toggles_turn(self, game):
        assert game.state.is_your_turn
        game.play_move("HIKE", 8, 6, True)
        assert not game.state.is_your_turn

    def test_play_move_with_new_rack(self, game):
        ok, score = game.play_move("ACE", 8, 7, True, new_rack="XYZWBDF")
        assert ok
        assert game.state.your_rack == "XYZWBDF"

    def test_play_records_move_history(self, game):
        assert len(game.state.board_moves) == 0
        game.play_move("HIKE", 8, 6, True)
        assert len(game.state.board_moves) == 1
        move = game.state.board_moves[0]
        assert move['word'] == 'HIKE'
        assert move['score'] == 14
        assert move['player'] == 'me'


class TestFinalTurnsTracking:
    def test_mid_game_no_final_turns(self, game):
        assert game.state.final_turns_remaining is None

    def test_constant_is_correct(self):
        assert ENDGAME_FINAL_TURNS == 2


class TestIsComplete:
    def test_new_game_not_complete(self, game):
        assert not game.is_complete()

    def test_explicit_complete_via_notes(self, game):
        game.state.notes = "COMPLETED"
        assert game.is_complete()

    def test_final_turns_zero_is_complete(self, game):
        game.state.final_turns_remaining = 0
        assert game.is_complete()

    def test_final_turns_one_not_complete(self, game):
        game.state.final_turns_remaining = 1
        assert not game.is_complete()
