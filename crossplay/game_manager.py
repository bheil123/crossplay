#!/usr/bin/env python3
"""
CROSSPLAY V16.0 - Game Manager

Highlights:
  - Cython-accelerated GADDAG move generation (gaddag_accel.so)
  - MC 2-ply with fast path: cached bytes + SetDict + inline scoring
  - Tiered N×K scales by game phase (10K-22K sims, 30s budget)
  - 3-ply exhaustive endgame (bag <= 12, adaptive timing)
  - Bingo probability DB for composite leave evaluation
  - MC-integrated exchange evaluation
  - Blank correction factor (Crossplay 3-blank support)
  - Parallel 2-ply lookahead (4 workers)
  - Risk analysis with two-tier approach + exhaustive endgame (bag <= 2)

Modes:
  1. Play against Claude AI
  2. AI-assisted games against human opponents (up to 4 concurrent)
  3. AI vs AI simulation with different strategies

Usage:
    python3 game_manager.py
"""

import sys
import os
import json
import random
import time
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from .board import Board, tiles_used as _board_tiles_used
from .move_finder_gaddag import GADDAGMoveFinder
from .gaddag import get_gaddag
from .dictionary import Dictionary
from .config import TILE_DISTRIBUTION, TILE_VALUES, BONUS_SQUARES, RACK_SIZE, BOARD_SIZE, CENTER_ROW, CENTER_COL
from .tile_tracker import TileTracker
from .log import get_logger
from . import __version__
from .game_analysis import GameAnalysisMixin
from .game_moves import GameMovesMixin

logger = get_logger(__name__)

# =============================================================================
# GAME STATE
# =============================================================================

SAVE_DIR = os.path.join(os.path.expanduser('~'), 'crossplay_saves')
GADDAG = None  # Loaded on demand
DICTIONARY = None


def get_resources():
    """Load GADDAG and dictionary (cached)."""
    global GADDAG, DICTIONARY
    if GADDAG is None:
        logger.info("Loading GADDAG...")
        GADDAG = get_gaddag()
        import os as _os
        _dict_path = _os.path.join(_os.path.dirname(__file__), 'crossplay_dict.pkl')
        DICTIONARY = Dictionary.load(_dict_path)
    return GADDAG, DICTIONARY


class Strategy(Enum):
    """AI strategy types."""
    HIGHEST_SCORE = "highest"      # Always play highest scoring move
    BALANCED = "balanced"          # Balance score vs leave quality
    DUMP_TILES = "dump"            # Prefer using more tiles
    HIGH_VALUE = "high_value"      # Target high-value letters (Q, Z, X, J)
    DEFENSIVE = "defensive"        # Minimize opponent opportunities


@dataclass
class GameState:
    """Serializable game state."""
    name: str
    board_moves: List[Tuple]       # [(word, row, col, horizontal), ...]
    blank_positions: List[Tuple]   # [(row, col, letter), ...]
    your_score: int
    opp_score: int
    your_rack: str
    is_your_turn: bool
    opponent_name: str
    created_at: str
    updated_at: str
    notes: str = ""
    final_turns_remaining: Optional[int] = None

    def __repr__(self) -> str:
        spread = self.your_score - self.opp_score
        turn = "yours" if self.is_your_turn else self.opponent_name
        return (f"GameState({self.name}, {self.your_score}-{self.opp_score} "
                f"({'+' if spread >= 0 else ''}{spread}), "
                f"{len(self.board_moves)} moves, turn={turn})")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        # Normalize board_moves: support both old tuple format and enriched dicts
        raw_moves = d.get('board_moves', [])
        normalized = []
        for m in raw_moves:
            if isinstance(m, dict):
                # Already enriched — keep as-is
                normalized.append(m)
            elif isinstance(m, (list, tuple)):
                if len(m) == 4:
                    # Old format: (word, row, col, horizontal)
                    normalized.append({
                        'word': m[0], 'row': m[1], 'col': m[2],
                        'dir': 'H' if m[3] else 'V',
                        'player': None, 'score': None, 'bag': None,
                        'blanks': [], 'cumulative': None, 'note': '',
                    })
                else:
                    # Unknown tuple length — keep raw for debugging
                    normalized.append(m)
            else:
                normalized.append(m)
        d = dict(d)  # Don't mutate caller's dict
        d['board_moves'] = normalized
        # Strip legacy 'bag' field (bag is now reconstructed from board state)
        d.pop('bag', None)
        return cls(**d)


class Game(GameAnalysisMixin, GameMovesMixin):
    """A single game instance."""
    
    def __init__(self, state: Optional[GameState] = None,
                 gaddag=None, dictionary=None):
        if gaddag is not None and dictionary is not None:
            self.gaddag, self.dictionary = gaddag, dictionary
        else:
            self.gaddag, self.dictionary = get_resources()
        self.auto_save = False      # Enable auto-save after each move
        self.save_filename = None   # Consistent filename for auto-save
        self.game_id = None         # Library game ID (e.g., 'canjam_002')
        self.last_analysis = None   # MC top 3 from most recent analyze()
        
        # Auto-calibrate MC throughput (cached, runs 3s benchmark once)
        from .mc_calibrate import calibrate
        calibrate(quiet=False)
        
        if state:
            self.state = state
            self.board = Board()
            for move in state.board_moves:
                if isinstance(move, dict):
                    if move.get('is_exchange'):
                        continue  # Skip exchange moves (no board placement)
                    word = move['word']
                    row, col = move['row'], move['col']
                    horiz = move.get('dir', 'H') == 'H' if 'dir' in move else move.get('horizontal', True)
                else:
                    word, row, col, horiz = move[0], move[1], move[2], move[3]
                self.board.place_word(word, row, col, horiz)
            # Reconstruct bag from board state (never trust stored bag)
            self.bag = self._reconstruct_bag()
            # Infer final_turns_remaining for old saves missing it
            if state.final_turns_remaining is None:
                _, _, _bag = self._get_tile_context()
                if _bag == 0:
                    # Bag already empty but field not set -- assume 2 final turns left
                    # (both players get one final turn after bag empties)
                    state.final_turns_remaining = 2
        else:
            self.state = GameState(
                name="New Game",
                board_moves=[],
                blank_positions=[],
                your_score=0,
                opp_score=0,
                your_rack="",
                is_your_turn=True,
                opponent_name="Opponent",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            )
            self.board = Board()
            self.bag = self._new_bag()
    
    def _get_tile_context(self, rack: str = None):
        """Build a synced TileTracker, unseen dict, and bag count.

        This consolidates the repeated pattern of creating a TileTracker,
        syncing it with the board, and extracting unseen tiles. Call once
        at the start of any method that needs tile distribution info.

        Args:
            rack: Rack to use for tracking. If None, uses self.state.your_rack.

        Returns:
            (tracker, unseen_dict, bag_size) where:
            - tracker: synced TileTracker instance
            - unseen_dict: dict of {letter: count} for unseen tiles (A-Z + ?)
            - bag_size: tiles in bag (unseen minus opponent rack)
        """
        rack = rack or self.state.your_rack or ""
        tracker = TileTracker()
        tracker.sync_with_board(
            self.board, your_rack=rack,
            blanks_in_rack=rack.count('?'),
            blank_positions=self.state.blank_positions
        )
        unseen = {}
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ?':
            remaining = tracker.get_remaining(letter)
            if remaining > 0:
                unseen[letter] = remaining
        bag_size = tracker.get_bag_count()
        return tracker, unseen, bag_size

    def __repr__(self) -> str:
        gid = self.game_id or "no-id"
        s = self.state
        spread = s.your_score - s.opp_score
        return (f"Game({gid}, vs {s.opponent_name}, "
                f"{s.your_score}-{s.opp_score} "
                f"({'+' if spread >= 0 else ''}{spread}), "
                f"rack={s.your_rack or 'none'})")

    def _new_bag(self) -> List[str]:
        """Create a new shuffled bag."""
        bag = []
        for letter, count in TILE_DISTRIBUTION.items():
            bag.extend([letter] * count)
        random.shuffle(bag)
        return bag

    def _reconstruct_bag(self) -> List[str]:
        """Reconstruct bag from tile distribution minus board tiles minus rack.

        The bag is derived state: full distribution minus tiles on board
        (adjusted for blanks) minus your rack. The result is the unseen
        pool (bag + opponent rack), shuffled for random draws.
        """
        from collections import Counter
        remaining = Counter(TILE_DISTRIBUTION)

        # Subtract tiles on board
        for r in range(1, 16):
            for c in range(1, 16):
                tile = self.board.get_tile(r, c)
                if tile:
                    remaining[tile] -= 1

        # Adjust for blanks on board (they show as letters but are actually '?')
        for (r, c, letter) in self.state.blank_positions:
            remaining[letter] += 1   # Undo the letter subtraction
            remaining['?'] -= 1      # Subtract a blank instead

        # Subtract your rack
        for tile in (self.state.your_rack or ""):
            remaining[tile] -= 1

        # Build bag list from remaining counts
        bag = []
        for letter, count in remaining.items():
            if count > 0:
                bag.extend([letter] * count)

        random.shuffle(bag)
        return bag

    def draw_tiles(self, n: int) -> List[str]:
        """Draw n tiles from bag."""
        drawn = []
        for _ in range(min(n, len(self.bag))):
            drawn.append(self.bag.pop())
        return drawn
    
    def enable_auto_save(self, filename: str = None):
        """Enable auto-save after each move.
        
        Args:
            filename: Optional specific filename. If not provided, generates from game name.
        """
        self.auto_save = True
        if filename:
            self.save_filename = filename
        else:
            safe_name = self.state.name.replace(' ', '_').lower()
            opp_name = self.state.opponent_name.replace(' ', '_').lower()
            self.save_filename = f"{safe_name}_{opp_name}.json"
        print(f"[SAVE] Auto-save enabled: {self.save_filename}")
    
    def disable_auto_save(self):
        """Disable auto-save."""
        self.auto_save = False
        print("[SAVE] Auto-save disabled")
    
    def _auto_save(self):
        """Save game if auto-save is enabled."""
        if self.auto_save and self.game_id:
            from . import game_library as lib
            lib.save_active(self.game_id, self)
        elif self.auto_save and self.save_filename:
            self.save(self.save_filename, quiet=True)
    
    def is_complete(self) -> bool:
        """Check if this game is complete, using game-over rules.

        A game is over when:
        - Both players have taken their final turns after the bag emptied, OR
        - 6 consecutive passes (detected via notes), OR
        - Notes explicitly say COMPLETED

        Crossplay endgame: both players get one final turn after the bag
        empties. final_turns_remaining tracks this: None = mid-game,
        2 = bag just emptied (2 final turns left), 1 = one final turn
        left, 0 = game over. We must NOT archive while final turns remain.
        """
        s = self.state
        # Explicit completion marker
        if 'COMPLETED' in s.notes.upper():
            return True
        # If final_turns_remaining is actively tracking endgame, respect it
        if s.final_turns_remaining is not None:
            return s.final_turns_remaining <= 0
        # Mid-game checks (final_turns_remaining is None = bag not empty yet)
        # Compute remaining unseen tiles (bag + opp_rack) from board state
        _trk, _, _ = self._get_tile_context()
        remaining = _trk.get_unseen_count()  # bag + opp_rack
        # You went out (empty rack AND bag was already empty)
        # Note: empty rack with tiles remaining just means rack hasn't been set
        if len(s.your_rack or "") == 0 and remaining <= RACK_SIZE:
            return True
        # Opponent went out (no unaccounted tiles remain)
        if remaining == 0 and len(s.your_rack or "") > 0:
            return True
        return False

    def show_board(self):
        """Display the board with bonus squares."""
        print("\n     1  2  3  4  5  6  7  8  9 10 11 12 13 14 15")
        print("   " + "-" * 47)
        
        for row in range(1, 16):
            row_str = f"{row:2} |"
            for col in range(1, 16):
                tile = self.board.get_tile(row, col)
                if tile:
                    is_blank = any(r == row and c == col for r, c, _ in self.state.blank_positions)
                    row_str += f" {tile.lower() if is_blank else tile} "
                else:
                    row_str += " . "
            print(row_str + "|")
        print("   " + "-" * 47)
    
    def show_status(self):
        """Show game status."""
        spread = self.state.your_score - self.state.opp_score
        spread_str = f"+{spread}" if spread >= 0 else str(spread)
        if self.is_complete():
            result = "WIN" if spread > 0 else "LOSS" if spread < 0 else "TIE"
            status = f"COMPLETED - {result}"
        else:
            status = "YOUR TURN" if self.state.is_your_turn else f"{self.state.opponent_name.upper()}'S TURN"

        print(f"\n[INFO] {self.state.name} vs {self.state.opponent_name}")
        print(f"   Score: You {self.state.your_score} - {self.state.opponent_name} {self.state.opp_score} ({spread_str})")
        # Compute bag count and unseen tiles from board state
        tracker, unseen_dict, bag_tiles = self._get_tile_context()
        ftr = self.state.final_turns_remaining
        if bag_tiles == 0 and ftr is not None:
            if ftr == 2 and self.state.is_your_turn:
                status_extra = f"   Bag: 0 tiles | {status} (2 final turns -- you play, then opp responds)"
            elif ftr == 2 and not self.state.is_your_turn:
                status_extra = f"   Bag: 0 tiles | {status} (2 final turns -- opp plays, then you respond)"
            elif ftr == 1 and self.state.is_your_turn:
                status_extra = f"   Bag: 0 tiles | {status} (final move -- no opp response)"
            elif ftr == 1 and not self.state.is_your_turn:
                status_extra = f"   Bag: 0 tiles | {status} (final move -- you respond after)"
            elif ftr <= 0:
                status_extra = f"   Bag: 0 tiles | {status} (no turns remaining)"
            else:
                status_extra = f"   Bag: 0 tiles | {status} (final turns: {ftr})"
            print(status_extra)
        else:
            print(f"   Bag: {bag_tiles} tiles | {status}")
        if self.state.your_rack:
            rack_val = sum(TILE_VALUES.get(t, 0) for t in self.state.your_rack)
            print(f"   Your rack: [{' '.join(self.state.your_rack)}] (value: {rack_val})")
        
        # Show unseen tiles (reuse tracker from above)
        unseen_parts = []
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ?':
            count = unseen_dict.get(letter, 0)
            if count > 0:
                unseen_parts.append(f"{letter}:{count}" if count > 1 else letter)

        total_unseen = sum(unseen_dict.values())
        print(f"   Unseen ({total_unseen}): {' '.join(unseen_parts)}")

        # Show played out high-value tiles (5+ points)
        high_value_letters = {'J': 10, 'Q': 10, 'X': 8, 'Z': 10, 'K': 6, 'V': 6, 'W': 5}
        played_out = []
        for letter, value in sorted(high_value_letters.items(), key=lambda x: -x[1]):
            if unseen_dict.get(letter, 0) == 0:
                played_out.append(f"{letter}({value})")
        
        if played_out:
            print(f"   Played out: {' '.join(played_out)}")
        
        # Show power tile draw probabilities if any remain
        from .power_tiles import format_power_tile_display, get_power_tiles_in_pool
        power_tiles = get_power_tiles_in_pool(unseen_dict)
        if power_tiles:
            print(f"   {format_power_tile_display(unseen_dict, bag_tiles)}")
    
    def _tiles_used(self, move: dict, rack: str = None) -> str:
        """Determine which rack tiles a move uses."""
        return ''.join(_board_tiles_used(
            self.board, move['word'], move['row'], move['col'],
            move['direction'] == 'H'
        ))
    
    def _get_new_tile_word_indices(self, move: dict) -> List[int]:
        """Get list of word indices that correspond to new tiles placed."""
        word = move['word']
        row, col = move['row'], move['col']
        horiz = move['direction'] == 'H'
        
        indices = []
        for i, letter in enumerate(word):
            r = row if horiz else row + i
            c = col + i if horiz else col
            if not self.board.get_tile(r, c):
                indices.append(i)
        return indices
    
    def _used_idx_to_word_idx(self, move: dict, used_idx: int) -> int:
        """Map an index in 'used' tiles to the corresponding word index."""
        indices = self._get_new_tile_word_indices(move)
        if used_idx < len(indices):
            return indices[used_idx]
        return -1
    
    def _check_archive(self):
        """Auto-archive the game when it's complete. Uses game library."""
        if self.is_complete() and self.game_id:
            try:
                from . import game_library as lib
                lib.archive_completed(self.game_id, self)
            except Exception as e:
                print(f"Warning: auto-archive failed: {e}")
        elif self.is_complete():
            # Legacy fallback for games without game_id
            try:
                from .game_archive import enrich_move_history, archive_game
                has_enriched = all(isinstance(m, dict) and m.get('score') is not None
                                  for m in self.state.board_moves)
                if has_enriched:
                    archive_game(self.state, self.state.board_moves)
                else:
                    enriched = enrich_move_history(
                        self.state.board_moves,
                        self.state.blank_positions,
                        first_player='me'
                    )
                    archive_game(self.state, enriched)
            except Exception as e:
                print(f"Warning: auto-archive failed: {e}")

    def set_rack(self, rack: str):
        """Set your rack."""
        self.state.your_rack = rack.upper().replace(' ', '')
        print(f"--> Rack set to: [{' '.join(self.state.your_rack)}]")
        self._auto_save()
    
    def save(self, filename: str = None, quiet: bool = False) -> str:
        """Save game to file.
        
        Args:
            filename: Optional filename. If not provided, generates timestamped name.
            quiet: If True, don't print confirmation message.
        """
        os.makedirs(SAVE_DIR, exist_ok=True)

        self.state.updated_at = datetime.now().isoformat()
        
        if not filename:
            safe_name = self.state.name.replace(' ', '_').lower()
            filename = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(SAVE_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
        
        if not quiet:
            print(f"[SAVE] Saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'Game':
        """Load game from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        state = GameState.from_dict(data)
        return cls(state)


# =============================================================================
# AI STRATEGIES
# =============================================================================

def ai_select_move(moves: List[dict], strategy: Strategy, rack: str) -> Optional[dict]:
    """Select a move based on strategy."""
    if not moves:
        return None
    
    if strategy == Strategy.HIGHEST_SCORE:
        return moves[0]  # Already sorted by score
    
    elif strategy == Strategy.DUMP_TILES:
        # Prefer moves that use more tiles
        def tiles_used(m):
            used = 0
            word = m['word']
            # Simplified - just use word length as proxy
            return len(word)
        
        sorted_moves = sorted(moves, key=lambda m: (-tiles_used(m), -m['score']))
        return sorted_moves[0]
    
    elif strategy == Strategy.HIGH_VALUE:
        # Prefer moves that use high-value tiles (Q, Z, X, J, K, V)
        high_value_tiles = set('QZXJKV')
        
        def high_value_score(m):
            word = m['word']
            hv_count = sum(1 for c in word if c in high_value_tiles)
            return hv_count * 100 + m['score']
        
        sorted_moves = sorted(moves, key=lambda m: -high_value_score(m))
        return sorted_moves[0]
    
    elif strategy == Strategy.BALANCED:
        # Balance score with leave quality (simplified)
        def balanced_score(m):
            score = m['score']
            # Penalize leaving bad tiles (rough heuristic)
            return score
        
        return moves[0]  # For now, same as highest
    
    elif strategy == Strategy.DEFENSIVE:
        # Avoid opening bonus squares (simplified)
        return moves[0]  # TODO: implement properly
    
    return moves[0]


# =============================================================================
# SIMULATION MODE
# =============================================================================

def run_simulation(strategy1: Strategy, strategy2: Strategy, num_games: int = 10):
    """Run V7 vs V7 simulation with different strategies."""
    gaddag, dictionary = get_resources()
    
    results = {
        'player1_wins': 0,
        'player2_wins': 0,
        'ties': 0,
        'player1_total_score': 0,
        'player2_total_score': 0,
        'games': []
    }
    
    print(f"\n{'='*60}")
    print(f"SIMULATION: {strategy1.value.upper()} vs {strategy2.value.upper()}")
    print(f"Running {num_games} games...")
    print(f"{'='*60}")
    
    for game_num in range(1, num_games + 1):
        print(f"\n--- Game {game_num}/{num_games} ---")
        
        # Initialize game
        board = Board()
        bag = []
        for letter, count in TILE_DISTRIBUTION.items():
            bag.extend([letter] * count)
        random.shuffle(bag)
        
        rack1 = [bag.pop() for _ in range(7)]
        rack2 = [bag.pop() for _ in range(7)]
        score1, score2 = 0, 0
        
        turn = 1
        consecutive_passes = 0
        max_turns = 100
        final_turns = None  # Crossplay: both get 1 more turn after bag empties

        while consecutive_passes < 4 and turn < max_turns:
            # Crossplay endgame: once final countdown reaches 0, game is over
            if final_turns is not None and final_turns <= 0:
                break

            current_rack = rack1 if turn % 2 == 1 else rack2
            current_strategy = strategy1 if turn % 2 == 1 else strategy2
            player_name = f"P1({strategy1.value})" if turn % 2 == 1 else f"P2({strategy2.value})"

            rack_str = ''.join(current_rack)
            finder = GADDAGMoveFinder(board, gaddag)
            moves = finder.find_all_moves(rack_str)

            if not moves:
                consecutive_passes += 1
                if final_turns is not None:
                    final_turns -= 1
                turn += 1
                continue

            # Select move based on strategy
            move = ai_select_move(moves, current_strategy, rack_str)

            if move:
                word = move['word']
                row, col = move['row'], move['col']
                horizontal = move['direction'] == 'H'

                # Calculate tiles used
                tiles_used = []
                for i, letter in enumerate(word):
                    r = row if horizontal else row + i
                    c = col + i if horizontal else col
                    if not board.get_tile(r, c):
                        tiles_used.append(letter)

                # Place word
                board.place_word(word, row, col, horizontal)
                points = move['score']
                # Bingo bonus already included in move['score']

                # Update score
                if turn % 2 == 1:
                    score1 += points
                else:
                    score2 += points

                # Update rack
                for t in tiles_used:
                    if t in current_rack:
                        current_rack.remove(t)

                # Draw new tiles
                for _ in range(min(len(tiles_used), len(bag))):
                    current_rack.append(bag.pop())

                # Crossplay: start final-turn countdown when bag empties
                if len(bag) == 0 and final_turns is None:
                    final_turns = 2  # Both players get one more turn
                if final_turns is not None:
                    final_turns -= 1

                consecutive_passes = 0

                if turn <= 10 or turn % 10 == 0:
                    print(f"  Turn {turn}: {player_name} plays {word} for {points} pts")
            else:
                consecutive_passes += 1
                if final_turns is not None:
                    final_turns -= 1

            turn += 1

        # Crossplay final scoring: no tile penalties, no transfer bonus.
        # Final scores are simply accumulated move scores.
        
        # Record results
        results['player1_total_score'] += score1
        results['player2_total_score'] += score2
        
        if score1 > score2:
            results['player1_wins'] += 1
            winner = f"P1({strategy1.value})"
        elif score2 > score1:
            results['player2_wins'] += 1
            winner = f"P2({strategy2.value})"
        else:
            results['ties'] += 1
            winner = "TIE"
        
        results['games'].append({
            'score1': score1,
            'score2': score2,
            'turns': turn,
            'winner': winner
        })
        
        print(f"  Final: P1={score1}, P2={score2} -> {winner}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SIMULATION RESULTS")
    print(f"{'='*60}")
    print(f"Strategy 1: {strategy1.value.upper()}")
    print(f"Strategy 2: {strategy2.value.upper()}")
    print(f"\nP1 Wins: {results['player1_wins']}")
    print(f"P2 Wins: {results['player2_wins']}")
    print(f"Ties: {results['ties']}")
    print(f"\nP1 Avg Score: {results['player1_total_score'] / num_games:.1f}")
    print(f"P2 Avg Score: {results['player2_total_score'] / num_games:.1f}")
    
    return results


# =============================================================================
# GAME MANAGER
# =============================================================================

class GameManager:
    """Manages multiple concurrent games."""
    
    MAX_GAMES = 8
    
    def __init__(self):
        self.games: Dict[int, Game] = {}
        self.current_slot: int = 1
        
        # Initialize with preloaded games
        self._init_default_games()
    
    def _init_default_games(self):
        """Initialize game slots from game library.

        Loads slot assignments from games/index.json, then loads each
        game from games/active/{game_id}.json. On first run, migrates
        from factory functions automatically.
        """
        from . import game_library as lib

        index = lib.ensure_library_initialized()
        slots = index.get('slots', {})

        print("Initializing game slots...")
        for slot_str, game_id in slots.items():
            slot = int(slot_str)
            if slot > self.MAX_GAMES:
                continue
            try:
                game = lib.load_active(game_id)
                if game is None:
                    print(f"  Slot {slot}: {game_id} not found in library")
                    continue
                game.game_id = game_id
                game.auto_save = True  # Auto-save enabled for library games
                self.games[slot] = game
                s = game.state
                spread = s.your_score - s.opp_score
                if game.is_complete():
                    result = "WIN" if spread > 0 else "LOSS" if spread < 0 else "TIE"
                    status = f"COMPLETED - {result}"
                else:
                    status = "Your turn" if s.is_your_turn else f"{s.opponent_name}'s turn"
                print(f"  Slot {slot}: {s.name} vs {s.opponent_name} "
                      f"({s.your_score}-{s.opp_score}, {'+' if spread >= 0 else ''}{spread}) "
                      f"| {status} [{game_id}]")
            except Exception as e:
                logger.error("Slot %d: failed to load %s: %s", slot, game_id, e)
                print(f"  Slot {slot}: ! Failed to load {game_id}: {e}")
                self.games[slot] = None

        # Detect orphaned games (active files not assigned to any slot)
        assigned_ids = {v for v in slots.values() if v}
        try:
            import os as _os
            active_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                       'games', 'active')
            if _os.path.isdir(active_dir):
                on_disk = {f.replace('.json', '') for f in _os.listdir(active_dir)
                           if f.endswith('.json')}
                orphaned = on_disk - assigned_ids
                if orphaned:
                    print(f"\n  [!] {len(orphaned)} orphaned game(s) not in any slot:")
                    for gid in sorted(orphaned):
                        g = lib.load_active(gid)
                        if g:
                            s = g.state
                            spread = s.your_score - s.opp_score
                            print(f"      {gid}: vs {s.opponent_name} "
                                  f"({s.your_score}-{s.opp_score}, "
                                  f"{'+' if spread >= 0 else ''}{spread})"
                                  f" | {len(s.board_moves)} moves")
                        else:
                            print(f"      {gid}: (failed to load)")
                    print(f"      Use 'load GAME_ID' in assisted mode to assign to a slot")
        except Exception:
            pass  # Non-critical -- don't block startup

    def reload_games(self, slot=None):
        """Reload game(s) from disk after git pull or external edits.

        Re-reads game JSON files and reconstructs in-memory Game objects,
        replacing any stale state. Use this after 'git pull' when game
        files were updated on another computer.

        Args:
            slot: If provided, reload only that slot. Otherwise reload all slots.
        """
        from . import game_library as lib

        # Re-read index (may have changed on the other computer)
        index = lib.load_index()
        slots_to_reload = {}

        if slot is not None:
            # Reload single slot
            game_id = index.get('slots', {}).get(str(slot))
            if game_id:
                slots_to_reload[slot] = game_id
            else:
                print(f"Slot {slot}: no game assigned in index")
                return
        else:
            # Reload all slots
            for slot_str, game_id in index.get('slots', {}).items():
                if game_id:
                    s = int(slot_str)
                    if s <= self.MAX_GAMES:
                        slots_to_reload[s] = game_id

        if not slots_to_reload:
            print("No games to reload.")
            return

        print(f"\n[RELOAD] Reloading {len(slots_to_reload)} game(s) from disk...")
        for s, game_id in sorted(slots_to_reload.items()):
            try:
                game = lib.load_active(game_id)
                if game is None:
                    # File might have been archived on the other computer
                    print(f"  Slot {s}: {game_id} -- file not found (archived?)")
                    self.games[s] = None
                    continue
                game.game_id = game_id
                game.auto_save = True
                self.games[s] = game
                gs = game.state
                spread = gs.your_score - gs.opp_score
                sign = '+' if spread >= 0 else ''
                moves = len(gs.board_moves)
                turn = "Your turn" if gs.is_your_turn else f"{gs.opponent_name}'s turn"
                print(f"  Slot {s}: {game_id} | {gs.your_score}-{gs.opp_score} "
                      f"({sign}{spread}) | {moves} moves | {turn}")
            except Exception as e:
                print(f"  Slot {s}: {game_id} -- reload failed: {e}")

        print("[RELOAD] Done.")

    def show_slots(self):
        """Show all game slots."""
        print(f"\n{'='*60}")
        print("GAME SLOTS")
        print(f"{'='*60}")
        
        for slot in range(1, self.MAX_GAMES + 1):
            game = self.games.get(slot)
            marker = "-> " if slot == self.current_slot else "  "
            
            if game:
                spread = game.state.your_score - game.state.opp_score
                spread_str = f"+{spread}" if spread >= 0 else str(spread)
                if game.is_complete():
                    result = "WIN" if spread > 0 else "LOSS" if spread < 0 else "TIE"
                    status = f"COMPLETED - {result}"
                else:
                    status = "Your turn" if game.state.is_your_turn else f"{game.state.opponent_name}'s turn"
                print(f"{marker}Slot {slot}: {game.state.name} vs {game.state.opponent_name}")
                print(f"         Score: {game.state.your_score}-{game.state.opp_score} ({spread_str}) | {status}")
            else:
                print(f"{marker}Slot {slot}: [Empty]")
    
    def select_slot(self, slot: int):
        """Select a game slot."""
        if 1 <= slot <= self.MAX_GAMES:
            self.current_slot = slot
            if self.games.get(slot):
                print(f"[OK] Selected Slot {slot}: {self.games[slot].state.name}")
            else:
                print(f"[OK] Selected Slot {slot} (empty)")
        else:
            print(f"[X] Invalid slot. Use 1-{self.MAX_GAMES}")
    
    def _show_all_games(self):
        """Show all active games, highlighting which slot they're in (if any)."""
        from . import game_library as lib

        index = lib.load_index()
        slot_to_id = {v: int(k) for k, v in index.get('slots', {}).items() if v}

        all_games = lib.list_active()
        if not all_games:
            print("\nNo active games.")
            return

        print(f"\n{'='*60}")
        print("ALL ACTIVE GAMES")
        print(f"{'='*60}")
        for g in all_games:
            gid = g['game_id']
            opp = g.get('opponent', '?')
            ys = g.get('your_score', 0)
            os_ = g.get('opp_score', 0)
            spread = ys - os_
            spread_str = f"+{spread}" if spread >= 0 else str(spread)
            moves = g.get('move_count', 0)
            slot = slot_to_id.get(gid)
            if slot:
                loc = f"Slot {slot}"
            else:
                loc = "[not in slot]"
            print(f"  {gid}: vs {opp} ({ys}-{os_}, {spread_str}) "
                  f"| {moves} moves | {loc}")

        orphaned = [g for g in all_games if g['game_id'] not in slot_to_id]
        if orphaned:
            print(f"\nTo load an orphaned game: load GAME_ID")

    def _load_game_into_slot(self, game_id: str, slot: int):
        """Load an active game by ID into the given slot."""
        from . import game_library as lib

        if slot < 1 or slot > self.MAX_GAMES:
            print(f"[X] Invalid slot. Use 1-{self.MAX_GAMES}")
            return

        game = lib.load_active(game_id)
        if game is None:
            print(f"[X] Game '{game_id}' not found in games/active/")
            return

        game.game_id = game_id
        game.auto_save = True
        self.games[slot] = game
        self.current_slot = slot

        # Update index
        index = lib.load_index()
        index['slots'][str(slot)] = game_id
        lib.save_index(index)

        s = game.state
        spread = s.your_score - s.opp_score
        print(f"[OK] Loaded {game_id} into Slot {slot}: vs {s.opponent_name} "
              f"({s.your_score}-{s.opp_score}, {'+' if spread >= 0 else ''}{spread})")

    def new_game(self, slot: int, opponent_name: str = "Opponent"):
        """Start a new game in a slot."""
        if slot < 1 or slot > self.MAX_GAMES:
            print(f"[X] Invalid slot. Use 1-{self.MAX_GAMES}")
            return

        # Overwrite protection: refuse to clobber an in-progress game
        existing = self.games.get(slot)
        if existing and len(existing.state.board_moves) > 0:
            s = existing.state
            gid = getattr(existing, 'game_id', '?')
            spread = s.your_score - s.opp_score
            print(f"[X] Slot {slot} has an in-progress game: {gid} vs {s.opponent_name}")
            print(f"    Score: {s.your_score}-{s.opp_score} ({'+' if spread >= 0 else ''}{spread})"
                  f" | {len(s.board_moves)} moves")
            print(f"    Use 'reset {slot}' first to clear it, or 'slot {slot}' to resume")
            return

        from . import game_library as lib

        # Allocate a game ID from the library
        index = lib.load_index()
        game_id = lib.get_game_id(opponent_name, index)

        state = GameState(
            name=f"Game {game_id}",
            board_moves=[],
            blank_positions=[],
            your_score=0,
            opp_score=0,
            your_rack="",
            is_your_turn=True,
            opponent_name=opponent_name,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        game = Game(state)
        game.game_id = game_id
        game.auto_save = True

        # Save to library
        lib.save_active(game_id, game)
        index['slots'][str(slot)] = game_id
        lib.save_index(index)

        self.games[slot] = game
        self.current_slot = slot
        print(f"[OK] New game created in Slot {slot} vs {opponent_name} [{game_id}]")
    
    def end_game(self, slot: int = None, result: str = None,
                 your_score: int = None, opp_score: int = None,
                 resignation: bool = False):
        """Complete a game: update final scores, archive, and clear slot.

        Args:
            slot: Slot number (defaults to current_slot)
            result: 'win', 'loss', or 'tie' (auto-detected from scores if omitted)
            your_score: Final score (updates game state if provided)
            opp_score: Final opponent score (updates game state if provided)
            resignation: If True, opponent resigned
        """
        if slot is None:
            slot = self.current_slot
        if slot < 1 or slot > self.MAX_GAMES:
            print(f"[X] Invalid slot. Use 1-{self.MAX_GAMES}")
            return False

        game = self.games.get(slot)
        if not game:
            print(f"[X] Slot {slot} is empty")
            return False

        game_id = game.game_id

        # Update final scores if provided
        if your_score is not None:
            game.state.your_score = your_score
        if opp_score is not None:
            game.state.opp_score = opp_score

        # Auto-detect result from scores
        spread = game.state.your_score - game.state.opp_score
        if result is None:
            if spread > 0:
                result = 'win'
            elif spread < 0:
                result = 'loss'
            else:
                result = 'tie'

        game.state.is_your_turn = False
        game.state.your_rack = ''

        # Record resignation in notes
        if resignation:
            resign_note = "Opponent resigned"
            if game.state.notes:
                game.state.notes = f"{game.state.notes}; {resign_note}"
            else:
                game.state.notes = resign_note

        # Save final state then archive
        from .game_library import save_active, archive_completed, load_index, save_index
        save_active(game_id, game)

        # Archive (appends to archive.jsonl, deletes active JSON)
        archive_completed(game_id, game, resignation=resignation)

        # Clear slot in index
        index = load_index()
        index['slots'][str(slot)] = None
        save_index(index)

        # Clear in-memory slot
        self.games[slot] = None

        result_label = f"{result.upper()} (resignation)" if resignation else result.upper()
        print(f"[OK] Game over: {result_label} {game.state.your_score}-{game.state.opp_score} "
              f"({'+' if spread >= 0 else ''}{spread}) vs {game.state.opponent_name}")
        print(f"     Archived as {game_id}, Slot {slot} is now free")
        return True

    def reset_slot(self, slot: int):
        """Reset a game slot."""
        if slot < 1 or slot > self.MAX_GAMES:
            print(f"[X] Invalid slot. Use 1-{self.MAX_GAMES}")
            return

        self.games[slot] = None
        print(f"[OK] Slot {slot} reset")
    
    def save_slot(self, slot: int, filename: str = None):
        """Save a game slot."""
        game = self.games.get(slot)
        if game:
            game.save(filename)
        else:
            print(f"[X] Slot {slot} is empty")
    
    def load_slot(self, slot: int, filepath: str):
        """Load a game into a slot."""
        if slot < 1 or slot > self.MAX_GAMES:
            print(f"[X] Invalid slot. Use 1-{self.MAX_GAMES}")
            return
        
        try:
            self.games[slot] = Game.load(filepath)
            self.current_slot = slot
            print(f"[OK] Loaded into Slot {slot}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[X] Failed to load: {e}")
        except Exception as e:
            print(f"[X] Failed to load: {e}")
            traceback.print_exc()
    
    def list_saves(self):
        """List all saved games."""
        if not os.path.exists(SAVE_DIR):
            print("[DIR] No saves directory yet")
            return []
        
        saves = []
        for filename in sorted(os.listdir(SAVE_DIR)):
            if filename.endswith('.json'):
                filepath = os.path.join(SAVE_DIR, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    name = data.get('name', 'Unknown')
                    opponent = data.get('opponent_name', 'Unknown')
                    your_score = data.get('your_score', 0)
                    opp_score = data.get('opp_score', 0)
                    updated = data.get('updated_at', '')[:10]  # Just date
                    spread = your_score - opp_score
                    spread_str = f"+{spread}" if spread >= 0 else str(spread)
                    saves.append({
                        'filename': filename,
                        'filepath': filepath,
                        'name': name,
                        'opponent': opponent,
                        'score': f"{your_score}-{opp_score} ({spread_str})",
                        'updated': updated
                    })
                except (json.JSONDecodeError, KeyError, OSError):
                    pass

        if saves:
            print(f"\n{'='*70}")
            print("SAVED GAMES")
            print(f"{'='*70}")
            for i, s in enumerate(saves, 1):
                print(f"{i}. {s['filename']}")
                print(f"   {s['name']} vs {s['opponent']} | {s['score']} | {s['updated']}")
        else:
            print("[DIR] No saved games found")
        
        return saves
    
    def current_game(self) -> Optional[Game]:
        """Get the current game."""
        return self.games.get(self.current_slot)
    
    def run(self):
        """Main menu loop."""
        print("\n" + "="*60)
        print("CROSSPLAY V16 - GAME MANAGER")
        print("="*60)
        
        while True:
            print("\n" + "-"*40)
            print("MAIN MENU")
            print("-"*40)
            print("1. Assisted Games (vs human opponents)")
            print("2. Play vs Claude AI")
            print("3. Simulate AI vs AI")
            print("4. Quit")
            
            try:
                choice = input("\nSelect mode (1-4): ").strip()
            except EOFError:
                break
            
            if choice == '1':
                self._assisted_mode()
            elif choice == '2':
                self._vs_ai_mode()
            elif choice == '3':
                self._simulation_mode()
            elif choice == '4':
                print("Goodbye!")
                break
            else:
                print("Invalid choice")
    
    def _assisted_mode(self):
        """V7-assisted games against human opponents."""
        self.show_slots()
        
        while True:
            game = self.current_game()
            if game:
                game.show_status()
            
            print("\nCommands: slot N | new N NAME | load ID | games | board | rack TILES | analyze")
            print("          play WORD R C H/V | opp WORD R C H/V SCORE | save | reload | reset N | back")
            
            try:
                cmd = input(f"\n[Slot {self.current_slot}]> ").strip().lower()
            except EOFError:
                break
            
            if not cmd:
                continue
            
            parts = cmd.split()
            action = parts[0]
            
            if action == 'back':
                break
            
            elif action == 'slot' and len(parts) >= 2:
                try:
                    self.select_slot(int(parts[1]))
                except (ValueError, IndexError):
                    print("Usage: slot N")
            
            elif action == 'new' and len(parts) >= 2:
                try:
                    slot = int(parts[1])
                    name = ' '.join(parts[2:]) if len(parts) > 2 else "Opponent"
                    self.new_game(slot, name)
                except (ValueError, IndexError):
                    print("Usage: new SLOT OPPONENT_NAME")
            
            elif action == 'slots':
                self.show_slots()

            elif action == 'games':
                self._show_all_games()

            elif action == 'load' and len(parts) >= 2:
                game_id = parts[1]
                self._load_game_into_slot(game_id, self.current_slot)
                game = self.current_game()  # refresh after load

            elif action == 'board':
                if game:
                    game.show_board()
                else:
                    print("No game in current slot")
            
            elif action == 'rack' and len(parts) >= 2:
                if game:
                    game.set_rack(parts[1])
                else:
                    print("No game in current slot")
            
            elif action == 'analyze':
                if game:
                    game.analyze()
                else:
                    print("No game in current slot")
            
            elif action == 'play' and len(parts) >= 5:
                if game:
                    if not game.state.is_your_turn:
                        print("[!] It's not your turn. Use 'opp' to record opponent's move.")
                        continue
                    try:
                        word = parts[1].upper()
                        row = int(parts[2])
                        col = int(parts[3])
                        horiz = parts[4].upper() == 'H'
                        # Optional 6th arg: new rack after drawing
                        post_rack = parts[5].upper() if len(parts) >= 6 else None
                        # Always pass rack for blank detection; optionally
                        # pass new_rack so the post-draw rack is set directly.
                        game.play_move(word, row, col, horiz,
                                       rack=game.state.your_rack,
                                       new_rack=post_rack)
                    except (ValueError, IndexError) as e:
                        print(f"Error: {e}")
                    except Exception as e:
                        print(f"Error: {e}")
                        traceback.print_exc()
                else:
                    print("No game in current slot")

            elif action in ('exchange', 'swap') and len(parts) >= 3:
                if game:
                    if not game.state.is_your_turn:
                        print("[!] It's not your turn. Use 'opp' to record opponent's move first.")
                        continue
                    try:
                        tiles_dumped = parts[1].upper()
                        new_rack = parts[2].upper()
                        game.record_exchange(tiles_dumped, new_rack)
                    except (ValueError, IndexError) as e:
                        print(f"Error: {e}")
                    except Exception as e:
                        print(f"Error: {e}")
                        traceback.print_exc()
                else:
                    print("No game in current slot")

            elif action in ('opp', 'opp!') and len(parts) >= 6:
                if game:
                    if game.state.is_your_turn:
                        print("[!] It's your turn. Use 'play' to record your move.")
                        continue
                    try:
                        force = (action == 'opp!')
                        word = parts[1].upper()
                        row = int(parts[2])
                        col = int(parts[3])
                        horiz = parts[4].upper() == 'H'
                        score = int(parts[5])
                        game.record_opponent_move(word, row, col, horiz, score,
                                                  force=force)
                    except (ValueError, IndexError) as e:
                        print(f"Error: {e}")
                    except Exception as e:
                        print(f"Error: {e}")
                        traceback.print_exc()
                else:
                    print("No game in current slot")
            
            elif action == 'end' and len(parts) >= 3:
                # end YOUR_SCORE OPP_SCORE [RESULT]
                # e.g.: end 501 346        -> auto-detects win
                # e.g.: end 369 436 loss   -> explicit result
                try:
                    your_sc = int(parts[1])
                    opp_sc = int(parts[2])
                    result = parts[3].lower() if len(parts) >= 4 else None
                    self.end_game(your_score=your_sc, opp_score=opp_sc, result=result)
                except (ValueError, IndexError) as e:
                    print(f"Error: {e}")
                    print("Usage: end YOUR_SCORE OPP_SCORE [win/loss/tie]")
                except Exception as e:
                    print(f"Error: {e}")
                    traceback.print_exc()

            elif action == 'resign':
                # resign [SLOT] -- opponent resigned, end game with current scores
                # e.g.: resign      -> current slot
                # e.g.: resign 3    -> specific slot
                try:
                    resign_slot = int(parts[1]) if len(parts) >= 2 else None
                    self.end_game(slot=resign_slot, result='win', resignation=True)
                except (ValueError, IndexError) as e:
                    print(f"Error: {e}")
                    print("Usage: resign [SLOT]")
                except Exception as e:
                    print(f"Error: {e}")
                    traceback.print_exc()

            elif action == 'reload':
                # Reload game(s) from disk after git pull or external edits
                if len(parts) >= 2:
                    try:
                        self.reload_games(slot=int(parts[1]))
                    except ValueError:
                        print("Usage: reload [SLOT]")
                else:
                    self.reload_games()

            elif action == 'save':
                if game:
                    game.save()
                else:
                    print("No game in current slot")

            elif action == 'reset' and len(parts) >= 2:
                try:
                    self.reset_slot(int(parts[1]))
                except (ValueError, IndexError):
                    print("Usage: reset N")
            
            else:
                print("Unknown command")
    
    def _vs_ai_mode(self):
        """Play against Claude AI."""
        from .play_game import CrossplayGame
        game = CrossplayGame()
        game.play()
    
    def _simulation_mode(self):
        """V7 vs V7 simulation."""
        print("\n" + "="*60)
        print("SIMULATION MODE")
        print("="*60)
        print("\nStrategies:")
        print("  1. highest  - Always highest scoring move")
        print("  2. dump     - Prefer using more tiles")
        print("  3. high_value - Target Q, Z, X, J tiles")
        
        strategy_map = {
            '1': Strategy.HIGHEST_SCORE,
            'highest': Strategy.HIGHEST_SCORE,
            '2': Strategy.DUMP_TILES,
            'dump': Strategy.DUMP_TILES,
            '3': Strategy.HIGH_VALUE,
            'high_value': Strategy.HIGH_VALUE,
        }
        
        try:
            s1 = input("\nPlayer 1 strategy (1/2/3): ").strip().lower()
            s2 = input("Player 2 strategy (1/2/3): ").strip().lower()
            n = int(input("Number of games to simulate: ").strip())
        except (EOFError, ValueError):
            return
        
        strat1 = strategy_map.get(s1, Strategy.HIGHEST_SCORE)
        strat2 = strategy_map.get(s2, Strategy.DUMP_TILES)
        
        run_simulation(strat1, strat2, n)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    manager = GameManager()
    manager.run()
