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
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from .board import Board, tiles_used as _board_tiles_used
from .move_finder_gaddag import GADDAGMoveFinder
from .gaddag import get_gaddag
from .scoring import calculate_move_score, find_crosswords
from .dictionary import Dictionary
from .config import TILE_DISTRIBUTION, TILE_VALUES, BONUS_SQUARES, RACK_SIZE, BOARD_SIZE, CENTER_ROW, CENTER_COL
from .leave_eval import evaluate_leave
from .tile_tracker import TileTracker
from .blocked_cache import BlockedSquareCache
from .lookahead import evaluate_with_lookahead
from .parallel_eval import evaluate_with_lookahead_parallel
from .mc_eval import mc_evaluate_2ply
from .nyt_filter import is_nyt_curated, nyt_warning
from .analysis_lock import acquire_lock, release_lock
from . import __version__

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
        print("Loading GADDAG...")
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
    bag: List[str]
    is_your_turn: bool
    opponent_name: str
    created_at: str
    updated_at: str
    notes: str = ""
    final_turns_remaining: Optional[int] = None

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
        return cls(**d)


class Game:
    """A single game instance."""
    
    def __init__(self, state: Optional[GameState] = None):
        self.gaddag, self.dictionary = get_resources()
        self.blocked_cache = BlockedSquareCache()
        self._threats_cache = None  # Cached existing threats (invalidated after moves)
        self._cached_baseline_risk = 0.0   # Board-wide baseline risk (EV of top existing threat)
        self._cached_baseline_threats = [] # Board-wide threat list for Phase 2 context
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
            self.bag = state.bag.copy() if state.bag else self._new_bag()
            # Infer final_turns_remaining for old saves missing it
            if state.final_turns_remaining is None:
                _tracker = TileTracker()
                _tracker.sync_with_board(self.board, your_rack=state.your_rack or "",
                                         blanks_in_rack=(state.your_rack or "").count('?'),
                                         blank_positions=state.blank_positions)
                if _tracker.get_bag_count() == 0:
                    # Bag already empty but field not set -- assume 2 final turns left
                    # (both players get one final turn after bag empties)
                    state.final_turns_remaining = 2
            # Initialize blocked cache with current board state
            self.blocked_cache.initialize(self.board, self.dictionary)
        else:
            self.state = GameState(
                name="New Game",
                board_moves=[],
                blank_positions=[],
                your_score=0,
                opp_score=0,
                your_rack="",
                bag=[],
                is_your_turn=True,
                opponent_name="Opponent",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            )
            self.board = Board()
            self.bag = self._new_bag()
            # Empty board - all bonus squares playable
            self.blocked_cache.initialize(self.board, self.dictionary)
    
    def _new_bag(self) -> List[str]:
        """Create a new shuffled bag."""
        bag = []
        for letter, count in TILE_DISTRIBUTION.items():
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
        - Bag is empty and one player has no tiles, OR
        - 6 consecutive passes (detected via notes)

        For saved games, remaining = bag + opp_rack. We detect:
        - your_rack < RACK_SIZE means bag was empty on last draw
        - remaining = 0 means opponent also has no tiles (they went out)
        - your_rack empty means you went out
        - Notes explicitly say COMPLETED
        """
        s = self.state
        remaining = len(self.bag)  # bag + opp_rack (unaccounted tiles)
        # Explicit completion marker
        if 'COMPLETED' in s.notes.upper():
            return True
        # You went out (empty rack, bag was already empty)
        if len(s.your_rack) == 0:
            return True
        # Opponent went out (no unaccounted tiles remain)
        if remaining == 0 and len(s.your_rack) > 0:
            return True
        # Rack below full size means bag was empty on last draw;
        # if remaining tiles fit in one rack, game is in final stage
        if len(s.your_rack) < RACK_SIZE and remaining <= RACK_SIZE:
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
        # Compute bag from tracker (self.bag is stale in assisted mode)
        _tracker = TileTracker()
        _tracker.sync_with_board(self.board, your_rack=self.state.your_rack or "",
                                 blanks_in_rack=(self.state.your_rack or "").count('?'),
                                 blank_positions=self.state.blank_positions)
        bag_tiles = _tracker.get_bag_count()
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
        
        # Show unseen tiles
        tracker = TileTracker()
        tracker.sync_with_board(self.board, your_rack=self.state.your_rack or "",
                                blanks_in_rack=(self.state.your_rack or "").count('?'),
                                blank_positions=self.state.blank_positions)
        
        # Build unseen dict and string
        unseen_dict = {}
        unseen_parts = []
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            remaining = tracker.get_remaining(letter)
            if remaining > 0:
                unseen_dict[letter] = remaining
                unseen_parts.append(f"{letter}:{remaining}" if remaining > 1 else letter)
        blanks_remaining = tracker.get_remaining('?')
        if blanks_remaining > 0:
            unseen_dict['?'] = blanks_remaining
            unseen_parts.append(f"?:{blanks_remaining}" if blanks_remaining > 1 else "?")
        
        total_unseen = tracker.get_unseen_count()
        print(f"   Unseen ({total_unseen}): {' '.join(unseen_parts)}")
        
        # Show played out high-value tiles (5+ points)
        high_value_letters = {'J': 10, 'Q': 10, 'X': 8, 'Z': 10, 'K': 6, 'V': 6, 'W': 5}
        played_out = []
        for letter, value in sorted(high_value_letters.items(), key=lambda x: -x[1]):
            if tracker.get_remaining(letter) == 0:
                played_out.append(f"{letter}({value})")
        
        if played_out:
            print(f"   Played out: {' '.join(played_out)}")
        
        # Show power tile draw probabilities if any remain
        from .power_tiles import format_power_tile_display, get_power_tiles_in_pool
        power_tiles = get_power_tiles_in_pool(unseen_dict)
        if power_tiles:
            bag_size = len(self.bag)
            print(f"   {format_power_tile_display(unseen_dict, bag_size)}")
    
    def show_existing_threats(self, top_n: int = 10, force_refresh: bool = False):
        """Show threats that already exist on the current board.
        
        Results are cached and only recomputed after moves are played.
        Use force_refresh=True to force recalculation.
        """
        from .real_risk import analyze_existing_threats
        
        # Check cache first
        if self._threats_cache is not None and not force_refresh:
            risk_str, expected_dmg, max_dmg, threats = self._threats_cache
        else:
            # Build unseen tiles
            tracker = TileTracker()
            tracker.sync_with_board(self.board, your_rack=self.state.your_rack or "",
                                    blanks_in_rack=(self.state.your_rack or "").count('?'),
                                    blank_positions=self.state.blank_positions)
            
            unseen = {}
            for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ?':
                remaining = tracker.get_remaining(letter)
                if remaining > 0:
                    unseen[letter] = remaining
            
            risk_str, expected_dmg, max_dmg, threats = analyze_existing_threats(
                self.board, unseen, self.dictionary,
                BONUS_SQUARES, TILE_VALUES, self.blocked_cache
            )
            
            # Cache results
            self._threats_cache = (risk_str, expected_dmg, max_dmg, threats)

        # Cache baseline risk for use in Phase 2 per-move analysis
        self._cached_baseline_risk = expected_dmg
        self._cached_baseline_threats = threats
        
        print(f"\n{'='*70}")
        print(f"EXISTING THREATS ON BOARD")
        print(f"{'='*70}")
        
        if not threats:
            print("No significant threats detected.")
            return []
        
        # Group threats by position to find max damage per position
        by_position = {}
        for t in threats:
            key = (t['row'], t['col'], t['horizontal'])
            if key not in by_position:
                by_position[key] = []
            by_position[key].append(t)
        
        # For each position, find top EV and max score
        position_summary = []
        for key, pos_threats in by_position.items():
            top_ev = max(pos_threats, key=lambda x: x['ev'])
            max_score = max(pos_threats, key=lambda x: x['score'])
            position_summary.append({
                'top_ev': top_ev,
                'max_score': max_score,
                'key': key
            })
        
        # Sort by top EV descending
        position_summary.sort(key=lambda x: -x['top_ev']['ev'])
        
        print(f"Summary: {risk_str} | Top EV: {expected_dmg:.1f} | Max: {max_dmg}")
        print(f"\n{'#':<3} {'Word':<10} {'Position':<10} {'Score':>5} {'Prob':>6} {'EV':>6}  {'Max Dmg':<18}")
        print("-" * 75)
        
        for i, ps in enumerate(position_summary[:top_n], 1):
            t = ps['top_ev']
            mx = ps['max_score']
            d = 'H' if t['horizontal'] else 'V'
            pos = f"R{t['row']}C{t['col']} {d}"
            
            # Max damage column: show word and score if different from top EV
            if mx['word'] != t['word']:
                max_str = f"{mx['word']}({mx['score']})"
            else:
                max_str = f"({mx['score']})"
            
            print(f"{i:<3} {t['word']:<10} {pos:<10} {t['score']:>5} {t['prob']*100:>5.1f}% {t['ev']:>6.1f}  {max_str:<18}")
        
        if len(position_summary) > top_n:
            print(f"  ... and {len(position_summary) - top_n} more threat positions")
        
        print(f"{'='*70}\n")
        return threats
    
    def analyze(self, rack: str = None, top_n: int = 15, lookahead_n: int = None):
        """Analyze best moves for rack with full risk/leave analysis.

        Args:
            rack: Tile rack to analyze (uses saved rack if not provided)
            top_n: Number of moves to show in main analysis (default 15)
            lookahead_n: Number of candidates for 2-ply evaluation.
                         If None (default), uses N=40 (all within equity
                         spread). MC early stopping controls actual sim
                         count per candidate (~150-530 avg vs K=2000 cap).
        """
        # Request pause from training for duration of analysis
        acquire_lock('game_analysis')
        try:
            return self._analyze_impl(rack, top_n, lookahead_n)
        finally:
            release_lock()

    def _analyze_impl(self, rack: str, top_n: int, lookahead_n: int):
        """Internal analyze implementation."""
        rack = rack or self.state.your_rack
        if not rack:
            print("No rack to analyze!")
            return []
        
        # Show existing threats FIRST so user sees board vulnerabilities
        # Skip when opponent has no remaining turns (ftr==1 and your turn = last move)
        ftr = self.state.final_turns_remaining
        skip_threats = (ftr == 1 and self.state.is_your_turn) or (ftr is not None and ftr <= 0)
        if not skip_threats:
            self.show_existing_threats(top_n=5)
        
        # Move generation — prefer C-accelerated, fallback to Python
        try:
            from .move_finder_c import find_all_moves_c, is_available
            if is_available():
                t_mg = time.time()
                moves = find_all_moves_c(self.board, self.gaddag, rack,
                                          board_blanks=self.state.blank_positions)
                t_mg = time.time() - t_mg
                print(f"  {len(moves)} moves found in {t_mg*1000:.0f}ms (C accel)")
            else:
                raise ImportError("C not available")
        except ImportError:
            finder = GADDAGMoveFinder(self.board, self.gaddag, board_blanks=self.state.blank_positions)
            t_mg = time.time()
            moves = finder.find_all_moves(rack)
            t_mg = time.time() - t_mg
            print(f"  {len(moves)} moves found in {t_mg*1000:.0f}ms (Python)")
        
        if not moves:
            print("No valid moves found!")
            return []
        
        # Build tile tracker to know what's unseen
        tracker = TileTracker()
        tracker.sync_with_board(self.board, your_rack=rack, blanks_in_rack=rack.count('?'),
                                blank_positions=self.state.blank_positions)
        
        # Build unseen tiles counter
        unseen = {}
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ?':
            remaining = tracker.get_remaining(letter)
            if remaining > 0:
                unseen[letter] = remaining
        
        bag_size = tracker.get_bag_count()
        bag_empty = bag_size < 7
        endgame_exact = (bag_size == 0)
        
        # Check for power tiles
        from .power_tiles import format_power_tile_display, get_power_tiles_in_pool, prob_draw_any_power_tile
        power_tiles = get_power_tiles_in_pool(unseen)
        has_power_tiles = bool(power_tiles) and bag_size > 0
        
        # Header
        print(f"\n{'='*70}")
        print(f"ANALYSIS FOR RACK: {rack}  |  Bag: {bag_size} tiles  |  Unseen: {sum(unseen.values())} tiles")
        if has_power_tiles:
            print(f"{format_power_tile_display(unseen, bag_size)}")
        print(f"{'='*70}")
        print(f"\n{'#':<3} {'Word':<12} {'Position':<12} {'Pts':>4} {'Risk (exp/max)':<20} {'Leave':>6} {'DLS':>5} {'DD':>4} {'Trn':>4} {'TW':>5} {'Equity':>6} {'Worst':>6}")
        print("-" * 95)
        print("* = Top Equity    Safe = Best Worst-Case    ! = Negative Worst-Case")
        print("-" * 95)
        
        # OPTIMIZATION: Two-phase analysis
        # Phase 1: Quick score + leave for all moves (no risk calculation)
        preliminary = []
        for move in moves:
            word = move['word']
            pts = move['score']
            
            # Calculate tiles used and leave, accounting for blanks
            used = self._tiles_used(move, rack)
            blanks_used = move.get('blanks_used', [])
            
            leave_tiles = list(rack)
            blanks_consumed = 0
            for i, letter in enumerate(used):
                # Check if this position in the word uses a blank
                # Need to map 'used' index to word index
                word_idx = self._used_idx_to_word_idx(move, i)
                if word_idx in blanks_used:
                    # This tile uses a blank
                    if '?' in leave_tiles:
                        leave_tiles.remove('?')
                        blanks_consumed += 1
                else:
                    # Regular tile
                    if letter in leave_tiles:
                        leave_tiles.remove(letter)
                    elif '?' in leave_tiles:
                        # Shouldn't happen if blanks_used is correct, but fallback
                        leave_tiles.remove('?')
            
            leave_str = ''.join(sorted(leave_tiles)) if leave_tiles else "-"
            
            # Proper leave evaluation (quality score, not just point value)
            # Crossplay: when bag=0, leftover tiles don't penalize — leave is meaningless
            if endgame_exact:
                leave_value = 0.0
            else:
                leave_value = evaluate_leave(leave_str, bag_empty=bag_empty)
            
            # Preliminary equity (without risk)
            prelim_equity = pts + leave_value
            
            preliminary.append({
                **move,
                'used': used,
                'tiles_used': used,
                'leave': leave_str,
                'leave_value': leave_value,
                'prelim_equity': prelim_equity
            })
        
        # Sort by preliminary equity and take top candidates for full analysis
        # Need enough to feed both display (top_n) and MC (up to ~111 candidates)
        preliminary.sort(key=lambda m: -m['prelim_equity'])
        candidates = preliminary[:max(130, top_n * 3)]  # Ensure enough for MC stage
        
        # ENDGAME FAST PATH: When bag=0, skip risk analysis entirely.
        # The endgame 2-ply solver gives exact results (opponent rack is known).
        # Risk analysis is redundant and extremely slow with blanks in opp rack.
        if endgame_exact:
            ftr = self.state.final_turns_remaining
            is_final_move = (ftr == 1 and self.state.is_your_turn)

            # Show 1-ply table (score only, no risk/leave since they're meaningless)
            for i, move in enumerate(candidates[:top_n]):
                word = move['word']
                pts = move['score']
                pos = f"R{move['row']}C{move['col']} {move['direction']}"
                flags = '*' if i == 0 else ''
                print(f"{i+1:<3} {word:<12} {pos:<12} {pts:>4} {'--':<20} {'--':>6} {'--':>5} {'--':>4} {'--':>4} {pts:>+6.0f} {pts:>+6.0f} {flags}")

            if is_final_move:
                # 1-ply: this is the absolute last move of the game.
                # Opponent has no response -- just maximize score.
                print(f"\nENDGAME 1-PLY: your final move, no opponent response.")
                print(f"Best: {candidates[0]['word']} ({candidates[0]['score']} pts)")
                return moves

            # Build unseen tile string for opponent rack (same logic as _show_2ply_analysis)
            opp_tiles = []
            blank_count = unseen.get('?', 0)
            for letter in sorted(unseen.keys()):
                if letter == '?':
                    continue
                opp_tiles.extend([letter] * unseen[letter])
            opp_tiles.extend(['?'] * blank_count)
            opp_rack_str = ''.join(opp_tiles)

            # Jump directly to endgame 2-ply (exact solver)
            self._show_endgame_2ply(rack, opp_rack_str, sum(unseen.values()))
            return moves

        # Risk analysis with comparative pruning (historical: was "Phase 2" during development)
        # Two-tier approach: full risk for top candidates within ~3s budget,
        # remaining candidates keep prelim_equity for MC evaluation.
        # Skip moves whose best-case equity can't beat our current best
        # But always analyze at least top_n moves to ensure we have enough results
        analyzed_moves = []
        best_equity = float('-inf')
        skipped_count = 0
        min_to_analyze = max(top_n + 5, 115)  # Analyze enough for both display and MC candidates (up to N=111)
        risk_time_budget = 90.0 if bag_size <= 5 else 3.0  # generous budget for exhaustive endgame
        t_risk_start = time.time()
        risk_budget_exhausted = False
        
        for i, move in enumerate(candidates):
            # COMPARATIVE PRUNING: Only prune after we have enough moves analyzed
            # Allow small margin (5 pts) to avoid missing close alternatives
            if i >= min_to_analyze and move['prelim_equity'] < best_equity - 5:
                skipped_count += 1
                continue
            
            # TIME-BUDGET CHECK: If risk analysis budget exhausted, use prelim_equity
            # This ensures we still pass plenty of candidates to MC even when
            # full risk analysis is too expensive (e.g. opening with 639ms/move)
            if not risk_budget_exhausted and (time.time() - t_risk_start) > risk_time_budget and i >= top_n:
                risk_budget_exhausted = True
            
            if risk_budget_exhausted:
                # Lightweight: use prelim_equity (score + leave) without risk/positional
                analyzed_moves.append({
                    **move,
                    'risk_str': '-',
                    'expected_risk': 0,
                    'max_damage': 0,
                    'blocking_bonus': 0,
                    'blocked_squares': [],
                    'dls_penalty': 0,
                    'dd_bonus': 0,
                    'turnover_bonus': 0,
                    'hvt_bonus': 0,
                    'tw_dw_penalty': 0,
                    'dls_details': [],
                    'dd_desc': '',
                    'hvt_details': [],
                    'tw_dw_details': [],
                    'positional_adj': 0,
                    'equity': move['prelim_equity'],
                    'worst_case': move['prelim_equity'],
                })
                continue
            
            # Calculate risk with probability weighting
            # For small bags (<=5), use exhaustive opponent analysis (all possible racks).
            # This is exact: enumerate every rack the opp could hold and find their best move.
            # bag=0: ~1ms/move (1 rack), bag=1: ~80ms (6), bag=2: ~300ms (23),
            # bag=3: ~900ms (70), bag=4: ~2.4s (183), bag=5: ~5.6s (428)
            if bag_size <= 5:
                risk_str, expected_risk, max_damage = self._calculate_exhaustive_opp_risk(move, unseen)
            else:
                risk_str, expected_risk, max_damage = self._calculate_probabilistic_risk(move, unseen)
            
            # Calculate defensive bonus for blocking bonus squares
            blocking_bonus, blocked_squares = self._calculate_blocking_bonus(move)
            
            # Calculate positional heuristics (DLS exposure, DD lanes, turnover)
            from .opening_heuristics import evaluate_opening_heuristics
            oh = evaluate_opening_heuristics(self.board, move, rack, unseen, bag_size)
            dls_penalty = oh['dls_penalty']
            dd_bonus = oh['dd_bonus']
            turnover_bonus = oh['turnover_bonus']
            hvt_bonus = oh['hvt_bonus']
            tw_dw_penalty = oh['tw_dw_penalty']

            # Full equity = score + leave_value - expected_risk + blocking_bonus + positional
            equity = move['score'] + move['leave_value'] - expected_risk + blocking_bonus + dls_penalty + dd_bonus + turnover_bonus + hvt_bonus + tw_dw_penalty

            # Worst case equity = score + leave_value - max_damage + blocking_bonus + positional
            worst_case = move['score'] + move['leave_value'] - max_damage + blocking_bonus + dls_penalty + dd_bonus + turnover_bonus + hvt_bonus + tw_dw_penalty

            # Update best equity found
            if equity > best_equity:
                best_equity = equity

            # Net positional adjustment from Phase 2 (blocking, risk, heuristics)
            # This will be carried into MC with dampening to avoid double-counting
            positional_adj = blocking_bonus - expected_risk + dls_penalty + dd_bonus + turnover_bonus + hvt_bonus + tw_dw_penalty

            analyzed_moves.append({
                **move,
                'risk_str': risk_str,
                'expected_risk': expected_risk,
                'max_damage': max_damage,
                'blocking_bonus': blocking_bonus,
                'blocked_squares': blocked_squares,
                'dls_penalty': dls_penalty,
                'dd_bonus': dd_bonus,
                'turnover_bonus': turnover_bonus,
                'hvt_bonus': hvt_bonus,
                'tw_dw_penalty': tw_dw_penalty,
                'dls_details': oh['dls_details'],
                'dd_desc': oh['dd_desc'],
                'hvt_details': oh['hvt_details'],
                'tw_dw_details': oh['tw_dw_details'],
                'positional_adj': positional_adj,
                'equity': equity,
                'worst_case': worst_case,
                'baseline_risk': self._cached_baseline_risk,
            })
        
        if skipped_count > 0:
            print(f"(Skipped {skipped_count} dominated moves via comparative pruning)")
        
        # Sort by EQUITY (not raw score!)
        analyzed_moves.sort(key=lambda m: -m['equity'])
        
        # Find best worst-case among top_n for indicator
        top_moves = analyzed_moves[:top_n]
        best_worst_case = max(m['worst_case'] for m in top_moves) if top_moves else 0
        
        # Display top N by equity
        for i, move in enumerate(top_moves, 1):
            word = move['word']
            pos = f"R{move['row']} C{move['col']} {move['direction']}"
            pts = move['score']
            risk_str = move['risk_str']
            max_damage = move['max_damage']
            leave_value = move['leave_value']
            equity = move['equity']
            worst_case = move['worst_case']
            # Show risk as "type(expected/max)" e.g. "3W(6/26)"
            if risk_str != "-" and max_damage > 0:
                exp_risk = move['expected_risk']
                risk_display = f"{risk_str}({exp_risk:.0f}/{max_damage})"
            else:
                risk_display = risk_str
            
            # Determine indicator
            indicator = ""
            if i == 1:
                indicator = "*"
            elif worst_case == best_worst_case and worst_case >= 0:
                indicator = "Safe"
            elif worst_case < 0:
                indicator = "!"

            # NYT curated word warning
            nyt_tag = nyt_warning(word)

            print(f"{i:<3} {word:<12} {pos:<12} {pts:>4} {risk_display:<20} {leave_value:>+6.1f} {move['dls_penalty']:>+5.1f} {move['dd_bonus']:>+4.1f} {move['turnover_bonus']:>+3.1f} {move['tw_dw_penalty']:>+5.1f} {equity:>+6.0f} {worst_case:>+6.0f}  {indicator}{nyt_tag}")
        
        # Top 3 detailed view
        by_equity = analyzed_moves[:3]
        
        # Detailed top 3 by equity
        print(f"\n{'='*90}")
        print("TOP 3 BY EQUITY (Score + Leave - Risk + Positional)")
        print("=" * 90)
        
        for i, move in enumerate(by_equity, 1):
            word = move['word']
            row, col = move['row'], move['col']
            direction = move['direction']
            pts = move['score']
            
            nyt_tag = nyt_warning(word)
            print(f"\n{i}. {word} @ R{row} C{col} {direction}{nyt_tag}")
            print(f"   Score: {pts} | Leave: {move['leave']} ({move['leave_value']:+.1f}) | Risk: {move['expected_risk']:.1f}")
            
            # Show positional heuristics
            dls_p = move.get('dls_penalty', 0)
            dd_b = move.get('dd_bonus', 0)
            turn_b = move.get('turnover_bonus', 0)
            hvt_b = move.get('hvt_bonus', 0)
            tw_dw_p = move.get('tw_dw_penalty', 0)
            if dls_p != 0 or dd_b != 0 or turn_b != 0 or hvt_b != 0 or tw_dw_p != 0:
                parts = []
                if dls_p != 0:
                    parts.append(f"DLS:{dls_p:+.1f}")
                if dd_b != 0:
                    parts.append(f"DD:{dd_b:+.1f}")
                if turn_b != 0:
                    parts.append(f"Turn:{turn_b:+.1f}")
                if hvt_b != 0:
                    parts.append(f"HVT:{hvt_b:+.1f}")
                if tw_dw_p != 0:
                    parts.append(f"TW/DW:{tw_dw_p:+.1f}")
                print(f"   [INFO] Positional: {' | '.join(parts)}")

                # Show DLS exposure details
                for d in move.get('dls_details', []):
                    tile_r, tile_c, tile_l = d['tile_pos']
                    dls_r, dls_c = d['dls_pos']
                    print(f"      ! {tile_l}@R{tile_r}C{tile_c} -> DLS@R{dls_r}C{dls_c}: opp {d['worst_tile']} for {d['max_damage']}pts ({d['probability']*100:.0f}%)")

                # Show HVT premium details
                for d in move.get('hvt_details', []):
                    print(f"      + {d}")

                # Show TW/DW exposure details
                for d in move.get('tw_dw_details', []):
                    sq_r, sq_c = d['square']
                    print(f"      ! Opens {d['bonus_type']}@R{sq_r}C{sq_c} ({d['access_type']}): -{d['penalty']:.1f}pt")

                # Show DD description if notable
                dd_desc = move.get('dd_desc', '')
                if 'DOUBLE-DOUBLE' in dd_desc:
                    print(f"      --> {dd_desc}")
            
            # Show power tile draw probability if power tiles exist
            if has_power_tiles:
                tiles_used = len(move.get('tiles_used', move.get('used', word)))
                leave_len = len(move['leave'])
                draw_count = 7 - leave_len  # How many tiles we'd draw
                if draw_count > 0 and bag_size > 0:
                    power_prob = prob_draw_any_power_tile(unseen, bag_size, draw_count)
                    if power_prob > 0.05:  # Only show if >5%
                        print(f"   --> Power tile chance: {power_prob*100:.0f}% (drawing {draw_count})")
            
            # Show blocking bonus if any
            if move.get('blocking_bonus', 0) > 0:
                blocked = move.get('blocked_squares', [])
                blocked_str = ', '.join(f"{b[2]}@R{b[0]}C{b[1]}" for b in blocked)
                print(f"   [SAFE] Blocks: {blocked_str} (+{move['blocking_bonus']:.1f} defensive)")
            
            print(f"   EQUITY: {move['equity']:+.1f}")
            
            # Show opened bonus squares with threat analysis
            opened_3w, opened_2w = self._get_opened_bonus_squares(move)
            if opened_3w:
                for sq in opened_3w:
                    threat = self._analyze_threat(sq, unseen, is_3w=True)
                    print(f"   WARNING:  3W {sq}: {threat}")
            if opened_2w:
                for sq in opened_2w:
                    threat = self._analyze_threat(sq, unseen, is_3w=False)
                    print(f"   WARNING:  2W {sq}: {threat}")
            if not opened_3w and not opened_2w and move.get('blocking_bonus', 0) == 0:
                print(f"   [OK] No bonus squares opened")
        
        # =================================================================
        # EXCHANGE CANDIDATE GENERATION
        # Generate top exchange options to compete in MC pipeline alongside
        # regular moves. Only when bag >= 7 and best play looks weak.
        # =================================================================
        exchange_candidates = None
        if analyzed_moves and bag_size >= 7:
            best_move = analyzed_moves[0]
            best_play_equity = best_move['score'] + best_move.get('leave_value', 0)
            
            # Only consider exchange when best play equity < 35
            # (strong plays will always dominate exchange)
            if best_play_equity < 35:
                exchange_candidates = self._generate_exchange_candidates(rack, unseen)
                if exchange_candidates:
                    n_exch = len(exchange_candidates)
                    print(f"\n  [EXCH] {n_exch} exchange option{'s' if n_exch > 1 else ''} "
                          f"added to MC evaluation (best play equity: {best_play_equity:.1f})")
        
        # =================================================================
        # 2-PLY LOOKAHEAD SECTION
        # MC needs more candidates than the display top_n — pass full
        # analyzed_moves (already equity-sorted) so _show_2ply_analysis
        # can slice to lookahead_n independently of the display count.
        # Exchange candidates are passed through to compete head-to-head.
        # =================================================================
        self._show_2ply_analysis(rack, analyzed_moves, unseen, bag_empty, lookahead_n,
                                exchange_candidates=exchange_candidates)
        
        return moves
    
    def _generate_exchange_candidates(self, rack: str, unseen: dict) -> list:
        """Generate top exchange options for MC evaluation.
        
        Tries full exchange and partial exchanges (keep 1-4 tiles).
        Returns top 5 exchange options sorted by expected new rack leave.
        Each returned dict has: 'keep', 'dump', 'rack'.
        """
        from .leave_eval import evaluate_leave
        from itertools import combinations
        import random
        
        rack_list = list(rack)
        pool = []
        for letter, count in unseen.items():
            pool.extend([letter] * count)
        
        if len(pool) < 7:
            return None
        
        # Quick MC to estimate E[new rack leave] for each exchange option
        N_QUICK = 500  # fast estimate
        options = []
        
        # Full exchange (keep nothing)
        total = 0
        for _ in range(N_QUICK):
            drawn = random.sample(pool, min(7, len(pool)))
            total += evaluate_leave(''.join(drawn))
        avg_leave = total / N_QUICK
        options.append({
            'keep': '', 'dump': rack, 'rack': rack,
            'expected_leave': avg_leave,
        })
        
        # Partial exchanges: keep 1-4 tiles
        for keep_n in range(1, min(5, len(rack_list))):
            seen_keeps = set()
            for keep_combo in combinations(range(len(rack_list)), keep_n):
                keep = tuple(sorted(rack_list[i] for i in keep_combo))
                if keep in seen_keeps:
                    continue
                seen_keeps.add(keep)
                
                keep_str = ''.join(keep)
                keep_leave = evaluate_leave(keep_str)
                
                # Skip clearly bad keeps
                if keep_leave < -5 and keep_n >= 3:
                    continue
                
                draw_n = 7 - keep_n
                total = 0
                for _ in range(N_QUICK):
                    drawn = random.sample(pool, min(draw_n, len(pool)))
                    new_rack = list(keep) + drawn
                    total += evaluate_leave(''.join(new_rack))
                avg_leave = total / N_QUICK
                
                remaining = list(rack)
                for t in keep:
                    remaining.remove(t)
                
                options.append({
                    'keep': keep_str, 'dump': ''.join(remaining), 'rack': rack,
                    'expected_leave': avg_leave,
                })
        
        # Sort by expected leave, take top 5
        options.sort(key=lambda x: -x['expected_leave'])
        return options[:5]
    
    def _analyze_threat(self, square: tuple, unseen: dict, is_3w: bool) -> str:
        """Analyze threat level for an opened bonus square."""
        row, col = square
        
        # Check what high-value tiles are still unseen
        high_value = ['Z', 'Q', 'X', 'J', 'K']
        hv_unseen = sum(unseen.get(t, 0) for t in high_value)
        total_unseen = sum(unseen.values())
        
        if total_unseen == 0:
            return "no tiles unseen"
        
        # Probability opponent has at least one high-value tile
        # P(at least 1 HV in 7 tiles) ≈ 1 - (1 - hv/total)^7
        if hv_unseen > 0 and total_unseen >= 7:
            prob_hv = 1 - ((total_unseen - hv_unseen) / total_unseen) ** 7
        else:
            prob_hv = 0
        
        # Estimate max damage
        if is_3w:
            # 3W with high value tile could be 45-90 points
            max_damage = 60 if prob_hv > 0.3 else 40
        else:
            # 2W typically 20-40 points
            max_damage = 30 if prob_hv > 0.3 else 20
        
        return f"~{prob_hv*100:.0f}% HV tile, est. {max_damage} pts"
    
    def _show_2ply_analysis(self, rack: str, top_moves: list, unseen: dict, bag_empty: bool, lookahead_n: int = None, exchange_candidates: list = None):
        """Show Monte Carlo 2-ply analysis for top moves.
        
        Pre-ranked candidates from 1-ply are passed directly to MC evaluation.
        N and K are computed adaptively based on:
        - Calibrated throughput on current hardware
        - Candidate equity spread (include all within 15 equity of best)
        - Game phase (determines MC time budget)
        
        Args:
            rack: Current rack
            top_moves: All analyzed moves from 1-ply (equity-sorted)
            unseen: Dict of unseen tile counts
            bag_empty: Whether bag is empty
            lookahead_n: Override N (None = adaptive)
            exchange_candidates: Exchange moves to compete with regular plays
        """
        # Build unseen tiles string (limit blanks to 2 for speed)
        unseen_tiles = []
        blank_count = 0
        for letter, count in sorted(unseen.items()):
            if letter == '?':
                blanks_to_add = min(count, 2)
                unseen_tiles.extend(['?'] * blanks_to_add)
                blank_count = count
            else:
                unseen_tiles.extend([letter] * count)
        unseen_str = ''.join(unseen_tiles)
        total_unseen = sum(unseen.values())
        
        # Routing based on bag size:
        #   bag == 0: deterministic 2-ply (opp rack known exactly)
        #   bag 1-8:  hybrid near-endgame (exhaustive for bag-emptying moves,
        #             parity-adjusted 1-ply for non-emptying moves)
        #   bag 9+:   Monte Carlo 2-ply sampling
        bag_size = total_unseen - 7  # unseen minus opponent rack
        if bag_size == 0:
            # Bag empty: opponent rack is known exactly — use deterministic 2-ply
            self._show_endgame_2ply(rack, unseen_str, total_unseen)
            return
        if 1 <= bag_size <= 8:
            # Near-endgame: hybrid evaluation. Bag-emptying moves get exact
            # 3-ply enumeration over all C(unseen,7) opponent racks.
            # Non-emptying moves get parity-adjusted 1-ply equity.
            self._show_near_endgame(rack, unseen_str, total_unseen, top_moves)
            return

        # Adaptive N×K based on calibrated throughput
        blanks_unseen = blank_count
        
        # MC time budget -- with early stopping, MC typically finishes in 2-8s
        # regardless of K. Budget is a safety cap, not a planning target.
        mc_budget = 27.0
        
        if lookahead_n is None:
            from .mc_calibrate import compute_adaptive_n
            lookahead_n, k_sims, nk_reason = compute_adaptive_n(
                top_moves, mc_budget, bag_size, top_n_display=15)
        else:
            # User specified N; compute K from calibration
            from .mc_calibrate import estimate_throughput
            sps = estimate_throughput(bag_size)
            k_sims = max(100, min(2000, int(sps * mc_budget / lookahead_n)))
            nk_reason = "user-specified N"
        
        print(f"\n{'='*70}")
        print(f"MONTE CARLO 2-PLY (K={k_sims} sims x N={lookahead_n} candidates) [{total_unseen} unseen]")
        print(f"  ({nk_reason})")
        print("=" * 70)
        if blank_count > 2:
            from .mc_eval import _blank_correction_factor
            bcf = _blank_correction_factor(total_unseen, blank_count)
            print(f"Note: {blank_count} blanks unseen, capped to 2 for speed (correction: {bcf:.3f}x)")
        
        try:
            t_mc_start = time.time()
            results = mc_evaluate_2ply(
                self.board, rack, unseen_str,
                board_moves=self.state.board_moves,
                gaddag=self.gaddag,
                top_n=lookahead_n,
                k_sims=k_sims,
                include_blockers=False,
                board_blanks=self.state.blank_positions,
                pre_ranked_candidates=top_moves,  # pass 1-ply equity-ranked moves
                exchange_candidates=exchange_candidates,
            )
            t_mc = time.time() - t_mc_start
            
            if not results:
                print("No MC 2-ply results available.")
                return
            
            print(f"  >> MC 2-ply completed in {t_mc*1000:.0f}ms ({k_sims * len(results)} total sims)")
            
            # Check for bingo threats BEFORE any move (baseline)
            baseline_bingo = self._find_opponent_bingo(unseen_str)
            
            # Check if any result has non-zero positional adjustment or risk
            has_pos_adj = any(m.get('pos_adj_dampened', 0) != 0 for m in results[:min(lookahead_n, 30)])
            has_risk = any(m.get('expected_risk', 0) > 0 for m in results[:min(lookahead_n, 30)])

            if has_pos_adj:
                print(f"\n{'#':<3} {'Word':<12} {'Pos':<10} {'Pts':>4} "
                      f"{'AvgOpp':>6} {'MaxOpp':>6} {'Std':>5} "
                      f"{'%Beats':>6} {'Leave':>6} {'PosAdj':>6} {'MC Eq':>7}"
                      + (f" {'RiskEq':>7}" if has_risk else ""))
                print("-" * (85 + (8 if has_risk else 0)))
            else:
                print(f"\n{'#':<3} {'Word':<12} {'Pos':<10} {'Pts':>4} "
                      f"{'AvgOpp':>6} {'MaxOpp':>6} {'Std':>5} "
                      f"{'%Beats':>6} {'Leave':>6} {'MC Eq':>7}"
                      + (f" {'RiskEq':>7}" if has_risk else ""))
                print("-" * (78 + (8 if has_risk else 0)))

            # Track which moves block bingos
            blocking_moves = []

            # Count exchange results for summary
            exchange_in_results = [m for m in results if m.get('is_exchange')]

            for i, m in enumerate(results[:min(lookahead_n, 30)], 1):
                is_exch = m.get('is_exchange', False)

                if is_exch:
                    pos = "EXCHANGE"
                    word_display = f"<-> dump {m.get('exchange_dump', '?')}"
                    if m.get('exchange_keep'):
                        word_display = f"<-> keep {m['exchange_keep']}"
                else:
                    pos = f"R{m['row']}C{m['col']} {m['direction']}"
                    word_display = m['word']

                # Check if this move blocks opponent's bingo
                blocks_bingo = False
                opp_avg = m['mc_avg_opp']

                if not is_exch and baseline_bingo and baseline_bingo['score'] >= 41:
                    if opp_avg < baseline_bingo['score'] - 20:
                        blocks_bingo = True
                        blocking_moves.append((m, baseline_bingo))

                # Markers
                bingo_marker = ""
                if is_exch:
                    bingo_marker = "[EXCH]"
                else:
                    opp_word = m.get('opp_word', '')
                    if len(opp_word) >= 7 and m['mc_max_opp'] >= 41:
                        bingo_marker = "[!B]"
                    elif blocks_bingo:
                        bingo_marker = "[DB]"

                leave_display = m['leave'] if not is_exch else m['leave'][:8]

                risk_eq_str = ""
                if has_risk:
                    risk_eq = m.get('risk_adj_equity', m['total_equity'])
                    risk_eq_str = f" {risk_eq:>+7.1f}"

                # NYT curated word warning
                nyt_tag = ""
                if not is_exch and is_nyt_curated(m['word']):
                    nyt_tag = " [NYT?]"

                if has_pos_adj:
                    pos_adj = m.get('pos_adj_dampened', 0)
                    pos_str = f"{pos_adj:>+5.1f}" if pos_adj != 0 else "    -"
                    print(f"{i:<3} {word_display:<12} {pos:<10} {m['score']:>4} "
                          f"{m['mc_avg_opp']:>6.1f} {m['mc_max_opp']:>6} {m['mc_std_opp']:>5.1f} "
                          f"{m['pct_opp_beats']:>5.1f}% {leave_display:>6} {pos_str:>6} "
                          f"{m['total_equity']:>+7.1f}{risk_eq_str} {bingo_marker}{nyt_tag}")
                else:
                    print(f"{i:<3} {word_display:<12} {pos:<10} {m['score']:>4} "
                          f"{m['mc_avg_opp']:>6.1f} {m['mc_max_opp']:>6} {m['mc_std_opp']:>5.1f} "
                          f"{m['pct_opp_beats']:>5.1f}% {leave_display:>6} "
                          f"{m['total_equity']:>+7.1f}{risk_eq_str} {bingo_marker}{nyt_tag}")
            
            # Exchange recommendation if any exchange ranked highly
            best_exch_in_top = None
            best_play_in_top = None
            for m in results[:min(lookahead_n, 30)]:
                if m.get('is_exchange') and best_exch_in_top is None:
                    best_exch_in_top = m
                elif not m.get('is_exchange') and best_play_in_top is None:
                    best_play_in_top = m
            
            if best_exch_in_top and best_play_in_top:
                exch_rank = next(i for i, m in enumerate(results) if m.get('is_exchange')) + 1
                if exch_rank == 1:
                    margin = best_exch_in_top['total_equity'] - best_play_in_top['total_equity']
                    keep = best_exch_in_top.get('exchange_keep', '') or '(none)'
                    dump = best_exch_in_top.get('exchange_dump', '')
                    print(f"\n  WARNING:  EXCHANGE RECOMMENDED: dump [{dump}], keep [{keep}]")
                    print(f"      MC advantage: +{margin:.1f} equity over {best_play_in_top['word']}")
                elif exch_rank <= 3:
                    keep = best_exch_in_top.get('exchange_keep', '') or '(none)'
                    print(f"\n  [EXCH] Exchange (keep [{keep}]) ranked #{exch_rank} -- competitive option")
            
            # Show top opponent responses for #1 ranked move
            best = results[0]
            if best.get('top_opp_responses') and not best.get('is_exchange'):
                print(f"\n  Top opponent responses to {best['word']}:")
                for resp in best['top_opp_responses'][:3]:
                    rpos = f"R{resp['row']}C{resp['col']} {resp['direction']}"
                    print(f"    {resp['word']} @ {rpos} = {resp['score']} pts")
            
            # Compare with 1-ply recommendation
            best_1ply = top_moves[0]['word'] if top_moves else None
            best_mc = results[0]['word'] if results else None
            best_mc_is_exch = results[0].get('is_exchange', False) if results else False
            
            if best_1ply and best_mc:
                if best_mc_is_exch:
                    print(f"\n>> MC 2-ply recommends EXCHANGE over 1-ply pick: {best_1ply}{nyt_warning(best_1ply)}")
                elif best_1ply == best_mc:
                    print(f"\n[OK] 1-ply and MC 2-ply agree: {best_1ply}{nyt_warning(best_1ply)}")
                else:
                    print(f"\n>> Different recommendations:")
                    print(f"   1-ply: {best_1ply}{nyt_warning(best_1ply)}")
                    print(f"   MC 2-ply: {best_mc}{nyt_warning(best_mc)}")

            # Store engine recommendation for next play_move() call
            if results:
                self.last_analysis = {
                    'top3': [
                        {
                            'word': r['word'],
                            'row': r['row'],
                            'col': r['col'],
                            'dir': r['direction'],
                            'score': r['score'],
                            'equity': r.get('total_equity', 0),
                            'risk_eq': r.get('risk_adj_equity', r.get('total_equity', 0)),
                        }
                        for r in results[:3]
                        if not r.get('is_exchange', False)
                    ]
                }

            # Bingo blocking analysis
            if baseline_bingo and baseline_bingo['score'] >= 41:
                self._show_bingo_blocking_analysis(rack, unseen_str, baseline_bingo, results)
            
            # Catch-up advice for late game when far behind
            self._show_catchup_advice(unseen, total_unseen, results)
            
            # 3-ply analysis for late game (<=12 in bag)
            # Budget: 20s total minus MC time already spent
            bag_size = total_unseen - 7  # unseen minus opponent rack
            if bag_size <= 12 and bag_size >= 0:
                remaining_budget = max(2.0, 20.0 - t_mc)
                self._show_3ply_analysis(rack, unseen_str, total_unseen, time_budget=remaining_budget)
                    
        except Exception as e:
            import traceback
            print(f"MC 2-ply analysis error: {e}")
            traceback.print_exc()
    
    def _find_opponent_bingo(self, unseen_str: str) -> dict:
        """Find opponent's best bingo (7+ tiles) on current board.
        
        Only runs when unseen tiles <= 21 (late game), otherwise returns None.
        """
        # Skip if too many unseen - combinatorially expensive and less useful
        if len(unseen_str) > 21:
            return None
            
        finder = GADDAGMoveFinder(self.board, self.gaddag, board_blanks=self.state.blank_positions)
        opp_moves = finder.find_all_moves(unseen_str)
        
        # Find best bingo (7+ tiles used)
        bingos = []
        for m in opp_moves:
            # Count tiles actually placed (word length minus tiles already on board)
            tiles_placed = len(m['word'])  # Simplified - actual calculation would check hooks
            if m['score'] >= 41:  # Likely a bingo (min: 7 × 1pt + 40 bonus)
                bingos.append(m)
        
        if bingos:
            best = max(bingos, key=lambda m: m['score'])
            return {
                'word': best['word'],
                'score': best['score'],
                'row': best['row'],
                'col': best['col'],
                'direction': best['direction']
            }
        return None
    
    def _show_bingo_blocking_analysis(self, rack: str, unseen_str: str, baseline_bingo: dict, results: list):
        """Show analysis of moves that block opponent's bingo."""
        print(f"\n{'-'*70}")
        print(f"WARNING:  BINGO THREAT: {baseline_bingo['word']} at R{baseline_bingo['row']}C{baseline_bingo['col']} "
              f"{baseline_bingo['direction']} for {baseline_bingo['score']} pts")
        print(f"{'-'*70}")
        
        # Find ALL moves that block the bingo lane
        bingo_row = baseline_bingo['row']
        bingo_col = baseline_bingo['col']
        bingo_horiz = baseline_bingo['direction'] == 'H'
        bingo_word = baseline_bingo['word']
        
        # Calculate squares the bingo occupies
        bingo_squares = set()
        for i in range(len(bingo_word)):
            if bingo_horiz:
                bingo_squares.add((bingo_row, bingo_col + i))
            else:
                bingo_squares.add((bingo_row + i, bingo_col))
        
        # Find all your moves
        finder = GADDAGMoveFinder(self.board, self.gaddag, board_blanks=self.state.blank_positions)
        all_moves = finder.find_all_moves(rack)
        
        # Find moves that occupy any bingo square
        blocking_candidates = []
        for m in all_moves:
            move_squares = set()
            for i in range(len(m['word'])):
                if m['direction'] == 'H':
                    move_squares.add((m['row'], m['col'] + i))
                else:
                    move_squares.add((m['row'] + i, m['col']))
            
            # Check if move blocks bingo
            if move_squares & bingo_squares:
                blocking_candidates.append(m)
        
        if not blocking_candidates:
            print("\n   No blocking moves available for this bingo.")
            return
        
        # Evaluate each blocking move with 2-ply
        print(f"\nEvaluating {len(blocking_candidates)} blocking moves...")
        
        blocking_results = []
        for move in blocking_candidates:
            horiz = move['direction'] == 'H'
            placed = self.board.place_move(move['word'], move['row'], move['col'], horiz)
            
            # Find opponent's new best
            opp_finder = GADDAGMoveFinder(self.board, self.gaddag, board_blanks=self.state.blank_positions)
            opp_moves = opp_finder.find_all_moves(unseen_str)
            
            if opp_moves:
                opp_best = max(opp_moves, key=lambda m: m['score'])
                opp_score = opp_best['score']
                opp_word = opp_best['word']
            else:
                opp_score = 0
                opp_word = "(none)"
            
            self.board.undo_move(placed)
            
            net = move['score'] - opp_score
            saved = baseline_bingo['score'] - opp_score
            
            blocking_results.append({
                'word': move['word'],
                'row': move['row'],
                'col': move['col'],
                'direction': move['direction'],
                'score': move['score'],
                'opp_score': opp_score,
                'opp_word': opp_word,
                'net': net,
                'saved': saved
            })
        
        # Sort by net outcome (best first)
        blocking_results.sort(key=lambda m: -m['net'])
        
        print(f"\nMoves that BLOCK {baseline_bingo['word']}:")
        print(f"{'Word':<12} {'Pos':<10} {'Pts':>4} {'Opp After':>12} {'OppPts':>6} {'Saved':>6} {'Net':>6}")
        print("-" * 65)
        
        for m in blocking_results[:8]:
            pos = f"R{m['row']}C{m['col']} {m['direction']}"
            print(f"{m['word']:<12} {pos:<10} {m['score']:>4} {m['opp_word'][:12]:>12} "
                  f"{m['opp_score']:>6} {m['saved']:>+6} {m['net']:>+6}")
        
        # Compare best blocker to best overall 2-ply
        best_blocker = blocking_results[0]
        best_overall = results[0] if results else None
        
        if best_overall:
            overall_net = best_overall['score'] - best_overall['opp_best']
            blocker_net = best_blocker['net']
            
            print(f"\n[INFO] Comparison:")
            print(f"   Best 2-ply (no block): {best_overall['word']} ({best_overall['score']} pts)")
            print(f"      -> Opp plays {best_overall.get('opp_word', '?')} ({best_overall['opp_best']} pts)")
            print(f"      -> Net: {overall_net:+.0f}")
            print(f"   Best BLOCKER: {best_blocker['word']} ({best_blocker['score']} pts)")
            print(f"      -> Opp plays {best_blocker['opp_word']} ({best_blocker['opp_score']} pts)")
            print(f"      -> Net: {blocker_net:+d}")
            
            if blocker_net > overall_net:
                advantage = blocker_net - overall_net
                print(f"\n   --> BLOCKING IS BETTER! {best_blocker['word']} gains {advantage} pts net")
                print(f"      Blocks {baseline_bingo['score']} pt bingo, opp limited to {best_blocker['opp_score']} pts")
            else:
                disadvantage = overall_net - blocker_net
                print(f"\n   OK Non-blocking play is better by {disadvantage} pts net")
                print(f"      But blocking keeps game closer if you're ahead")
    
    def _show_catchup_advice(self, unseen: dict, total_unseen: int, results: list):
        """Show strategic advice when far behind in late game."""
        from .power_tiles import get_power_tiles_in_pool, prob_draw_any_power_tile
        
        score_diff = self.state.your_score - self.state.opp_score
        bag_size = total_unseen - 7  # Unseen minus opponent's rack
        
        # Only show if: behind by 50+, unseen <= 21 (late game), power tiles available
        if score_diff >= -50:
            return
        if total_unseen > 21:
            return
        
        power_tiles = get_power_tiles_in_pool(unseen)
        if not power_tiles:
            return
        
        print(f"\n{'-'*70}")
        print(f"TIP: CATCH-UP STRATEGY (down {abs(score_diff)} pts, {bag_size} in bag)")
        print(f"{'-'*70}")
        
        # Find moves with different draw counts
        draw_options = {}
        for r in results[:8]:
            leave_len = len(r.get('leave', ''))
            draw_count = 7 - leave_len
            if draw_count > 0 and draw_count not in draw_options:
                prob = prob_draw_any_power_tile(unseen, bag_size, draw_count)
                draw_options[draw_count] = (r, prob)
        
        # Show power tile odds for different plays
        power_str = ", ".join(f"{t}({v})" for t, (c, v) in sorted(power_tiles.items(), key=lambda x: -x[1][1]))
        print(f"   Power tiles in pool: {power_str}")
        print()
        
        if draw_options:
            print("   Play style trade-offs:")
            for draw_count in sorted(draw_options.keys()):
                move, prob = draw_options[draw_count]
                word = move['word']
                leave = move.get('leave', '')
                pts = move['score']
                print(f"   * {word} ({pts} pts) -> draw {draw_count} -> {prob*100:.0f}% power tile, leave: {leave or '(empty)'}")
        
        # Strategic advice based on situation
        print()
        if score_diff <= -100:
            print("   WARNING:  DOWN 100+: High variance is your friend!")
            print("      Consider plays that draw more tiles for better power tile odds.")
            print("      Keeping a great leave matters less when you need big swings.")
        elif score_diff <= -50:
            print("   [INFO] DOWN 50-100: Balance scoring with power tile chances.")
            print("      Plays that empty your rack give more lottery tickets.")
    
    def _show_3ply_analysis(self, rack: str, unseen_str: str, total_unseen: int, time_budget: float = 30.0):
        """Show 3-ply lookahead analysis for endgame (<=12 in bag)."""
        from .lookahead_3ply import evaluate_3ply
        
        bag_size = total_unseen - 7
        
        print(f"\n{'='*70}")
        print(f"3-PLY LOOKAHEAD (You -> Opp -> You) [bag: {bag_size}, unseen: {total_unseen}]")
        print("=" * 70)
        print("Your move -> Opponent's best -> Your counter-response")
        print()
        
        try:
            results = evaluate_3ply(
                self.board,
                your_rack=rack,
                unseen_tiles=unseen_str,
                gaddag=self.gaddag,
                board_blanks=self.state.blank_positions,
                time_budget=time_budget,
            )
            
            if not results:
                print("No 3-ply results available.")
                return
            
            # Show metadata
            meta = results[0].get('_meta', {})
            evald = meta.get('candidates_evaluated', '?')
            pruned_ct = meta.get('candidates_pruned', 0)
            elapsed = meta.get('elapsed_s', '?')
            print(f"  Evaluated {evald} candidates, pruned {pruned_ct}, in {elapsed}s")
            print()
            
            print(f"{'#':<3} {'Your Move':<10} {'Pts':>4} {'Opp':>12} {'Opp Pts':>7} {'Counter':>10} {'+Pts':>5} {'Net':>5}")
            print("-" * 70)
            
            for i, r in enumerate(results[:10], 1):
                your_move = r['word']
                your_pts = r['score']
                opp_move = r['opp_word'][:10]
                opp_pts = r['opp_score']
                counter = r['your_response'][:10] if r['your_response'] else "-"
                counter_pts = r['your_response_score']
                net = r['net_3ply']
                
                print(f"{i:<3} {your_move:<10} {your_pts:>4} {opp_move:>12} {opp_pts:>7} {counter:>10} {counter_pts:>+5} {net:>+5}")
            
            # Show best 3-ply recommendation
            best_3ply = results[0]
            print(f"\n--> 3-ply best: {best_3ply['word']} ({best_3ply['score']} pts)")
            print(f"   -> Opp plays {best_3ply['opp_word']} ({best_3ply['opp_score']} pts)")
            print(f"   -> You counter with {best_3ply['your_response']} ({best_3ply['your_response_score']} pts)")
            print(f"   -> Net after 3 plies: {best_3ply['net_3ply']:+d}")
            
            # Show top opp responses for awareness
            opp_resps = best_3ply.get('opp_responses', [])
            if len(opp_resps) > 1:
                print(f"\n  Top opponent responses to {best_3ply['word']}:")
                for resp in opp_resps[:3]:
                    print(f"    {resp['word']} @ {resp['pos']} = {resp['score']} pts")
            
        except Exception as e:
            print(f"3-ply analysis error: {e}")
            import traceback
            traceback.print_exc()
    
    def _show_endgame_2ply(self, rack: str, opp_rack: str, total_unseen: int):
        """Show exact 2-ply endgame analysis when bag=0 (Crossplay rules).

        Opponent rack is known exactly (= all unseen tiles).  Both players
        get one final turn; leftover tiles don't penalize.

        Uses upper-bound pruning: evaluates ALL moves but only does full
        opponent search for moves that could actually be optimal.
        """
        from .lookahead_3ply import evaluate_endgame_2ply

        print(f"\n{'='*70}")
        print(f"ENDGAME 2-PLY (exact) [bag: 0, opp rack: {len(opp_rack)} tiles]")
        print("=" * 70)
        print(f"Opponent rack known: {opp_rack}")
        print("Your move -> Opponent's best response -> Net")
        print()

        try:
            results = evaluate_endgame_2ply(
                self.board,
                your_rack=rack,
                opp_rack=opp_rack,
                gaddag=self.gaddag,
                board_blanks=self.state.blank_positions,
            )

            if not results:
                print("No endgame results available (no valid moves).")
                return

            meta = results[0].get('_meta', {})
            total_moves = meta.get('total_moves', '?')
            n_interfering = meta.get('interfering', '?')
            fully_evald = meta.get('fully_evaluated', '?')
            pruned_ct = meta.get('pruned_by_bound', 0)
            elapsed = meta.get('elapsed_s', '?')
            opp_bl = meta.get('opp_baseline', '?')
            bl_time = meta.get('baseline_time_s', '?')

            print(f"  {total_moves} total moves, {n_interfering} interfering, "
                  f"{fully_evald} fully evaluated, {pruned_ct} pruned by bound")
            print(f"  Opp baseline: {opp_bl} (found in {bl_time}s)")
            print(f"  Total time: {elapsed}s")
            print()

            print(f"{'#':<3} {'Your Move':<12} {'Pos':<12} {'Pts':>4} "
                  f"{'Opp Move':<12} {'Opp Pts':>7} {'Net':>6} {'':>5}")
            print("-" * 67)

            for i, r in enumerate(results[:15], 1):
                pos = f"R{r['row']}C{r['col']} {r['direction']}"
                opp_move = r['opp_word'][:10]
                tag = "" if r.get('exact', True) else " (ub)"
                print(f"{i:<3} {r['word']:<12} {pos:<12} {r['score']:>4} "
                      f"{opp_move:<12} {r['opp_score']:>7} {r['net_2ply']:>+6}{tag}")

            best = results[0]
            print(f"\n  Best: {best['word']} ({best['score']} pts)")
            print(f"    -> Opp plays {best['opp_word']} ({best['opp_score']} pts)")
            print(f"    -> Net: {best['net_2ply']:+d}")

        except Exception as e:
            print(f"Endgame 2-ply error: {e}")
            import traceback
            traceback.print_exc()

    def _show_near_endgame(self, rack: str, unseen_str: str, total_unseen: int, top_moves: list):
        """Show hybrid near-endgame analysis for bag 1-8.

        Bag-emptying moves get exact 3-ply evaluation (exhaustive enumeration
        over all possible opponent rack assignments). Non-emptying moves get
        parity-adjusted 1-ply equity (penalized by P(opp empties bag) ×
        structural advantage). This correctly captures the structural advantage
        of emptying the bag.
        """
        from .lookahead_3ply import evaluate_near_endgame

        bag_size = total_unseen - 7

        print(f"\n{'='*70}")
        print(f"NEAR-ENDGAME HYBRID (bag: {bag_size}, unseen: {total_unseen})")
        print(f"  EXH = EMPTIES BAG (exact 3-ply over all opp racks)")
        print(f"  PAR = leaves tiles in bag (parity penalty applied)")
        print(f"  Emptying the bag = you know opp's rack + control endgame")
        print("=" * 70)

        try:
            t_start = time.time()
            results = evaluate_near_endgame(
                self.board,
                your_rack=rack,
                unseen_tiles=unseen_str,
                candidates=top_moves,
                gaddag=self.gaddag,
                board_blanks=self.state.blank_positions,
                top_n=25,
            )
            t_elapsed = time.time() - t_start

            if not results:
                print("No near-endgame results available.")
                return

            meta = results[0].get('_meta', {})
            n_exhaust = meta.get('exhaust_evaluated', 0)
            n_1ply = meta.get('oneply_evaluated', 0)
            solver_time = meta.get('elapsed_s', round(t_elapsed, 2))
            bcf = meta.get('blank_correction', 1.0)

            n_parity = sum(1 for r in results if r.get('eval_type') == 'parity')
            n_plain_1ply = n_1ply - n_parity

            print(f"  {n_exhaust} exhaust + {n_parity} parity + "
                  f"{n_plain_1ply} 1ply candidates | "
                  f"{solver_time}s | blank_corr={bcf}")
            print()

            # Table header
            print(f"{'#':<3} {'Word':<12} {'Pos':<12} {'Pts':>4} "
                  f"{'Type':<7} {'AvgOpp':>6} {'MaxOpp':>6} {'YourResp':>8} "
                  f"{'Net Eq':>7} {'Bag>':>5} {'Parity':>7}")
            print("-" * 90)

            for i, r in enumerate(results[:20], 1):
                pos = f"R{r['row']}C{r['col']} {r['direction']}"
                etype = r['eval_type']
                if etype == 'exhaust':
                    tag = "EXH"
                elif etype == 'parity':
                    tag = "PAR"
                else:
                    tag = "1PL"

                if etype == 'exhaust':
                    print(f"{i:<3} {r['word']:<12} {pos:<12} {r['score']:>4} "
                          f"{tag:<7} {r['opp_avg']:>6.1f} {r['opp_max']:>6} "
                          f"{r['your_resp_avg']:>8.1f} {r['net_equity']:>+7.1f} "
                          f"{'>0':>5} {'EMPTY':>7}")
                elif etype == 'parity':
                    # Parity-adjusted: show leave value and penalty
                    lv = r.get('leave_value', 0.0)
                    bag_after = r.get('bag_after', '?')
                    penalty = r.get('parity_penalty', 0.0)
                    p_opp = r.get('p_opp_empties', 0.0)
                    print(f"{i:<3} {r['word']:<12} {pos:<12} {r['score']:>4} "
                          f"{tag:<7} {'':>6} {'':>6} "
                          f"{'lv=' + str(lv):>8} {r['net_equity']:>+7.1f} "
                          f"{'>' + str(bag_after):>5} {penalty:>+7.1f}")
                else:
                    # Plain 1-ply: show leave value
                    lv = r.get('leave_value', 0.0)
                    print(f"{i:<3} {r['word']:<12} {pos:<12} {r['score']:>4} "
                          f"{tag:<7} {'':>6} {'':>6} "
                          f"{'lv=' + str(lv):>8} {r['net_equity']:>+7.1f} "
                          f"{'':>5} {'':>7}")

            # Best move summary
            best = results[0]
            print(f"\n  Best: {best['word']} @ R{best['row']}C{best['col']} "
                  f"{best['direction']} ({best['score']} pts)")
            if best['eval_type'] == 'exhaust':
                print(f"    Exhaustive 3-ply: score {best['score']} "
                      f"- avg_opp {best['opp_avg']:.1f} "
                      f"+ avg_resp {best['your_resp_avg']:.1f} "
                      f"= net {best['net_equity']:+.1f}")
                print(f"    Evaluated {best['n_racks']} opponent rack assignments")
                print(f"    ** Empties bag -> you control endgame")
                if best['top_opp_responses']:
                    print(f"    Top opp responses:")
                    for resp in best['top_opp_responses'][:3]:
                        print(f"      {resp['word']}: {resp['count']}x "
                              f"(max {resp['max_score']} pts)")
            elif best['eval_type'] == 'parity':
                bag_after = best.get('bag_after', '?')
                penalty = best.get('parity_penalty', 0.0)
                p_opp = best.get('p_opp_empties', 0.0)
                print(f"    Parity-adjusted 1-ply: score {best['score']} + "
                      f"leave {best['leave_value']:+.1f} "
                      f"+ parity {penalty:+.1f} "
                      f"= {best['net_equity']:+.1f}")
                print(f"    !! Leaves {bag_after} in bag -> "
                      f"P(opp empties)={p_opp:.0%} -> penalty {penalty:+.1f}")
            else:
                print(f"    1-ply: score {best['score']} + "
                      f"leave {best['leave_value']:+.1f} "
                      f"= {best['net_equity']:+.1f}")

            # Store near-endgame recommendation for next play_move() call
            if results:
                self.last_analysis = {
                    'top3': [
                        {
                            'word': r['word'],
                            'row': r['row'],
                            'col': r['col'],
                            'dir': r['direction'],
                            'score': r['score'],
                            'equity': r.get('net_equity', 0),
                            'risk_eq': r.get('net_equity', 0),
                        }
                        for r in results[:3]
                    ]
                }

        except Exception as e:
            print(f"Near-endgame error: {e}")
            import traceback
            traceback.print_exc()

    def _calculate_probabilistic_risk(self, move: dict, unseen: dict) -> tuple:
        """Calculate risk using real word analysis of opened bonus squares."""
        from .real_risk import calculate_real_risk
        
        risk_str, expected_risk, max_damage, threats = calculate_real_risk(
            self.board, move, unseen, self.dictionary,
            BONUS_SQUARES, TILE_VALUES, self.blocked_cache
        )
        
        return risk_str, expected_risk, max_damage
    
    def _calculate_exhaustive_opp_risk(self, move: dict, unseen: dict) -> tuple:
        """Calculate risk by exhaustively enumerating all possible opponent racks
        and finding the best move for each. Used when bag is small enough (<=4).
        
        Returns (risk_str, expected_risk, max_damage) matching the interface
        of _calculate_probabilistic_risk.
        
        expected_risk = probability-weighted expected best opponent score
        max_damage = maximum opponent score across all possible racks
        """
        import math
        from collections import Counter
        from itertools import combinations
        
        total_unseen = sum(unseen.values())
        opp_hand = min(7, total_unseen)
        
        if opp_hand == 0:
            return "exhaust:0", 0.0, 0
        
        # Simulate our move on a copy
        sim_board = self.board.copy()
        word = move['word']
        row, col = move['row'], move['col']
        horiz = move['direction'] == 'H'
        sim_board.place_move(word, row, col, horiz)
        
        # Enumerate all unique opponent racks
        tiles = []
        for t, c in sorted(unseen.items()):
            tiles.extend([t] * c)
        
        total_combos = math.comb(total_unseen, opp_hand)
        
        seen_racks = set()
        weighted_best = 0.0
        max_score = 0
        
        blank_positions = self.state.blank_positions.copy()
        # Add blanks from our move
        blanks_used = move.get('blanks_used', [])
        for bi in blanks_used:
            r = row + (0 if horiz else bi)
            c = col + (bi if horiz else 0)
            blank_positions.append((r, c, word[bi]))
        
        for combo in combinations(range(total_unseen), opp_hand):
            rack_tiles = [tiles[i] for i in combo]
            rack_sorted = ''.join(sorted(rack_tiles))
            
            if rack_sorted in seen_racks:
                continue
            seen_racks.add(rack_sorted)
            
            # Count how many ways this rack can be drawn
            rc = Counter(rack_sorted)
            ways = 1
            for t, needed in rc.items():
                ways *= math.comb(unseen.get(t, 0), needed)
            prob = ways / total_combos
            
            # Generate all moves for this rack
            try:
                from .move_finder_c import find_all_moves_c, is_available
                if is_available():
                    opp_moves = find_all_moves_c(
                        sim_board, self.gaddag, rack_sorted,
                        board_blanks=blank_positions
                    )
                else:
                    raise ImportError("C accel not available")
            except ImportError:
                from .move_finder_gaddag import GADDAGMoveFinder
                finder = GADDAGMoveFinder(
                    sim_board, self.gaddag,
                    board_blanks=blank_positions
                )
                opp_moves = finder.find_all_moves(rack_sorted)
            
            if opp_moves:
                best = max(m['score'] for m in opp_moves)
                weighted_best += best * prob
                max_score = max(max_score, best)
        
        risk_str = f"exhaust:{len(seen_racks)}"
        return risk_str, weighted_best, max_score
    
    def _calculate_blocking_bonus(self, move: dict) -> tuple:
        """Calculate defensive bonus for blocking bonus squares.
        
        Returns (bonus_value, list_of_blocked_squares)
        
        Blocking values (based on typical damage prevented):
        - 3W: +15 pts (prevents 40-80 pt plays)
        - 2W: +8 pts (prevents 25-40 pt plays)
        - 3L: +3 pts (prevents high-value letter multipliers)
        - 2L: +2 pts (minor benefit)
        """
        word = move['word']
        row, col = move['row'], move['col']
        horiz = move['direction'] == 'H'
        
        BLOCKING_VALUES = {
            '3W': 15,
            '2W': 8,
            '3L': 3,
            '2L': 2
        }
        
        total_bonus = 0
        blocked = []
        
        for i in range(len(word)):
            if horiz:
                r, c = row, col + i
            else:
                r, c = row + i, col
            
            # Only count if this is a NEW tile (square currently empty)
            existing = self.board.get_tile(r, c)
            if existing is not None and existing != '.':
                continue  # Already occupied, not blocking anything new
            
            # Check if this square has a bonus
            if (r, c) in BONUS_SQUARES:
                bonus_type = BONUS_SQUARES[(r, c)]
                if bonus_type in BLOCKING_VALUES:
                    total_bonus += BLOCKING_VALUES[bonus_type]
                    blocked.append((r, c, bonus_type))
        
        return total_bonus, blocked
    
    def _get_opened_bonus_squares(self, move: dict) -> tuple:
        """Find which bonus squares a move opens up (not squares the word uses)."""
        word = move['word']
        row, col = move['row'], move['col']
        horiz = move['direction'] == 'H'
        
        opened_3w = []
        opened_2w = []
        
        # Build set of squares the word occupies (these are USED, not opened)
        word_squares = set()
        for i in range(len(word)):
            if horiz:
                word_squares.add((row, col + i))
            else:
                word_squares.add((row + i, col))
        
        # Check squares adjacent to the word
        for i in range(len(word)):
            if horiz:
                r, c = row, col + i
            else:
                r, c = row + i, col
            
            # Check all 4 adjacent squares
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 1 <= nr <= 15 and 1 <= nc <= 15:
                    # Skip if this square is part of the word itself
                    if (nr, nc) in word_squares:
                        continue
                    if not self.board.get_tile(nr, nc):
                        bonus = BONUS_SQUARES.get((nr, nc))
                        if bonus == '3W' and (nr, nc) not in opened_3w:
                            opened_3w.append((nr, nc))
                        elif bonus == '2W' and (nr, nc) not in opened_2w:
                            opened_2w.append((nr, nc))
        
        # Also check squares at word ends
        if horiz:
            # Before word
            if col > 1 and not self.board.get_tile(row, col - 1):
                bonus = BONUS_SQUARES.get((row, col - 1))
                if bonus == '3W' and (row, col - 1) not in opened_3w:
                    opened_3w.append((row, col - 1))
                elif bonus == '2W' and (row, col - 1) not in opened_2w:
                    opened_2w.append((row, col - 1))
            # After word
            end_col = col + len(word)
            if end_col <= 15 and not self.board.get_tile(row, end_col):
                bonus = BONUS_SQUARES.get((row, end_col))
                if bonus == '3W' and (row, end_col) not in opened_3w:
                    opened_3w.append((row, end_col))
                elif bonus == '2W' and (row, end_col) not in opened_2w:
                    opened_2w.append((row, end_col))
        else:
            # Before word
            if row > 1 and not self.board.get_tile(row - 1, col):
                bonus = BONUS_SQUARES.get((row - 1, col))
                if bonus == '3W' and (row - 1, col) not in opened_3w:
                    opened_3w.append((row - 1, col))
                elif bonus == '2W' and (row - 1, col) not in opened_2w:
                    opened_2w.append((row - 1, col))
            # After word
            end_row = row + len(word)
            if end_row <= 15 and not self.board.get_tile(end_row, col):
                bonus = BONUS_SQUARES.get((end_row, col))
                if bonus == '3W' and (end_row, col) not in opened_3w:
                    opened_3w.append((end_row, col))
                elif bonus == '2W' and (end_row, col) not in opened_2w:
                    opened_2w.append((end_row, col))
        
        return opened_3w, opened_2w
    
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
    
    def play_move(self, word: str, row: int, col: int, horizontal: bool,
                  is_opponent: bool = False, rack: str = None,
                  new_rack: str = None) -> Tuple[bool, int]:
        """Play a move. Returns (success, score).

        Args:
            rack: Pre-move rack for tile validation. If None, skips validation.
            new_rack: Post-move rack (after drawing). If provided, the rack is
                set to this value after the move instead of simulating a draw.
                This supports the assisted-play workflow where the user reports
                their new rack after playing on the real board.
        """
        word = word.upper()

        # Capture rack BEFORE the move (for enriched record)
        rack_before = self.state.your_rack if not is_opponent else None

        # Validate word
        if len(word) > 2 and not self.dictionary.is_valid(word):
            print(f"[X] '{word}' is not a valid word!")
            return False, 0

        # Get tiles used
        move_dict = {'word': word, 'row': row, 'col': col, 'direction': 'H' if horizontal else 'V'}
        tiles_used = self._tiles_used(move_dict)

        # Check rack and detect blanks
        blank_word_indices = []
        if rack:
            rack_list = list(rack)
            new_indices = self._get_new_tile_word_indices(move_dict)
            used_idx = 0
            for t in tiles_used:
                if t in rack_list:
                    rack_list.remove(t)
                elif '?' in rack_list:
                    rack_list.remove('?')
                    # Map this used tile to its word index
                    if used_idx < len(new_indices):
                        blank_word_indices.append(new_indices[used_idx])
                else:
                    print(f"[X] Missing tile '{t}' for word '{word}'!")
                    return False, 0
                used_idx += 1

        # Calculate score
        try:
            score, crosswords = calculate_move_score(
                self.board, word, row, col, horizontal,
                blanks_used=blank_word_indices,
                board_blanks=self.state.blank_positions
            )
        except Exception as e:
            print(f"[X] Invalid placement: {e}")
            return False, 0

        # Bingo bonus -- already included in score from calculate_move_score()
        if len(tiles_used) == 7:
            print("[WIN] BINGO! (40 bonus included in score)")

        # Compute bag count BEFORE placing the move (for final_turns tracking)
        _trk_pre = TileTracker()
        _trk_pre.sync_with_board(self.board, your_rack=self.state.your_rack or "",
                                 blanks_in_rack=(self.state.your_rack or "").count('?'),
                                 blank_positions=self.state.blank_positions)
        bag_before_play = _trk_pre.get_bag_count()

        # Place word on board
        self.board.place_word(word, row, col, horizontal)

        # Update score and track drawn tiles
        drawn_tiles = None
        if is_opponent:
            self.state.opp_score += score
        else:
            self.state.your_score += score
            # Update rack
            if new_rack is not None:
                # User provided post-draw rack directly (assisted play workflow)
                # Infer drawn tiles by comparing old rack (minus used) to new rack
                if self.state.your_rack:
                    old_remaining = list(self.state.your_rack)
                    for t in tiles_used:
                        if t in old_remaining:
                            old_remaining.remove(t)
                        elif '?' in old_remaining:
                            old_remaining.remove('?')
                    # drawn = new_rack minus old_remaining
                    new_list = list(new_rack)
                    for t in old_remaining:
                        if t in new_list:
                            new_list.remove(t)
                    drawn_tiles = ''.join(new_list)
                self.state.your_rack = new_rack
            elif self.state.your_rack:
                rack_list = list(self.state.your_rack)
                for t in tiles_used:
                    if t in rack_list:
                        rack_list.remove(t)
                    elif '?' in rack_list:
                        rack_list.remove('?')
                # Draw new tiles
                new_tiles = self.draw_tiles(len(tiles_used))
                drawn_tiles = ''.join(new_tiles)
                rack_list.extend(new_tiles)
                self.state.your_rack = ''.join(rack_list)

        # Record blank positions
        for wi in blank_word_indices:
            if horizontal:
                br, bc = row, col + wi
            else:
                br, bc = row + wi, col
            self.state.blank_positions.append((br, bc, word[wi]))

        # Compute bag count (tiles visible to opponent: bag minus their rack)
        bag_count = max(0, len(self.bag) - RACK_SIZE)

        # Build engine recommendation record (from last analyze())
        engine_rec = None
        if not is_opponent and self.last_analysis:
            top_word = self.last_analysis['top3'][0]['word'] if self.last_analysis.get('top3') else None
            engine_rec = {
                'top3': self.last_analysis.get('top3', []),
                'followed': (word.upper() == top_word.upper()) if top_word else None,
            }
            self.last_analysis = None  # Clear after use

        # Build enriched move record
        player = self.state.opponent_name if is_opponent else 'me'
        enriched = {
            'word': word,
            'row': row,
            'col': col,
            'dir': 'H' if horizontal else 'V',
            'player': player,
            'score': score,
            'bag': bag_count,
            'blanks': blank_word_indices,
            'cumulative': [self.state.your_score, self.state.opp_score],
            'rack': rack_before,
            'drawn': drawn_tiles,
            'note': 'bingo' if len(tiles_used) >= RACK_SIZE else '',
            'timestamp': datetime.now().isoformat(),
            'engine': engine_rec,
            'engine_version': __version__,
            'win_pct': None,
            'nyt': None,
        }
        self.state.board_moves.append(enriched)

        # Track final_turns_remaining
        _trk_post = TileTracker()
        _trk_post.sync_with_board(self.board, your_rack=self.state.your_rack or "",
                                  blanks_in_rack=(self.state.your_rack or "").count('?'),
                                  blank_positions=self.state.blank_positions)
        bag_after_play = _trk_post.get_bag_count()
        if bag_before_play > 0 and bag_after_play == 0:
            # Bag just emptied: both players get one final turn each = 2 turns remain
            self.state.final_turns_remaining = 2
        elif bag_before_play == 0 and self.state.final_turns_remaining is not None:
            self.state.final_turns_remaining = max(0, self.state.final_turns_remaining - 1)

        # Toggle turn: after your move it's opponent's turn and vice versa
        self.state.is_your_turn = is_opponent  # If opponent played, now it's your turn (True); if you played, their turn (False)

        self.state.updated_at = datetime.now().isoformat()

        # Update blocked square cache and invalidate threats cache
        self.blocked_cache.update_after_move(self.board, word, row, col, horizontal)
        self._threats_cache = None

        d = 'H' if horizontal else 'V'
        who = self.state.opponent_name if is_opponent else "You"
        print(f"[OK] {who} played {word} at R{row}C{col} {d} for {score} points!")

        # Show existing threats (skip when opponent has no turns left)
        # Note: is_your_turn hasn't been toggled yet, so use is_opponent param
        ftr = self.state.final_turns_remaining
        if ftr is not None and ftr <= 0:
            print("\n[ENDGAME] Game over -- no turns remaining.")
        elif ftr == 2 and not is_opponent:
            # User just played and bag emptied -- both players get final turns
            print("\n[ENDGAME] Bag just emptied -- opponent plays next, then you get final move.")
            self.show_existing_threats(top_n=5)
        elif ftr == 2 and is_opponent:
            # Opponent just played and bag emptied -- both players get final turns
            print("\n[ENDGAME] Bag just emptied -- you play next, then opponent gets final move.")
            self.show_existing_threats(top_n=5)
        elif ftr == 1 and not is_opponent:
            # User just played their final turn -- opponent gets last move
            print("\n[ENDGAME] Your final turn done -- opponent gets last move.")
            self.show_existing_threats(top_n=5)
        elif ftr == 1 and is_opponent:
            # Opponent just played -- user's final move, no opp response
            print("\n[ENDGAME] Bag empty -- your final move (no opponent response). Use 'analyze' for 1-ply ranking.")
        elif bag_after_play == 0:
            print("\n[ENDGAME] Bag empty -- use 'analyze' for endgame solver.")
        else:
            self.show_existing_threats(top_n=5)

        # Auto-save if enabled
        self._auto_save()

        # Check if game should be archived
        self._check_archive()

        return True, score

    # -----------------------------------------------------------------
    # Opponent move validation (V17.2)
    # -----------------------------------------------------------------
    def _validate_opponent_move(self, word: str, row: int, col: int,
                                horizontal: bool):
        """Validate an opponent move BEFORE placing it on the board.

        Checks bounds, tile conflicts, connectivity, main-word validity,
        and cross-word validity.  Returns (is_valid, errors, warnings).
        All checks are read-only — board state is never modified.
        """
        errors = []
        warnings = []
        new_tile_positions = []
        connects = False
        covers_center = False

        # --- Check 1 & 2: bounds + tile conflicts ---
        for i, letter in enumerate(word):
            r = row + (0 if horizontal else i)
            c = col + (i if horizontal else 0)

            # Bounds
            if r < 1 or r > BOARD_SIZE or c < 1 or c > BOARD_SIZE:
                errors.append(f"Out of bounds at R{r}C{c}")
                continue

            if r == CENTER_ROW and c == CENTER_COL:
                covers_center = True

            existing = self.board.get_tile(r, c)
            if existing is not None:
                if existing != letter:
                    errors.append(
                        f"Tile conflict at R{r}C{c}: board has '{existing}', "
                        f"word has '{letter}'")
                else:
                    connects = True          # overlapping existing tile
            else:
                new_tile_positions.append((r, c))
                if self.board.has_adjacent_tile(r, c):
                    connects = True

        if errors:
            return False, errors, warnings

        # --- Check 3: must place at least one new tile ---
        if not new_tile_positions:
            errors.append("No new tiles placed — word is already on the board")
            return False, errors, warnings

        # --- Check 4: connectivity ---
        if self.board.is_board_empty():
            if not covers_center:
                errors.append(
                    f"First move must cover the center square "
                    f"(R{CENTER_ROW}C{CENTER_COL})")
        else:
            if not connects:
                pos_str = ", ".join(f"R{r}C{c}" for r, c in new_tile_positions)
                errors.append(
                    f"Move does not connect to existing tiles\n"
                    f"        New tile positions: {pos_str}")

        if errors:
            return False, errors, warnings

        # --- Check 5: main word validity ---
        if not self.gaddag.is_word(word):
            errors.append(f"'{word}' is not a valid dictionary word")

        # --- Check 6: cross-word validity ---
        crosswords = find_crosswords(
            self.board, word, row, col, horizontal, new_tile_positions)
        for cw in crosswords:
            cw_word = cw['word']
            cw_dir = 'H' if cw['horizontal'] else 'V'
            if not self.gaddag.is_word(cw_word):
                errors.append(
                    f"Invalid cross-word '{cw_word}' at "
                    f"R{cw['row']}C{cw['col']} {cw_dir}")

        # --- NYT warnings (non-blocking) ---
        if is_nyt_curated(word):
            warnings.append(f"'{word}' is NYT-curated (may not be in WWF)")
        for cw in crosswords:
            if is_nyt_curated(cw['word']):
                warnings.append(
                    f"Cross-word '{cw['word']}' is NYT-curated")

        return len(errors) == 0, errors, warnings

    def record_opponent_move(self, word: str, row: int, col: int, horizontal: bool,
                             score: int, new_opp_score: int = None, new_bag_count: int = None,
                             blanks: List[str] = None, force: bool = False):
        """Record opponent's move with validation.

        Args:
            word: Word played
            row, col: Starting position (1-indexed)
            horizontal: True if horizontal, False if vertical
            score: Points scored for this move
            new_opp_score: Opponent's total score after move (for validation)
            new_bag_count: Tiles remaining in bag after move (for validation)
            blanks: List of letters in the word that are blanks, e.g. ['M'] if M is blank
            force: If True, accept even if validation fails (opp! command)

        Validation checks:
            - Board connectivity, word validity, cross-words (pre-placement)
            - Calculated score vs reported score (catches unreported blanks)
            - Expected opp total vs reported total
            - Expected bag count vs reported count
        """
        word = word.upper()
        blanks = [b.upper() for b in (blanks or [])]

        # --- Move validation gate (V17.2) ---
        direction = 'H' if horizontal else 'V'
        valid, errors, warnings = self._validate_opponent_move(
            word, row, col, horizontal)

        for w in warnings:
            print(f"  [NOTE] {w}")

        if not valid:
            if force:
                print(f"  [FORCED] Accepting {word} at R{row}C{col} "
                      f"{direction} despite errors:")
                for e in errors:
                    print(f"    [X] {e}")
            else:
                print(f"\n  [INVALID MOVE] {word} at R{row}C{col} {direction}")
                for e in errors:
                    print(f"    [X] {e}")
                print(f"\n    To force-accept anyway, use:")
                print(f"    opp! {word.lower()} {row} {col} {direction} {score}")
                print()
                return

        from .scoring import calculate_move_score
        from .config import TILE_VALUES

        # Identify which tile positions are new (placed from rack)
        new_tile_indices = []
        for i, letter in enumerate(word):
            if horizontal:
                check_row, check_col = row, col + i
            else:
                check_row, check_col = row + i, col
            if self.board.is_empty(check_row, check_col):
                new_tile_indices.append(i)
        
        # --- Score calculation helper ---
        def _calc_score_with_blanks(blank_indices):
            """Calculate score with specific indices treated as blanks."""
            try:
                s, _ = calculate_move_score(
                    self.board, word, row, col, horizontal,
                    blanks_used=blank_indices,
                    board_blanks=self.state.blank_positions
                )
                return s
            except Exception:
                return None
        
        # --- Blank detection logic ---
        detected_blanks = []  # list of word indices detected as blanks
        
        if blanks:
            # User specified blanks explicitly — convert letters to indices
            blanks_remaining = list(blanks)
            for i in new_tile_indices:
                if word[i] in blanks_remaining:
                    detected_blanks.append(i)
                    blanks_remaining.remove(word[i])
            
            expected_score = _calc_score_with_blanks(detected_blanks)
            if expected_score is not None and expected_score != score:
                print(f"\nWARNING:  SCORE MISMATCH (with specified blanks)")
                print(f"    Reported: {score} pts, Calculated: {expected_score} pts")
        else:
            # No blanks specified — check if score matches without blanks
            no_blank_score = _calc_score_with_blanks([])
            
            if no_blank_score is not None and no_blank_score != score:
                # Score doesn't match — try to auto-detect which tile is blank
                # Only new tiles from rack can be blanks, and only tiles with
                # point value > 0 would cause a score difference
                candidates = []
                for i in new_tile_indices:
                    if TILE_VALUES.get(word[i], 0) > 0:
                        test_score = _calc_score_with_blanks([i])
                        if test_score == score:
                            candidates.append((i, word[i]))
                
                # Also try 2-blank combinations if single didn't match
                if not candidates and len(new_tile_indices) >= 2:
                    for j in range(len(new_tile_indices)):
                        for k in range(j + 1, len(new_tile_indices)):
                            i1, i2 = new_tile_indices[j], new_tile_indices[k]
                            if TILE_VALUES.get(word[i1], 0) > 0 or TILE_VALUES.get(word[i2], 0) > 0:
                                test_score = _calc_score_with_blanks([i1, i2])
                                if test_score == score:
                                    candidates.append(((i1, i2), f"{word[i1]},{word[i2]}"))
                
                if len(candidates) == 1:
                    # Exactly one blank assignment makes the score work
                    idx_or_pair, letter_info = candidates[0]
                    if isinstance(idx_or_pair, tuple):
                        detected_blanks = list(idx_or_pair)
                        print(f"\n[?] AUTO-DETECTED BLANKS: {word[idx_or_pair[0]]} and {word[idx_or_pair[1]]}")
                        print(f"    Score {no_blank_score} -> {score} (only valid blank assignment)")
                    else:
                        detected_blanks = [idx_or_pair]
                        print(f"\n[?] AUTO-DETECTED BLANK: {word[idx_or_pair]} (position {idx_or_pair + 1} in word)")
                        print(f"    Score {no_blank_score} -> {score} (only valid blank assignment)")
                elif len(candidates) > 1:
                    # Multiple possible blank assignments
                    print(f"\nWARNING:  SCORE MISMATCH -- possible blank detected!")
                    print(f"    Reported: {score} pts, Without blanks: {no_blank_score} pts")
                    print(f"    Possible blanks (each makes score match):")
                    for idx_or_pair, letter_info in candidates:
                        if isinstance(idx_or_pair, tuple):
                            print(f"      blanks=['{word[idx_or_pair[0]]}', '{word[idx_or_pair[1]]}']")
                        else:
                            pos = idx_or_pair
                            if horizontal:
                                print(f"      blanks=['{word[pos]}']  (R{row}C{col + pos})")
                            else:
                                print(f"      blanks=['{word[pos]}']  (R{row + pos}C{col})")
                    print(f"    TIP: Re-record with blanks=['X'] to confirm")
                else:
                    # No single/double blank makes the score match
                    print(f"\nWARNING:  SCORE MISMATCH!")
                    print(f"    Reported: {score} pts, Calculated: {no_blank_score} pts")
                    print(f"    No blank assignment found -- check move details")
            else:
                expected_score = no_blank_score
        
        # Compute bag count BEFORE placing the move (for final_turns tracking)
        _trk_pre = TileTracker()
        _trk_pre.sync_with_board(self.board, your_rack=self.state.your_rack or "",
                                 blanks_in_rack=(self.state.your_rack or "").count('?'),
                                 blank_positions=self.state.blank_positions)
        bag_before = _trk_pre.get_bag_count()

        # Place the word on board
        self.board.place_word(word, row, col, horizontal)

        # Record blank positions (from explicit blanks or auto-detected)
        for i in detected_blanks:
            if horizontal:
                blank_row, blank_col = row, col + i
            else:
                blank_row, blank_col = row + i, col
            self.state.blank_positions.append((blank_row, blank_col, word[i]))
            print(f"    [>] Blank recorded at R{blank_row}C{blank_col} = {word[i]}")

        # Update opponent score
        old_opp_score = self.state.opp_score
        self.state.opp_score += score

        # Validate opponent total score
        if new_opp_score is not None:
            if self.state.opp_score != new_opp_score:
                print(f"\nWARNING:  OPP SCORE MISMATCH!")
                print(f"    Expected: {old_opp_score} + {score} = {self.state.opp_score}")
                print(f"    Reported: {new_opp_score}")
                print(f"    Using reported score.")
                self.state.opp_score = new_opp_score

        # Validate bag count
        if new_bag_count is not None:
            # Sync tracker to get current count
            tracker = TileTracker()
            tracker.sync_with_board(self.board, self.state.your_rack, 0, self.state.blank_positions)

            # Expected: unseen - 7 (opp rack) = bag
            unseen = tracker.get_unseen_count()
            expected_bag = unseen - 7  # Assuming opp has 7 tiles

            if expected_bag != new_bag_count:
                diff = expected_bag - new_bag_count
                print(f"\nWARNING:  BAG COUNT MISMATCH!")
                print(f"    Expected: ~{expected_bag} tiles")
                print(f"    Reported: {new_bag_count} tiles")
                print(f"    Difference: {diff} tiles")
                if diff > 0:
                    print(f"    TIP: Missing tiles - check if blanks were reported")

        # Compute bag count for enriched record
        bag_count = max(0, len(self.bag) - RACK_SIZE)

        # Build enriched move record (after score/bag updates for accurate cumulative)
        enriched = {
            'word': word,
            'row': row,
            'col': col,
            'dir': 'H' if horizontal else 'V',
            'player': 'opp',
            'score': score,
            'bag': new_bag_count if new_bag_count is not None else bag_count,
            'blanks': detected_blanks,
            'cumulative': [self.state.your_score, self.state.opp_score],
            'rack': None,       # Unknown for opponent
            'drawn': None,      # Unknown for opponent
            'note': 'bingo' if len(new_tile_indices) >= RACK_SIZE else '',
            'timestamp': datetime.now().isoformat(),
            'engine': None,     # No engine analysis for opponent moves
            'engine_version': __version__,
            'win_pct': None,
            'nyt': None,
        }
        self.state.board_moves.append(enriched)

        # Track final_turns_remaining
        _trk_post = TileTracker()
        _trk_post.sync_with_board(self.board, your_rack=self.state.your_rack or "",
                                  blanks_in_rack=(self.state.your_rack or "").count('?'),
                                  blank_positions=self.state.blank_positions)
        bag_after = _trk_post.get_bag_count()
        if bag_before > 0 and bag_after == 0:
            # Bag just emptied: both players get one final turn each = 2 turns remain
            self.state.final_turns_remaining = 2
        elif bag_before == 0 and self.state.final_turns_remaining is not None:
            self.state.final_turns_remaining = max(0, self.state.final_turns_remaining - 1)

        self.state.is_your_turn = True
        self.state.updated_at = datetime.now().isoformat()

        # Update blocked square cache and invalidate threats cache
        self.blocked_cache.update_after_move(self.board, word, row, col, horizontal)
        self._threats_cache = None

        print(f"\n[NOTE] Recorded: {self.state.opponent_name} played {word} for {score} pts")
        print(f"   Score: You {self.state.your_score} - {self.state.opponent_name} {self.state.opp_score}")

        # Auto-analyze after opponent move: always run full analysis so the
        # engine recommendation is available before any move is played.
        # The analyze() method already shows threats as its first step.
        ftr = self.state.final_turns_remaining
        if ftr is not None and ftr <= 0:
            print("\n[ENDGAME] Game over -- no turns remaining.")
        elif self.state.your_rack:
            # We have a rack — run full analysis automatically
            print("\n[AUTO-ANALYZE] Running engine analysis...")
            try:
                self.analyze()
            except Exception as e:
                print(f"[AUTO-ANALYZE] Analysis failed: {e}")
                # Fall back to showing threats only
                self.show_existing_threats(top_n=5)
        else:
            # No rack set yet — just show threats
            if ftr == 2 and self.state.is_your_turn:
                print("\n[ENDGAME] Bag just emptied -- you play next, then opponent gets final move.")
                self.show_existing_threats(top_n=5)
            elif ftr == 1 and self.state.is_your_turn:
                print("\n[ENDGAME] Bag empty -- your final move (no opponent response).")
            elif ftr == 1 and not self.state.is_your_turn:
                print("\n[ENDGAME] Your final turn done -- opponent gets last move.")
                self.show_existing_threats(top_n=5)
            elif bag_after == 0:
                print("\n[ENDGAME] Bag empty.")
            else:
                self.show_existing_threats(top_n=5)
            print("\n[TIP] Set your rack with 'rack LETTERS' then 'analyze' for full engine analysis.")

        # Auto-save if enabled
        self._auto_save()

        # Check if game should be archived
        self._check_archive()
    
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
        
        self.state.bag = self.bag
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
            bag=[],
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
                 your_score: int = None, opp_score: int = None):
        """Complete a game: update final scores, archive, and clear slot.

        Args:
            slot: Slot number (defaults to current_slot)
            result: 'win', 'loss', or 'tie' (auto-detected from scores if omitted)
            your_score: Final score (updates game state if provided)
            opp_score: Final opponent score (updates game state if provided)
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

        # Save final state then archive
        from .game_library import save_active, archive_completed, load_index, save_index
        save_active(game_id, game)

        # Archive (appends to archive.jsonl, deletes active JSON)
        archive_completed(game_id, game)

        # Clear slot in index
        index = load_index()
        index['slots'][str(slot)] = None
        save_index(index)

        # Clear in-memory slot
        self.games[slot] = None

        print(f"[OK] Game over: {result.upper()} {game.state.your_score}-{game.state.opp_score} "
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
        except Exception as e:
            print(f"[X] Failed to load: {e}")
    
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
                except:
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
            print("          play WORD R C H/V | opp WORD R C H/V SCORE | save | reset N | back")
            
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
                except:
                    print("Usage: slot N")
            
            elif action == 'new' and len(parts) >= 2:
                try:
                    slot = int(parts[1])
                    name = ' '.join(parts[2:]) if len(parts) > 2 else "Opponent"
                    self.new_game(slot, name)
                except:
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
                    except Exception as e:
                        print(f"Error: {e}")
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
                    except Exception as e:
                        print(f"Error: {e}")
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
                except Exception as e:
                    print(f"Error: {e}")
                    print("Usage: end YOUR_SCORE OPP_SCORE [win/loss/tie]")

            elif action == 'save':
                if game:
                    game.save()
                else:
                    print("No game in current slot")

            elif action == 'reset' and len(parts) >= 2:
                try:
                    self.reset_slot(int(parts[1]))
                except:
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
