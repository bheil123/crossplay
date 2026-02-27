"""Game analysis mixin -- analysis, risk, MC display, and endgame methods.

Extracted from game_manager.py to reduce file size. Mixed into Game
via multiple inheritance. All methods access shared Game instance
state through self.
"""

import time
import math
import random
import traceback
from collections import Counter
from itertools import combinations
from typing import List, Optional

from .board import Board, tiles_used as _board_tiles_used
from .move_finder_gaddag import GADDAGMoveFinder
from .config import (
    TILE_DISTRIBUTION, TILE_VALUES, BONUS_SQUARES, RACK_SIZE, BOARD_SIZE,
    RISK_TIME_BUDGET_EXHAUSTIVE, RISK_TIME_BUDGET_NORMAL, MC_TIME_BUDGET,
    THREE_PLY_TIME_BUDGET, ANALYSIS_MIN_CANDIDATES_MC, ANALYSIS_MIN_CANDIDATES_RISK,
    ANALYSIS_ENDGAME_THREATS, EXCHANGE_EQUITY_THRESHOLD, EXCHANGE_TOP_CANDIDATES,
    EXCHANGE_QUICK_MC_SIMS, BINGO_SCORE_THRESHOLD, BINGO_BLOCKING_DELTA,
    LATE_GAME_UNSEEN_THRESHOLD, CATCHUP_DEFICIT_THRESHOLD,
    HIGH_VARIANCE_DEFICIT_THRESHOLD, POWER_TILE_PROB_THRESHOLD,
    BLOCKING_BONUSES,
    BLANK_3PLY_MIN_BLANKS, BLANK_3PLY_MIN_BAG,
    BLANK_3PLY_TOP_N, BLANK_3PLY_FORCE_SAVERS,
)
from .nyt_filter import nyt_warning, is_nyt_curated
from .analysis_lock import acquire_lock, release_lock
from .leave_eval import evaluate_leave
from .log import get_logger
from .mc_eval import mc_evaluate_2ply

logger = get_logger(__name__)


class GameAnalysisMixin:
    """Analysis methods for the Game class.
    
    Mixed into Game via multiple inheritance. All methods access
    shared Game instance state through self.
    """
    
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
            _, unseen, _ = self._get_tile_context()
            
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
            self.show_existing_threats(top_n=ANALYSIS_ENDGAME_THREATS)

        # Move generation — prefer C-accelerated, fallback to Python
        try:
            from .move_finder_c import find_all_moves_c, is_available
            if is_available():
                t_mg = time.time()
                moves = find_all_moves_c(self.board, self.gaddag, rack,
                                          board_blanks=self.state.blank_positions)
                t_mg = time.time() - t_mg
                logger.debug("%d moves found in %dms (C accel)", len(moves), t_mg*1000)
            else:
                raise ImportError("C not available")
        except ImportError:
            finder = GADDAGMoveFinder(self.board, self.gaddag, board_blanks=self.state.blank_positions)
            t_mg = time.time()
            moves = finder.find_all_moves(rack)
            t_mg = time.time() - t_mg
            logger.debug("%d moves found in %dms (Python)", len(moves), t_mg*1000)
        
        if not moves:
            print("No valid moves found!")
            return []
        
        # Build tile tracker to know what's unseen
        _, unseen, bag_size = self._get_tile_context(rack=rack)
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
        candidates = preliminary[:max(ANALYSIS_MIN_CANDIDATES_MC, top_n * 3)]  # Ensure enough for MC stage
        
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
        min_to_analyze = max(top_n + 5, ANALYSIS_MIN_CANDIDATES_RISK)  # Analyze enough for both display and MC candidates (up to N=111)
        risk_time_budget = RISK_TIME_BUDGET_EXHAUSTIVE if bag_size <= 5 else RISK_TIME_BUDGET_NORMAL
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
                    if power_prob > POWER_TILE_PROB_THRESHOLD:  # Only show if >5%
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
            if best_play_equity < EXCHANGE_EQUITY_THRESHOLD:
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
        N_QUICK = EXCHANGE_QUICK_MC_SIMS
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
        return options[:EXCHANGE_TOP_CANDIDATES]
    
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
        mc_budget = MC_TIME_BUDGET
        
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
            
            logger.debug("MC 2-ply completed in %dms (%d total sims)", t_mc*1000, k_sims * len(results))
            
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

                if not is_exch and baseline_bingo and baseline_bingo['score'] >= BINGO_SCORE_THRESHOLD:
                    if opp_avg < baseline_bingo['score'] - BINGO_BLOCKING_DELTA:
                        blocks_bingo = True
                        blocking_moves.append((m, baseline_bingo))

                # Markers
                bingo_marker = ""
                if is_exch:
                    bingo_marker = "[EXCH]"
                else:
                    opp_word = m.get('opp_word', '')
                    if len(opp_word) >= 7 and m['mc_max_opp'] >= BINGO_SCORE_THRESHOLD:
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
                    ],
                    'mc_top': [
                        {
                            'word': r.get('word', 'EXCHANGE'),
                            'row': r.get('row', 0),
                            'col': r.get('col', 0),
                            'dir': r.get('direction', '-'),
                            'score': r['score'],
                            'equity': r.get('total_equity', 0),
                            'risk_eq': r.get('risk_adj_equity', r.get('total_equity', 0)),
                            'is_exchange': r.get('is_exchange', False),
                        }
                        for r in results[:5]
                    ],
                }

            # Bingo blocking analysis
            if baseline_bingo and baseline_bingo['score'] >= BINGO_SCORE_THRESHOLD:
                self._show_bingo_blocking_analysis(rack, unseen_str, baseline_bingo, results)
            
            # Catch-up advice for late game when far behind
            self._show_catchup_advice(unseen, total_unseen, results)

            # Blank strategy 3-ply: when rack has 2+ blanks, compare spend vs save
            bag_size = total_unseen - 7  # unseen minus opponent rack
            blanks_in_rack = rack.count('?')
            if blanks_in_rack >= BLANK_3PLY_MIN_BLANKS and bag_size > BLANK_3PLY_MIN_BAG:
                self._show_blank_3ply_analysis(rack, unseen_str, total_unseen, results, bag_size)

            # 3-ply analysis for late-to-mid game (C extension makes this fast)
            # Budget: 20s total minus MC time already spent
            if bag_size <= 21 and bag_size >= 0:
                remaining_budget = max(2.0, THREE_PLY_TIME_BUDGET - t_mc)
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
        if len(unseen_str) > LATE_GAME_UNSEEN_THRESHOLD:
            return None

        # Use C extension for speed (13x faster with large tile sets)
        try:
            from .move_finder_c import find_all_moves_c
            opp_moves = find_all_moves_c(
                self.board, self.gaddag, unseen_str,
                board_blanks=self.state.blank_positions)
        except Exception:
            finder = GADDAGMoveFinder(self.board, self.gaddag, board_blanks=self.state.blank_positions)
            opp_moves = finder.find_all_moves(unseen_str)
        
        # Find best bingo (7+ tiles used)
        bingos = []
        for m in opp_moves:
            # Count tiles actually placed (word length minus tiles already on board)
            tiles_placed = len(m['word'])  # Simplified - actual calculation would check hooks
            if m['score'] >= BINGO_SCORE_THRESHOLD:  # Likely a bingo (min: 7x1pt + 40 bonus)
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

        # Sort by score and limit candidates (opponent movegen is expensive)
        blocking_candidates.sort(key=lambda m: -m['score'])
        max_eval = 8  # only display 8 anyway
        blocking_candidates = blocking_candidates[:max_eval]

        # Evaluate each blocking move with 2-ply (time-budgeted)
        print(f"\nEvaluating top {len(blocking_candidates)} blocking moves...")
        t_block_start = time.time()
        block_budget = 15.0  # seconds

        # Try C extension for opponent movegen (13x faster than Python)
        try:
            from .move_finder_c import find_all_moves_c as _block_find_c
            _use_c_block = True
        except Exception:
            _use_c_block = False

        blocking_results = []
        for move in blocking_candidates:
            if time.time() - t_block_start > block_budget:
                print(f"  (time budget reached, {len(blocking_results)} evaluated)")
                break

            horiz = move['direction'] == 'H'
            placed = self.board.place_move(move['word'], move['row'], move['col'], horiz)

            # Find opponent's new best (C extension preferred for speed)
            if _use_c_block:
                opp_moves = _block_find_c(
                    self.board, self.gaddag, unseen_str,
                    board_blanks=self.state.blank_positions)
            else:
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
        
        # Compare best blocker to best overall MC result
        if blocking_results:
            best_blocker = blocking_results[0]
            best_overall = results[0] if results else None

            if best_overall and not best_overall.get('is_exchange'):
                opp_avg = best_overall.get('mc_avg_opp', 0)
                overall_net = best_overall['score'] - opp_avg
                blocker_net = best_blocker['net']

                print(f"\n[INFO] Comparison:")
                print(f"   Best MC: {best_overall['word']} ({best_overall['score']} pts)")
                print(f"      -> Avg opp: {opp_avg:.0f} pts  -> Net: {overall_net:+.0f}")
                print(f"   Best BLOCKER: {best_blocker['word']} ({best_blocker['score']} pts)")
                print(f"      -> Opp plays {best_blocker['opp_word']} ({best_blocker['opp_score']} pts)")
                print(f"      -> Net: {blocker_net:+d}")

                if blocker_net > overall_net:
                    advantage = blocker_net - overall_net
                    print(f"\n   --> BLOCKING IS BETTER! {best_blocker['word']} gains {advantage:.0f} pts net")
                    print(f"      Blocks {baseline_bingo['score']} pt bingo, opp limited to {best_blocker['opp_score']} pts")
                else:
                    disadvantage = overall_net - blocker_net
                    print(f"\n   OK Non-blocking play is better by {disadvantage:.0f} pts net")
    
    def _show_catchup_advice(self, unseen: dict, total_unseen: int, results: list):
        """Show strategic advice when far behind in late game."""
        from .power_tiles import get_power_tiles_in_pool, prob_draw_any_power_tile
        
        score_diff = self.state.your_score - self.state.opp_score
        bag_size = total_unseen - 7  # Unseen minus opponent's rack
        
        # Only show if: behind by 50+, unseen <= 21 (late game), power tiles available
        if score_diff >= -CATCHUP_DEFICIT_THRESHOLD:
            return
        if total_unseen > LATE_GAME_UNSEEN_THRESHOLD:
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
        if score_diff <= -HIGH_VARIANCE_DEFICIT_THRESHOLD:
            print("   WARNING:  DOWN 100+: High variance is your friend!")
            print("      Consider plays that draw more tiles for better power tile odds.")
            print("      Keeping a great leave matters less when you need big swings.")
        elif score_diff <= -CATCHUP_DEFICIT_THRESHOLD:
            print("   [INFO] DOWN 50-100: Balance scoring with power tile chances.")
            print("      Plays that empty your rack give more lottery tickets.")
    
    def _show_blank_3ply_analysis(self, rack: str, unseen_str: str, total_unseen: int,
                                   mc_results: list, bag_size: int):
        """Show 3-ply blank strategy comparison when rack has 2+ blanks.

        Compares spending both blanks now vs saving one for next turn.
        Runs after MC 2-ply to provide supplementary 3-ply insight.
        """
        from .mc_eval import mc_evaluate_3ply_blanks, _get_leave
        from .board import tiles_used as _board_tiles_used

        blanks_total = rack.count('?')

        # Select candidates: top N from MC + force-include blank-savers
        top_mc = mc_results[:BLANK_3PLY_TOP_N]
        top_mc_keys = {(m['word'], m['row'], m['col']) for m in top_mc}

        blank_savers = []
        for m in mc_results:
            if len(blank_savers) >= BLANK_3PLY_FORCE_SAVERS:
                break
            key = (m['word'], m['row'], m['col'])
            if key not in top_mc_keys and '?' in m.get('leave', ''):
                blank_savers.append(m)

        # If no blank-savers in MC results, generate them from the move finder
        if not blank_savers:
            try:
                finder = GADDAGMoveFinder(self.board, self.gaddag,
                                          board_blanks=self.state.blank_positions)
                all_moves = finder.find_all_moves(rack)
                # Find moves that leave at least one blank
                saver_candidates = []
                for m in all_moves:
                    horizontal = m['direction'] == 'H'
                    used = _board_tiles_used(self.board, m['word'], m['row'],
                                             m['col'], horizontal)
                    used_str = ''.join(used)
                    leave = _get_leave(rack, used_str)
                    if '?' in leave:
                        m['tiles_used'] = used_str
                        m['leave'] = leave
                        saver_candidates.append(m)
                # Take best by score
                saver_candidates.sort(key=lambda x: -x['score'])
                for m in saver_candidates[:BLANK_3PLY_FORCE_SAVERS]:
                    key = (m['word'], m['row'], m['col'])
                    if key not in top_mc_keys:
                        blank_savers.append(m)
            except Exception:
                pass  # If move generation fails, proceed without savers

        candidates = top_mc + blank_savers
        if not candidates:
            return

        n_cands = len(candidates)
        print(f"\n{'='*70}")
        print(f"BLANK STRATEGY 3-PLY ({blanks_total} blanks in rack, {n_cands} candidates)")
        print(f"  Should you spend both blanks or save one?")
        print("=" * 70)

        try:
            results = mc_evaluate_3ply_blanks(
                self.board, rack, unseen_str, candidates,
                board_moves=self.state.board_moves,
                gaddag=self.gaddag,
                board_blanks=self.state.blank_positions,
            )
        except Exception as e:
            print(f"  [!] Blank 3-ply failed: {e}")
            return

        if not results:
            print("  No results.")
            return

        # Build lookup for MC 2-ply equity
        mc_eq_lookup = {}
        for m in mc_results:
            key = (m['word'], m['row'], m['col'])
            mc_eq_lookup[key] = m.get('total_equity', 0)

        # Display table
        print()
        print(f"{'#':<3} {'Word':<13} {'Pos':<11} {'Pts':>4} {'Blanks':>6} "
              f"{'AvgOpp':>6} {'Follow':>6} {'3ply Net':>8} {'vs MC':>7}")
        print("-" * 73)

        best_3ply = results[0]['net_3ply']
        best_saver = None
        best_spender = None

        for i, r in enumerate(results[:15]):
            pos = f"R{r['row']}C{r['col']} {r['direction']}"
            blanks_used = r.get('blanks_used', 0)
            blanks_str = f"{blanks_used}/{blanks_total}"
            mc_key = (r['word'], r['row'], r['col'])
            mc_eq = mc_eq_lookup.get(mc_key, 0)
            delta = r['net_3ply'] - mc_eq

            marker = ''
            if r['leave_has_blank'] and best_saver is None:
                best_saver = r
                marker = ' <-- saves blank'
            elif not r['leave_has_blank'] and best_spender is None:
                best_spender = r

            print(f"{i+1:<3} {r['word']:<13} {pos:<11} {r['score']:>4} {blanks_str:>6} "
                  f"{r['avg_opp']:>6.1f} {r['avg_followup']:>6.1f} "
                  f"{r['net_3ply']:>+8.1f} {delta:>+7.1f}{marker}")

        # Summary comparison
        if best_saver and best_spender:
            saver_net = best_saver['net_3ply']
            spender_net = best_spender['net_3ply']
            diff = saver_net - spender_net
            print()
            if diff > 2.0:
                print(f"  ** SAVE A BLANK: {best_saver['word']} nets {diff:+.1f} more over 2 turns"
                      f" than {best_spender['word']}")
                print(f"     {best_spender['word']} scores +{best_spender['score'] - best_saver['score']}"
                      f" more now but loses on follow-up ({best_saver['avg_followup']:.0f}"
                      f" vs {best_spender['avg_followup']:.0f} avg next turn)")
            elif diff < -2.0:
                print(f"  ** SPEND BOTH: {best_spender['word']} nets {-diff:+.1f} more over 2 turns"
                      f" than saving with {best_saver['word']}")
            else:
                print(f"  ** CLOSE CALL: Spending vs saving blanks within {abs(diff):.1f} pts"
                      f" -- either approach is viable")
        elif not blank_savers:
            print()
            print("  [INFO] No blank-saving moves in top candidates -- all plays use both blanks")

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
        from .mc_eval import mc_evaluate_endgame

        print(f"\n{'='*70}")
        print(f"ENDGAME 2-PLY (exact) [bag: 0, opp rack: {len(opp_rack)} tiles]")
        print("=" * 70)
        print(f"Opponent rack known: {opp_rack}")
        print("Your move -> Opponent's best response -> Net")
        print()

        try:
            results = mc_evaluate_endgame(
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

            # Store endgame recommendation for deep dive / play_move()
            self.last_analysis = {
                'top3': [
                    {
                        'word': r['word'],
                        'row': r['row'],
                        'col': r['col'],
                        'dir': r['direction'],
                        'score': r['score'],
                        'equity': r['net_2ply'],
                        'risk_eq': r['net_2ply'],
                    }
                    for r in results[:3]
                ],
                'mc_top': None,  # Not MC-based, exact endgame evaluation
            }

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
        from .mc_eval import mc_evaluate_near_endgame

        bag_size = total_unseen - 7

        print(f"\n{'='*70}")
        print(f"NEAR-ENDGAME HYBRID (bag: {bag_size}, unseen: {total_unseen})")
        print(f"  EXH = EMPTIES BAG (exact 3-ply over all opp racks)")
        print(f"  PAR = leaves tiles in bag (parity penalty applied)")
        print(f"  Emptying the bag = you know opp's rack + control endgame")
        print("=" * 70)

        try:
            t_start = time.time()
            results = mc_evaluate_near_endgame(
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
                    ],
                    'mc_top': None,  # Not MC-based, near-endgame evaluation
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
        
        BLOCKING_VALUES = BLOCKING_BONUSES
        
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
    
