"""Game analysis mixin -- analysis, MC display, and endgame methods.

Extracted from game_manager.py to reduce file size. Mixed into Game
via multiple inheritance. All methods access shared Game instance
state through self.

V21.1: Removed all dead heuristic computation and display code
(risk analysis, blocking bonus, opening heuristics, exchange
evaluation, bingo blocking, threat display). Equity = score + leave.
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
    MC_TIME_BUDGET,
    THREE_PLY_TIME_BUDGET, ANALYSIS_MIN_CANDIDATES_MC,
    LATE_GAME_UNSEEN_THRESHOLD, CATCHUP_DEFICIT_THRESHOLD,
    HIGH_VARIANCE_DEFICIT_THRESHOLD, POWER_TILE_PROB_THRESHOLD,
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

    def analyze(self, rack: str = None, top_n: int = 15, lookahead_n: int = None):
        """Analyze best moves for rack with leave analysis.

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

        # Move generation -- prefer C-accelerated, fallback to Python
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
        print(f"\n{'#':<3} {'Word':<12} {'Position':<12} {'Pts':>4} {'Leave':>6} {'Equity':>7}")
        print("-" * 50)
        print("* = Top Equity")
        print("-" * 50)

        # Score + leave for all moves
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
                word_idx = self._used_idx_to_word_idx(move, i)
                if word_idx in blanks_used:
                    if '?' in leave_tiles:
                        leave_tiles.remove('?')
                        blanks_consumed += 1
                else:
                    if letter in leave_tiles:
                        leave_tiles.remove(letter)
                    elif '?' in leave_tiles:
                        leave_tiles.remove('?')

            leave_str = ''.join(sorted(leave_tiles)) if leave_tiles else "-"

            # Proper leave evaluation
            # Crossplay: when bag=0, leftover tiles don't penalize -- leave is meaningless
            if endgame_exact:
                leave_value = 0.0
            else:
                leave_value = evaluate_leave(leave_str, bag_empty=bag_empty)

            equity = pts + leave_value

            preliminary.append({
                **move,
                'used': used,
                'tiles_used': used,
                'leave': leave_str,
                'leave_value': leave_value,
                'equity': equity,
            })

        # Sort by equity and take top candidates
        preliminary.sort(key=lambda m: -m['equity'])
        candidates = preliminary[:max(ANALYSIS_MIN_CANDIDATES_MC, top_n * 3)]

        # ENDGAME FAST PATH: When bag=0, skip to endgame solver.
        if endgame_exact:
            ftr = self.state.final_turns_remaining
            is_final_move = (ftr == 1 and self.state.is_your_turn)

            # Show 1-ply table (score only, leave meaningless at bag=0)
            for i, move in enumerate(candidates[:top_n]):
                word = move['word']
                pts = move['score']
                pos = f"R{move['row']}C{move['col']} {move['direction']}"
                flags = '*' if i == 0 else ''
                print(f"{i+1:<3} {word:<12} {pos:<12} {pts:>4} {'--':>6} {pts:>+7.0f} {flags}")

            if is_final_move:
                print(f"\nENDGAME 1-PLY: your final move, no opponent response.")
                print(f"Best: {candidates[0]['word']} ({candidates[0]['score']} pts)")
                return moves

            # Build unseen tile string for opponent rack
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

        # Sort by equity for display
        analyzed_moves = candidates
        analyzed_moves.sort(key=lambda m: -m['equity'])

        # Display top N by equity
        for i, move in enumerate(analyzed_moves[:top_n], 1):
            word = move['word']
            pos = f"R{move['row']} C{move['col']} {move['direction']}"
            pts = move['score']
            leave_value = move['leave_value']
            equity = move['equity']

            indicator = ""
            if i == 1:
                indicator = "*"

            nyt_tag = nyt_warning(word)

            print(f"{i:<3} {word:<12} {pos:<12} {pts:>4} {leave_value:>+6.1f} {equity:>+7.0f}  {indicator}{nyt_tag}")

        # Top 3 detailed view
        by_equity = analyzed_moves[:3]

        print(f"\n{'='*90}")
        print("TOP 3 BY EQUITY (Score + Leave)")
        print("=" * 90)

        for i, move in enumerate(by_equity, 1):
            word = move['word']
            row, col = move['row'], move['col']
            direction = move['direction']
            pts = move['score']

            nyt_tag = nyt_warning(word)
            print(f"\n{i}. {word} @ R{row} C{col} {direction}{nyt_tag}")
            print(f"   Score: {pts} | Leave: {move['leave']} ({move['leave_value']:+.1f})")

            # Show power tile draw probability if power tiles exist
            if has_power_tiles:
                tiles_used = len(move.get('tiles_used', move.get('used', word)))
                leave_len = len(move['leave'])
                draw_count = 7 - leave_len
                if draw_count > 0 and bag_size > 0:
                    power_prob = prob_draw_any_power_tile(unseen, bag_size, draw_count)
                    if power_prob > POWER_TILE_PROB_THRESHOLD:
                        print(f"   --> Power tile chance: {power_prob*100:.0f}% (drawing {draw_count})")

            print(f"   EQUITY: {move['equity']:+.1f}")

        # =================================================================
        # 2-PLY LOOKAHEAD SECTION
        # MC needs more candidates than the display top_n -- pass full
        # analyzed_moves (already equity-sorted) so _show_2ply_analysis
        # can slice to lookahead_n independently of the display count.
        # =================================================================
        self._show_2ply_analysis(rack, analyzed_moves, unseen, bag_empty, lookahead_n)

        return moves

    def _show_2ply_analysis(self, rack: str, top_moves: list, unseen: dict, bag_empty: bool, lookahead_n: int = None):
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
            self._show_endgame_2ply(rack, unseen_str, total_unseen)
            return
        if 1 <= bag_size <= 8:
            self._show_near_endgame(rack, unseen_str, total_unseen, top_moves)
            return

        # Adaptive N*K based on calibrated throughput
        blanks_unseen = blank_count

        mc_budget = MC_TIME_BUDGET

        if lookahead_n is None:
            from .mc_calibrate import compute_adaptive_n
            lookahead_n, k_sims, nk_reason = compute_adaptive_n(
                top_moves, mc_budget, bag_size, top_n_display=15)
        else:
            from .mc_calibrate import estimate_throughput
            sps = estimate_throughput(bag_size)
            k_sims = max(100, min(2000, int(sps * mc_budget / lookahead_n)))
            nk_reason = "user-specified N"

        print(f"\n{'='*70}")
        print(f"MONTE CARLO 2-PLY (K={k_sims} sims x N={lookahead_n} candidates) [{total_unseen} unseen]")
        print(f"  ({nk_reason})")
        print("=" * 70)
        if blank_count > 2:
            print(f"Note: {blank_count} blanks unseen, capped to 2 for speed")

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
                pre_ranked_candidates=top_moves,
            )
            t_mc = time.time() - t_mc_start

            if not results:
                print("No MC 2-ply results available.")
                return

            logger.debug("MC 2-ply completed in %dms (%d total sims)", t_mc*1000, k_sims * len(results))

            # Table header
            print(f"\n{'#':<3} {'Word':<12} {'Pos':<10} {'Pts':>4} "
                  f"{'AvgOpp':>6} {'MaxOpp':>6} {'Std':>5} "
                  f"{'%Beats':>6} {'Leave':>6} {'MC Eq':>7}")
            print("-" * 78)

            for i, m in enumerate(results[:min(lookahead_n, 30)], 1):
                pos = f"R{m['row']}C{m['col']} {m['direction']}"
                word_display = m['word']

                # Markers
                bingo_marker = ""
                opp_word = m.get('opp_word', '')
                if len(opp_word) >= 7 and m['mc_max_opp'] >= 41:
                    bingo_marker = "[!B]"

                leave_display = m['leave']

                nyt_tag = ""
                if is_nyt_curated(m['word']):
                    nyt_tag = " [NYT?]"

                print(f"{i:<3} {word_display:<12} {pos:<10} {m['score']:>4} "
                      f"{m['mc_avg_opp']:>6.1f} {m['mc_max_opp']:>6} {m['mc_std_opp']:>5.1f} "
                      f"{m['pct_opp_beats']:>5.1f}% {leave_display:>6} "
                      f"{m['total_equity']:>+7.1f} {bingo_marker}{nyt_tag}")

            # Show top opponent responses for #1 ranked move
            best = results[0]
            if best.get('top_opp_responses'):
                print(f"\n  Top opponent responses to {best['word']}:")
                for resp in best['top_opp_responses'][:3]:
                    rpos = f"R{resp['row']}C{resp['col']} {resp['direction']}"
                    print(f"    {resp['word']} @ {rpos} = {resp['score']} pts")

            # Compare with 1-ply recommendation
            best_1ply = top_moves[0]['word'] if top_moves else None
            best_mc = results[0]['word'] if results else None

            if best_1ply and best_mc:
                if best_1ply == best_mc:
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
                        }
                        for r in results[:3]
                    ],
                    'mc_top': [
                        {
                            'word': r['word'],
                            'row': r['row'],
                            'col': r['col'],
                            'dir': r['direction'],
                            'score': r['score'],
                            'equity': r.get('total_equity', 0),
                        }
                        for r in results[:5]
                    ],
                }

            # Catch-up advice for late game when far behind
            self._show_catchup_advice(unseen, total_unseen, results)

            # Blank strategy 3-ply: when rack has 2+ blanks, compare spend vs save
            bag_size = total_unseen - 7
            blanks_in_rack = rack.count('?')
            if blanks_in_rack >= BLANK_3PLY_MIN_BLANKS and bag_size > BLANK_3PLY_MIN_BAG:
                self._show_blank_3ply_analysis(rack, unseen_str, total_unseen, results, bag_size)

            # 3-ply analysis for late-to-mid game (C extension makes this fast)
            if bag_size <= 21 and bag_size >= 0:
                remaining_budget = max(2.0, THREE_PLY_TIME_BUDGET - t_mc)
                self._show_3ply_analysis(rack, unseen_str, total_unseen, time_budget=remaining_budget)

        except Exception as e:
            import traceback
            print(f"MC 2-ply analysis error: {e}")
            traceback.print_exc()

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

            # Store endgame recommendation
            self.last_analysis = {
                'top3': [
                    {
                        'word': r['word'],
                        'row': r['row'],
                        'col': r['col'],
                        'dir': r['direction'],
                        'score': r['score'],
                        'equity': r['net_2ply'],
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
        parity-adjusted 1-ply equity (penalized by P(opp empties bag) *
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
            n_parity = sum(1 for r in results if r.get('eval_type') == 'parity')
            n_plain_1ply = n_1ply - n_parity

            print(f"  {n_exhaust} exhaust + {n_parity} parity + "
                  f"{n_plain_1ply} 1ply candidates | "
                  f"{solver_time}s")
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
                    lv = r.get('leave_value', 0.0)
                    bag_after = r.get('bag_after', '?')
                    penalty = r.get('parity_penalty', 0.0)
                    p_opp = r.get('p_opp_empties', 0.0)
                    print(f"{i:<3} {r['word']:<12} {pos:<12} {r['score']:>4} "
                          f"{tag:<7} {'':>6} {'':>6} "
                          f"{'lv=' + str(lv):>8} {r['net_equity']:>+7.1f} "
                          f"{'>' + str(bag_after):>5} {penalty:>+7.1f}")
                else:
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
                        }
                        for r in results[:3]
                    ],
                    'mc_top': None,  # Not MC-based, near-endgame evaluation
                }

        except Exception as e:
            print(f"Near-endgame error: {e}")
            import traceback
            traceback.print_exc()
