"""
CROSSPLAY V15 - Monte Carlo 2-Ply Evaluation Engine

Proper MC simulation for move evaluation:
    1-ply generates all moves, ranks top N by equity.
    2-ply evaluates each of those N candidates with K Monte Carlo
    simulations: sample a random 7-tile opponent rack from unseen
    tiles, find opponent's best response, record score differential.
    Average across K trials gives the simulated 2-ply equity.

Architecture:
    - Each candidate move can be evaluated independently (embarrassingly parallel)
    - Workers reconstruct Board from move list, apply candidate, run K sims
    - Uses GADDAG move finder for fast opponent response search
    - Falls back to sequential if parallel fails

Usage:
    from .mc_eval import mc_evaluate_2ply

    results = mc_evaluate_2ply(
        board, rack, unseen_tiles_str,
        board_moves=game.state.board_moves,
        board_blanks=game.state.blank_positions,
        top_n=10,         # candidates from 1-ply
        k_sims=50,        # MC iterations per candidate
        max_workers=4,
    )
"""

import os
import random
import time
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from .board import Board
from .move_finder_gaddag import GADDAGMoveFinder
from .gaddag import get_gaddag
from .leave_eval import evaluate_leave
from .config import TILE_VALUES, VALID_TWO_LETTER, BINGO_BONUS, RACK_SIZE


# ---------------------------------------------------------------------------
# Pre-computed scoring tables (module-level, shared by all workers)
# ---------------------------------------------------------------------------
_MC_TV = [0] * 26  # tile values indexed by ord(ch) - ord('A')
for _ch, _val in TILE_VALUES.items():
    if _ch != '?':
        _MC_TV[ord(_ch) - ord('A')] = _val
_MC_OA = ord('A')

# Bonus grid: _MC_BONUS[r][c] = (letter_mult, word_mult)
# Dampening factor for Phase 2 positional adjustments carried into MC.
# MC avg_opp already captures ~30-50% of blocking/risk, so dampen to
# avoid double-counting. 0.5 = conservative; tune via self-play.
MC_POSITIONAL_DAMPEN = 0.5

from .config import BONUS_SQUARES
_MC_BONUS = [[(1, 1)] * 15 for _ in range(15)]
for (_r1, _c1), _btype in BONUS_SQUARES.items():
    _r0, _c0 = _r1 - 1, _c1 - 1
    if _btype == '2L': _MC_BONUS[_r0][_c0] = (2, 1)
    elif _btype == '3L': _MC_BONUS[_r0][_c0] = (3, 1)
    elif _btype == '2W': _MC_BONUS[_r0][_c0] = (1, 2)
    elif _btype == '3W': _MC_BONUS[_r0][_c0] = (1, 3)


def _mc_find_best_score(grid, gdata_bytes, rack_str, set_dict, board_blank_set):
    """
    Fast best-score finder for MC simulations.
    Returns (best_score, best_word, best_row, best_col, best_dir) or (0, None, 0, 0, None).
    
    Optimizations vs find_all_moves_c:
    - Uses pre-cached bytes(gdata) (saves ~3ms/call)
    - Uses SetDict with __contains__ (saves ~0.5ms/call)
    - Inline scoring, no dict/list construction for non-best moves
    - Returns only best move, not all moves
    """
    from .move_finder_c import _accel
    if _accel is None:
        from .move_finder_opt import find_best_score_opt
        return find_best_score_opt(grid, gdata_bytes, rack_str, board_blank_set)
    
    tv = _MC_TV
    bonus = _MC_BONUS
    OA = _MC_OA
    bingo = BINGO_BONUS
    rs = RACK_SIZE
    
    raw_moves = _accel.find_moves_c(
        gdata_bytes, grid, rack_str, set_dict, VALID_TWO_LETTER,
    )
    
    best_score = 0
    best_info = None
    
    for word_str, row1, col1, is_horiz, blanks_list in raw_moves:
        r0 = row1 - 1
        c0 = col1 - 1
        wlen = len(word_str)
        blanks_set = set(blanks_list) if blanks_list else set()
        
        main_score = 0
        word_mult = 1
        new_count = 0
        cw_total = 0
        
        if is_horiz:
            for i in range(wlen):
                cc = c0 + i
                is_new = grid[r0][cc] is None
                if i in blanks_set:
                    lv = 0
                elif not is_new and (r0, cc) in board_blank_set:
                    lv = 0
                else:
                    lv = tv[ord(word_str[i]) - OA]
                if is_new:
                    new_count += 1
                    lm, wm = bonus[r0][cc]
                    lv *= lm
                    word_mult *= wm
                    # Crossword scoring (inline)
                    if (r0 > 0 and grid[r0-1][cc] is not None) or \
                       (r0 < 14 and grid[r0+1][cc] is not None):
                        cw_s = 0
                        r2 = r0 - 1
                        while r2 >= 0 and grid[r2][cc] is not None:
                            cw_s += 0 if (r2, cc) in board_blank_set else tv[ord(grid[r2][cc]) - OA]
                            r2 -= 1
                        plv = 0 if i in blanks_set else tv[ord(word_str[i]) - OA]
                        lm2, wm2 = bonus[r0][cc]
                        cw_s += plv * lm2
                        r2 = r0 + 1
                        while r2 < 15 and grid[r2][cc] is not None:
                            cw_s += 0 if (r2, cc) in board_blank_set else tv[ord(grid[r2][cc]) - OA]
                            r2 += 1
                        cw_total += cw_s * wm2
                main_score += lv
        else:
            for i in range(wlen):
                cr = r0 + i
                is_new = grid[cr][c0] is None
                if i in blanks_set:
                    lv = 0
                elif not is_new and (cr, c0) in board_blank_set:
                    lv = 0
                else:
                    lv = tv[ord(word_str[i]) - OA]
                if is_new:
                    new_count += 1
                    lm, wm = bonus[cr][c0]
                    lv *= lm
                    word_mult *= wm
                    if (c0 > 0 and grid[cr][c0-1] is not None) or \
                       (c0 < 14 and grid[cr][c0+1] is not None):
                        cw_s = 0
                        c2 = c0 - 1
                        while c2 >= 0 and grid[cr][c2] is not None:
                            cw_s += 0 if (cr, c2) in board_blank_set else tv[ord(grid[cr][c2]) - OA]
                            c2 -= 1
                        plv = 0 if i in blanks_set else tv[ord(word_str[i]) - OA]
                        lm2, wm2 = bonus[cr][c0]
                        cw_s += plv * lm2
                        c2 = c0 + 1
                        while c2 < 15 and grid[cr][c2] is not None:
                            cw_s += 0 if (cr, c2) in board_blank_set else tv[ord(grid[cr][c2]) - OA]
                            c2 += 1
                        cw_total += cw_s * wm2
                main_score += lv
        
        total = main_score * word_mult + cw_total
        if new_count >= rs:
            total += bingo
        
        if total > best_score:
            best_score = total
            best_info = (word_str, row1, col1, is_horiz)
    
    if best_info:
        w, r, c, h = best_info
        return (best_score, w, r, c, 'H' if h else 'V')
    return (0, None, 0, 0, None)


# ---------------------------------------------------------------------------
# Worker process globals (initialized once per worker via initializer)
# ---------------------------------------------------------------------------
_mc_worker_gaddag = None
_mc_worker_gdata_bytes = None  # cached bytes(gaddag._data) — avoids 3ms conversion per call
_mc_worker_set_dict = None     # SetDict with __contains__ — avoids is_valid overhead
_mc_worker_word_set = None     # raw set for Cython fast path
_mc_worker_cython_fast = False # True if Cython MC fast path available


def _mc_init_worker():
    """Load GADDAG once per worker process."""
    global _mc_worker_gaddag, _mc_worker_gdata_bytes, _mc_worker_set_dict
    global _mc_worker_word_set, _mc_worker_cython_fast
    from .gaddag import get_gaddag
    _mc_worker_gaddag = get_gaddag()
    _mc_worker_gdata_bytes = bytes(_mc_worker_gaddag._data)
    # SetDict: is_valid = set.__contains__ (skips .upper() overhead)
    from .move_finder_c import _get_dict
    _d = _get_dict()
    class _SetDict:
        __slots__ = ('is_valid',)
        def __init__(self, s): self.is_valid = s.__contains__
    _mc_worker_set_dict = _SetDict(_d._words)
    _mc_worker_word_set = _d._words  # raw set for Cython fast path
    # Detect Cython MC fast path
    try:
        from .move_finder_c import is_mc_fast_available
        _mc_worker_cython_fast = is_mc_fast_available()
    except ImportError:
        _mc_worker_cython_fast = False


# ---------------------------------------------------------------------------
# Core MC logic — runs K simulations for ONE candidate move
# ---------------------------------------------------------------------------

def _mc_eval_single_candidate(args: tuple) -> dict:
    """
    Monte Carlo evaluation of one candidate move.

    For this candidate, run K simulations:
        1. Sample a random 7-tile opponent rack from unseen pool
        2. Find opponent's best response on the post-move board
        3. Record opponent's best score
    Aggregate stats across all K trials.

    Args (packed as tuple for pool.map):
        board_moves:    list of (word, row, col, horizontal) to reconstruct board
        move:           dict with word, row, col, direction, score, tiles_used
        unseen_pool:    list of individual unseen tile characters (pre-expanded)
        your_rack:      your rack string (for leave calculation)
        board_blanks:   list of (row, col, letter) for blanks on board
        k_sims:         number of MC iterations
        seed:           random seed for reproducibility (None = random)

    Returns:
        dict with MC statistics:
            word, row, col, direction, score,
            mc_avg_opp, mc_max_opp, mc_min_opp, mc_std_opp,
            mc_equity (score - mc_avg_opp),
            pct_opp_beats (% of sims where opp scores more than us),
            leave, leave_value, total_equity,
            top_opp_responses (top 5 unique opponent plays seen),
            k_sims (actual number of sims run)
    """
    board_moves, move, unseen_pool, your_rack, board_blanks, k_sims, seed = args

    # Seed RNG for this worker (deterministic if seed provided)
    if seed is not None:
        random.seed(seed)

    # Use module-level cached GADDAG
    global _mc_worker_gaddag, _mc_worker_gdata_bytes, _mc_worker_set_dict
    if _mc_worker_gaddag is None:
        from .gaddag import get_gaddag
        _mc_worker_gaddag = get_gaddag()
        _mc_worker_gdata_bytes = bytes(_mc_worker_gaddag._data)
        from .move_finder_c import _get_dict
        _d = _get_dict()
        class _SetDict:
            __slots__ = ('is_valid',)
            def __init__(self, s): self.is_valid = s.__contains__
        _mc_worker_set_dict = _SetDict(_d._words)
    gaddag = _mc_worker_gaddag

    # Reconstruct board
    from .board import Board
    board = Board()
    for word, row, col, horiz in board_moves:
        board.place_word(word, row, col, horiz)

    # Simulate our move
    horizontal = move['direction'] == 'H'
    placed = board.place_move(move['word'], move['row'], move['col'], horizontal)

    # Run K simulations
    opp_scores = []
    opp_responses = {}  # (word, row, col, dir) -> max_score seen

    pool_size = len(unseen_pool)
    rack_size = min(7, pool_size)

    # Pre-compute board blank set for fast scoring
    _bb_set = {(r-1, c-1) for r, c, _ in (board_blanks or [])}
    _grid = board._grid

    # Determine which path to use (priority: Cython fast > Python best-score)
    global _mc_worker_cython_fast, _mc_worker_word_set
    _use_cython_fast = _mc_worker_cython_fast and _mc_worker_gdata_bytes is not None
    _use_c = False  # legacy C path (find_moves_c) — disabled in favor of above

    _ctx = None
    if _use_cython_fast:
        import gaddag_accel as _accel
        _ctx = _accel.prepare_board_context(
            _grid, _mc_worker_gdata_bytes, _bb_set,
            _mc_worker_word_set, VALID_TWO_LETTER,
            _MC_TV, _MC_BONUS, BINGO_BONUS, RACK_SIZE)
    else:
        from .move_finder_opt import find_best_score_opt
        from .dictionary import get_dictionary as _get_dict_py
        _dict_py = _get_dict_py()
        _cross_cache = {}

    # Adaptive K: time first few sims to detect slow boards.
    # Target: each candidate completes in <=8s so 15 candidates / 10 workers
    # finishes in ~12s total (2 waves of work).
    _PER_CANDIDATE_BUDGET = 8.0  # seconds
    _PROBE_COUNT = 3             # sims to time before adapting
    effective_k = k_sims
    _t_sim_start = time.perf_counter()

    # Early stopping: if avg_opp has converged (SE < threshold), stop early.
    # Running stats for O(1) SE computation (no list re-scan).
    _ES_MIN_SIMS = 100       # minimum sims before checking convergence
    _ES_CHECK_EVERY = 10     # check every N sims (amortize overhead)
    _ES_SE_THRESHOLD = 1.0   # stop when SE of mean < 1 pt (95% CI +/-2 pts)
    _running_sum = 0.0
    _running_sum_sq = 0.0

    sim_i = 0
    while sim_i < effective_k:
        # After probe sims, check if we need to reduce K
        if sim_i == _PROBE_COUNT and not _use_cython_fast:
            elapsed = time.perf_counter() - _t_sim_start
            ms_per_sim = elapsed / _PROBE_COUNT * 1000
            if ms_per_sim > 20:  # slow board (>20ms/sim)
                max_k = max(20, int(_PER_CANDIDATE_BUDGET / (ms_per_sim / 1000)))
                if max_k < effective_k:
                    effective_k = max_k

        # Early stopping convergence check
        if (sim_i >= _ES_MIN_SIMS and sim_i % _ES_CHECK_EVERY == 0):
            _variance = (_running_sum_sq / sim_i) - (_running_sum / sim_i) ** 2
            if _variance > 0:
                _se = (_variance / sim_i) ** 0.5
                if _se < _ES_SE_THRESHOLD:
                    break  # converged -- more sims won't change avg_opp meaningfully

        # Sample random opponent rack
        opp_rack_list = random.sample(unseen_pool, rack_size)
        opp_rack = ''.join(opp_rack_list)

        # Find opponent's best move
        if _use_cython_fast:
            opp_score, opp_word, opp_row, opp_col, opp_dir = _accel.find_best_score_c(
                _ctx, opp_rack)
        else:
            opp_score, opp_word, opp_row, opp_col, opp_dir = find_best_score_opt(
                _grid, gaddag._data, opp_rack, _bb_set,
                cross_cache=_cross_cache, dictionary=_dict_py)

        if opp_score > 0:
            opp_scores.append(opp_score)
            _running_sum += opp_score
            _running_sum_sq += opp_score * opp_score
            key = (opp_word, opp_row, opp_col, opp_dir)
            if key not in opp_responses or opp_score > opp_responses[key]:
                opp_responses[key] = opp_score
        else:
            opp_scores.append(0)
            # 0 contributes to sum but not sum_sq (already 0)
        sim_i += 1

    # Undo our move (not strictly necessary since board is local, but clean)
    board.undo_move(placed)

    # Compute stats
    n = len(opp_scores)
    if n == 0:
        avg_opp = 0.0
        max_opp = 0
        min_opp = 0
        std_opp = 0.0
        pct_beats = 0.0
    else:
        avg_opp = _running_sum / n
        max_opp = max(opp_scores)
        min_opp = min(opp_scores)
        variance = (_running_sum_sq / n) - (avg_opp * avg_opp)
        if variance < 0:
            variance = 0.0  # numerical stability
        std_opp = variance ** 0.5
        pct_beats = sum(1 for s in opp_scores if s > move['score']) / n * 100

    # Top 5 unique opponent responses by score
    top_responses = sorted(opp_responses.items(), key=lambda x: -x[1])[:5]
    top_opp = [
        {'word': k[0], 'row': k[1], 'col': k[2], 'direction': k[3], 'score': v}
        for k, v in top_responses
    ]

    # Calculate leave
    from .leave_eval import evaluate_leave
    tiles_used = move.get('tiles_used', move.get('used', move['word']))
    leave = _get_leave(your_rack, tiles_used)
    leave_value = evaluate_leave(leave)

    mc_equity = move['score'] - avg_opp
    total_eq = mc_equity + leave_value + move.get('positional_adj', 0) * MC_POSITIONAL_DAMPEN
    exp_risk = move.get('expected_risk', 0)

    return {
        'word': move['word'],
        'row': move['row'],
        'col': move['col'],
        'direction': move['direction'],
        'score': move['score'],
        # MC opponent stats
        'mc_avg_opp': round(avg_opp, 1),
        'mc_max_opp': max_opp,
        'mc_min_opp': min_opp,
        'mc_std_opp': round(std_opp, 1),
        'pct_opp_beats': round(pct_beats, 1),
        # Equity (with dampened Phase 2 positional adjustment)
        'mc_equity': round(mc_equity, 1),
        'leave': leave,
        'leave_value': round(leave_value, 1),
        'positional_adj': move.get('positional_adj', 0),
        'pos_adj_dampened': round(move.get('positional_adj', 0) * MC_POSITIONAL_DAMPEN, 1),
        'total_equity': round(total_eq, 1),
        'expected_risk': round(exp_risk, 1),
        'risk_adj_equity': round(total_eq - exp_risk, 1),
        'baseline_risk': round(move.get('baseline_risk', 0), 1),
        # Detail
        'top_opp_responses': top_opp,
        'k_sims': n,
        # Keep legacy field names for compatibility with display code
        'opp_best': round(avg_opp, 0),        # avg instead of deterministic best
        'opp_word': top_opp[0]['word'] if top_opp else '',
        'lookahead_equity': round(mc_equity, 1),
    }


def _mc_eval_exchange_candidate(args: tuple) -> dict:
    """
    Monte Carlo evaluation of one exchange candidate.

    For this exchange candidate, run K simulations:
        1. Board stays UNCHANGED (exchange doesn't place tiles)
        2. Sample a random 7-tile opponent rack from unseen pool
        3. Find opponent's best response on the unchanged board
        4. Simulate our post-exchange draw (remove opp rack from pool,
           remove kept tiles, draw from remainder)
        5. Record opponent score and our expected new rack leave

    Args (packed as tuple for pool.map):
        board_moves:    list of (word, row, col, horizontal) to reconstruct board
        exchange_info:  dict with 'keep' (tiles kept), 'dump' (tiles exchanged),
                        'rack' (full original rack)
        unseen_pool:    list of individual unseen tile characters (pre-expanded)
        board_blanks:   list of (row, col, letter) for blanks on board
        k_sims:         number of MC iterations
        seed:           random seed for reproducibility (None = random)

    Returns:
        dict with MC statistics formatted like regular move results,
        plus exchange-specific fields.
    """
    board_moves, exchange_info, unseen_pool, board_blanks, k_sims, seed = args

    if seed is not None:
        random.seed(seed)

    global _mc_worker_gaddag, _mc_worker_gdata_bytes, _mc_worker_set_dict
    if _mc_worker_gaddag is None:
        from .gaddag import get_gaddag
        _mc_worker_gaddag = get_gaddag()
        _mc_worker_gdata_bytes = bytes(_mc_worker_gaddag._data)
        from .move_finder_c import _get_dict
        _d = _get_dict()
        class _SetDict:
            __slots__ = ('is_valid',)
            def __init__(self, s): self.is_valid = s.__contains__
        _mc_worker_set_dict = _SetDict(_d._words)
    gaddag = _mc_worker_gaddag

    # Reconstruct board (unchanged — no move placed)
    from .board import Board
    board = Board()
    for word, row, col, horiz in board_moves:
        board.place_word(word, row, col, horiz)

    keep_tiles = list(exchange_info['keep'])
    dump_tiles = list(exchange_info['dump'])
    draw_n = len(dump_tiles)  # how many we draw after dumping

    pool_size = len(unseen_pool)
    rack_size = min(7, pool_size)

    _bb_set = {(r-1, c-1) for r, c, _ in (board_blanks or [])}
    _grid = board._grid

    # Determine which path to use (priority: Cython fast > Python best-score)
    global _mc_worker_cython_fast, _mc_worker_word_set
    _use_cython_fast = _mc_worker_cython_fast and _mc_worker_gdata_bytes is not None

    _ctx = None
    if _use_cython_fast:
        import gaddag_accel as _accel
        _ctx = _accel.prepare_board_context(
            _grid, _mc_worker_gdata_bytes, _bb_set,
            _mc_worker_word_set, VALID_TWO_LETTER,
            _MC_TV, _MC_BONUS, BINGO_BONUS, RACK_SIZE)
    else:
        from .move_finder_opt import find_best_score_opt
        from .dictionary import get_dictionary as _get_dict_py
        _dict_py = _get_dict_py()
        _cross_cache = {}

    from .leave_eval import evaluate_leave

    opp_scores = []
    opp_responses = {}
    new_rack_leaves = []

    # Adaptive K (same logic as _mc_eval_single_candidate)
    _PER_CANDIDATE_BUDGET = 8.0
    _PROBE_COUNT = 3
    effective_k = k_sims
    _t_sim_start = time.perf_counter()

    # Early stopping (same as _mc_eval_single_candidate)
    _ES_MIN_SIMS = 100
    _ES_CHECK_EVERY = 10
    _ES_SE_THRESHOLD = 1.0
    _running_sum = 0.0
    _running_sum_sq = 0.0

    sim_i = 0
    while sim_i < effective_k:
        if sim_i == _PROBE_COUNT and not _use_cython_fast:
            elapsed = time.perf_counter() - _t_sim_start
            ms_per_sim = elapsed / _PROBE_COUNT * 1000
            if ms_per_sim > 20:
                max_k = max(20, int(_PER_CANDIDATE_BUDGET / (ms_per_sim / 1000)))
                if max_k < effective_k:
                    effective_k = max_k

        # Early stopping convergence check
        if (sim_i >= _ES_MIN_SIMS and sim_i % _ES_CHECK_EVERY == 0):
            _variance = (_running_sum_sq / sim_i) - (_running_sum / sim_i) ** 2
            if _variance > 0:
                _se = (_variance / sim_i) ** 0.5
                if _se < _ES_SE_THRESHOLD:
                    break

        # 1. Sample opponent rack from unseen pool
        opp_rack_list = random.sample(unseen_pool, rack_size)
        opp_rack = ''.join(opp_rack_list)

        # 2. Find opponent's best move on UNCHANGED board
        if _use_cython_fast:
            opp_score, opp_word, opp_row, opp_col, opp_dir = _accel.find_best_score_c(
                _ctx, opp_rack)
        else:
            opp_score, opp_word, opp_row, opp_col, opp_dir = find_best_score_opt(
                _grid, gaddag._data, opp_rack, _bb_set,
                cross_cache=_cross_cache, dictionary=_dict_py)

        if opp_score > 0:
            opp_scores.append(opp_score)
            _running_sum += opp_score
            _running_sum_sq += opp_score * opp_score
            key = (opp_word, opp_row, opp_col, opp_dir)
            if key not in opp_responses or opp_score > opp_responses[key]:
                opp_responses[key] = opp_score
        else:
            opp_scores.append(0)

        # 3. Simulate our exchange draw:
        # Remaining pool = unseen - opp_rack + dump_tiles (dumped go back to bag)
        # We draw draw_n tiles from this pool
        remaining = list(unseen_pool)
        for t in opp_rack_list:
            try:
                remaining.remove(t)
            except ValueError:
                pass
        # Dumped tiles go back into bag
        remaining.extend(dump_tiles)

        if len(remaining) >= draw_n and draw_n > 0:
            drawn = random.sample(remaining, draw_n)
            new_rack = keep_tiles + drawn
        else:
            new_rack = keep_tiles + remaining[:draw_n]

        new_rack_leaves.append(evaluate_leave(''.join(new_rack)))
        sim_i += 1

    # Compute stats
    n = len(opp_scores)
    if n == 0:
        avg_opp = 0.0
        max_opp = 0
        min_opp = 0
        std_opp = 0.0
        pct_beats = 0.0
    else:
        avg_opp = sum(opp_scores) / n
        max_opp = max(opp_scores)
        min_opp = min(opp_scores)
        variance = sum((s - avg_opp) ** 2 for s in opp_scores) / n
        std_opp = variance ** 0.5
        # For exchange, our score is 0 — opp always "beats" us on points
        pct_beats = 100.0

    avg_new_leave = sum(new_rack_leaves) / len(new_rack_leaves) if new_rack_leaves else 0.0

    top_responses = sorted(opp_responses.items(), key=lambda x: -x[1])[:5]
    top_opp = [
        {'word': k[0], 'row': k[1], 'col': k[2], 'direction': k[3], 'score': v}
        for k, v in top_responses
    ]

    mc_equity = 0 - avg_opp  # score=0 for exchange

    keep_str = ''.join(keep_tiles) or '(none)'
    dump_str = ''.join(dump_tiles)

    return {
        'word': f'xchg -{dump_str}',
        'row': -1,
        'col': -1,
        'direction': 'X',  # exchange marker
        'score': 0,
        'is_exchange': True,
        'exchange_keep': ''.join(keep_tiles),
        'exchange_dump': dump_str,
        # MC opponent stats
        'mc_avg_opp': round(avg_opp, 1),
        'mc_max_opp': max_opp,
        'mc_min_opp': min_opp,
        'mc_std_opp': round(std_opp, 1),
        'pct_opp_beats': round(pct_beats, 1),
        # Equity — leave is E[new rack leave from draw simulation]
        'mc_equity': round(mc_equity, 1),
        'leave': f'>{keep_str}+{len(dump_tiles)}',
        'leave_value': round(avg_new_leave, 1),
        'total_equity': round(mc_equity + avg_new_leave, 1),
        'expected_risk': 0,
        'risk_adj_equity': round(mc_equity + avg_new_leave, 1),
        'baseline_risk': round(exchange_info.get('baseline_risk', 0), 1),
        # Detail
        'top_opp_responses': top_opp,
        'k_sims': n,
        'opp_best': round(avg_opp, 0),
        'opp_word': top_opp[0]['word'] if top_opp else '',
        'lookahead_equity': round(mc_equity, 1),
    }


def _get_leave(rack: str, tiles_used: str) -> str:
    """Get remaining tiles after playing a move."""
    rack_list = list(rack.upper())
    for tile in tiles_used.upper():
        if tile in rack_list:
            rack_list.remove(tile)
        elif '?' in rack_list:
            rack_list.remove('?')
    return ''.join(rack_list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank_correction_factor(total_unseen: int, blanks_unseen: int) -> float:
    """
    Calculate correction multiplier for MC opponent scores when blanks are capped.

    The MC simulation caps blanks at 2 in opponent racks for performance.
    This is only needed when 3 blanks are unseen (Crossplay has 3 blanks).
    With cap=2, the correction is small (~1.01-1.04x) since 3-blank racks
    are extremely rare (~0.3% of draws at unseen=44).

    Based on empirical measurement (Crossplay board, N=20 per category):
      0-blank avg best = 39
      1-blank avg best = 57
      2-blank avg best = 83  (baseline for cap=2)
      3-blank avg best = 86  (1.04x vs 2-blank)

    Uses hypergeometric distribution to calculate P(opp draws k blanks).
    Returns multiplier to apply to avg_opp score (1.0 = no correction needed).

    Typical corrections (Crossplay, 3 blanks total, cap=2):
      unseen=24, 3 blanks: ~1.04x
      unseen=44, 3 blanks: ~1.02x
      unseen=65, 3 blanks: ~1.01x
      unseen=93, 3 blanks: ~1.01x
    """
    if blanks_unseen <= 2 or total_unseen < 7:
        return 1.0

    from math import comb

    # Empirical scoring ratios: score[k_blanks] / score[2_blank]
    # Calibrated on Crossplay board (Turn 13 position, N=20 per category)
    RATIO_0v2 = 0.470   # 0-blank scores ~47% of 2-blank
    RATIO_1v2 = 0.687   # 1-blank scores ~69% of 2-blank
    RATIO_3v2 = 1.036   # 3-blank scores ~104% of 2-blank

    draw = min(7, total_unseen)

    def p_draw_k(k):
        """P(exactly k blanks in draw from unseen)"""
        non = total_unseen - blanks_unseen
        if k > blanks_unseen or k > draw or (draw - k) > non:
            return 0.0
        return comb(blanks_unseen, k) * comb(non, draw - k) / comb(total_unseen, draw)

    # True expected score (relative to 2-blank baseline):
    # E_true = P(0)*ratio_0v2 + P(1)*ratio_1v2 + P(2)*1.0 + P(3)*ratio_3v2
    p0 = p_draw_k(0)
    p1 = p_draw_k(1)
    p2 = p_draw_k(2)
    p3 = p_draw_k(3) if blanks_unseen >= 3 else 0.0

    e_true = p0 * RATIO_0v2 + p1 * RATIO_1v2 + p2 * 1.0 + p3 * RATIO_3v2

    # MC-capped expected score:
    # MC pool has (total - blanks_removed) tiles with 2 blanks max.
    # Recalculate draw probabilities for the capped pool.
    blanks_removed = blanks_unseen - 2
    cap_total = total_unseen - blanks_removed
    cap_blanks = 2  # 2 blanks remain in capped pool
    cap_draw = min(7, cap_total)

    if cap_total < 1 or cap_draw < 1:
        return 1.0

    cap_non = cap_total - cap_blanks

    def p_draw_k_capped(k):
        """P(exactly k blanks in draw from capped pool)"""
        if k > cap_blanks or k > cap_draw or (cap_draw - k) > cap_non:
            return 0.0
        return comb(cap_blanks, k) * comb(cap_non, cap_draw - k) / comb(cap_total, cap_draw)

    cp0 = p_draw_k_capped(0)
    cp1 = p_draw_k_capped(1)
    cp2 = p_draw_k_capped(2)

    e_capped = cp0 * RATIO_0v2 + cp1 * RATIO_1v2 + cp2 * 1.0

    if e_capped <= 0:
        return 1.0

    return e_true / e_capped


def _limit_blanks(tiles: str, max_blanks: int = 2) -> str:
    """Limit blanks in tile string to avoid exponential move generation."""
    blank_count = tiles.count('?')
    if blank_count <= max_blanks:
        return tiles
    result = []
    blanks_kept = 0
    for c in tiles:
        if c == '?':
            if blanks_kept < max_blanks:
                result.append(c)
                blanks_kept += 1
        else:
            result.append(c)
    return ''.join(result)


def _expand_unseen_to_pool(unseen_str: str) -> list:
    """Convert unseen tile string to a list of individual tiles for sampling."""
    return list(unseen_str)


def _select_candidates(
    moves: List[Dict],
    board: Board,
    top_n: int,
    include_blockers: bool
) -> List[Dict]:
    """Select candidate moves for deep evaluation (top N by score + blockers)."""
    by_score = sorted(moves, key=lambda m: -m['score'])
    candidates = by_score[:top_n]
    candidate_ids = {id(m) for m in candidates}

    if include_blockers:
        blockers_added = 0
        for m in moves:
            if blockers_added >= 5:
                break
            if id(m) not in candidate_ids and _blocks_premium(m, board):
                candidates.append(m)
                candidate_ids.add(id(m))
                blockers_added += 1
    return candidates


def _blocks_premium(move: Dict, board: Board) -> bool:
    """Check if move covers a premium square."""
    row, col = move['row'], move['col']
    horizontal = move['direction'] == 'H'
    for i in range(len(move['word'])):
        r = row + (0 if horizontal else i)
        c = col + (i if horizontal else 0)
        if board.is_empty(r, c):
            bonus = board.get_bonus(r, c)
            if bonus in ('3W', '2W', '3L'):
                return True
    return False


# ---------------------------------------------------------------------------
# Persistent process pool
# ---------------------------------------------------------------------------
_mc_pool: Optional[ProcessPoolExecutor] = None
_mc_pool_workers: int = 0


def _get_mc_pool(max_workers: int) -> ProcessPoolExecutor:
    """Get or create a persistent MC process pool."""
    global _mc_pool, _mc_pool_workers
    if _mc_pool is None or _mc_pool_workers != max_workers:
        if _mc_pool is not None:
            _mc_pool.shutdown(wait=False)
        _mc_pool = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_mc_init_worker
        )
        _mc_pool_workers = max_workers
    return _mc_pool


def shutdown_mc_pool():
    """Shut down the persistent MC pool."""
    global _mc_pool
    if _mc_pool is not None:
        _mc_pool.shutdown(wait=True)
        _mc_pool = None


# ---------------------------------------------------------------------------
# Sequential fallback
# ---------------------------------------------------------------------------

def _mc_eval_sequential(
    board: Board,
    candidates: List[Dict],
    unseen_pool: list,
    your_rack: str,
    board_blanks: list,
    gaddag,
    k_sims: int,
    blank_corr: float = 1.0,
) -> List[Dict]:
    """Sequential MC evaluation — runs in the main process."""
    results = []
    pool_size = len(unseen_pool)
    rack_size = min(7, pool_size)
    total = len(candidates)
    t0 = time.time()

    # Determine which path to use (once, not per sim)
    # Priority: Cython fast path > Python find_best_score_opt
    _use_cython_fast = False
    try:
        from .move_finder_c import is_mc_fast_available
        _use_cython_fast = is_mc_fast_available()
    except ImportError:
        pass

    if _use_cython_fast:
        import gaddag_accel as _accel_seq
        from .dictionary import get_dictionary as _get_dict_seq
        _dict_seq = _get_dict_seq()
        _word_set_seq = _dict_seq._words
    else:
        from .move_finder_opt import find_best_score_opt
        from .dictionary import get_dictionary as _get_dict_seq
        _dict_seq = _get_dict_seq()

    _bb_set = {(r-1, c-1) for r, c, _ in (board_blanks or [])}
    _gdata_bytes_seq = bytes(gaddag._data) if _use_cython_fast else None

    for idx, move in enumerate(candidates):
        horizontal = move['direction'] == 'H'
        placed = board.place_move(move['word'], move['row'], move['col'], horizontal)

        opp_scores = []
        opp_responses = {}

        # Pre-compute context per candidate (board changes after place_move)
        if _use_cython_fast:
            _ctx = _accel_seq.prepare_board_context(
                board._grid, _gdata_bytes_seq, _bb_set,
                _word_set_seq, VALID_TWO_LETTER,
                _MC_TV, _MC_BONUS, BINGO_BONUS, RACK_SIZE)
        else:
            _cross_cache = {}

        # Adaptive K for sequential path
        _PER_CANDIDATE_BUDGET = 8.0
        _PROBE_COUNT = 3
        effective_k = k_sims
        _t_cand_start = time.perf_counter()

        # Early stopping (same as parallel path)
        _ES_MIN_SIMS = 100
        _ES_CHECK_EVERY = 10
        _ES_SE_THRESHOLD = 1.0
        _running_sum = 0.0
        _running_sum_sq = 0.0

        sim_i = 0
        while sim_i < effective_k:
            if sim_i == _PROBE_COUNT and not _use_cython_fast:
                elapsed_probe = time.perf_counter() - _t_cand_start
                ms_per_sim = elapsed_probe / _PROBE_COUNT * 1000
                if ms_per_sim > 20:
                    max_k = max(20, int(_PER_CANDIDATE_BUDGET / (ms_per_sim / 1000)))
                    if max_k < effective_k:
                        effective_k = max_k

            # Early stopping convergence check
            if (sim_i >= _ES_MIN_SIMS and sim_i % _ES_CHECK_EVERY == 0):
                _variance = (_running_sum_sq / sim_i) - (_running_sum / sim_i) ** 2
                if _variance > 0:
                    _se = (_variance / sim_i) ** 0.5
                    if _se < _ES_SE_THRESHOLD:
                        break

            opp_rack_list = random.sample(unseen_pool, rack_size)
            opp_rack = ''.join(opp_rack_list)

            if _use_cython_fast:
                opp_score, opp_word, opp_row, opp_col, opp_dir = _accel_seq.find_best_score_c(
                    _ctx, opp_rack)
            else:
                opp_score, opp_word, opp_row, opp_col, opp_dir = find_best_score_opt(
                    board._grid, gaddag._data, opp_rack, _bb_set,
                    cross_cache=_cross_cache, dictionary=_dict_seq)
            if opp_score > 0:
                opp_scores.append(opp_score)
                _running_sum += opp_score
                _running_sum_sq += opp_score * opp_score
                key = (opp_word, opp_row, opp_col, opp_dir)
                if key not in opp_responses or opp_score > opp_responses[key]:
                    opp_responses[key] = opp_score
            else:
                opp_scores.append(0)
            sim_i += 1

        board.undo_move(placed)
        elapsed = time.time() - t0
        pct = (idx + 1) / total
        filled = int(20 * pct)
        bar = '#' * filled + '-' * (20 - filled)
        # Print at 25% milestones and final
        if idx + 1 == total or int(pct * 4) > int((idx) / total * 4):
            print(f"  MC [{bar}] {idx+1}/{total}  {elapsed:.0f}s")

        # Stats
        n = len(opp_scores)
        avg_opp_raw = sum(opp_scores) / n if n else 0
        avg_opp = avg_opp_raw * blank_corr  # Apply blank correction
        max_opp = max(opp_scores) if n else 0
        min_opp = min(opp_scores) if n else 0
        mean = avg_opp_raw  # Use raw for variance calc
        variance = sum((s - mean) ** 2 for s in opp_scores) / n if n else 0
        std_opp = variance ** 0.5
        pct_beats = (sum(1 for s in opp_scores if s > move['score']) / n * 100) if n else 0

        top_responses = sorted(opp_responses.items(), key=lambda x: -x[1])[:5]
        top_opp = [
            {'word': k[0], 'row': k[1], 'col': k[2], 'direction': k[3], 'score': v}
            for k, v in top_responses
        ]

        tiles_used = move.get('tiles_used', move.get('used', move['word']))
        leave = _get_leave(your_rack, tiles_used)
        leave_value = evaluate_leave(leave)
        mc_equity = move['score'] - avg_opp
        total_eq = mc_equity + leave_value + move.get('positional_adj', 0) * MC_POSITIONAL_DAMPEN
        exp_risk = move.get('expected_risk', 0)

        results.append({
            'word': move['word'],
            'row': move['row'],
            'col': move['col'],
            'direction': move['direction'],
            'score': move['score'],
            'mc_avg_opp': round(avg_opp, 1),
            'mc_max_opp': max_opp,
            'mc_min_opp': min_opp,
            'mc_std_opp': round(std_opp, 1),
            'pct_opp_beats': round(pct_beats, 1),
            'mc_equity': round(mc_equity, 1),
            'leave': leave,
            'leave_value': round(leave_value, 1),
            'positional_adj': move.get('positional_adj', 0),
            'pos_adj_dampened': round(move.get('positional_adj', 0) * MC_POSITIONAL_DAMPEN, 1),
            'total_equity': round(total_eq, 1),
            'expected_risk': round(exp_risk, 1),
            'risk_adj_equity': round(total_eq - exp_risk, 1),
            'baseline_risk': round(move.get('baseline_risk', 0), 1),
            'top_opp_responses': top_opp,
            'k_sims': n,
            'opp_best': round(avg_opp, 0),
            'opp_word': top_opp[0]['word'] if top_opp else '',
            'lookahead_equity': round(mc_equity, 1),
        })

    results.sort(key=lambda x: -x['total_equity'])
    return results
# ---------------------------------------------------------------------------

def mc_evaluate_2ply(
    board: Board,
    your_rack: str,
    unseen_tiles_str: str,
    board_moves: List[Tuple] = None,
    gaddag=None,
    top_n: int = 10,
    k_sims: int = 50,
    include_blockers: bool = True,
    board_blanks: List[Tuple[int, int, str]] = None,
    max_workers: int = None,
    seed: int = None,
    pre_ranked_candidates: List[Dict] = None,
    exchange_candidates: List[Dict] = None,
) -> List[Dict]:
    """
    Monte Carlo 2-ply evaluation.

    1-ply: find all moves, select top N by score (+ optional blockers)
    2-ply: for each candidate, run K MC simulations with random opponent racks

    Args:
        board:              Current Board object
        your_rack:          Your tile rack
        unseen_tiles_str:   All unseen tiles as a string (bag + opponent rack)
        board_moves:        List of (word, row, col, horizontal) placed so far.
                            Required for parallel mode; if None, uses sequential.
        gaddag:             GADDAG instance (loaded if not provided)
        top_n:              Number of candidate moves to evaluate (from 1-ply)
        k_sims:             Number of MC simulations per candidate move
        include_blockers:   Also evaluate moves that block premium squares
        board_blanks:       List of (row, col, letter) for blanks on board
        max_workers:        Max parallel workers (default: min(cpu, 4))
        seed:               Random seed for reproducibility (None = random)
        pre_ranked_candidates: Optional list of pre-ranked move dicts (e.g. from
                            1-ply equity analysis). If provided, skips internal
                            move generation and uses these directly as MC candidates.
                            Each dict must have: word, row, col, direction, score,
                            tiles_used. Top top_n entries are used.
        exchange_candidates: Optional list of exchange dicts to evaluate alongside
                            regular moves. Each dict must have: 'keep' (str of kept
                            tiles), 'dump' (str of dumped tiles), 'rack' (full rack).
                            These are evaluated via _mc_eval_exchange_candidate and
                            results are merged with regular move results.

    Returns:
        Sorted list of result dicts with MC statistics, best first by total_equity
    """
    if gaddag is None:
        gaddag = get_gaddag()
    if board_blanks is None:
        board_blanks = []

    # Calculate blank correction factor BEFORE limiting blanks
    blanks_in_unseen = unseen_tiles_str.count('?')
    blank_corr = _blank_correction_factor(len(unseen_tiles_str), blanks_in_unseen)

    # Limit blanks in unseen to avoid exponential blowup in move generation
    unseen_limited = _limit_blanks(unseen_tiles_str, max_blanks=2)

    # Expand to a list for random.sample
    unseen_pool = _expand_unseen_to_pool(unseen_limited)

    if len(unseen_pool) < 1:
        return []

    # 1-ply: find all your moves, select top N candidates
    # If pre-ranked candidates provided (e.g. from 1-ply equity), use those
    # directly — skips redundant move generation entirely
    if pre_ranked_candidates is not None:
        candidates = pre_ranked_candidates[:top_n]
    else:
        try:
            from .move_finder_c import find_all_moves_c, is_available
            if is_available():
                your_moves = find_all_moves_c(board, gaddag, your_rack,
                                               board_blanks=board_blanks)
            else:
                raise ImportError("C accel not available")
        except ImportError:
            finder = GADDAGMoveFinder(board, gaddag, board_blanks=board_blanks)
            your_moves = finder.find_all_moves(your_rack)
        if not your_moves:
            return []

        candidates = _select_candidates(your_moves, board, top_n, include_blockers)

    # Decide parallel vs sequential
    if max_workers is None:
        cpu = os.cpu_count() or 1
        max_workers = min(cpu - 2, 10)  # Leave 2 threads for main process
        max_workers = max(max_workers, 2)  # At least 2 workers

    # Sequential for small batches or no board_moves
    if board_moves is None or len(candidates) <= 2 or max_workers <= 1:
        return _mc_eval_sequential(
            board, candidates, unseen_pool, your_rack, board_blanks,
            gaddag, k_sims, blank_corr
        )

    # Convert board_moves dicts to (word, row, col, horizontal) tuples for workers.
    # Workers unpack as: for word, row, col, horiz in board_moves
    # GameState stores enriched dicts with 12+ keys; passing those raw causes
    # "too many values to unpack" in workers and silent fallback to sequential.
    board_move_tuples = []
    for m in board_moves:
        if isinstance(m, dict):
            if m.get('is_exchange'):
                continue  # Skip exchange moves (no board placement)
            board_move_tuples.append((m['word'], m['row'], m['col'], m['dir'] == 'H'))
        else:
            board_move_tuples.append(m)

    # Build picklable argument tuples for parallel execution
    args_list = []
    exchange_args_list = []
    for i, move in enumerate(candidates):
        clean_move = {
            'word': move['word'],
            'row': move['row'],
            'col': move['col'],
            'direction': move['direction'],
            'score': move['score'],
            'tiles_used': move.get('tiles_used', move.get('used', move['word'])),
            'positional_adj': move.get('positional_adj', 0),
            'expected_risk': move.get('expected_risk', 0),
            'baseline_risk': move.get('baseline_risk', 0),
        }
        # Each worker gets a unique seed derived from the base seed
        worker_seed = (seed + i * 1000) if seed is not None else None
        args_list.append((
            board_move_tuples,
            clean_move,
            unseen_pool,
            your_rack,
            board_blanks,
            k_sims,
            worker_seed,
        ))

    # Build exchange candidate args (if any)
    if exchange_candidates:
        for j, exch in enumerate(exchange_candidates):
            exch_seed = (seed + (len(candidates) + j) * 1000) if seed is not None else None
            exchange_args_list.append((
                board_move_tuples,
                exch,
                unseen_pool,
                board_blanks,
                k_sims,
                exch_seed,
            ))

    # Run in parallel
    try:
        pool = _get_mc_pool(max_workers)
        t0 = time.time()
        total = len(args_list) + len(exchange_args_list)
        futures = {pool.submit(_mc_eval_single_candidate, args): i for i, args in enumerate(args_list)}
        # Submit exchange candidates to the same pool
        for j, exch_args in enumerate(exchange_args_list):
            futures[pool.submit(_mc_eval_exchange_candidate, exch_args)] = len(args_list) + j
        results = []
        for i, future in enumerate(as_completed(futures)):
            results.append(future.result())
            elapsed = time.time() - t0
            pct = (i + 1) / total
            filled = int(20 * pct)
            bar = '#' * filled + '-' * (20 - filled)
            # Print at 25% milestones and final
            if i + 1 == total or int(pct * 4) > int(i / total * 4):
                print(f"  MC [{bar}] {i+1}/{total}  {elapsed:.0f}s")
        elapsed = time.time() - t0

        # Apply blank correction to parallel results
        if blank_corr != 1.0:
            for r in results:
                raw_avg = r['mc_avg_opp']
                corrected_avg = raw_avg * blank_corr
                delta = corrected_avg - raw_avg
                r['mc_avg_opp'] = round(corrected_avg, 1)
                r['mc_equity'] = round(r['score'] - corrected_avg, 1)
                pos_adj = r.get('pos_adj_dampened', 0)
                r['total_equity'] = round(r['mc_equity'] + r['leave_value'] + pos_adj, 1)
                r['risk_adj_equity'] = round(r['total_equity'] - r.get('expected_risk', 0), 1)
                r['opp_best'] = round(corrected_avg, 0)
                r['lookahead_equity'] = r['mc_equity']

        results.sort(key=lambda x: -x['total_equity'])
        return results

    except Exception as e:
        print(f"  (MC parallel eval failed: {e} -- falling back to sequential)")
        return _mc_eval_sequential(
            board, candidates, unseen_pool, your_rack, board_blanks,
            gaddag, k_sims, blank_corr
        )


# ---------------------------------------------------------------------------
# Convenience: quick MC analysis printer
# ---------------------------------------------------------------------------

def quick_mc_analysis(board: Board, rack: str, unseen_str: str,
                      top_n: int = 10, k_sims: int = 50) -> None:
    """Print a quick MC 2-ply analysis."""
    print(f"MONTE CARLO 2-PLY ANALYSIS (K={k_sims} sims, N={top_n} candidates)")
    print(f"Rack: {rack} | Unseen: {len(unseen_str)} tiles")
    print("=" * 80)

    results = mc_evaluate_2ply(
        board, rack, unseen_str,
        top_n=top_n, k_sims=k_sims
    )

    if not results:
        print("No valid moves found.")
        return

    print(f"\n{'#':<3} {'Word':<12} {'Pos':<10} {'Pts':>4} "
          f"{'AvgOpp':>6} {'MaxOpp':>6} {'StdD':>5} "
          f"{'%Beats':>6} {'Leave':>6} {'MC Eq':>7}")
    print("-" * 80)

    for i, m in enumerate(results[:top_n], 1):
        pos = f"R{m['row']}C{m['col']} {m['direction']}"
        print(f"{i:<3} {m['word']:<12} {pos:<10} {m['score']:>4} "
              f"{m['mc_avg_opp']:>6.1f} {m['mc_max_opp']:>6} {m['mc_std_opp']:>5.1f} "
              f"{m['pct_opp_beats']:>5.1f}% {m['leave']:>6} {m['total_equity']:>+7.1f}")

    # Show top opponent response for #1
    best = results[0]
    if best['top_opp_responses']:
        print(f"\nTop opponent responses to {best['word']}:")
        for resp in best['top_opp_responses'][:3]:
            pos = f"R{resp['row']}C{resp['col']} {resp['direction']}"
            print(f"  {resp['word']} @ {pos} = {resp['score']} pts")
