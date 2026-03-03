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
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

from .board import Board
from .move_finder_gaddag import GADDAGMoveFinder
from .gaddag import get_gaddag
from .leave_eval import evaluate_leave
from .config import (
    TILE_VALUES, VALID_TWO_LETTER, BINGO_BONUS, RACK_SIZE,
    MC_ES_MIN_SIMS, MC_ES_CHECK_EVERY, MC_ES_SE_THRESHOLD,
    MC_CEILING_K, MC_PROBE_COUNT, MC_SLOW_BOARD_MS, MC_TOTAL_TIMEOUT,
    BLANK_3PLY_K_SIMS, BLANK_3PLY_TIME_BUDGET,
)


# ---------------------------------------------------------------------------
# Pre-computed scoring tables (module-level, shared by all workers)
# ---------------------------------------------------------------------------
_MC_TV = [0] * 26  # tile values indexed by ord(ch) - ord('A')
for _ch, _val in TILE_VALUES.items():
    if _ch != '?':
        _MC_TV[ord(_ch) - ord('A')] = _val
_MC_OA = ord('A')

# Bonus grid: _MC_BONUS[r][c] = (letter_mult, word_mult)
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
    effective_k = k_sims
    _t_sim_start = time.perf_counter()

    # Early stopping: if avg_opp has converged (SE < threshold), stop early.
    # Running stats for O(1) SE computation (no list re-scan).
    _running_sum = 0.0
    _running_sum_sq = 0.0

    sim_i = 0
    while sim_i < effective_k:
        # After probe sims, check if we need to reduce K
        if sim_i == MC_PROBE_COUNT and not _use_cython_fast:
            elapsed = time.perf_counter() - _t_sim_start
            ms_per_sim = elapsed / MC_PROBE_COUNT * 1000
            if ms_per_sim > MC_SLOW_BOARD_MS:
                max_k = max(20, int(_PER_CANDIDATE_BUDGET / (ms_per_sim / 1000)))
                if max_k < effective_k:
                    effective_k = max_k

        # Early stopping convergence check
        if (sim_i >= MC_ES_MIN_SIMS and sim_i % MC_ES_CHECK_EVERY == 0):
            _variance = (_running_sum_sq / sim_i) - (_running_sum / sim_i) ** 2
            if _variance > 0:
                _se = (_variance / sim_i) ** 0.5
                if _se < MC_ES_SE_THRESHOLD:
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
    total_eq = mc_equity + leave_value

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
        # Equity
        'mc_equity': round(mc_equity, 1),
        'leave': leave,
        'leave_value': round(leave_value, 1),
        'total_equity': round(total_eq, 1),
        # Detail
        'top_opp_responses': top_opp,
        'k_sims': n,
        # Keep legacy field names for compatibility with display code
        'opp_best': round(avg_opp, 0),        # avg instead of deterministic best
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


# Register atexit handler to clean up MC pool on interpreter exit
import atexit
atexit.register(shutdown_mc_pool)


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
        effective_k = k_sims
        _t_cand_start = time.perf_counter()

        # Early stopping (same as parallel path)
        _running_sum = 0.0
        _running_sum_sq = 0.0

        sim_i = 0
        while sim_i < effective_k:
            if sim_i == MC_PROBE_COUNT and not _use_cython_fast:
                elapsed_probe = time.perf_counter() - _t_cand_start
                ms_per_sim = elapsed_probe / MC_PROBE_COUNT * 1000
                if ms_per_sim > MC_SLOW_BOARD_MS:
                    max_k = max(20, int(_PER_CANDIDATE_BUDGET / (ms_per_sim / 1000)))
                    if max_k < effective_k:
                        effective_k = max_k

            # Early stopping convergence check
            if (sim_i >= MC_ES_MIN_SIMS and sim_i % MC_ES_CHECK_EVERY == 0):
                _variance = (_running_sum_sq / sim_i) - (_running_sum / sim_i) ** 2
                if _variance > 0:
                    _se = (_variance / sim_i) ** 0.5
                    if _se < MC_ES_SE_THRESHOLD:
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
        avg_opp = sum(opp_scores) / n if n else 0
        max_opp = max(opp_scores) if n else 0
        min_opp = min(opp_scores) if n else 0
        variance = sum((s - avg_opp) ** 2 for s in opp_scores) / n if n else 0
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
        total_eq = mc_equity + leave_value

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
            'total_equity': round(total_eq, 1),
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

    Returns:
        Sorted list of result dicts with MC statistics, best first by total_equity
    """
    if gaddag is None:
        gaddag = get_gaddag()
    if board_blanks is None:
        board_blanks = []

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
            gaddag, k_sims
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
    for i, move in enumerate(candidates):
        clean_move = {
            'word': move['word'],
            'row': move['row'],
            'col': move['col'],
            'direction': move['direction'],
            'score': move['score'],
            'tiles_used': move.get('tiles_used', move.get('used', move['word'])),
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

    # Run in parallel
    try:
        pool = _get_mc_pool(max_workers)
        t0 = time.time()
        total = len(args_list)
        futures = {pool.submit(_mc_eval_single_candidate, args): i for i, args in enumerate(args_list)}
        results = []
        timed_out = 0
        try:
            for i, future in enumerate(as_completed(futures, timeout=MC_TOTAL_TIMEOUT)):
                results.append(future.result())
                elapsed = time.time() - t0
                pct = (i + 1) / total
                filled = int(20 * pct)
                bar = '#' * filled + '-' * (20 - filled)
                # Print at 25% milestones and final
                if i + 1 == total or int(pct * 4) > int(i / total * 4):
                    print(f"  MC [{bar}] {i+1}/{total}  {elapsed:.0f}s")
        except TimeoutError:
            # Cancel remaining futures stuck on dense board positions
            for f in futures:
                f.cancel()
            timed_out = total - len(results)
            elapsed = time.time() - t0
            print(f"  MC timeout after {elapsed:.0f}s -- {len(results)}/{total} candidates"
                  f" ({timed_out} skipped)")
        elapsed = time.time() - t0

        results.sort(key=lambda x: -x['total_equity'])
        return results

    except Exception as e:
        print(f"[!] MC parallel eval failed: {e} -- falling back to sequential")
        return _mc_eval_sequential(
            board, candidates, unseen_pool, your_rack, board_blanks,
            gaddag, k_sims
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


# ---------------------------------------------------------------------------
# Blank Strategy 3-Ply: MC evaluation with ply 3 (your follow-up move)
# ---------------------------------------------------------------------------

def _mc_eval_3ply_blank_candidate(args: tuple) -> dict:
    """
    MC 3-ply evaluation of one candidate for blank strategy analysis.

    For each simulation:
      1. Reconstruct board, place candidate move
      2. Sample opponent rack from unseen
      3. Find opponent's best response (ply 2)
      4. Place opponent's move on the board
      5. Simulate your tile draw (leave + drawn from remaining bag)
      6. Find your best follow-up (ply 3) on the double-modified board
      7. Record net: your_score - opp_score + followup_score
      8. Undo both moves

    Args (packed as tuple):
        board_moves:    list of (word, row, col, horizontal) to reconstruct board
        move:           dict with word, row, col, direction, score, tiles_used, leave
        unseen_pool:    list of individual unseen tile characters
        your_rack:      your rack string
        board_blanks:   list of (row, col, letter) for blanks on board
        k_sims:         number of MC iterations
        seed:           random seed
        bag_size:       tiles in bag (unseen - 7)

    Returns:
        dict with 3-ply MC statistics
    """
    (board_moves, move, unseen_pool, your_rack, board_blanks,
     k_sims, seed, bag_size) = args

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

    # Place our candidate move (ply 1)
    horizontal = move['direction'] == 'H'
    placed_1 = board.place_move(move['word'], move['row'], move['col'], horizontal)

    # Compute leave and tiles used
    tiles_used = move.get('tiles_used', move.get('used', move['word']))
    leave = _get_leave(your_rack, tiles_used)
    tiles_used_count = len(tiles_used)

    # Update blank set after our move (blanks we placed)
    blanks_after_1 = set()
    for r, c, _ in (board_blanks or []):
        blanks_after_1.add((r - 1, c - 1))
    blanks_used_indices = move.get('blanks_used', [])
    for idx in blanks_used_indices:
        if horizontal:
            br, bc = move['row'] - 1, move['col'] - 1 + idx
        else:
            br, bc = move['row'] - 1 + idx, move['col'] - 1
        blanks_after_1.add((br, bc))

    # Build ply-2 board context (after our move)
    global _mc_worker_cython_fast, _mc_worker_word_set
    _use_cython = _mc_worker_cython_fast and _mc_worker_gdata_bytes is not None

    _grid = board._grid
    pool_size = len(unseen_pool)
    rack_size = min(7, pool_size)

    if _use_cython:
        import gaddag_accel as _accel
        ctx_ply2 = _accel.prepare_board_context(
            _grid, _mc_worker_gdata_bytes, blanks_after_1,
            _mc_worker_word_set, VALID_TWO_LETTER,
            _MC_TV, _MC_BONUS, BINGO_BONUS, RACK_SIZE)
    else:
        from .move_finder_opt import find_best_score_opt
        from .dictionary import get_dictionary as _get_dict_py
        _dict_py = _get_dict_py()

    # Early stopping tracking
    _running_sum = 0.0
    _running_sum_sq = 0.0
    opp_score_total = 0.0
    followup_score_total = 0.0
    n_sims = 0

    sim_i = 0
    effective_k = k_sims

    while sim_i < effective_k:
        # Early stopping convergence check (SE < 2.0 for 3-ply higher variance)
        if sim_i >= MC_ES_MIN_SIMS and sim_i % MC_ES_CHECK_EVERY == 0:
            _variance = (_running_sum_sq / sim_i) - (_running_sum / sim_i) ** 2
            if _variance > 0:
                _se = (_variance / sim_i) ** 0.5
                if _se < 2.0:
                    break

        # Sample opponent rack (ply 2)
        opp_rack_list = random.sample(unseen_pool, rack_size)
        opp_rack = ''.join(opp_rack_list)

        # Find opponent's best move on post-ply1 board
        if _use_cython:
            opp_score, opp_word, opp_row, opp_col, opp_dir = _accel.find_best_score_c(
                ctx_ply2, opp_rack)
        else:
            opp_score, opp_word, opp_row, opp_col, opp_dir = find_best_score_opt(
                _grid, gaddag._data, opp_rack, blanks_after_1,
                dictionary=_dict_py)

        # Place opponent's move (ply 2) if they scored
        placed_2 = None
        if opp_score > 0 and opp_word:
            opp_horiz = (opp_dir == 'H' if isinstance(opp_dir, str) else opp_dir)
            try:
                placed_2 = board.place_move(opp_word, opp_row, opp_col, opp_horiz)
            except Exception:
                placed_2 = None

        # Simulate tile draw for ply 3
        # Bag = unseen minus opponent's rack
        opp_rack_set = list(opp_rack_list)  # copy for removal
        bag_tiles = list(unseen_pool)
        for t in opp_rack_list:
            if t in bag_tiles:
                bag_tiles.remove(t)

        draw_count = min(tiles_used_count, len(bag_tiles))
        if draw_count > 0:
            drawn = random.sample(bag_tiles, draw_count)
        else:
            drawn = []
        ply3_rack = leave + ''.join(drawn)

        # Find our best follow-up (ply 3) on the double-modified board
        followup_score = 0
        if ply3_rack and placed_2 is not None:
            # Need fresh context for the board with both moves placed
            blanks_ply3 = set(blanks_after_1)
            # Add opponent blanks if they used any
            # (simplified: opponent blanks are rare, skip tracking for perf)
            if _use_cython:
                ctx_ply3 = _accel.prepare_board_context(
                    _grid, _mc_worker_gdata_bytes, blanks_ply3,
                    _mc_worker_word_set, VALID_TWO_LETTER,
                    _MC_TV, _MC_BONUS, BINGO_BONUS, RACK_SIZE)
                followup_score, _, _, _, _ = _accel.find_best_score_c(
                    ctx_ply3, ply3_rack)
            else:
                followup_score, _, _, _, _ = find_best_score_opt(
                    _grid, gaddag._data, ply3_rack, blanks_ply3,
                    dictionary=_dict_py)
        elif ply3_rack:
            # Opponent didn't play (score 0) -- find our move on post-ply1 board
            if _use_cython:
                followup_score, _, _, _, _ = _accel.find_best_score_c(
                    ctx_ply2, ply3_rack)
            else:
                followup_score, _, _, _, _ = find_best_score_opt(
                    _grid, gaddag._data, ply3_rack, blanks_after_1,
                    dictionary=_dict_py)

        # Undo opponent's move
        if placed_2 is not None:
            board.undo_move(placed_2)

        # Net equity for this sim
        net = move['score'] - opp_score + followup_score
        _running_sum += net
        _running_sum_sq += net * net
        opp_score_total += opp_score
        followup_score_total += followup_score
        n_sims += 1
        sim_i += 1

    # Undo our move
    board.undo_move(placed_1)

    # Compute stats
    if n_sims == 0:
        return {
            'word': move['word'], 'row': move['row'], 'col': move['col'],
            'direction': move['direction'], 'score': move['score'],
            'blanks_used': len(blanks_used_indices),
            'leave': leave, 'leave_has_blank': '?' in leave,
            'avg_opp': 0.0, 'avg_followup': 0.0, 'net_3ply': float(move['score']),
            'k_sims': 0,
        }

    avg_net = _running_sum / n_sims
    avg_opp = opp_score_total / n_sims
    avg_followup = followup_score_total / n_sims

    return {
        'word': move['word'],
        'row': move['row'],
        'col': move['col'],
        'direction': move['direction'],
        'score': move['score'],
        'blanks_used': len(blanks_used_indices),
        'leave': leave,
        'leave_has_blank': '?' in leave,
        'avg_opp': round(avg_opp, 1),
        'avg_followup': round(avg_followup, 1),
        'net_3ply': round(avg_net, 1),
        'k_sims': n_sims,
    }


# ---------------------------------------------------------------------------
# Endgame: deterministic 2-ply when bag=0 (parallel worker)
# ---------------------------------------------------------------------------

def _mc_eval_endgame_candidate(args: tuple) -> dict:
    """
    Deterministic 2-ply endgame evaluation for one candidate move (bag=0).

    Opponent rack is known exactly (all unseen tiles). Evaluates:
      net = our_score - opponent_best_response_score

    Uses grid-copy board reconstruction (O(225) constant time, faster than
    board_moves replay for late-game positions with 20+ moves played).

    Args (packed tuple):
        grid:           15x15 list of lists (board._grid snapshot)
        bb_set_list:    list of (r0, c0) tuples for blanks on board (0-indexed)
        move:           dict with word, row, col, direction, score, blanks_used
        opp_rack:       opponent rack string (known exactly when bag=0)

    Returns:
        dict matching evaluate_endgame_2ply() result format
    """
    grid, bb_set_list, move, opp_rack = args

    global _mc_worker_gaddag, _mc_worker_gdata_bytes
    global _mc_worker_cython_fast, _mc_worker_word_set
    if _mc_worker_gaddag is None:
        from .gaddag import get_gaddag
        _mc_worker_gaddag = get_gaddag()
        _mc_worker_gdata_bytes = bytes(_mc_worker_gaddag._data)

    from .board import Board

    # Reconstruct board from grid snapshot
    board = Board()
    for r in range(15):
        for c in range(15):
            if grid[r][c] is not None:
                board._grid[r][c] = grid[r][c]

    # Place our move
    horizontal = move['direction'] == 'H'
    placed = board.place_move(move['word'], move['row'], move['col'], horizontal)

    # Build blank set including blanks from this move
    bb_set = set(bb_set_list)
    for idx in move.get('blanks_used', []):
        if horizontal:
            bb_set.add((move['row'] - 1, move['col'] - 1 + idx))
        else:
            bb_set.add((move['row'] - 1 + idx, move['col'] - 1))

    # Find opponent's best response
    _use_cython = _mc_worker_cython_fast and _mc_worker_gdata_bytes is not None

    opp_score = 0
    opp_word = '(pass)'

    if _use_cython:
        import gaddag_accel as _accel
        ctx = _accel.prepare_board_context(
            board._grid, _mc_worker_gdata_bytes, bb_set,
            _mc_worker_word_set, VALID_TWO_LETTER,
            _MC_TV, _MC_BONUS, BINGO_BONUS, RACK_SIZE)
        score, word, _, _, _ = _accel.find_best_score_c(ctx, opp_rack)
        if score > 0:
            opp_score = score
            opp_word = word if word else '(pass)'
    else:
        from .move_finder_opt import find_best_score_opt
        score, word, _, _, _ = find_best_score_opt(
            board._grid, _mc_worker_gaddag._data, opp_rack, bb_set)
        if score > 0:
            opp_score = score
            opp_word = word if word else '(pass)'

    board.undo_move(placed)

    return {
        'word': move['word'],
        'row': move['row'],
        'col': move['col'],
        'direction': move['direction'],
        'score': move['score'],
        'opp_word': opp_word,
        'opp_score': opp_score,
        'opp_responses': [],
        'net_2ply': move['score'] - opp_score,
        'exact': True,
    }


# ---------------------------------------------------------------------------
# Near-endgame: exhaustive 3-ply for one bag-emptying move (parallel worker)
# ---------------------------------------------------------------------------

def _mc_eval_near_endgame_candidate(args: tuple) -> dict:
    """
    Exhaustive 3-ply evaluation for one bag-emptying move (bag 1-8).

    Iterates over ALL C(unseen, opp_rack_size) opponent rack combinations.
    For each: our_score - opp_best_response + our_follow_up.

    Args (packed tuple):
        grid:           15x15 list of lists (board._grid snapshot)
        bb_set_list:    list of (r0, c0) tuples for blanks on board (0-indexed)
        move:           dict with word, row, col, direction, score,
                        blanks_used, tiles_used
        unseen_pool:    string of all unseen tiles
        your_rack:      your rack string

    Returns:
        dict matching evaluate_near_endgame() 'exhaust' result format
    """
    from itertools import combinations

    grid, bb_set_list, move, unseen_pool, your_rack = args

    global _mc_worker_gaddag, _mc_worker_gdata_bytes
    global _mc_worker_cython_fast, _mc_worker_word_set
    if _mc_worker_gaddag is None:
        from .gaddag import get_gaddag
        _mc_worker_gaddag = get_gaddag()
        _mc_worker_gdata_bytes = bytes(_mc_worker_gaddag._data)

    from .board import Board

    # Reconstruct board from grid snapshot
    board = Board()
    for r in range(15):
        for c in range(15):
            if grid[r][c] is not None:
                board._grid[r][c] = grid[r][c]

    # Place our move (ply 1)
    horizontal = move['direction'] == 'H'
    placed_1 = board.place_move(move['word'], move['row'], move['col'], horizontal)

    # Build blank set including blanks from this move
    bb_set = set(bb_set_list)
    for idx in move.get('blanks_used', []):
        if horizontal:
            bb_set.add((move['row'] - 1, move['col'] - 1 + idx))
        else:
            bb_set.add((move['row'] - 1 + idx, move['col'] - 1))

    # Compute our leave (tiles remaining after playing)
    tiles_used = move.get('tiles_used', move.get('used', list(move['word'])))
    rack_list = list(your_rack.upper())
    for t in tiles_used:
        t_upper = t.upper() if isinstance(t, str) else t
        if t_upper in rack_list:
            rack_list.remove(t_upper)
        elif '?' in rack_list:
            rack_list.remove('?')
    your_leave = ''.join(rack_list)

    # Build ply-2 board context (after our move)
    _use_cython = _mc_worker_cython_fast and _mc_worker_gdata_bytes is not None

    if _use_cython:
        import gaddag_accel as _accel
        ctx_ply2 = _accel.prepare_board_context(
            board._grid, _mc_worker_gdata_bytes, bb_set,
            _mc_worker_word_set, VALID_TWO_LETTER,
            _MC_TV, _MC_BONUS, BINGO_BONUS, RACK_SIZE)
    else:
        from .move_finder_opt import find_best_score_opt

    unseen_list = list(unseen_pool)
    n_unseen = len(unseen_list)
    opp_rack_size = min(RACK_SIZE, n_unseen)

    net_scores = []
    opp_scores_all = []
    your_resp_scores_all = []
    opp_response_counts = {}  # word -> [count, max_score]

    for combo_indices in combinations(range(n_unseen), opp_rack_size):
        opp_rack = ''.join(unseen_list[i] for i in combo_indices)

        # Limit blanks in opp rack to avoid exponential blowup
        opp_rack_limited = _limit_blanks(opp_rack, max_blanks=2)

        # Remaining unseen tiles go to bag -> your draw
        drawn_indices = set(range(n_unseen)) - set(combo_indices)
        drawn_tiles = ''.join(unseen_list[i] for i in drawn_indices)
        your_full_rack = your_leave + drawn_tiles

        # Ply 2: opponent's best response
        if _use_cython:
            opp_score, opp_word, opp_r, opp_c, opp_d = _accel.find_best_score_c(
                ctx_ply2, opp_rack_limited)
        else:
            opp_score, opp_word, opp_r, opp_c, opp_d = find_best_score_opt(
                board._grid, _mc_worker_gaddag._data, opp_rack_limited, bb_set)

        opp_score_val = opp_score if opp_score > 0 else 0

        # Track opponent responses
        if opp_score > 0 and opp_word:
            if opp_word not in opp_response_counts:
                opp_response_counts[opp_word] = [0, 0]
            opp_response_counts[opp_word][0] += 1
            opp_response_counts[opp_word][1] = max(
                opp_response_counts[opp_word][1], opp_score)

        # Ply 3: our follow-up after opponent's move
        your_resp_score = 0
        if opp_score > 0 and opp_word:
            opp_horiz = opp_d == 'H' if isinstance(opp_d, str) else opp_d
            placed_2 = board.place_move(opp_word, opp_r, opp_c, opp_horiz)

            if _use_cython:
                ctx_ply3 = _accel.prepare_board_context(
                    board._grid, _mc_worker_gdata_bytes, bb_set,
                    _mc_worker_word_set, VALID_TWO_LETTER,
                    _MC_TV, _MC_BONUS, BINGO_BONUS, RACK_SIZE)
                your_resp_score, _, _, _, _ = _accel.find_best_score_c(
                    ctx_ply3, your_full_rack)
            else:
                your_resp_score, _, _, _, _ = find_best_score_opt(
                    board._grid, _mc_worker_gaddag._data, your_full_rack, bb_set)

            board.undo_move(placed_2)
        else:
            # Opponent passed -- find our best on ply-2 board (reuse ctx)
            if _use_cython:
                your_resp_score, _, _, _, _ = _accel.find_best_score_c(
                    ctx_ply2, your_full_rack)
            else:
                your_resp_score, _, _, _, _ = find_best_score_opt(
                    board._grid, _mc_worker_gaddag._data, your_full_rack, bb_set)

        net = move['score'] - opp_score_val + your_resp_score
        net_scores.append(net)
        opp_scores_all.append(opp_score_val)
        your_resp_scores_all.append(your_resp_score)

    board.undo_move(placed_1)

    # Aggregate stats
    n_racks = len(net_scores)
    if n_racks > 0:
        avg_net = sum(net_scores) / n_racks
        avg_opp = sum(opp_scores_all) / n_racks
        max_opp = max(opp_scores_all) if opp_scores_all else 0
        avg_resp = sum(your_resp_scores_all) / n_racks
    else:
        avg_net = float(move['score'])
        avg_opp = 0.0
        max_opp = 0
        avg_resp = 0.0

    # Top opponent responses by frequency
    top_opp = sorted(opp_response_counts.items(),
                     key=lambda x: (-x[1][0], -x[1][1]))[:5]
    top_opp_list = [
        {'word': w, 'count': c, 'max_score': s}
        for w, (c, s) in top_opp
    ]

    return {
        'word': move['word'],
        'row': move['row'],
        'col': move['col'],
        'direction': move['direction'],
        'score': move['score'],
        'eval_type': 'exhaust',
        'leave': your_leave,
        'leave_value': 0.0,
        'opp_avg': round(avg_opp, 1),
        'opp_max': max_opp,
        'your_resp_avg': round(avg_resp, 1),
        'net_equity': round(avg_net, 1),
        'n_racks': n_racks,
        'top_opp_responses': top_opp_list,
    }


def mc_evaluate_3ply_blanks(
    board: Board,
    your_rack: str,
    unseen_tiles_str: str,
    candidates: List[Dict],
    board_moves: List = None,
    gaddag=None,
    board_blanks: List[Tuple[int, int, str]] = None,
    k_sims: int = BLANK_3PLY_K_SIMS,
    time_budget: float = BLANK_3PLY_TIME_BUDGET,
    max_workers: int = None,
) -> List[Dict]:
    """
    3-ply MC evaluation for blank strategy analysis.

    For each candidate, simulates: your move -> opp response -> your follow-up.
    Compares net equity across blank-spending vs blank-saving plays.

    Args:
        board: Current board state
        your_rack: Your rack string (with '?' for blanks)
        unseen_tiles_str: All unseen tiles as string
        candidates: Pre-selected candidate moves (from 2-ply MC results)
        board_moves: Move history for board reconstruction in workers
        gaddag: GADDAG instance (optional, for board_moves generation)
        board_blanks: Blank positions on board
        k_sims: Simulations per candidate
        time_budget: Max seconds for the entire pass
        max_workers: Worker count (None = auto)

    Returns:
        List of result dicts sorted by net_3ply descending
    """
    if not candidates:
        return []

    # Build board_moves tuples for worker board reconstruction
    if board_moves is not None:
        bm_tuples = []
        for m in board_moves:
            word = m.get('word', '')
            row = m.get('row', 0)
            col = m.get('col', 0)
            d = m.get('dir', m.get('direction', 'H'))
            horiz = d in ('H', True)
            if word and row > 0 and col > 0:
                bm_tuples.append((word, row, col, horiz))
    else:
        bm_tuples = []

    unseen_pool = _expand_unseen_to_pool(unseen_tiles_str)
    bag_size = len(unseen_pool) - 7  # unseen minus opponent rack

    # Build args for each candidate
    args_list = []
    for i, move in enumerate(candidates):
        args_list.append((
            bm_tuples, move, unseen_pool, your_rack,
            board_blanks, k_sims, i * 31337, bag_size
        ))

    # Run in parallel using existing pool
    t0 = time.time()
    try:
        pool = _get_mc_pool(max_workers)
        total = len(args_list)
        futures = {pool.submit(_mc_eval_3ply_blank_candidate, args): i
                   for i, args in enumerate(args_list)}
        results = []
        try:
            for i, future in enumerate(as_completed(futures, timeout=time_budget)):
                results.append(future.result())
                elapsed = time.time() - t0
                pct = (i + 1) / total
                filled = int(20 * pct)
                bar = '#' * filled + '-' * (20 - filled)
                if i + 1 == total or int(pct * 4) > int(i / total * 4):
                    print(f"  3ply [{bar}] {i+1}/{total}  {elapsed:.0f}s")
        except TimeoutError:
            for f in futures:
                f.cancel()
            elapsed = time.time() - t0
            print(f"  3ply timeout after {elapsed:.0f}s -- {len(results)}/{total}")

        results.sort(key=lambda x: -x['net_3ply'])
        return results

    except Exception as e:
        print(f"[!] Blank 3-ply eval failed: {e}")
        import traceback
        traceback.print_exc()
        return []


# ---------------------------------------------------------------------------
# Parallel endgame evaluation (bag=0, opponent rack known exactly)
# ---------------------------------------------------------------------------

def mc_evaluate_endgame(
    board: Board,
    your_rack: str,
    opp_rack: str,
    gaddag=None,
    board_blanks: List[Tuple[int, int, str]] = None,
    top_n: int = 20,
    max_workers: int = None,
) -> List[Dict]:
    """
    Parallel 2-ply endgame solver (bag=0, opp rack known exactly).

    Fans out ALL candidate moves to worker pool. Each worker places the move,
    finds opponent's best response via Cython, returns net = score - opp_score.

    With 10 workers, brute-force evaluation of N moves completes in ceil(N/10)
    evaluations, competitive with sequential upper-bound pruning.

    Falls back to sequential evaluate_endgame_2ply() for small move counts
    or on failure.

    Returns:
        Same format as evaluate_endgame_2ply(): list of dicts sorted by net_2ply.
    """
    t_start = time.perf_counter()

    if board_blanks is None:
        board_blanks = []

    # Generate all candidate moves
    try:
        from .move_finder_c import find_all_moves_c, is_available
        if is_available():
            if gaddag is None:
                gaddag = get_gaddag()
            your_moves = find_all_moves_c(
                board, gaddag, your_rack, board_blanks=board_blanks)
        else:
            raise ImportError("C extension not available")
    except (ImportError, Exception):
        if gaddag is None:
            gaddag = get_gaddag()
        finder = GADDAGMoveFinder(board, gaddag, board_blanks=board_blanks)
        your_moves = finder.find_all_moves(your_rack)

    if not your_moves:
        return []

    # For small move counts, use sequential (pool overhead not worth it)
    if len(your_moves) < 4:
        from .lookahead_3ply import evaluate_endgame_2ply
        return evaluate_endgame_2ply(
            board, your_rack=your_rack, opp_rack=opp_rack,
            gaddag=gaddag, board_blanks=board_blanks)

    # Serialize board grid (O(225) constant, faster than move replay)
    grid = [row[:] for row in board._grid]
    bb_set_list = [(r - 1, c - 1) for r, c, _ in board_blanks]

    # Build clean move dicts (only picklable fields)
    args_list = []
    for move in your_moves:
        clean_move = {
            'word': move['word'],
            'row': move['row'],
            'col': move['col'],
            'direction': move['direction'],
            'score': move['score'],
            'blanks_used': move.get('blanks_used', []),
        }
        args_list.append((grid, bb_set_list, clean_move, opp_rack))

    # Run in parallel
    try:
        if max_workers is None:
            cpu = os.cpu_count() or 4
            max_workers = max(2, min(cpu - 2, 10))
        pool = _get_mc_pool(max_workers)
        total = len(args_list)

        futures = {pool.submit(_mc_eval_endgame_candidate, args): i
                   for i, args in enumerate(args_list)}

        results = []
        try:
            for future in as_completed(futures, timeout=30):
                results.append(future.result())
        except TimeoutError:
            for f in futures:
                f.cancel()
            print(f"  Endgame parallel: {len(results)}/{total} completed before timeout")

        elapsed = time.perf_counter() - t_start
        results.sort(key=lambda r: -r['net_2ply'])

        # Attach metadata (matching evaluate_endgame_2ply format)
        meta = {
            'elapsed_s': round(elapsed, 2),
            'total_moves': len(your_moves),
            'interfering': len(your_moves),  # all evaluated in parallel
            'non_interfering': 0,
            'fully_evaluated': len(results),
            'pruned_by_bound': 0,
            'opp_baseline': '(parallel)',
            'baseline_time_s': 0,
            'opp_rack_size': len(opp_rack),
            'solver': 'endgame_2ply_parallel',
        }
        for r in results:
            r['_meta'] = meta

        return results

    except Exception as e:
        print(f"  Endgame parallel failed: {e} -- falling back to sequential")
        from .lookahead_3ply import evaluate_endgame_2ply
        return evaluate_endgame_2ply(
            board, your_rack=your_rack, opp_rack=opp_rack,
            gaddag=gaddag, board_blanks=board_blanks)


# ---------------------------------------------------------------------------
# Parallel near-endgame evaluation (bag 1-8, hybrid exhaustive + parity)
# ---------------------------------------------------------------------------

# Parity penalty lookup table (same as lookahead_3ply.py)
_PARITY_P_OPP_EMPTIES = {
    1: 0.97, 2: 0.94, 3: 0.88, 4: 0.78,
    5: 0.62, 6: 0.40, 7: 0.18,
}
_PARITY_STRUCTURAL_ADV = 10.0


def mc_evaluate_near_endgame(
    board: Board,
    your_rack: str,
    unseen_tiles: str,
    candidates: List[Dict],
    gaddag=None,
    board_blanks: List[Tuple[int, int, str]] = None,
    top_n: int = 25,
    time_budget: float = 45.0,
    max_workers: int = None,
) -> List[Dict]:
    """
    Parallel hybrid near-endgame evaluator (bag 1-8).

    Pass 1 (main process, instant): Non-emptying moves get parity-adjusted
    1-ply equity -- penalized by P(opp empties bag) x structural advantage.

    Pass 2 (parallel workers): Bag-emptying moves fan out to worker pool.
    Each worker computes exhaustive 3-ply over all C(unseen, 7) opponent
    rack combinations.

    Falls back to sequential evaluate_near_endgame() on failure.

    Returns:
        Same format as evaluate_near_endgame(): list of dicts sorted by net_equity.
    """
    t_start = time.perf_counter()

    if not candidates:
        return []

    if gaddag is None:
        gaddag = get_gaddag()
    if board_blanks is None:
        board_blanks = []

    unseen_count = len(unseen_tiles)
    bag_size = max(0, unseen_count - 7)

    if bag_size < 1 or bag_size > 8:
        return []

    cands = candidates[:top_n]
    results = []
    exhaust_count = 0
    oneply_count = 0

    # --- PASS 1: Non-emptying moves (instant, parity-adjusted 1-ply) ---
    exhaust_cands = []
    for move in cands:
        tiles_used = move.get('tiles_used', move.get('used', move['word']))
        n_tiles_used = len(tiles_used)

        if n_tiles_used >= bag_size:
            exhaust_cands.append(move)
        else:
            # Parity-adjusted 1-ply equity
            oneply_count += 1
            leave_val = move.get('leave_value', 0.0)
            if isinstance(leave_val, str):
                leave_val = 0.0
            equity = move['score'] + leave_val

            n_draw = min(n_tiles_used, bag_size)
            bag_after = bag_size - n_draw
            parity_penalty = 0.0
            p_opp_empties = 0.0
            if 1 <= bag_after <= 7:
                p_opp_empties = _PARITY_P_OPP_EMPTIES[bag_after]
                parity_penalty = -p_opp_empties * _PARITY_STRUCTURAL_ADV
                equity += parity_penalty

            eval_type = 'parity' if parity_penalty != 0 else '1ply'
            results.append({
                'word': move['word'],
                'row': move['row'],
                'col': move['col'],
                'direction': move['direction'],
                'score': move['score'],
                'eval_type': eval_type,
                'leave': move.get('leave_str', ''),
                'leave_value': round(leave_val, 1),
                'opp_avg': 0.0,
                'opp_max': 0,
                'your_resp_avg': 0.0,
                'net_equity': round(equity, 1),
                'n_racks': 0,
                'top_opp_responses': [],
                'parity_penalty': round(parity_penalty, 1),
                'p_opp_empties': round(p_opp_empties, 2),
                'bag_after': bag_after,
            })

    # --- PASS 2: Bag-emptying moves (parallel exhaustive 3-ply) ---
    if exhaust_cands:
        grid = [row[:] for row in board._grid]
        bb_set_list = [(r - 1, c - 1) for r, c, _ in board_blanks]

        work_list = []
        for move in exhaust_cands:
            clean_move = {
                'word': move['word'],
                'row': move['row'],
                'col': move['col'],
                'direction': move['direction'],
                'score': move['score'],
                'blanks_used': move.get('blanks_used', []),
                'tiles_used': move.get('tiles_used',
                                       move.get('used', list(move['word']))),
            }
            work_list.append((
                grid, bb_set_list, clean_move,
                unseen_tiles, your_rack
            ))

        try:
            if max_workers is None:
                cpu = os.cpu_count() or 4
                max_workers = max(2, min(cpu - 2, 10))
            pool = _get_mc_pool(max_workers)
            total = len(work_list)

            futures = {pool.submit(_mc_eval_near_endgame_candidate, args): i
                       for i, args in enumerate(work_list)}

            try:
                remaining_budget = time_budget - (time.perf_counter() - t_start)
                for future in as_completed(futures, timeout=max(5, remaining_budget)):
                    result = future.result()
                    results.append(result)
                    exhaust_count += 1
            except TimeoutError:
                # Fall back to 1-ply for timed-out candidates
                completed_words = {r['word'] for r in results
                                   if r.get('eval_type') == 'exhaust'}
                for move in exhaust_cands:
                    if move['word'] not in completed_words:
                        leave_val = move.get('leave_value', 0.0)
                        if isinstance(leave_val, str):
                            leave_val = 0.0
                        oneply_count += 1
                        results.append({
                            'word': move['word'],
                            'row': move['row'],
                            'col': move['col'],
                            'direction': move['direction'],
                            'score': move['score'],
                            'eval_type': '1ply',
                            'leave': move.get('leave_str', ''),
                            'leave_value': round(leave_val, 1),
                            'opp_avg': 0.0,
                            'opp_max': 0,
                            'your_resp_avg': 0.0,
                            'net_equity': round(move['score'] + leave_val, 1),
                            'n_racks': 0,
                            'top_opp_responses': [],
                            'parity_penalty': 0.0,
                            'p_opp_empties': 0.0,
                            'bag_after': 0,
                        })

                for f in futures:
                    f.cancel()
                elapsed = time.perf_counter() - t_start
                print(f"  Near-endgame: {exhaust_count}/{total} exhaust completed "
                      f"in {elapsed:.1f}s (timeout)")

        except Exception as e:
            print(f"  Near-endgame parallel failed: {e} -- falling back to sequential")
            from .lookahead_3ply import evaluate_near_endgame
            return evaluate_near_endgame(
                board, your_rack=your_rack, unseen_tiles=unseen_tiles,
                candidates=candidates, gaddag=gaddag,
                board_blanks=board_blanks, top_n=top_n,
                time_budget=time_budget)

    elapsed = time.perf_counter() - t_start
    results.sort(key=lambda r: -r['net_equity'])

    # Attach metadata (matching evaluate_near_endgame format)
    if results:
        results[0]['_meta'] = {
            'elapsed_s': round(elapsed, 2),
            'bag_size': bag_size,
            'unseen_count': unseen_count,
            'exhaust_evaluated': exhaust_count,
            'oneply_evaluated': oneply_count,
            'total_candidates': len(cands),
            'solver': 'near_endgame_parallel',
        }

    return results
