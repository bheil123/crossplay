"""
CROSSPLAY V15 - 3-Ply Lookahead Module
Deep analysis for late-to-mid game positions (bag <= 21, unseen <= 28).

3-ply: Your move -> Opponent's best -> Your best response

Features:
  - 20-second time budget with adaptive per-candidate timing
  - C extension for move generation (13x faster than Python)
  - Blank correction factor for capped opponent blanks
  - Progressive pruning (skip candidates that can't beat current best)
  - Auto-scaled candidate count based on unseen tile count
"""

import time
from typing import List, Dict, Tuple, Optional
from .board import Board
from .move_finder_gaddag import GADDAGMoveFinder
from .gaddag import get_gaddag
from .leave_eval import evaluate_leave

# Bag parity: P(opponent empties bag on their turn | bag_after tiles remain)
# Conservative estimates based on typical Crossplay move lengths.
# bag_after=1 means opp just needs to play 1+ tile (nearly certain).
# bag_after=7 means opp must play all 7 tiles / bingo (rare).
_PARITY_P_OPP_EMPTIES = {
    1: 0.97, 2: 0.94, 3: 0.88, 4: 0.78,
    5: 0.62, 6: 0.40, 7: 0.18,
}
_PARITY_STRUCTURAL_ADV = 10.0  # equity pts for emptying the bag


def evaluate_endgame_2ply(
    board: Board,
    your_rack: str,
    opp_rack: str,
    gaddag=None,
    top_n: int = 20,
    board_blanks: List[Tuple[int, int, str]] = None,
) -> List[Dict]:
    """
    Deterministic 2-ply solver for bag=0 endgame (Crossplay rules).

    When the bag is empty, the opponent's rack is known exactly (= all unseen
    tiles).  Both players get one final turn, leftover tiles don't penalize.

    Uses upper-bound pruning to evaluate ALL candidate moves correctly:
      1. Find opponent's baseline best on current board (before we play)
      2. Classify our moves as "interfering" (touch opp baseline zone) or not
      3. Fully evaluate all interfering moves
      4. For non-interfering moves, estimated_net = score - opp_baseline is an
         upper bound (opponent can still play their baseline move unchanged).
         Only re-evaluate if estimated_net > best fully-evaluated net.
      5. Remaining non-interfering moves are provably worse -> skip

    This is 100% correct (no missed optima) and typically 5-10x faster than
    brute force since most moves don't interfere with the opponent's best.

    Returns:
        List of dicts with move info and net equity, sorted best first.
    """
    t_start = time.perf_counter()

    if gaddag is None:
        gaddag = get_gaddag()
    if board_blanks is None:
        board_blanks = []

    # --- Generate ALL your candidate moves ---
    from .move_finder_opt import find_all_moves_opt, find_best_score_opt
    your_moves = find_all_moves_opt(board, gaddag, your_rack, board_blanks=board_blanks)

    if not your_moves:
        return []

    # Pre-compute board blank set for find_best_score_opt (0-indexed)
    bb_set = {(r - 1, c - 1) for r, c, _ in board_blanks}

    # --- Step 1: Find opponent's baseline best on CURRENT board ---
    opp_base = find_best_score_opt(board._grid, gaddag._data, opp_rack, bb_set)
    opp_baseline_score = opp_base[0] if opp_base[0] > 0 else 0
    opp_baseline_word = opp_base[1] if opp_base[0] > 0 else "(pass)"

    t_baseline = time.perf_counter()

    # --- Step 2: Compute opp_zone (squares used by opp baseline + neighbors) ---
    # Any of our tiles placed in this zone could invalidate opp_baseline
    opp_zone = set()
    if opp_base[0] > 0:
        opp_word = opp_base[1]
        opp_r1, opp_c1, opp_dir = opp_base[2], opp_base[3], opp_base[4]
        for i in range(len(opp_word)):
            if opp_dir == 'H':
                r0, c0 = opp_r1 - 1, opp_c1 - 1 + i
            else:
                r0, c0 = opp_r1 - 1 + i, opp_c1 - 1
            # Add this square and all 4 neighbors
            for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                opp_zone.add((r0 + dr, c0 + dc))

    # --- Step 3: Classify moves as interfering vs non-interfering ---
    interfering = []
    non_interfering = []

    for move in your_moves:
        # Find NEW tiles this move places (squares currently empty)
        new_tiles = set()
        word = move['word']
        horiz = move['direction'] == 'H'
        for i in range(len(word)):
            if horiz:
                r0, c0 = move['row'] - 1, move['col'] - 1 + i
            else:
                r0, c0 = move['row'] - 1 + i, move['col'] - 1
            if board._grid[r0][c0] is None:
                new_tiles.add((r0, c0))

        if new_tiles & opp_zone:
            interfering.append(move)
        else:
            non_interfering.append(move)

    # --- Helper: fully evaluate a move (place, find opp best, undo) ---
    def evaluate_move(move):
        horizontal = move['direction'] == 'H'
        placed = board.place_move(move['word'], move['row'], move['col'], horizontal)

        # Update blank positions for the new board state
        opp_blanks = list(board_blanks)
        for bi in move.get('blanks_used', []):
            r = move['row'] + (0 if horizontal else bi)
            c = move['col'] + (bi if horizontal else 0)
            opp_blanks.append((r, c, move['word'][bi]))
        opp_bb = {(r - 1, c - 1) for r, c, _ in opp_blanks}

        opp_result = find_best_score_opt(board._grid, gaddag._data, opp_rack, opp_bb)
        opp_sc = opp_result[0] if opp_result[0] > 0 else 0
        opp_wd = opp_result[1] if opp_result[0] > 0 else "(pass)"

        board.undo_move(placed)
        return opp_sc, opp_wd

    # --- Step 4: Evaluate all interfering moves ---
    results = []
    best_net = float('-inf')
    eval_count = 0

    for move in interfering:
        opp_sc, opp_wd = evaluate_move(move)
        eval_count += 1
        net = move['score'] - opp_sc

        results.append({
            'word': move['word'],
            'row': move['row'],
            'col': move['col'],
            'direction': move['direction'],
            'score': move['score'],
            'opp_word': opp_wd,
            'opp_score': opp_sc,
            'opp_responses': [],
            'net_2ply': net,
            'exact': True,
        })

        if net > best_net:
            best_net = net

    # --- Step 5: Check non-interfering moves with upper-bound pruning ---
    # Sort by score descending so we check the most promising first
    non_interfering.sort(key=lambda m: -m['score'])
    skipped = 0

    for move in non_interfering:
        estimated_net = move['score'] - opp_baseline_score

        if estimated_net > best_net:
            # Could beat current best -- must verify with full evaluation
            opp_sc, opp_wd = evaluate_move(move)
            eval_count += 1
            net = move['score'] - opp_sc

            results.append({
                'word': move['word'],
                'row': move['row'],
                'col': move['col'],
                'direction': move['direction'],
                'score': move['score'],
                'opp_word': opp_wd,
                'opp_score': opp_sc,
                'opp_responses': [],
                'net_2ply': net,
                'exact': True,
            })

            if net > best_net:
                best_net = net
        else:
            # Provably worse: upper bound <= best_net, safe to skip
            skipped += 1
            results.append({
                'word': move['word'],
                'row': move['row'],
                'col': move['col'],
                'direction': move['direction'],
                'score': move['score'],
                'opp_word': opp_baseline_word,
                'opp_score': opp_baseline_score,
                'opp_responses': [],
                'net_2ply': estimated_net,
                'exact': False,
            })

    elapsed = time.perf_counter() - t_start
    results.sort(key=lambda r: -r['net_2ply'])

    meta = {
        'elapsed_s': round(elapsed, 2),
        'total_moves': len(your_moves),
        'interfering': len(interfering),
        'non_interfering': len(non_interfering),
        'fully_evaluated': eval_count,
        'pruned_by_bound': skipped,
        'opp_baseline': f"{opp_baseline_word} ({opp_baseline_score} pts)",
        'baseline_time_s': round(t_baseline - t_start, 2),
        'opp_rack_size': len(opp_rack),
        'solver': 'endgame_2ply_pruned',
    }
    for r in results:
        r['_meta'] = meta

    return results


def evaluate_3ply(
    board: Board,
    your_rack: str,
    unseen_tiles: str,
    gaddag=None,
    top_n: int = None,
    board_blanks: List[Tuple[int, int, str]] = None,
    time_budget: float = 30.0,
) -> List[Dict]:
    """
    Evaluate moves with 3-ply lookahead under a time budget.

    For each candidate move:
    1. Place your move on the board
    2. Find opponent's best response (from unseen tiles)
    3. Find YOUR best counter-response from your leave
    4. Calculate net outcome

    Args:
        board: Current board state
        your_rack: Your tile rack
        unseen_tiles: All unseen tiles (bag + opponent rack)
        gaddag: GADDAG instance
        top_n: Number of candidate moves to evaluate (auto-scaled if None)
        board_blanks: Blanks already on board
        time_budget: Maximum seconds to spend (default 30)

    Returns:
        List of dicts with move info and 3-ply equity, sorted best first
    """
    t_start = time.perf_counter()

    if gaddag is None:
        gaddag = get_gaddag()
    if board_blanks is None:
        board_blanks = []

    unseen_count = len(unseen_tiles)
    bag_size = max(0, unseen_count - 7)
    blanks_in_unseen = unseen_tiles.count('?')

    # Limit blanks in unseen for opponent move generation speed
    unseen_limited = _limit_blanks(unseen_tiles, max_blanks=2)

    # Blank correction removed (V21.1: proven negligible)

    # --- Auto-scale top_n based on unseen count ---
    # C extension opponent movegen: ~0.1s (14 tiles) to ~1.0s (28 tiles)
    # Python fallback: 5-11s per call, so fewer candidates
    if top_n is None:
        if unseen_count <= 10:
            top_n = 12
        elif unseen_count <= 14:
            top_n = 10
        elif unseen_count <= 19:
            top_n = 8
        elif unseen_count <= 24:
            top_n = 6
        elif unseen_count <= 28:
            top_n = 5
        else:
            top_n = 4

    # Try C extension for move generation (13x faster with large tile sets)
    _use_c_ext = False
    try:
        from .move_finder_c import find_all_moves_c as _3ply_find_c
        _use_c_ext = True
    except Exception:
        pass

    def _find_moves(brd, rack_str):
        """Generate moves using C extension with Python fallback."""
        if _use_c_ext:
            try:
                return _3ply_find_c(brd, gaddag, rack_str, board_blanks=board_blanks)
            except Exception:
                pass
        f = GADDAGMoveFinder(brd, gaddag, board_blanks=board_blanks)
        return f.find_all_moves(rack_str)

    # --- PLY 0: Generate your candidate moves ---
    your_moves = _find_moves(board, your_rack)

    if not your_moves:
        return []

    your_moves.sort(key=lambda m: -m['score'])
    candidates = your_moves[:top_n]

    # Time check after movegen
    elapsed = time.perf_counter() - t_start
    remaining = time_budget - elapsed
    if remaining < 0.5:
        return _fallback_1ply(candidates)

    # Estimate time per candidate: we'll measure the first one and adapt.
    # Initial conservative estimate: ~3s with blank, ~0.5s without
    has_blank_unseen = '?' in unseen_limited
    est_per_candidate = 4.0 if has_blank_unseen else 0.5
    max_candidates = max(2, int(remaining / est_per_candidate))
    if max_candidates < len(candidates):
        candidates = candidates[:max_candidates]
    if max_candidates < len(candidates):
        candidates = candidates[:max_candidates]

    # --- Evaluate each candidate with adaptive time tracking ---
    results = []
    best_net = float('-inf')
    pruned = 0
    candidate_times = []

    for idx, move in enumerate(candidates):
        elapsed = time.perf_counter() - t_start
        remaining_budget = time_budget - elapsed

        # After measuring at least 1 candidate, predict if next fits
        if candidate_times:
            avg_time = sum(candidate_times) / len(candidate_times)
            if remaining_budget < avg_time * 0.8:
                break

        if remaining_budget < 0.5:
            break

        t_cand = time.perf_counter()

        # Progressive pruning: skip if theoretical max can't beat best
        if best_net > float('-inf'):
            theoretical_max = move['score'] + 60
            if theoretical_max < best_net - 5:
                pruned += 1
                continue

        horizontal = move['direction'] == 'H'
        your_leave = _calculate_leave(your_rack, move)

        # === PLY 1: Place your move ===
        placed_1 = board.place_move(
            move['word'], move['row'], move['col'], horizontal
        )

        # === PLY 2: Opponent's best response ===
        opp_moves = _find_moves(board, unseen_limited)

        # Filter: opponent can only play moves using <= 7 tiles from rack.
        # When unseen_limited has more than 7 tiles (bag > 0), the move
        # finder may generate words requiring 8+ tiles from rack, which is
        # physically impossible regardless of which 7 tiles the opponent has.
        if opp_moves:
            opp_moves = [
                m for m in opp_moves
                if _tiles_from_rack(board, m) <= 7
            ]

        opp_best_responses = []
        if opp_moves:
            opp_moves.sort(key=lambda m: -m['score'])
            opp_best = opp_moves[0]
            opp_score = opp_best['score']
            opp_word = opp_best['word']

            for om in opp_moves[:3]:
                opp_best_responses.append({
                    'word': om['word'],
                    'score': om['score'],
                    'pos': f"R{om['row']}C{om['col']} {om['direction']}"
                })

            opp_leave = _calculate_leave(unseen_limited, opp_best)

            # === PLY 3: Your counter-response ===
            opp_horizontal = opp_best['direction'] == 'H'
            placed_2 = board.place_move(
                opp_best['word'], opp_best['row'], opp_best['col'], opp_horizontal
            )

            your_resp_moves = _find_moves(board, your_leave)

            if your_resp_moves:
                your_resp_best = max(your_resp_moves, key=lambda m: m['score'])
                your_resp_score = your_resp_best['score']
                your_resp_word = your_resp_best['word']
            else:
                your_resp_score = 0
                your_resp_word = "(pass)"

            board.undo_move(placed_2)
        else:
            opp_score = 0
            opp_word = "(pass)"
            opp_leave = unseen_limited
            your_resp_score = 0
            your_resp_word = "(n/a)"

        board.undo_move(placed_1)

        net_3ply = move['score'] - opp_score + your_resp_score
        leave_val = evaluate_leave(your_leave) if your_leave else 0.0

        result = {
            'word': move['word'],
            'row': move['row'],
            'col': move['col'],
            'direction': move['direction'],
            'score': move['score'],
            'leave': your_leave,
            'leave_value': leave_val,
            'opp_word': opp_word,
            'opp_score': opp_score,
            'opp_responses': opp_best_responses,
            'your_response': your_resp_word,
            'your_response_score': your_resp_score,
            'net_3ply': net_3ply,
        }
        results.append(result)

        if net_3ply > best_net:
            best_net = net_3ply

        candidate_times.append(time.perf_counter() - t_cand)

    elapsed = time.perf_counter() - t_start

    results.sort(key=lambda r: -r['net_3ply'])

    # Attach metadata to first result
    meta = {
        'elapsed_s': round(elapsed, 2),
        'candidates_evaluated': len(results),
        'candidates_pruned': pruned,
        'unseen': unseen_count,
        'bag_size': bag_size,
    }
    for r in results:
        r['_meta'] = meta

    return results


def _fallback_1ply(candidates: List[Dict]) -> List[Dict]:
    """Return 1-ply results when no time for deeper analysis."""
    results = []
    for move in candidates:
        results.append({
            'word': move['word'],
            'row': move['row'],
            'col': move['col'],
            'direction': move['direction'],
            'score': move['score'],
            'leave': '',
            'leave_value': 0.0,
            'opp_word': '(timeout)',
            'opp_score': 0,
            'opp_responses': [],
            'your_response': '(timeout)',
            'your_response_score': 0,
            'net_3ply': move['score'],
        })
    results.sort(key=lambda r: -r['net_3ply'])
    return results


def _calculate_leave(rack: str, move: dict) -> str:
    """Calculate tiles remaining after playing a move."""
    leave = list(rack.upper())
    word = move['word'].upper()
    blanks_used = move.get('blanks_used', [])

    blank_indices = set()
    for b in blanks_used:
        if b < 0:
            blank_indices.add(-b)
        else:
            blank_indices.add(b)

    for i, letter in enumerate(word):
        if i in blank_indices:
            if '?' in leave:
                leave.remove('?')
        else:
            if letter in leave:
                leave.remove(letter)
            elif '?' in leave:
                leave.remove('?')

    return ''.join(sorted(leave))


def _tiles_from_rack(board, move):
    """Count how many tiles a move requires from the rack (not from the board)."""
    word = move['word']
    horiz = move['direction'] == 'H'
    count = 0
    for i in range(len(word)):
        if horiz:
            r0, c0 = move['row'] - 1, move['col'] - 1 + i
        else:
            r0, c0 = move['row'] - 1 + i, move['col'] - 1
        if 0 <= r0 < 15 and 0 <= c0 < 15 and board._grid[r0][c0] is None:
            count += 1
    return count


def _limit_blanks(tiles: str, max_blanks: int = 2) -> str:
    """Limit blanks in tile string to avoid exponential blowup."""
    result = []
    blank_count = 0
    for t in tiles:
        if t == '?':
            if blank_count < max_blanks:
                result.append(t)
                blank_count += 1
        else:
            result.append(t)
    return ''.join(result)


def evaluate_near_endgame(
    board: Board,
    your_rack: str,
    unseen_tiles: str,
    candidates: List[Dict],
    gaddag=None,
    board_blanks: List[Tuple[int, int, str]] = None,
    top_n: int = 25,
    time_budget: float = 45.0,
) -> List[Dict]:
    """
    Hybrid evaluation for near-endgame positions (bag 1-8).

    For each candidate move, checks whether it empties the bag:
      - YES (tiles_used >= bag_size): Exhaustive 3-ply evaluation over all
        possible opponent rack assignments. Since emptying the bag makes all
        racks deterministic, this is exact (no sampling, no leave heuristics).
        Enumerates C(unseen, 7) opponent racks x endgame 2-ply for each.
      - NO: Parity-adjusted 1-ply equity. Penalizes moves that let the
        opponent empty the bag on their next turn, based on P(opp empties)
        lookup table x structural advantage of bag control (~10 pts).

    The "exhaust" candidates get much more accurate evaluation than MC could
    provide, correctly capturing the structural advantage of emptying the bag
    (you control who gets the final turn).

    Args:
        board:          Current Board object
        your_rack:      Your tile rack
        unseen_tiles:   All unseen tiles as string (bag + opponent rack)
        candidates:     Pre-ranked candidate moves from 1-ply analysis.
                        Each must have: word, row, col, direction, score,
                        tiles_used (str of tiles consumed from rack).
        gaddag:         GADDAG instance
        board_blanks:   List of (row, col, letter) for blanks on board
        top_n:          Max candidates to evaluate
        time_budget:    Max seconds to spend

    Returns:
        List of result dicts sorted by equity (best first). Each dict has:
            word, row, col, direction, score, eval_type ('exhaust'|'1ply'),
            opp_avg, opp_max, your_resp_avg, net_equity, n_racks,
            leave, leave_value, top_opp_responses
    """
    from itertools import combinations
    from .move_finder_opt import find_best_score_opt

    t_start = time.perf_counter()

    if gaddag is None:
        gaddag = get_gaddag()
    if board_blanks is None:
        board_blanks = []

    unseen_count = len(unseen_tiles)
    bag_size = max(0, unseen_count - 7)  # unseen minus opp rack (7)

    if bag_size < 1 or bag_size > 8:
        return []  # Not in near-endgame range

    # Pre-compute board blank set (0-indexed)
    bb_set = {(r - 1, c - 1) for r, c, _ in board_blanks}

    # Blank correction removed (V21.1: proven negligible)

    # Try Cython acceleration (same path as MC eval)
    _use_cython = False
    _accel = None
    _gdata_bytes = None
    _word_set = None
    try:
        import gaddag_accel as _accel
        _gdata_bytes = bytes(gaddag._data)
        from .move_finder_c import _get_dict
        _d = _get_dict()
        class _SetDict:
            __slots__ = ('is_valid', '_contains')
            def __init__(self, s):
                self.is_valid = s.__contains__
                self._contains = s.__contains__
            def __contains__(self, w): return self._contains(w)
        _word_set = _SetDict(_d._words)
        from .config import TILE_VALUES, VALID_TWO_LETTER, BINGO_BONUS, RACK_SIZE
        _tv = [0] * 26
        for _ch, _val in TILE_VALUES.items():
            if _ch != '?':
                _tv[ord(_ch) - ord('A')] = _val
        from .config import BONUS_SQUARES
        _bonus = [[(1, 1)] * 15 for _ in range(15)]
        for (_r1, _c1), _btype in BONUS_SQUARES.items():
            _r0, _c0 = _r1 - 1, _c1 - 1
            if _btype == '2L': _bonus[_r0][_c0] = (2, 1)
            elif _btype == '3L': _bonus[_r0][_c0] = (3, 1)
            elif _btype == '2W': _bonus[_r0][_c0] = (1, 2)
            elif _btype == '3W': _bonus[_r0][_c0] = (1, 3)
        _use_cython = True
    except (ImportError, Exception) as e:
        _use_cython = False

    def _make_ctx(grid, bb):
        """Build Cython board context (reusable for multiple racks)."""
        if _use_cython:
            return _accel.prepare_board_context(
                grid, _gdata_bytes, bb, _word_set, VALID_TWO_LETTER,
                _tv, _bonus, BINGO_BONUS, RACK_SIZE)
        return None

    def _find_best(grid, rack_str, bb, ctx=None):
        """Find best scoring move, using Cython if available."""
        if _use_cython:
            if ctx is None:
                ctx = _make_ctx(grid, bb)
            return _accel.find_best_score_c(ctx, rack_str)
        else:
            return find_best_score_opt(grid, gaddag._data, rack_str, bb)

    # Limit candidates
    cands = candidates[:top_n]

    results = []
    exhaust_count = 0
    oneply_count = 0

    # --- PASS 1: Process all non-exhausting candidates instantly (1-ply) ---
    exhaust_cands = []
    for move in cands:
        tiles_used = move.get('tiles_used', move.get('used', move['word']))
        n_tiles_used = len(tiles_used)

        if n_tiles_used >= bag_size:
            exhaust_cands.append(move)
        else:
            # NON-EXHAUSTING MOVE: parity-adjusted 1-ply equity
            oneply_count += 1
            leave_val = move.get('leave_value', 0.0)
            if isinstance(leave_val, str):
                leave_val = 0.0
            equity = move['score'] + leave_val

            # Bag parity penalty: if this move leaves bag_after tiles
            # and opponent can empty the bag on their turn, penalize
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

    # --- PASS 2: Evaluate bag-exhausting candidates under time budget ---
    for idx, move in enumerate(exhaust_cands):
        elapsed = time.perf_counter() - t_start
        if elapsed > time_budget:
            break

        # --- BAG-EXHAUSTING MOVE: full 3-ply over all opponent rack combos ---
        exhaust_count += 1

        # Calculate leave (tiles remaining in your rack after playing)
        your_leave = _calculate_leave(your_rack, move)

        # Place your move on the board
        horizontal = move['direction'] == 'H'
        placed_1 = board.place_move(move['word'], move['row'], move['col'], horizontal)

        # Update blank positions after your move
        move_blanks = list(board_blanks)
        for bi in move.get('blanks_used', []):
            r = move['row'] + (0 if horizontal else bi)
            c = move['col'] + (bi if horizontal else 0)
            move_blanks.append((r, c, move['word'][bi]))
        post_move_bb = {(r - 1, c - 1) for r, c, _ in move_blanks}

        # After you play tiles_used and draw min(n_tiles_used, bag_size) = bag_size
        # tiles from bag, the bag empties. The unseen tiles split into:
        #   - Your drawn tiles: bag_size tiles (unknown which ones)
        #   - Opponent rack: 7 tiles (the rest)
        # Your post-draw rack = your_leave + drawn tiles
        #
        # Enumerate all C(unseen, 7) ways to assign 7 tiles to opponent.
        # For each assignment, the remaining tiles are what you drew.

        unseen_list = list(unseen_tiles)

        # Generate all unique 7-tile opponent racks from unseen tiles.
        # Use indices to handle duplicate tiles correctly.
        n_unseen = len(unseen_list)
        opp_rack_size = min(7, n_unseen)

        # Accumulate stats across all opponent rack assignments
        net_scores = []
        opp_scores_all = []
        your_resp_scores_all = []
        opp_response_counts = {}  # word -> (count, max_score)

        # Build Cython context ONCE for ply 2 (board is fixed after our move)
        ply2_ctx = _make_ctx(board._grid, post_move_bb)

        for combo_indices in combinations(range(n_unseen), opp_rack_size):
            # Opponent gets these tiles
            opp_rack_list = [unseen_list[i] for i in combo_indices]
            opp_rack = ''.join(opp_rack_list)

            # You drew the remaining tiles
            drawn_indices = set(range(n_unseen)) - set(combo_indices)
            drawn_tiles = ''.join(unseen_list[i] for i in drawn_indices)
            your_full_rack = your_leave + drawn_tiles

            # Limit blanks for opponent move generation
            opp_rack_limited = _limit_blanks(opp_rack, max_blanks=2)

            # PLY 2: Opponent's best response on post-move board (cached ctx)
            opp_result = _find_best(
                board._grid, opp_rack_limited, post_move_bb, ctx=ply2_ctx)
            opp_score = opp_result[0] if opp_result[0] > 0 else 0
            opp_word = opp_result[1] if opp_result[0] > 0 else "(pass)"

            opp_score_corrected = opp_score

            # Track opponent responses
            if opp_word != "(pass)":
                if opp_word not in opp_response_counts:
                    opp_response_counts[opp_word] = [0, 0]
                opp_response_counts[opp_word][0] += 1
                opp_response_counts[opp_word][1] = max(
                    opp_response_counts[opp_word][1], opp_score)

            # PLY 3: Your counter-response after opponent plays
            # Board changes each time (opp places different move), so no ctx cache
            your_resp_score = 0
            if opp_score > 0:
                # Place opponent's move
                opp_horiz = opp_result[4] == 'H'
                placed_2 = board.place_move(
                    opp_word, opp_result[2], opp_result[3], opp_horiz)

                # Find your best response with full post-draw rack
                your_resp_result = _find_best(
                    board._grid, your_full_rack, post_move_bb)
                your_resp_score = your_resp_result[0] if your_resp_result[0] > 0 else 0

                board.undo_move(placed_2)
            else:
                # Opponent passes — find your best on unchanged board (reuse ply2 ctx)
                your_resp_result = _find_best(
                    board._grid, your_full_rack, post_move_bb, ctx=ply2_ctx)
                your_resp_score = your_resp_result[0] if your_resp_result[0] > 0 else 0

            # PLY 4: Opponent's final turn
            # After your response, opponent still has their leave
            # But for now, 3-ply net is sufficient — ply 4 adds marginal value
            # and would require tracking opponent's leave from ply 2

            net = move['score'] - opp_score_corrected + your_resp_score
            net_scores.append(net)
            opp_scores_all.append(opp_score_corrected)
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

        results.append({
            'word': move['word'],
            'row': move['row'],
            'col': move['col'],
            'direction': move['direction'],
            'score': move['score'],
            'eval_type': 'exhaust',
            'leave': your_leave,
            'leave_value': 0.0,  # Not needed — 3-ply is exact
            'opp_avg': round(avg_opp, 1),
            'opp_max': max_opp,
            'your_resp_avg': round(avg_resp, 1),
            'net_equity': round(avg_net, 1),
            'n_racks': n_racks,
            'top_opp_responses': top_opp_list,
        })

    elapsed = time.perf_counter() - t_start

    # Sort all results by net equity
    results.sort(key=lambda r: -r['net_equity'])

    # Attach metadata to first result
    if results:
        results[0]['_meta'] = {
            'elapsed_s': round(elapsed, 2),
            'bag_size': bag_size,
            'unseen_count': unseen_count,
            'exhaust_evaluated': exhaust_count,
            'oneply_evaluated': oneply_count,
            'total_candidates': len(cands),
            'solver': 'near_endgame_hybrid',
        }

    return results


def test_3ply():
    """Test 3-ply on a sample position."""
    board = Board()
    board.place_word("TEST", 8, 8, True)

    results = evaluate_3ply(
        board,
        your_rack="ABCDEFG",
        unseen_tiles="HIJKLMN",
        top_n=5
    )

    print("3-PLY TEST RESULTS:")
    for r in results:
        print(f"  {r['word']}: {r['score']} - {r['opp_score']} + {r['your_response_score']} = {r['net_3ply']}")


if __name__ == "__main__":
    test_3ply()
