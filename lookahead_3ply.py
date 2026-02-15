"""
CROSSPLAY V13 - 3-Ply Lookahead Module
Deep analysis for endgame situations (bag ≤ 12, unseen ≤ 19).

3-ply: Your move → Opponent's best → Your best response

Features:
  - 30-second time budget with adaptive per-candidate timing
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
    unseen_limited = _limit_blanks(unseen_tiles, max_blanks=1)
    
    # Blank correction: compensate for capping blanks at 1
    from .mc_eval import _blank_correction_factor
    blank_corr = _blank_correction_factor(unseen_count, blanks_in_unseen)

    # --- Auto-scale top_n based on unseen count ---
    if top_n is None:
        if unseen_count <= 10:
            top_n = 12
        elif unseen_count <= 14:
            top_n = 10
        elif unseen_count <= 17:
            top_n = 8
        elif unseen_count <= 19:
            top_n = 5   # ~30s budget, ~6s each
        else:
            top_n = 4   # tight budget

    # --- PLY 0: Generate your candidate moves ---
    finder = GADDAGMoveFinder(board, gaddag, board_blanks=board_blanks)
    your_moves = finder.find_all_moves(your_rack)

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
        opp_finder = GADDAGMoveFinder(board, gaddag, board_blanks=board_blanks)
        opp_moves = opp_finder.find_all_moves(unseen_limited)

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

            your_resp_finder = GADDAGMoveFinder(board, gaddag, board_blanks=board_blanks)
            your_resp_moves = your_resp_finder.find_all_moves(your_leave)

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

        net_3ply = move['score'] - int(opp_score * blank_corr) + your_resp_score
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


def _limit_blanks(tiles: str, max_blanks: int = 1) -> str:
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
