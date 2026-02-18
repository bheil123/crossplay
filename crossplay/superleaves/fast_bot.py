"""
CROSSPLAY V15 - Fast Greedy Bot for SuperLeaves Training

Selects argmax(score + leave_value) from top moves.
No MC, no risk analysis -- pure greedy for speed.
"""

from ..config import RACK_SIZE


def compute_tiles_used(board, move):
    """Determine which rack tiles a move consumes.

    Args:
        board: Board instance (1-indexed get_tile)
        move: move dict with word, row, col, direction

    Returns:
        list of letters used from rack
    """
    word = move['word']
    row, col = move['row'], move['col']
    horiz = move['direction'] == 'H'

    used = []
    for i, letter in enumerate(word):
        r = row if horiz else row + i
        c = col + i if horiz else col
        if not board.get_tile(r, c):
            used.append(letter)
    return used


def compute_leave(rack_list, tiles_used):
    """Compute leave (remaining tiles) after playing tiles_used.

    Args:
        rack_list: list of letters in rack (uppercase, '?' for blanks)
        tiles_used: list of letters consumed by the move

    Returns:
        sorted tuple of remaining tiles (leave key)
    """
    remaining = list(rack_list)
    for t in tiles_used:
        if t in remaining:
            remaining.remove(t)
        elif '?' in remaining:
            # Move used a blank (placed as a letter not in rack)
            remaining.remove('?')
    return tuple(sorted(remaining))


def select_best_move(board, moves, rack_list, leave_table, bag_size,
                     top_k=30):
    """Select the best move by greedy score + leave evaluation.

    Args:
        board: Board instance
        moves: list of move dicts sorted by score descending
        rack_list: list of rack tiles (uppercase, '?' for blanks)
        leave_table: LeaveTable instance
        bag_size: tiles remaining in bag
        top_k: only evaluate top K moves by raw score

    Returns:
        (best_move, best_leave_key, best_equity) or (None, None, 0)
        if no moves available
    """
    if not moves:
        return None, None, 0.0

    best_move = None
    best_leave = None
    best_equity = float('-inf')

    candidates = moves[:top_k]

    for move in candidates:
        tiles_used = compute_tiles_used(board, move)
        leave_key = compute_leave(rack_list, tiles_used)

        # move['score'] already includes bingo bonus from find_all_moves
        score = move['score']

        # Leave value handling per Crossplay rules
        if bag_size == 0:
            leave_val = 0.0
        elif bag_size < RACK_SIZE:
            leave_val = leave_table.get(leave_key, 0.0) * (bag_size / RACK_SIZE)
        else:
            leave_val = leave_table.get(leave_key, 0.0)

        equity = score + leave_val

        if equity > best_equity:
            best_equity = equity
            best_move = move
            best_leave = leave_key

    return best_move, best_leave, best_equity
