"""
CROSSPLAY V7 - Lookahead Module
2-ply lookahead for better move evaluation.

Uses incremental board updates for fast simulation.
"""

from typing import List, Dict, Tuple, Optional
from .board import Board
from .move_finder_gaddag import GADDAGMoveFinder
from .gaddag import get_gaddag
from .scoring import calculate_move_score
from .leave_eval import evaluate_leave


def evaluate_with_lookahead(
    board: Board,
    your_rack: str,
    opponent_tiles: str,
    gaddag=None,
    top_n: int = 12,
    include_blockers: bool = True,
    board_blanks: List[Tuple[int, int, str]] = None
) -> List[Dict]:
    """
    Evaluate moves with 2-ply lookahead.
    
    For each of your candidate moves:
    1. Simulate your move
    2. Find opponent's best response
    3. Calculate net outcome (your score - opponent's best)
    
    Args:
        board: Current board state
        your_rack: Your tile rack
        opponent_tiles: Opponent's tiles - can be:
            - Exact 7 tiles if known
            - All unseen tiles (opponent picks best from any 7)
            - Estimated tiles
        gaddag: GADDAG instance (loaded if not provided)
        top_n: Number of candidate moves to evaluate deeply
        include_blockers: Also evaluate moves that block premium squares
        board_blanks: List of (row, col, letter) for blanks already on board
        
    Returns:
        List of dicts with move info and lookahead equity
        
    Note:
        If opponent_tiles contains more than 7 tiles, this finds opponent's
        best possible move (as if they could choose any 7 from the pool).
        This is conservative - assumes worst case for you.
    """
    if gaddag is None:
        gaddag = get_gaddag()
    if board_blanks is None:
        board_blanks = []
    
    finder = GADDAGMoveFinder(board, gaddag, board_blanks=board_blanks)
    
    # Get your candidate moves
    your_moves = finder.find_all_moves(your_rack)
    if not your_moves:
        return []
    
    # Select candidates for deep evaluation
    candidates = _select_candidates(your_moves, board, top_n, include_blockers)
    
    results = []
    
    for move in candidates:
        # Get direction as boolean
        horizontal = move['direction'] == 'H'
        
        # Simulate your move
        placed = board.place_move(
            move['word'], 
            move['row'], 
            move['col'], 
            horizontal
        )
        
        # Limit blanks in opponent tiles to avoid exponential blowup
        # Move generation with 2+ blanks is extremely slow
        opp_tiles_limited = _limit_blanks(opponent_tiles, max_blanks=1)
        
        # Find opponent's best response
        opp_finder = GADDAGMoveFinder(board, gaddag, board_blanks=board_blanks)
        opp_moves = opp_finder.find_all_moves(opp_tiles_limited)
        
        if opp_moves:
            # Opponent's best is highest scoring
            opp_best = max(opp_moves, key=lambda m: m['score'])
            opp_best_score = opp_best['score']
            opp_best_word = opp_best['word']
        else:
            opp_best_score = 0
            opp_best_word = ""
        
        # Undo your move
        board.undo_move(placed)
        
        # Calculate lookahead equity
        # Your score - opponent's best response
        lookahead_equity = move['score'] - opp_best_score
        
        # Add leave value for tie-breaking
        tiles_used = move.get('tiles_used', move['word'])
        leave = _get_leave(your_rack, tiles_used)
        leave_value = evaluate_leave(leave)
        
        results.append({
            'word': move['word'],
            'row': move['row'],
            'col': move['col'],
            'direction': move['direction'],
            'score': move['score'],
            'opp_best': opp_best_score,
            'opp_word': opp_best_word,
            'lookahead_equity': lookahead_equity,
            'leave': leave,
            'leave_value': leave_value,
            'total_equity': lookahead_equity + leave_value,
        })
    
    # Sort by total equity
    results.sort(key=lambda x: -x['total_equity'])
    
    return results


def _select_candidates(
    moves: List[Dict], 
    board: Board, 
    top_n: int,
    include_blockers: bool
) -> List[Dict]:
    """
    Select candidate moves for deep evaluation.
    
    Includes:
    - Top N by score
    - Up to 5 additional moves that block premium squares (if enabled)
    """
    # Start with top N by score
    by_score = sorted(moves, key=lambda m: -m['score'])
    candidates = by_score[:top_n]
    candidate_ids = {id(m) for m in candidates}
    
    # Add a few blocking moves if not already included
    if include_blockers:
        blockers_added = 0
        max_blockers = 5  # Limit how many extra blockers to add
        
        for m in moves:
            if blockers_added >= max_blockers:
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
    word_len = len(move['word'])
    
    for i in range(word_len):
        if horizontal:
            r, c = row, col + i
        else:
            r, c = row + i, col
        
        # Check if this square is a premium and currently empty
        if board.is_empty(r, c):
            bonus = board.get_bonus(r, c)
            if bonus in ('3W', '2W', '3L'):
                return True
    
    return False


def _get_leave(rack: str, tiles_used: str) -> str:
    """Get remaining tiles after playing a move."""
    rack_list = list(rack.upper())
    for tile in tiles_used.upper():
        if tile in rack_list:
            rack_list.remove(tile)
        elif '?' in rack_list:
            rack_list.remove('?')
    return ''.join(rack_list)


def _limit_blanks(tiles: str, max_blanks: int = 1) -> str:
    """
    Limit the number of blanks in a tile string.
    
    Move generation with 2+ blanks is extremely slow (exponential).
    For practical 2-ply lookahead, we cap blanks.
    
    Args:
        tiles: Tile string (may contain '?' for blanks)
        max_blanks: Maximum blanks to keep (default 1)
        
    Returns:
        Tile string with excess blanks removed
    """
    blank_count = tiles.count('?')
    if blank_count <= max_blanks:
        return tiles
    
    # Remove excess blanks
    result = []
    blanks_kept = 0
    for c in tiles:
        if c == '?':
            if blanks_kept < max_blanks:
                result.append(c)
                blanks_kept += 1
            # else: skip this blank
        else:
            result.append(c)
    
    return ''.join(result)


# Convenience function for quick analysis
def quick_lookahead(board: Board, rack: str, opp_tiles: str, top_n: int = 10) -> None:
    """
    Print quick 2-ply lookahead analysis.
    
    Args:
        board: Current board
        rack: Your rack
        opp_tiles: Opponent's tiles (or estimate)
        top_n: Number of moves to show
    """
    print(f"2-PLY LOOKAHEAD ANALYSIS")
    print(f"Your rack: {rack} | Opponent tiles: {opp_tiles}")
    print("=" * 70)
    
    results = evaluate_with_lookahead(board, rack, opp_tiles, top_n=top_n)
    
    if not results:
        print("No valid moves found.")
        return
    
    print(f"{'#':<3} {'Word':<12} {'Position':<10} {'Pts':>4} {'Opp':>4} "
          f"{'Net':>5} {'Leave':>7} {'Equity':>7}")
    print("-" * 70)
    
    for i, m in enumerate(results[:top_n], 1):
        pos = f"R{m['row']}C{m['col']} {m['direction']}"
        print(f"{i:<3} {m['word']:<12} {pos:<10} {m['score']:>4} "
              f"{m['opp_best']:>4} {m['lookahead_equity']:>+5} "
              f"{m['leave']:>7} {m['total_equity']:>+7.1f}")


if __name__ == "__main__":
    # Test
    from .board import Board
    
    board = Board()
    board.place_word("HELLO", 8, 4, True)
    
    print("Board:")
    print(board)
    print()
    
    quick_lookahead(board, "AEINRST", "BDGLMOP")
