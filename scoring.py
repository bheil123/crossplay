"""
CROSSPLAY V7 - Scoring Module
Calculate scores for words and moves.
All positions are 1-indexed.
"""

from typing import List, Tuple, Dict, Optional
from .config import TILE_VALUES, BINGO_BONUS, RACK_SIZE
from .board import Board


def get_tile_value(letter: str) -> int:
    """Get point value of a tile."""
    return TILE_VALUES.get(letter.upper(), 0)


def calculate_word_score(
    board: Board,
    word: str,
    start_row: int,
    start_col: int,
    horizontal: bool,
    new_tile_positions: Optional[List[Tuple[int, int]]] = None,
    blanks_used: Optional[List[int]] = None
) -> int:
    """
    Calculate score for a single word.
    
    Args:
        board: The game board
        word: Word being scored
        start_row: Starting row (1-indexed)
        start_col: Starting column (1-indexed)
        horizontal: True if word is horizontal
        new_tile_positions: Positions of newly placed tiles (for bonus squares)
                           If None, assumes all tiles are new
        blanks_used: List of indices in word that use blank tiles (score 0)
    
    Returns:
        Score for the word
    """
    word = word.upper()
    
    if blanks_used is None:
        blanks_used = []
    blanks_set = set(blanks_used)
    
    if new_tile_positions is None:
        # Assume all positions are new
        new_tile_positions = []
        for i in range(len(word)):
            if horizontal:
                new_tile_positions.append((start_row, start_col + i))
            else:
                new_tile_positions.append((start_row + i, start_col))
    
    new_positions_set = set(new_tile_positions)
    
    word_score = 0
    word_multiplier = 1
    
    for i, letter in enumerate(word):
        if horizontal:
            row, col = start_row, start_col + i
        else:
            row, col = start_row + i, start_col
        
        # Blanks score 0
        if i in blanks_set:
            letter_value = 0
        else:
            letter_value = get_tile_value(letter)
        
        # Apply letter bonuses only for new tiles
        if (row, col) in new_positions_set:
            bonus = board.get_bonus(row, col)
            if bonus == '3L':
                letter_value *= 3
            elif bonus == '2L':
                letter_value *= 2
            elif bonus == '3W':
                word_multiplier *= 3
            elif bonus == '2W':
                word_multiplier *= 2
        
        word_score += letter_value
    
    return word_score * word_multiplier


def find_crosswords(
    board: Board,
    word: str,
    start_row: int,
    start_col: int,
    horizontal: bool,
    new_tile_positions: List[Tuple[int, int]]
) -> List[Dict]:
    """
    Find all crosswords formed by placing a word.
    
    Args:
        board: The game board (BEFORE the word is placed)
        word: Word being placed
        start_row: Starting row (1-indexed)
        start_col: Starting column (1-indexed)
        horizontal: True if main word is horizontal
        new_tile_positions: Positions where new tiles are placed
    
    Returns:
        List of crossword dicts: [{'word': str, 'row': int, 'col': int, 'horizontal': bool}]
    """
    crosswords = []
    
    for i, letter in enumerate(word):
        if horizontal:
            row, col = start_row, start_col + i
        else:
            row, col = start_row + i, start_col
        
        # Only check for crosswords at new tile positions
        if (row, col) not in new_tile_positions:
            continue
        
        # Look perpendicular to main word
        cross_horizontal = not horizontal
        
        # Build crossword by looking in both directions
        cross_word = letter
        cross_start_row, cross_start_col = row, col
        
        # Look backward
        if cross_horizontal:
            c = col - 1
            while c >= 1 and board.is_occupied(row, c):
                cross_word = board.get_tile(row, c) + cross_word
                cross_start_col = c
                c -= 1
        else:
            r = row - 1
            while r >= 1 and board.is_occupied(r, col):
                cross_word = board.get_tile(r, col) + cross_word
                cross_start_row = r
                r -= 1
        
        # Look forward
        if cross_horizontal:
            c = col + 1
            while c <= 15 and board.is_occupied(row, c):
                cross_word = cross_word + board.get_tile(row, c)
                c += 1
        else:
            r = row + 1
            while r <= 15 and board.is_occupied(r, col):
                cross_word = cross_word + board.get_tile(r, col)
                r += 1
        
        # Only count if crossword is 2+ letters
        if len(cross_word) >= 2:
            crosswords.append({
                'word': cross_word,
                'row': cross_start_row,
                'col': cross_start_col,
                'horizontal': cross_horizontal
            })
    
    return crosswords


def calculate_move_score(
    board: Board,
    word: str,
    start_row: int,
    start_col: int,
    horizontal: bool,
    blanks_used: List[int] = None,
    board_blanks: List[Tuple[int, int, str]] = None
) -> Tuple[int, List[Dict]]:
    """
    Calculate total score for a move including crosswords and bingo bonus.
    
    Args:
        board: The game board (BEFORE the move)
        word: Word being played
        start_row: Starting row (1-indexed)
        start_col: Starting column (1-indexed)
        horizontal: True if horizontal
        blanks_used: List of indices in word that use blank tiles
        board_blanks: List of (row, col, letter) for blanks already on board
    
    Returns:
        Tuple of (total_score, list of crosswords with scores)
    """
    word = word.upper()
    if blanks_used is None:
        blanks_used = []
    if board_blanks is None:
        board_blanks = []
    blanks_set = set(blanks_used)
    
    # Create set of board blank positions for quick lookup
    board_blank_positions = {(r, c) for r, c, _ in board_blanks}
    
    # Determine which positions are new (not already on board)
    new_tile_positions = []
    new_tile_indices = []  # Track which indices are new tiles
    for i, letter in enumerate(word):
        if horizontal:
            row, col = start_row, start_col + i
        else:
            row, col = start_row + i, start_col
        
        if board.is_empty(row, col):
            new_tile_positions.append((row, col))
            new_tile_indices.append(i)
    
    # Score the main word (board blanks don't affect main word since they're already placed)
    main_score = calculate_word_score(
        board, word, start_row, start_col, horizontal, new_tile_positions, blanks_used
    )
    
    # Find and score crosswords
    crosswords = find_crosswords(
        board, word, start_row, start_col, horizontal, new_tile_positions
    )
    
    crossword_total = 0
    crosswords_with_scores = []
    
    for cw in crosswords:
        # Figure out which positions in the crossword use blanks
        cw_blanks = []
        
        # Check for player's blanks in this crossword
        for main_idx in blanks_used:
            # Get position of blank in main word
            if horizontal:
                blank_row, blank_col = start_row, start_col + main_idx
            else:
                blank_row, blank_col = start_row + main_idx, start_col
            
            # Check if this position is in the crossword
            if cw['horizontal']:
                if blank_row == cw['row'] and cw['col'] <= blank_col < cw['col'] + len(cw['word']):
                    cw_blanks.append(blank_col - cw['col'])
            else:
                if blank_col == cw['col'] and cw['row'] <= blank_row < cw['row'] + len(cw['word']):
                    cw_blanks.append(blank_row - cw['row'])
        
        # Check for board blanks in this crossword
        for i, letter in enumerate(cw['word']):
            if cw['horizontal']:
                pos = (cw['row'], cw['col'] + i)
            else:
                pos = (cw['row'] + i, cw['col'])
            
            if pos in board_blank_positions:
                cw_blanks.append(i)
        
        cw_new_positions = [(r, c) for r, c in new_tile_positions 
             if (cw['horizontal'] and r == cw['row'] and cw['col'] <= c < cw['col'] + len(cw['word']))
             or (not cw['horizontal'] and c == cw['col'] and cw['row'] <= r < cw['row'] + len(cw['word']))]
        
        cw_score = calculate_word_score(
            board, cw['word'], cw['row'], cw['col'], cw['horizontal'],
            cw_new_positions, cw_blanks
        )
        crossword_total += cw_score
        crosswords_with_scores.append({
            'word': cw['word'],
            'row': cw['row'],
            'col': cw['col'],
            'horizontal': cw['horizontal'],
            'score': cw_score
        })
    
    total_score = main_score + crossword_total
    
    # Bingo bonus
    tiles_from_rack = len(new_tile_positions)
    
    if tiles_from_rack >= RACK_SIZE:
        total_score += BINGO_BONUS
    
    return (total_score, crosswords_with_scores)


if __name__ == "__main__":
    # Test scoring
    board = Board()
    
    # Test single word on empty board
    score, crosswords = calculate_move_score(board, "HELLO", 8, 4, True)
    print(f"HELLO at R8 C4: {score} pts")
    print(f"  H(3) + E(1) + L(2) + L(2) + O(1) = 9")
    print(f"  With 2W at R8 C4: 9 × 2 = 18")
    print()
    
    # Place the word
    board.place_word("HELLO", 8, 4, True)
    
    # Test crossword
    score, crosswords = calculate_move_score(board, "HI", 8, 4, False)
    print(f"HI at R8 C4 (V), crossing HELLO:")
    print(f"  Score: {score}")
    print(f"  Crosswords: {crosswords}")
