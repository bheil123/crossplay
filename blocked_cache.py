"""
Blocked Square Cache - tracks squares that can never have a valid letter placed.

Once a square is blocked (no letter forms valid crosswords), it stays blocked
because constraints only get tighter as more tiles are placed.
"""

from typing import Set, Tuple, Dict, Optional
from .config import BONUS_SQUARES, VALID_TWO_LETTER


class BlockedSquareCache:
    """
    Cache of permanently blocked squares.
    
    A square is blocked if NO letter A-Z can be legally placed there
    (i.e., no letter forms valid crosswords with all adjacent tiles).
    
    Once blocked, always blocked - constraints only tighten.
    """
    
    def __init__(self):
        self._blocked: Set[Tuple[int, int]] = set()
        self._occupied: Set[Tuple[int, int]] = set()
        self._playable: Set[Tuple[int, int]] = set()  # Known playable (can change)
        self._valid_crosses: Dict[str, Set[str]] = self._build_valid_crosses()
    
    def _build_valid_crosses(self) -> Dict[str, Set[str]]:
        """
        Build lookup: for each letter, what letters can go before/after it
        to form a valid 2-letter word.
        """
        before = {}  # before[X] = letters that can precede X (form ?X)
        after = {}   # after[X] = letters that can follow X (form X?)
        
        for word in VALID_TWO_LETTER:
            first, second = word[0], word[1]
            if second not in before:
                before[second] = set()
            before[second].add(first)
            
            if first not in after:
                after[first] = set()
            after[first].add(second)
        
        return {'before': before, 'after': after}
    
    def _can_any_letter_fit(self, r: int, c: int, get_tile) -> bool:
        """
        Check if ANY letter A-Z can legally be placed at (r, c).
        Uses fast 2-letter word lookup.
        """
        # Gather constraints from adjacent tiles
        constraints = []
        
        if r > 1:
            above = get_tile(r - 1, c)
            if above:
                constraints.append(('below', above))  # New letter goes BELOW 'above'
        
        if r < 15:
            below = get_tile(r + 1, c)
            if below:
                constraints.append(('above', below))  # New letter goes ABOVE 'below'
        
        if c > 1:
            left = get_tile(r, c - 1)
            if left:
                constraints.append(('right', left))  # New letter goes RIGHT of 'left'
        
        if c < 15:
            right = get_tile(r, c + 1)
            if right:
                constraints.append(('left', right))  # New letter goes LEFT of 'right'
        
        if not constraints:
            return True  # No adjacent tiles = any letter works
        
        # Find letters that satisfy ALL constraints
        valid_letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        for position, adj_letter in constraints:
            if position == 'below':
                # Need letter X where adj_letter + X is valid
                allowed = self._valid_crosses['after'].get(adj_letter, set())
            elif position == 'above':
                # Need letter X where X + adj_letter is valid
                allowed = self._valid_crosses['before'].get(adj_letter, set())
            elif position == 'right':
                # Need letter X where adj_letter + X is valid (horizontal)
                allowed = self._valid_crosses['after'].get(adj_letter, set())
            elif position == 'left':
                # Need letter X where X + adj_letter is valid (horizontal)
                allowed = self._valid_crosses['before'].get(adj_letter, set())
            else:
                allowed = set()
            
            valid_letters &= allowed
            
            if not valid_letters:
                return False  # No letter satisfies all constraints
        
        return len(valid_letters) > 0
    
    def update(self, board, squares_to_check: Optional[Set[Tuple[int, int]]] = None):
        """
        Update cache after board changes.
        
        Args:
            board: Current board state
            squares_to_check: Optional set of squares to check (for incremental update).
                            If None, checks all non-blocked, non-occupied squares.
        """
        def get_tile(r, c):
            return board.get_tile(r, c)
        
        # Determine which squares to check
        if squares_to_check is None:
            # Full scan - check all squares not already known blocked/occupied
            to_check = set()
            for r in range(1, 16):
                for c in range(1, 16):
                    if (r, c) not in self._blocked:
                        to_check.add((r, c))
        else:
            to_check = squares_to_check - self._blocked
        
        # Check each square
        for r, c in to_check:
            tile = board.get_tile(r, c)
            if tile:
                self._occupied.add((r, c))
                self._playable.discard((r, c))
            elif not self._can_any_letter_fit(r, c, get_tile):
                self._blocked.add((r, c))
                self._playable.discard((r, c))
            else:
                self._playable.add((r, c))
    
    def update_after_move(self, board, word: str, row: int, col: int, horizontal: bool):
        """
        Incremental update after a move is played.
        Only checks squares adjacent to the new word.
        """
        # Find all squares adjacent to the word
        adjacent = set()
        
        for i in range(len(word)):
            if horizontal:
                r, c = row, col + i
            else:
                r, c = row + i, col
            
            # Mark this square as occupied
            self._occupied.add((r, c))
            self._playable.discard((r, c))
            
            # Check adjacent squares
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 1 <= nr <= 15 and 1 <= nc <= 15:
                    if (nr, nc) not in self._occupied and (nr, nc) not in self._blocked:
                        adjacent.add((nr, nc))
        
        # Update only adjacent squares
        self.update(board, adjacent)
    
    def is_blocked(self, r: int, c: int) -> bool:
        """Check if a square is blocked (O(1) lookup)."""
        return (r, c) in self._blocked
    
    def is_occupied(self, r: int, c: int) -> bool:
        """Check if a square has a tile on it."""
        return (r, c) in self._occupied
    
    def is_unavailable(self, r: int, c: int) -> bool:
        """Check if square is blocked OR occupied."""
        return (r, c) in self._blocked or (r, c) in self._occupied
    
    def get_playable_bonus_squares(self) -> Set[Tuple[int, int]]:
        """Return bonus squares that are still playable."""
        result = set()
        for pos in BONUS_SQUARES:
            if pos not in self._blocked and pos not in self._occupied:
                result.add(pos)
        return result
    
    def get_stats(self) -> dict:
        """Return cache statistics."""
        total_bonus = len(BONUS_SQUARES)
        blocked_bonus = len([p for p in BONUS_SQUARES if p in self._blocked])
        occupied_bonus = len([p for p in BONUS_SQUARES if p in self._occupied])
        
        return {
            'total_blocked': len(self._blocked),
            'total_occupied': len(self._occupied),
            'total_playable': len(self._playable),
            'bonus_blocked': blocked_bonus,
            'bonus_occupied': occupied_bonus,
            'bonus_playable': total_bonus - blocked_bonus - occupied_bonus,
        }
    
    def clear(self):
        """Reset cache (for new game)."""
        self._blocked.clear()
        self._occupied.clear()
        self._playable.clear()
    
    def initialize(self, board, dictionary=None):
        """
        Initialize cache with current board state.
        Alias for update() with full board scan.
        """
        self.clear()
        self.update(board)
