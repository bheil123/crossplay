"""
CROSSPLAY V15 - GADDAG Move Finder
Fast move generation using GADDAG traversal.

Based on Gordon's algorithm from "A Faster Scrabble Move Generation Algorithm"

The algorithm:
1. Find anchor squares (empty squares adjacent to tiles)
2. For each anchor, in each direction:
   - Go LEFT from anchor, using reversed path in GADDAG (before delimiter)
   - Cross the delimiter 
   - Go RIGHT from anchor, using forward path in GADDAG (after delimiter)
3. At each step, check cross-constraints (perpendicular words must be valid)
"""

from typing import List, Dict, Tuple, Set, Optional
from collections import Counter
from .gaddag import GADDAG, GADDAGNode, DELIMITER, get_gaddag
from .board import Board
from .config import BOARD_SIZE, CENTER_ROW, CENTER_COL, VALID_TWO_LETTER
from .scoring import calculate_move_score
from .dictionary import get_dictionary

# Fast char-to-index lookup for offset-based GADDAG traversal
_CHAR_TO_IDX_FAST = {chr(65 + i): i for i in range(26)}
_CHAR_TO_IDX_FAST['+'] = 26
_IDX_TO_CHAR_FAST = {v: k for k, v in _CHAR_TO_IDX_FAST.items()}


# Module-level dictionary for word validation (consistent with rest of app)
_dictionary = None

def _get_dictionary():
    global _dictionary
    if _dictionary is None:
        _dictionary = get_dictionary()
    return _dictionary


class GADDAGMoveFinder:
    """Finds all valid moves using GADDAG traversal."""
    
    def __init__(self, board: Board, gaddag: Optional[GADDAG] = None, 
                 board_blanks: List[Tuple[int, int, str]] = None):
        self.board = board
        self.gaddag = gaddag or get_gaddag()
        self.board_blanks = board_blanks or []
        self._cross_cache: Dict[Tuple[int, int, bool], Optional[Set[str]]] = {}
        self._dict = _get_dictionary()
        
        # Detect compact GADDAG for fast path
        from .gaddag_compact import CompactGADDAG
        self._use_fast = isinstance(self.gaddag, CompactGADDAG)
        if self._use_fast:
            self._gdata = self.gaddag._data
            self._g = self.gaddag  # for _get_child, _is_terminal, etc.
    
    def find_all_moves(self, rack: str) -> List[Dict]:
        """Find all valid moves for a rack."""
        # Use optimized path for CompactGADDAG (1.7x+ faster)
        if self._use_fast:
            from .move_finder_opt import find_all_moves_opt
            return find_all_moves_opt(
                self.board, self.gaddag, rack,
                board_blanks=self.board_blanks
            )
        
        rack = rack.upper()
        
        # Count blanks separately
        self._num_blanks = rack.count('?')
        rack_letters = rack.replace('?', '')
        
        if not rack_letters and self._num_blanks == 0:
            return []
        
        self._cross_cache.clear()
        moves = []
        seen = set()
        
        # Pre-compute board state for fast access
        grid = self.board._grid  # 0-indexed direct access
        self._grid = grid
        self._empty_board = self.board.is_board_empty()
        
        # Find anchors
        anchors = self._get_anchors()
        
        if self._use_fast:
            # Fast offset-based path
            g = self._g
            _CHAR_IDX = _CHAR_TO_IDX_FAST  # local ref for speed
            DELIM_IDX = 26
            
            for anchor_row, anchor_col in anchors:
                for horizontal in [True, False]:
                    found = self._fast_gen_moves_from_anchor(
                        anchor_row, anchor_col, rack_letters, horizontal,
                        g, grid, _CHAR_IDX, DELIM_IDX
                    )
                    for move in found:
                        key = (move['word'], move['row'], move['col'], move['direction'])
                        if key not in seen:
                            seen.add(key)
                            moves.append(move)
        else:
            for anchor_row, anchor_col in anchors:
                for horizontal in [True, False]:
                    found = self._gen_moves_from_anchor(
                        anchor_row, anchor_col, rack_letters, horizontal
                    )
                    for move in found:
                        key = (move['word'], move['row'], move['col'], move['direction'])
                        if key not in seen:
                            seen.add(key)
                            moves.append(move)
        
        moves.sort(key=lambda m: -m['score'])
        return moves
    
    def _get_anchors(self) -> List[Tuple[int, int]]:
        """Find anchor squares."""
        if self._empty_board:
            return [(CENTER_ROW, CENTER_COL)]
        
        anchors = []
        grid = self._grid
        for r in range(15):
            for c in range(15):
                if grid[r][c] is None:
                    # Check adjacent tiles directly (0-indexed)
                    if ((r > 0 and grid[r-1][c] is not None) or
                        (r < 14 and grid[r+1][c] is not None) or
                        (c > 0 and grid[r][c-1] is not None) or
                        (c < 14 and grid[r][c+1] is not None)):
                        anchors.append((r + 1, c + 1))  # 1-indexed output
        return anchors
    
    def _get_cross_check(self, row: int, col: int, horizontal: bool) -> Optional[Set[str]]:
        """Get valid letters for a square based on perpendicular words."""
        key = (row, col, horizontal)
        if key in self._cross_cache:
            return self._cross_cache[key]
        
        grid = self._grid
        r0, c0 = row - 1, col - 1  # 0-indexed
        
        # Find perpendicular tiles using direct grid access
        if horizontal:
            above, below = [], []
            r = r0 - 1
            while r >= 0 and grid[r][c0] is not None:
                above.insert(0, grid[r][c0])
                r -= 1
            r = r0 + 1
            while r < 15 and grid[r][c0] is not None:
                below.append(grid[r][c0])
                r += 1
        else:
            above, below = [], []
            c = c0 - 1
            while c >= 0 and grid[r0][c] is not None:
                above.insert(0, grid[r0][c])
                c -= 1
            c = c0 + 1
            while c < 15 and grid[r0][c] is not None:
                below.append(grid[r0][c])
                c += 1
        
        if not above and not below:
            self._cross_cache[key] = None
            return None
        
        prefix = ''.join(above)
        suffix = ''.join(below)
        
        valid = set()
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            word = prefix + letter + suffix
            if self._is_valid_word(word):
                valid.add(letter)
        
        self._cross_cache[key] = valid
        return valid
    
    def _is_valid_word(self, word: str) -> bool:
        """Check if a word is valid using the dictionary."""
        word = word.upper()
        if len(word) == 2:
            return word in VALID_TWO_LETTER
        return self._dict.is_valid(word)
    
    def _gen_moves_from_anchor(
        self,
        anchor_row: int,
        anchor_col: int,
        rack: str,
        horizontal: bool
    ) -> List[Dict]:
        """Generate all moves through an anchor in one direction."""
        moves = []
        rack_counter = Counter(rack)
        blanks_remaining = self._num_blanks
        
        # Calculate how far left we can go from anchor
        left_limit = self._left_limit(anchor_row, anchor_col, horizontal)
        
        # Check if there's an existing tile immediately left of anchor
        if horizontal:
            left_tile = anchor_col > 1 and self.board.is_occupied(anchor_row, anchor_col - 1)
        else:
            left_tile = anchor_row > 1 and self.board.is_occupied(anchor_row - 1, anchor_col)
        
        if left_tile:
            # Existing tile to left - must extend from it
            self._extend_from_existing(
                moves, anchor_row, anchor_col, horizontal, rack_counter, blanks_remaining
            )
        else:
            # No tile to left - start fresh from GADDAG root
            self._gen_left_part(
                moves, anchor_row, anchor_col, horizontal,
                self.gaddag.root, '', rack_counter, left_limit, blanks_remaining, []
            )
        
        return moves
    
    def _left_limit(self, row: int, col: int, horizontal: bool) -> int:
        """How far left/up we can extend from anchor (limited by other anchors)."""
        limit = 0
        grid = self._grid
        if horizontal:
            r0 = row - 1  # 0-indexed
            c = col - 2   # 0-indexed, one left of anchor
            while c >= 0:
                if grid[r0][c] is not None:
                    break
                # Stop at another anchor
                if ((r0 > 0 and grid[r0-1][c] is not None) or
                    (r0 < 14 and grid[r0+1][c] is not None) or
                    (c > 0 and grid[r0][c-1] is not None) or
                    (c < 14 and grid[r0][c+1] is not None)):
                    break
                limit += 1
                c -= 1
        else:
            c0 = col - 1  # 0-indexed
            r = row - 2   # 0-indexed, one above anchor
            while r >= 0:
                if grid[r][c0] is not None:
                    break
                if ((r > 0 and grid[r-1][c0] is not None) or
                    (r < 14 and grid[r+1][c0] is not None) or
                    (c0 > 0 and grid[r][c0-1] is not None) or
                    (c0 < 14 and grid[r][c0+1] is not None)):
                    break
                limit += 1
                r -= 1
        return limit
    
    def _extend_from_existing(
        self,
        moves: List[Dict],
        anchor_row: int,
        anchor_col: int,
        horizontal: bool,
        rack: Counter,
        blanks_remaining: int
    ):
        """Extend from existing tiles on board."""
        # Collect existing tiles to the left
        prefix = []
        if horizontal:
            c = anchor_col - 1
            while c >= 1 and self.board.is_occupied(anchor_row, c):
                prefix.insert(0, self.board.get_tile(anchor_row, c))
                c -= 1
            start_col = c + 1
            start_row = anchor_row
        else:
            r = anchor_row - 1
            while r >= 1 and self.board.is_occupied(r, anchor_col):
                prefix.insert(0, self.board.get_tile(r, anchor_col))
                r -= 1
            start_row = r + 1
            start_col = anchor_col
        
        # Navigate GADDAG with reversed prefix
        prefix_str = ''.join(prefix)
        reversed_prefix = prefix_str[::-1]
        
        node = self.gaddag.root
        for char in reversed_prefix:
            if char not in node.children:
                return  # No valid extensions
            node = node.children[char]
        
        # Now we need to cross delimiter and extend right
        if DELIMITER not in node.children:
            return
        
        delim_node = node.children[DELIMITER]
        
        # Extend right from anchor
        self._extend_right(
            moves, anchor_row, anchor_col, horizontal,
            delim_node, prefix_str, rack,
            start_row, start_col, blanks_remaining, []
        )
    
    def _gen_left_part(
        self,
        moves: List[Dict],
        anchor_row: int,
        anchor_col: int,
        horizontal: bool,
        node: GADDAGNode,
        partial_word: str,
        rack: Counter,
        limit: int,
        blanks_remaining: int,
        blanks_used: List[int]
    ):
        """
        Generate the left part of a word (before anchor).
        Uses reversed path in GADDAG (before delimiter).
        
        The GADDAG stores paths like: A+A, BA+D, CBA+T
        - Letters before '+' are the reversed prefix (what's left of where we started)
        - Letters after '+' are the suffix (what's right of where we started)
        
        When partial_word is empty and we're at root, we need to try placing
        the first letter AT the anchor, not left of it.
        """
        # Try crossing delimiter (transition from left-part to right-part)
        if DELIMITER in node.children:
            delim_node = node.children[DELIMITER]
            # Calculate start position based on how many letters we've placed left
            if horizontal:
                start_row = anchor_row
                start_col = anchor_col - len(partial_word)
            else:
                start_row = anchor_row - len(partial_word)
                start_col = anchor_col
            
            pw_len = len(partial_word)
            fixed_blanks = [pw_len + bi if bi < 0 else bi for bi in blanks_used]
            self._extend_right(
                moves, anchor_row, anchor_col, horizontal,
                delim_node, partial_word, rack,
                start_row, start_col, blanks_remaining, fixed_blanks
            )
        
        # Try extending left (placing letters LEFT of current position)
        if limit > 0:
            # Position to place letter (LEFT of anchor minus partial_word length)
            if horizontal:
                pos_row = anchor_row
                pos_col = anchor_col - len(partial_word) - 1
            else:
                pos_row = anchor_row - len(partial_word) - 1
                pos_col = anchor_col
            
            if pos_row < 1 or pos_col < 1:
                return
            
            cross_check = self._get_cross_check(pos_row, pos_col, horizontal)
            
            # Track position index for blanks (negative for left part)
            pos_index = -(len(partial_word) + 1)
            
            # Try each letter from rack
            for letter in list(rack.keys()):
                if rack[letter] <= 0:
                    continue
                
                # Check cross constraint
                if cross_check is not None and letter not in cross_check:
                    continue
                
                # Check GADDAG path
                if letter not in node.children:
                    continue
                
                # Place letter and recurse
                rack[letter] -= 1
                new_partial = letter + partial_word  # Prepend
                
                self._gen_left_part(
                    moves, anchor_row, anchor_col, horizontal,
                    node.children[letter], new_partial, rack, limit - 1,
                    blanks_remaining, blanks_used
                )
                
                rack[letter] += 1
            
            # Try using blank as each letter
            if blanks_remaining > 0:
                for letter in node.children:
                    if letter == DELIMITER:
                        continue
                    # Check cross constraint
                    if cross_check is not None and letter not in cross_check:
                        continue
                    
                    new_partial = letter + partial_word  # Prepend
                    new_blanks_used = blanks_used + [pos_index]
                    
                    self._gen_left_part(
                        moves, anchor_row, anchor_col, horizontal,
                        node.children[letter], new_partial, rack, limit - 1,
                        blanks_remaining - 1, new_blanks_used
                    )
        
        # Try placing the first letter AT the anchor (not left of it).
        # This is different from "extending left" - it's starting the word
        # at the anchor. The GADDAG path for this: letter + DELIMITER + suffix
        # Must run whenever we haven't placed any left-part letters yet,
        # REGARDLESS of limit. When limit > 0, both strategies are valid:
        #   1. Extend left first (handled above)
        #   2. Start the word at the anchor (handled here)
        if len(partial_word) == 0:
            # We're at root with no room to go left, try each letter as word start
            cross_check = self._get_cross_check(anchor_row, anchor_col, horizontal)
            
            for letter in list(rack.keys()):
                if rack[letter] <= 0:
                    continue
                
                # Check cross constraint for anchor position
                if cross_check is not None and letter not in cross_check:
                    continue
                
                # Check GADDAG path: letter -> DELIMITER -> ...
                if letter not in node.children:
                    continue
                
                letter_node = node.children[letter]
                if DELIMITER not in letter_node.children:
                    continue
                
                # Place letter at anchor and extend right
                rack[letter] -= 1
                delim_node = letter_node.children[DELIMITER]
                
                # Move to position after anchor
                if horizontal:
                    next_row, next_col = anchor_row, anchor_col + 1
                else:
                    next_row, next_col = anchor_row + 1, anchor_col
                
                self._extend_right(
                    moves, next_row, next_col, horizontal,
                    delim_node, letter, rack,
                    anchor_row, anchor_col,  # Word starts at anchor
                    blanks_remaining, blanks_used[:]
                )
                
                rack[letter] += 1
            
            # Also try using blank as first letter at anchor
            if blanks_remaining > 0:
                for letter in node.children:
                    if letter == DELIMITER:
                        continue
                    
                    # Check cross constraint
                    if cross_check is not None and letter not in cross_check:
                        continue
                    
                    letter_node = node.children[letter]
                    if DELIMITER not in letter_node.children:
                        continue
                    
                    delim_node = letter_node.children[DELIMITER]
                    
                    if horizontal:
                        next_row, next_col = anchor_row, anchor_col + 1
                    else:
                        next_row, next_col = anchor_row + 1, anchor_col
                    
                    self._extend_right(
                        moves, next_row, next_col, horizontal,
                        delim_node, letter, rack,
                        anchor_row, anchor_col,
                        blanks_remaining - 1, [0]  # First letter uses blank
                    )
    
    def _extend_right(
        self,
        moves: List[Dict],
        row: int,
        col: int,
        horizontal: bool,
        node: GADDAGNode,
        word_so_far: str,
        rack: Counter,
        start_row: int,
        start_col: int,
        blanks_remaining: int,
        blanks_used: List[int]
    ):
        """
        Extend word to the right from current position.
        Uses forward path in GADDAG (after delimiter).
        blanks_used tracks which positions (indices in the word) used a blank.
        """
        # Check bounds
        if row < 1 or row > BOARD_SIZE or col < 1 or col > BOARD_SIZE:
            # Out of bounds - check if word is complete
            if node.is_terminal and len(word_so_far) >= 2:
                self._record_move(moves, word_so_far, start_row, start_col, horizontal, blanks_used)
            return
        
        existing = self.board.get_tile(row, col)
        
        if existing:
            # Square has a tile - must follow it
            if existing in node.children:
                new_word = word_so_far + existing
                next_node = node.children[existing]
                
                # Move to next position
                if horizontal:
                    next_row, next_col = row, col + 1
                else:
                    next_row, next_col = row + 1, col
                
                self._extend_right(
                    moves, next_row, next_col, horizontal,
                    next_node, new_word, rack, start_row, start_col,
                    blanks_remaining, blanks_used
                )
        else:
            # Empty square
            cross_check = self._get_cross_check(row, col, horizontal)
            
            # Check if word is already complete
            if node.is_terminal and len(word_so_far) >= 2:
                self._record_move(moves, word_so_far, start_row, start_col, horizontal, blanks_used)
            
            # Position index for tracking blanks
            pos_index = len(word_so_far)
            
            # Try placing letters from rack
            for letter in list(rack.keys()):
                if rack[letter] <= 0:
                    continue
                
                # Check cross constraint
                if cross_check is not None and letter not in cross_check:
                    continue
                
                # Check GADDAG path
                if letter not in node.children:
                    continue
                
                rack[letter] -= 1
                new_word = word_so_far + letter
                next_node = node.children[letter]
                
                # Move to next position
                if horizontal:
                    next_row, next_col = row, col + 1
                else:
                    next_row, next_col = row + 1, col
                
                self._extend_right(
                    moves, next_row, next_col, horizontal,
                    next_node, new_word, rack, start_row, start_col,
                    blanks_remaining, blanks_used
                )
                
                rack[letter] += 1
            
            # Try using blank as each possible letter
            if blanks_remaining > 0:
                for letter in node.children:
                    if letter == DELIMITER:
                        continue
                    
                    # Check cross constraint
                    if cross_check is not None and letter not in cross_check:
                        continue
                    
                    new_word = word_so_far + letter
                    next_node = node.children[letter]
                    new_blanks_used = blanks_used + [pos_index]
                    
                    # Move to next position
                    if horizontal:
                        next_row, next_col = row, col + 1
                    else:
                        next_row, next_col = row + 1, col
                    
                    self._extend_right(
                        moves, next_row, next_col, horizontal,
                        next_node, new_word, rack, start_row, start_col,
                        blanks_remaining - 1, new_blanks_used
                    )
    
    # ===== FAST PATH: offset-based GADDAG traversal (CompactGADDAG only) =====
    
    def _fast_gen_moves_from_anchor(self, anchor_row, anchor_col, rack, horizontal, g, grid, CI, DELIM_IDX):
        """Fast-path: generate moves from anchor using integer offsets."""
        moves = []
        rack_counter = Counter(rack)
        blanks_remaining = self._num_blanks
        left_limit = self._left_limit(anchor_row, anchor_col, horizontal)
        
        ar0, ac0 = anchor_row - 1, anchor_col - 1  # 0-indexed
        
        # Check for existing tile to the left
        if horizontal:
            left_tile = ac0 > 0 and grid[ar0][ac0 - 1] is not None
        else:
            left_tile = ar0 > 0 and grid[ar0 - 1][ac0] is not None
        
        if left_tile:
            self._fast_extend_from_existing(
                moves, anchor_row, anchor_col, horizontal, rack_counter, blanks_remaining,
                g, grid, CI, DELIM_IDX
            )
        else:
            self._fast_gen_left_part(
                moves, anchor_row, anchor_col, horizontal,
                0, '', rack_counter, left_limit, blanks_remaining, [],
                g, grid, CI, DELIM_IDX
            )
        return moves
    
    def _fast_extend_from_existing(self, moves, anchor_row, anchor_col, horizontal,
                                     rack, blanks_remaining, g, grid, CI, DELIM_IDX):
        """Fast-path: extend from existing tiles using offsets."""
        prefix = []
        ar0, ac0 = anchor_row - 1, anchor_col - 1
        if horizontal:
            c = ac0 - 1
            while c >= 0 and grid[ar0][c] is not None:
                prefix.insert(0, grid[ar0][c])
                c -= 1
            start_col = c + 2  # 1-indexed
            start_row = anchor_row
        else:
            r = ar0 - 1
            while r >= 0 and grid[r][ac0] is not None:
                prefix.insert(0, grid[r][ac0])
                r -= 1
            start_row = r + 2  # 1-indexed
            start_col = anchor_col
        
        # Navigate GADDAG with reversed prefix using offsets
        prefix_str = ''.join(prefix)
        reversed_prefix = prefix_str[::-1]
        
        offset = 0  # root
        for char in reversed_prefix:
            idx = CI.get(char)
            if idx is None:
                return
            child = g._get_child(offset, idx)
            if child < 0:
                return
            offset = child
        
        # Cross delimiter
        delim_child = g._get_child(offset, DELIM_IDX)
        if delim_child < 0:
            return
        
        self._fast_extend_right(
            moves, anchor_row, anchor_col, horizontal,
            delim_child, prefix_str, rack,
            start_row, start_col, blanks_remaining, [],
            g, grid, CI, DELIM_IDX
        )
    
    def _fast_gen_left_part(self, moves, anchor_row, anchor_col, horizontal,
                             offset, partial_word, rack, limit, blanks_remaining, blanks_used,
                             g, grid, CI, DELIM_IDX):
        """Fast-path: generate left part using integer offsets."""
        # Try crossing delimiter
        delim_child = g._get_child(offset, DELIM_IDX)
        if delim_child >= 0:
            pw_len = len(partial_word)
            if horizontal:
                start_row = anchor_row
                start_col = anchor_col - pw_len
            else:
                start_row = anchor_row - pw_len
                start_col = anchor_col
            
            fixed_blanks = [pw_len + bi if bi < 0 else bi for bi in blanks_used]
            self._fast_extend_right(
                moves, anchor_row, anchor_col, horizontal,
                delim_child, partial_word, rack,
                start_row, start_col, blanks_remaining, fixed_blanks,
                g, grid, CI, DELIM_IDX
            )
        
        # Try extending left
        if limit > 0:
            pw_len = len(partial_word)
            if horizontal:
                pos_row = anchor_row
                pos_col = anchor_col - pw_len - 1
            else:
                pos_row = anchor_row - pw_len - 1
                pos_col = anchor_col
            
            if pos_row < 1 or pos_col < 1:
                return
            
            cross_check = self._get_cross_check(pos_row, pos_col, horizontal)
            pos_index = -(pw_len + 1)
            
            for letter in list(rack.keys()):
                if rack[letter] <= 0:
                    continue
                if cross_check is not None and letter not in cross_check:
                    continue
                
                idx = CI.get(letter)
                if idx is None:
                    continue
                child = g._get_child(offset, idx)
                if child < 0:
                    continue
                
                rack[letter] -= 1
                self._fast_gen_left_part(
                    moves, anchor_row, anchor_col, horizontal,
                    child, letter + partial_word, rack, limit - 1,
                    blanks_remaining, blanks_used,
                    g, grid, CI, DELIM_IDX
                )
                rack[letter] += 1
            
            if blanks_remaining > 0:
                for letter, child_off in g._iter_children(offset):
                    if letter == DELIMITER:
                        continue
                    if cross_check is not None and letter not in cross_check:
                        continue
                    
                    self._fast_gen_left_part(
                        moves, anchor_row, anchor_col, horizontal,
                        child_off, letter + partial_word, rack, limit - 1,
                        blanks_remaining - 1, blanks_used + [pos_index],
                        g, grid, CI, DELIM_IDX
                    )
        
        # At root with limit=0: try placing first letter at anchor
        if len(partial_word) == 0 and limit == 0:
            cross_check = self._get_cross_check(anchor_row, anchor_col, horizontal)
            
            for letter in list(rack.keys()):
                if rack[letter] <= 0:
                    continue
                if cross_check is not None and letter not in cross_check:
                    continue
                
                idx = CI.get(letter)
                if idx is None:
                    continue
                letter_child = g._get_child(offset, idx)
                if letter_child < 0:
                    continue
                delim_child2 = g._get_child(letter_child, DELIM_IDX)
                if delim_child2 < 0:
                    continue
                
                rack[letter] -= 1
                if horizontal:
                    next_row, next_col = anchor_row, anchor_col + 1
                else:
                    next_row, next_col = anchor_row + 1, anchor_col
                
                self._fast_extend_right(
                    moves, next_row, next_col, horizontal,
                    delim_child2, letter, rack,
                    anchor_row, anchor_col, blanks_remaining, blanks_used[:],
                    g, grid, CI, DELIM_IDX
                )
                rack[letter] += 1
            
            if blanks_remaining > 0:
                for letter, letter_child in g._iter_children(offset):
                    if letter == DELIMITER:
                        continue
                    if cross_check is not None and letter not in cross_check:
                        continue
                    
                    delim_child2 = g._get_child(letter_child, DELIM_IDX)
                    if delim_child2 < 0:
                        continue
                    
                    if horizontal:
                        next_row, next_col = anchor_row, anchor_col + 1
                    else:
                        next_row, next_col = anchor_row + 1, anchor_col
                    
                    self._fast_extend_right(
                        moves, next_row, next_col, horizontal,
                        delim_child2, letter, rack,
                        anchor_row, anchor_col, blanks_remaining - 1, [0],
                        g, grid, CI, DELIM_IDX
                    )
    
    def _fast_extend_right(self, moves, row, col, horizontal,
                            offset, word_so_far, rack, start_row, start_col,
                            blanks_remaining, blanks_used,
                            g, grid, CI, DELIM_IDX):
        """Fast-path: extend right using integer offsets + direct grid access."""
        # Bounds check (1-indexed)
        if row < 1 or row > 15 or col < 1 or col > 15:
            if g._is_terminal(offset) and len(word_so_far) >= 2:
                self._record_move(moves, word_so_far, start_row, start_col, horizontal, blanks_used)
            return
        
        # Direct grid access (0-indexed)
        existing = grid[row - 1][col - 1]
        
        if existing:
            idx = CI.get(existing)
            if idx is not None:
                child = g._get_child(offset, idx)
                if child >= 0:
                    if horizontal:
                        nr, nc = row, col + 1
                    else:
                        nr, nc = row + 1, col
                    self._fast_extend_right(
                        moves, nr, nc, horizontal,
                        child, word_so_far + existing, rack, start_row, start_col,
                        blanks_remaining, blanks_used,
                        g, grid, CI, DELIM_IDX
                    )
        else:
            cross_check = self._get_cross_check(row, col, horizontal)
            
            if g._is_terminal(offset) and len(word_so_far) >= 2:
                self._record_move(moves, word_so_far, start_row, start_col, horizontal, blanks_used)
            
            pos_index = len(word_so_far)
            
            for letter in list(rack.keys()):
                if rack[letter] <= 0:
                    continue
                if cross_check is not None and letter not in cross_check:
                    continue
                
                idx = CI.get(letter)
                if idx is None:
                    continue
                child = g._get_child(offset, idx)
                if child < 0:
                    continue
                
                rack[letter] -= 1
                if horizontal:
                    nr, nc = row, col + 1
                else:
                    nr, nc = row + 1, col
                self._fast_extend_right(
                    moves, nr, nc, horizontal,
                    child, word_so_far + letter, rack, start_row, start_col,
                    blanks_remaining, blanks_used,
                    g, grid, CI, DELIM_IDX
                )
                rack[letter] += 1
            
            if blanks_remaining > 0:
                for letter, child_off in g._iter_children(offset):
                    if letter == DELIMITER:
                        continue
                    if cross_check is not None and letter not in cross_check:
                        continue
                    
                    if horizontal:
                        nr, nc = row, col + 1
                    else:
                        nr, nc = row + 1, col
                    self._fast_extend_right(
                        moves, nr, nc, horizontal,
                        child_off, word_so_far + letter, rack, start_row, start_col,
                        blanks_remaining - 1, blanks_used + [pos_index],
                        g, grid, CI, DELIM_IDX
                    )
    
    # ===== END FAST PATH =====
    
    def _record_move(
        self,
        moves: List[Dict],
        word: str,
        start_row: int,
        start_col: int,
        horizontal: bool,
        blanks_used: List[int] = None
    ):
        """Record a valid move. blanks_used is list of indices in word that use blanks."""
        if start_row < 1 or start_col < 1:
            return
        
        if blanks_used is None:
            blanks_used = []
        
        # Validate word is actually valid (esp. for 2-letter words)
        if not self._is_valid_word(word):
            return
        
        # Validate placement
        if not self._validate_placement(word, start_row, start_col, horizontal):
            return
        
        # Calculate score (blanks score 0)
        try:
            score, crosswords = calculate_move_score(
                self.board, word, start_row, start_col, horizontal, blanks_used,
                board_blanks=self.board_blanks
            )
        except:
            return
        
        moves.append({
            'word': word,
            'row': start_row,
            'col': start_col,
            'direction': 'H' if horizontal else 'V',
            'score': score,
            'crosswords': crosswords,
            'blanks_used': blanks_used
        })
    
    def _validate_placement(
        self,
        word: str,
        start_row: int,
        start_col: int,
        horizontal: bool
    ) -> bool:
        """Validate that placement is legal."""
        connects = False
        is_first = self.board.is_board_empty()
        covers_center = False
        uses_new_tile = False
        
        for i, letter in enumerate(word):
            if horizontal:
                r, c = start_row, start_col + i
            else:
                r, c = start_row + i, start_col
            
            if r < 1 or r > BOARD_SIZE or c < 1 or c > BOARD_SIZE:
                return False
            
            if r == CENTER_ROW and c == CENTER_COL:
                covers_center = True
            
            existing = self.board.get_tile(r, c)
            if existing:
                if existing != letter:
                    return False
                connects = True
            else:
                uses_new_tile = True
                if self.board.has_adjacent_tile(r, c):
                    connects = True
        
        if not uses_new_tile:
            return False
        
        if is_first:
            return covers_center
        
        return connects


def find_moves_gaddag(board: Board, rack: str) -> List[Dict]:
    """Convenience function to find moves using GADDAG."""
    finder = GADDAGMoveFinder(board)
    return finder.find_all_moves(rack)
