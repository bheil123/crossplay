"""
CROSSPLAY V15 - Board Module
Board representation and manipulation.
All positions are 1-indexed externally, 0-indexed internally.
"""

from typing import Optional, List, Tuple, Dict
from .config import (
    BOARD_SIZE, BONUS_SQUARES, CENTER_ROW, CENTER_COL,
    TRIPLE_WORD_SQUARES, DOUBLE_WORD_SQUARES
)


class Board:
    """
    15x15 Crossplay game board.
    
    External interface uses 1-indexed positions (row 1-15, col 1-15).
    Internal storage uses 0-indexed (0-14).
    """
    
    def __init__(self):
        """Initialize empty board."""
        self._grid: List[List[Optional[str]]] = [
            [None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)
        ]
    
    # -------------------------------------------------------------------------
    # Position conversion helpers
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _to_internal(row: int, col: int) -> Tuple[int, int]:
        """Convert 1-indexed to 0-indexed."""
        return (row - 1, col - 1)
    
    @staticmethod
    def _to_external(row: int, col: int) -> Tuple[int, int]:
        """Convert 0-indexed to 1-indexed."""
        return (row + 1, col + 1)
    
    @staticmethod
    def _is_valid_position(row: int, col: int) -> bool:
        """Check if 1-indexed position is valid."""
        return 1 <= row <= BOARD_SIZE and 1 <= col <= BOARD_SIZE
    
    # -------------------------------------------------------------------------
    # Tile access (1-indexed interface)
    # -------------------------------------------------------------------------
    
    def __repr__(self) -> str:
        tile_count = sum(
            1 for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
            if self._grid[r][c] is not None
        )
        return f"Board({tile_count} tiles)"

    def get_tile(self, row: int, col: int) -> Optional[str]:
        """
        Get tile at position (1-indexed).

        Returns:
            Letter at position, or None if empty.
        """
        if not self._is_valid_position(row, col):
            raise ValueError(f"Invalid position: R{row} C{col}")
        r, c = self._to_internal(row, col)
        return self._grid[r][c]
    
    def set_tile(self, row: int, col: int, letter: Optional[str]) -> None:
        """
        Set tile at position (1-indexed).
        
        Args:
            row: Row (1-15)
            col: Column (1-15)
            letter: Letter to place, or None to clear
        """
        if not self._is_valid_position(row, col):
            raise ValueError(f"Invalid position: R{row} C{col}")
        r, c = self._to_internal(row, col)
        self._grid[r][c] = letter.upper() if letter else None
    
    def is_empty(self, row: int, col: int) -> bool:
        """Check if position is empty (1-indexed)."""
        return self.get_tile(row, col) is None
    
    def is_occupied(self, row: int, col: int) -> bool:
        """Check if position has a tile (1-indexed)."""
        return self.get_tile(row, col) is not None
    
    # -------------------------------------------------------------------------
    # Bonus squares (1-indexed interface)
    # -------------------------------------------------------------------------
    
    def get_bonus(self, row: int, col: int) -> Optional[str]:
        """
        Get bonus type at position (1-indexed).
        
        Returns:
            '3W', '2W', '3L', '2L', or None
        """
        return BONUS_SQUARES.get((row, col))
    
    def is_triple_word(self, row: int, col: int) -> bool:
        """Check if position is a triple word square."""
        return self.get_bonus(row, col) == '3W'
    
    def is_double_word(self, row: int, col: int) -> bool:
        """Check if position is a double word square."""
        return self.get_bonus(row, col) == '2W'
    
    def is_triple_letter(self, row: int, col: int) -> bool:
        """Check if position is a triple letter square."""
        return self.get_bonus(row, col) == '3L'
    
    def is_double_letter(self, row: int, col: int) -> bool:
        """Check if position is a double letter square."""
        return self.get_bonus(row, col) == '2L'
    
    # -------------------------------------------------------------------------
    # Board state queries
    # -------------------------------------------------------------------------
    
    def is_board_empty(self) -> bool:
        """Check if board has no tiles."""
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self._grid[r][c] is not None:
                    return False
        return True
    
    def get_all_tiles(self) -> List[Tuple[int, int, str]]:
        """
        Get all tiles on board.
        
        Returns:
            List of (row, col, letter) tuples (1-indexed)
        """
        tiles = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self._grid[r][c]:
                    tiles.append((r + 1, c + 1, self._grid[r][c]))
        return tiles
    
    def count_tiles(self) -> int:
        """Count number of tiles on board."""
        return len(self.get_all_tiles())
    
    def has_adjacent_tile(self, row: int, col: int) -> bool:
        """Check if position has an adjacent tile (1-indexed)."""
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj_r, adj_c = row + dr, col + dc
            if self._is_valid_position(adj_r, adj_c):
                if self.is_occupied(adj_r, adj_c):
                    return True
        return False
    
    # -------------------------------------------------------------------------
    # Word placement
    # -------------------------------------------------------------------------
    
    def place_word(self, word: str, row: int, col: int, horizontal: bool) -> List[Tuple[int, int]]:
        """
        Place a word on the board.
        
        Args:
            word: Word to place (uppercase)
            row: Starting row (1-indexed)
            col: Starting column (1-indexed)
            horizontal: True for horizontal, False for vertical
            
        Returns:
            List of (row, col) positions where NEW tiles were placed
        """
        word = word.upper()
        new_tiles = []
        
        for i, letter in enumerate(word):
            if horizontal:
                r, c = row, col + i
            else:
                r, c = row + i, col
            
            if not self._is_valid_position(r, c):
                raise ValueError(f"Word extends off board at R{r} C{c}")
            
            current = self.get_tile(r, c)
            if current is None:
                self.set_tile(r, c, letter)
                new_tiles.append((r, c))
            elif current != letter:
                raise ValueError(f"Conflict at R{r} C{c}: board has '{current}', word has '{letter}'")
        
        return new_tiles
    
    def get_word_at(self, row: int, col: int, horizontal: bool) -> Tuple[str, int, int]:
        """
        Get the complete word passing through a position.
        
        Args:
            row: Row of any letter in word (1-indexed)
            col: Column of any letter in word (1-indexed)
            horizontal: True for horizontal word, False for vertical
            
        Returns:
            Tuple of (word, start_row, start_col)
        """
        if self.is_empty(row, col):
            return ('', row, col)
        
        # Find start of word
        start_r, start_c = row, col
        while True:
            if horizontal:
                prev_c = start_c - 1
                if prev_c >= 1 and self.is_occupied(start_r, prev_c):
                    start_c = prev_c
                else:
                    break
            else:
                prev_r = start_r - 1
                if prev_r >= 1 and self.is_occupied(prev_r, start_c):
                    start_r = prev_r
                else:
                    break
        
        # Build word
        word = ''
        r, c = start_r, start_c
        while self._is_valid_position(r, c) and self.is_occupied(r, c):
            word += self.get_tile(r, c)
            if horizontal:
                c += 1
            else:
                r += 1
        
        return (word, start_r, start_c)
    
    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------
    
    def display(self) -> str:
        """Generate string representation of board."""
        lines = []
        
        # Header
        header = "     " + "".join(f"{c:>3}" for c in range(1, 16))
        lines.append(header)
        lines.append("   +" + "-" * 46 + "+")
        
        for r in range(BOARD_SIZE):
            row_num = r + 1
            row_str = f"{row_num:2d} |"
            
            for c in range(BOARD_SIZE):
                tile = self._grid[r][c]
                if tile:
                    row_str += f" {tile} "
                else:
                    bonus = self.get_bonus(row_num, c + 1)
                    if bonus:
                        row_str += f" {bonus.lower()}"
                    else:
                        row_str += " . "
            
            row_str += "|"
            lines.append(row_str)
        
        lines.append("   +" + "-" * 46 + "+")
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return self.display()
    
    # -------------------------------------------------------------------------
    # Copy
    # -------------------------------------------------------------------------
    
    def copy(self) -> 'Board':
        """Create a deep copy of the board."""
        new_board = Board()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                new_board._grid[r][c] = self._grid[r][c]
        return new_board
    
    # -------------------------------------------------------------------------
    # Incremental updates (for fast lookahead simulation)
    # -------------------------------------------------------------------------
    
    def place_tiles(self, tiles: List[Tuple[int, int, str]]) -> None:
        """
        Place multiple tiles on the board (fast, no validation).
        
        Used for lookahead simulation where we make/unmake moves quickly.
        
        Args:
            tiles: List of (row, col, letter) tuples (1-indexed)
        """
        for row, col, letter in tiles:
            r, c = self._to_internal(row, col)
            self._grid[r][c] = letter.upper() if letter else None
    
    def remove_tiles(self, positions: List[Tuple[int, int]]) -> None:
        """
        Remove tiles from specified positions (fast, no validation).
        
        Used for lookahead simulation to undo moves.
        
        Args:
            positions: List of (row, col) tuples (1-indexed)
        """
        for row, col in positions:
            r, c = self._to_internal(row, col)
            self._grid[r][c] = None
    
    def place_move(self, word: str, row: int, col: int, horizontal: bool) -> List[Tuple[int, int, str]]:
        """
        Place a move and return the tiles that were placed (for later removal).
        
        This is the recommended method for lookahead - it returns exactly what
        you need to undo the move.
        
        Args:
            word: Word to place
            row: Starting row (1-indexed)
            col: Starting column (1-indexed)
            horizontal: Direction
            
        Returns:
            List of (row, col, letter) for tiles that were placed (not existing)
        """
        word = word.upper()
        placed = []
        
        for i, letter in enumerate(word):
            if horizontal:
                r, c = row, col + i
            else:
                r, c = row + i, col
            
            # Only place if square is empty
            ir, ic = self._to_internal(r, c)
            if self._grid[ir][ic] is None:
                self._grid[ir][ic] = letter
                placed.append((r, c, letter))
        
        return placed
    
    def undo_move(self, placed_tiles: List[Tuple[int, int, str]]) -> None:
        """
        Undo a move by removing the tiles that were placed.
        
        Args:
            placed_tiles: Return value from place_move()
        """
        for row, col, _ in placed_tiles:
            r, c = self._to_internal(row, col)
            self._grid[r][c] = None


def tiles_used(board: Board, word: str, row: int, col: int, horizontal: bool) -> list:
    """Determine which rack tiles a move consumes.

    Iterates the word positions and returns letters that land on empty
    squares (i.e. tiles that must come from the rack).  Letters that
    overlap existing board tiles are skipped.

    Args:
        board: Board instance (1-indexed get_tile).
        word: The word being played (uppercase).
        row: Starting row (1-indexed).
        col: Starting column (1-indexed).
        horizontal: True for horizontal, False for vertical.

    Returns:
        List of uppercase letters consumed from the rack.
    """
    used = []
    for i, letter in enumerate(word):
        r = row if horizontal else row + i
        c = col + i if horizontal else col
        if not board.get_tile(r, c):
            used.append(letter)
    return used


if __name__ == "__main__":
    # Test the board
    board = Board()
    print("Empty board:")
    print(board)
    print()
    
    # Place a word
    board.place_word("HELLO", 8, 4, horizontal=True)
    print("After placing HELLO at R8 C4 (H):")
    print(board)
    print()
    
    # Check tile
    print(f"Tile at R8 C4: {board.get_tile(8, 4)}")
    print(f"Bonus at R8 C4: {board.get_bonus(8, 4)}")
    print(f"Bonus at R4 C1: {board.get_bonus(4, 1)}")
    print(f"Board empty: {board.is_board_empty()}")
    print(f"Tile count: {board.count_tiles()}")
