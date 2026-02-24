"""
CROSSPLAY V15 - Tile Tracker
Tracks tiles played, in rack, and remaining in bag/opponent's hand.
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple, Set
from .config import TILE_DISTRIBUTION, TILE_VALUES, TOTAL_TILES
from .board import Board


class TileTracker:
    """
    Track tile distribution throughout the game.
    
    Knows:
    - Tiles on board (and which are blanks)
    - Your rack
    - Unseen tiles (bag + opponent rack)
    
    Can check if a word is playable from unseen tiles.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset to starting state."""
        self.remaining = Counter(TILE_DISTRIBUTION)
        self.on_board = Counter()
        self.your_rack = ""
        self.blanks_on_board: List[Tuple[int, int, str]] = []  # (row, col, represents)
        self.blanks_in_rack = 0
    
    def sync_with_board(self, board: Board, your_rack: str = "", 
                        blanks_in_rack: int = 0,
                        blank_positions: List[Tuple[int, int, str]] = None):
        """
        Sync tracker with actual board state.
        
        Args:
            board: Current board state
            your_rack: Your current rack (letters only, no blanks)
            blanks_in_rack: Number of blanks in your rack
            blank_positions: List of (row, col, letter) for blanks on board
                           e.g., [(11, 7, 'U'), (7, 11, 'E')] means blank at R11C7 shows as U
        """
        self.remaining = Counter(TILE_DISTRIBUTION)
        self.on_board = Counter()
        self.blanks_on_board = blank_positions or []
        self.blanks_in_rack = blanks_in_rack
        self.your_rack = your_rack.upper()
        
        # Count tiles on board
        for r in range(1, 16):
            for c in range(1, 16):
                tile = board.get_tile(r, c)
                if tile:
                    self.on_board[tile] += 1
        
        # Adjust for blanks on board (they show as letters but are actually blanks)
        for (r, c, letter) in self.blanks_on_board:
            self.on_board[letter] -= 1  # Not actually that letter
            # Blank itself is tracked via len(blanks_on_board)
        
        # Subtract board tiles from remaining
        for letter, count in self.on_board.items():
            self.remaining[letter] -= count
        
        # Subtract blanks used on board
        self.remaining['?'] -= len(self.blanks_on_board)
        
        # Subtract your rack from remaining
        for tile in self.your_rack:
            self.remaining[tile] -= 1
        self.remaining['?'] -= blanks_in_rack

        # Validate: total accounted tiles must not exceed TOTAL_TILES (100)
        # If any remaining count is negative, more tiles are accounted for
        # than exist -- likely a missing blank in blank_positions
        tiles_on_board = sum(self.on_board.values()) + len(self.blanks_on_board)
        tiles_in_rack = len(self.your_rack) + self.blanks_in_rack
        tiles_unseen = sum(v for v in self.remaining.values() if v > 0)
        total_accounted = tiles_on_board + tiles_in_rack + tiles_unseen

        negative_tiles = {k: v for k, v in self.remaining.items() if v < 0}
        if negative_tiles:
            over_letters = ", ".join(
                f"{k}(over by {-v})" for k, v in sorted(negative_tiles.items())
            )
            print(f"[!] TILE TRACKING ERROR: Over-counted tiles: {over_letters}")
            print(f"    Board: {tiles_on_board}, Rack: {tiles_in_rack}, "
                  f"Unseen: {tiles_unseen}, Total: {total_accounted} "
                  f"(max {TOTAL_TILES})")
            print(f"    -> Check blank_positions: are all blanks on the board tracked?")
            print(f"    -> Blanks tracked: {len(self.blanks_on_board)} on board, "
                  f"{self.blanks_in_rack} in rack, "
                  f"{self.remaining.get('?', 0)} unseen "
                  f"(should sum to {TILE_DISTRIBUTION['?']})")

    def get_unseen_count(self) -> int:
        """Total unseen tiles (bag + opponent rack)."""
        return sum(self.remaining.values())
    
    def get_bag_count(self) -> int:
        """Estimated tiles in bag (unseen - 7 for opponent)."""
        return max(0, self.get_unseen_count() - 7)
    
    def get_remaining(self, letter: str) -> int:
        """Get count of a specific letter remaining (unseen)."""
        return self.remaining.get(letter.upper(), 0)
    
    def get_blanks_remaining(self) -> int:
        """Get number of blanks remaining (unseen)."""
        return self.remaining.get('?', 0)
    
    def is_letter_available(self, letter: str) -> bool:
        """Check if at least one of this letter is unseen."""
        return self.get_remaining(letter) > 0 or self.get_blanks_remaining() > 0
    
    def can_opponent_play_word(self, word: str, 
                                tiles_on_board_used: str = "") -> Tuple[bool, str]:
        """
        Check if opponent could play this word from unseen tiles.
        
        Args:
            word: The word to check
            tiles_on_board_used: Letters in the word that come from the board
                                (through-play tiles)
        
        Returns:
            (can_play, reason) - True if playable, with explanation
        """
        word = word.upper()
        tiles_on_board_used = tiles_on_board_used.upper()
        
        # Letters opponent needs to provide
        needed = Counter(word)
        for tile in tiles_on_board_used:
            needed[tile] -= 1
        
        # Remove letters with count <= 0
        needed = Counter({k: v for k, v in needed.items() if v > 0})
        
        # Check if all needed letters are available
        blanks_needed = 0
        missing = []
        
        for letter, count in needed.items():
            available = self.remaining.get(letter, 0)
            if available < count:
                shortfall = count - available
                blanks_needed += shortfall
                if blanks_needed > self.get_blanks_remaining():
                    missing.append(f"{letter}(need {count}, have {available})")
        
        if missing:
            return False, f"Missing: {', '.join(missing)}"
        
        if blanks_needed > 0:
            return True, f"Requires {blanks_needed} blank(s)"
        
        return True, "Tiles available"
    
    def get_high_value_tiles_remaining(self) -> Dict[str, int]:
        """Get count of high-value tiles (5+ points) remaining."""
        high_value = {}
        for letter, value in TILE_VALUES.items():
            if value >= 5 and letter != '?':
                remaining = self.remaining.get(letter, 0)
                if remaining > 0:
                    high_value[letter] = remaining
        return high_value
    
    def get_tiles_out(self) -> List[str]:
        """Get list of letters completely played (none remaining)."""
        return [letter for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' 
                if self.remaining.get(letter, 0) == 0]
    
    def display(self) -> str:
        """Return formatted display of tile status."""
        lines = []
        lines.append("=" * 50)
        lines.append("TILE TRACKER")
        lines.append("=" * 50)
        
        tiles_on_board = sum(self.on_board.values()) + len(self.blanks_on_board)
        lines.append(f"On board: {tiles_on_board} tiles")
        lines.append(f"Your rack: {self.your_rack} + {self.blanks_in_rack} blank(s)")
        lines.append(f"Unseen: {self.get_unseen_count()} (bag ~{self.get_bag_count()} + opp 7)")
        
        # Blanks
        blanks_total = TILE_DISTRIBUTION['?']
        blanks_used = len(self.blanks_on_board) + self.blanks_in_rack
        lines.append(f"\nBLANKS: {self.get_blanks_remaining()}/{blanks_total} remaining")
        if self.blanks_on_board:
            blank_strs = [f"R{r}C{c}={letter}" for r, c, letter in self.blanks_on_board]
            lines.append(f"  On board: {', '.join(blank_strs)}")
        
        # Tiles out
        out = self.get_tiles_out()
        if out:
            lines.append(f"\nOUT: {', '.join(out)}")
        
        # High value remaining
        high_val = self.get_high_value_tiles_remaining()
        if high_val:
            hv_strs = [f"{l}({TILE_VALUES[l]}pts):{c}" for l, c in sorted(high_val.items())]
            lines.append(f"\nHIGH VALUE REMAINING: {', '.join(hv_strs)}")
        
        return "\n".join(lines)
    
    def display_full(self) -> str:
        """Return detailed display with all letter counts."""
        lines = [self.display(), ""]
        lines.append("ALL LETTERS:")
        
        row1 = "  "
        row2 = "  "
        for letter in 'ABCDEFGHIJKLM':
            total = TILE_DISTRIBUTION.get(letter, 0)
            remaining = self.remaining.get(letter, 0)
            row1 += f" {letter} "
            row2 += f"{remaining}/{total}".center(3)
        lines.append(row1)
        lines.append(row2)
        
        row1 = "  "
        row2 = "  "
        for letter in 'NOPQRSTUVWXYZ':
            total = TILE_DISTRIBUTION.get(letter, 0)
            remaining = self.remaining.get(letter, 0)
            row1 += f" {letter} "
            row2 += f"{remaining}/{total}".center(3)
        lines.append(row1)
        lines.append(row2)
        
        return "\n".join(lines)


def test_tile_tracker():
    """Test the TileTracker class."""
    from .board import Board
    
    # Build Game 2 board
    board = Board()
    board.place_word('FRAUD', 8, 4, True)
    board.place_word('STREW', 6, 5, False)
    board.place_word('DOGIE', 8, 8, False)
    board.place_word('SPEW', 13, 8, True)
    board.place_word('LOOGIE', 14, 10, True)
    board.place_word('QUIETS', 11, 6, True)
    board.place_word('TAEL', 12, 15, False)
    board.place_word('EH', 15, 11, True)
    board.place_word('MAKI', 12, 3, True)
    board.place_word('URN', 10, 11, True)
    board.place_word('VERJUS', 6, 11, False)
    
    # Track with blank positions
    tracker = TileTracker()
    tracker.sync_with_board(
        board,
        your_rack="ACCDRPN",  # Example rack
        blanks_in_rack=0,
        blank_positions=[
            (11, 7, 'U'),  # Blank as U in QUIETS
            (7, 11, 'E'),  # Blank as E in VERJUS
        ]
    )
    
    print(tracker.display_full())
    print()
    
    # Test word playability
    test_words = ['QUIZ', 'JINX', 'AXE', 'ZAP', 'WALTZ']
    print("CAN OPPONENT PLAY:")
    for word in test_words:
        can_play, reason = tracker.can_opponent_play_word(word)
        status = "Y" if can_play else "N"
        print(f"  {word}: {status} - {reason}")


if __name__ == "__main__":
    test_tile_tracker()
