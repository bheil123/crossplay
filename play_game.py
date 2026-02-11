#!/usr/bin/env python3
"""
CROSSPLAY V7 - Play Against Claude

Interactive game where you play against Claude. Claude uses V7 engine for its moves
and shows you analysis for your rack.

Usage:
    python3 play_game.py

Commands during game:
    play WORD ROW COL H/V  - Play a word (e.g., "play HELLO 8 4 H")
    rack                   - Show your current rack
    analyze                - Show best moves for your rack
    board                  - Show the board
    score                  - Show scores
    exchange ABC           - Exchange tiles (e.g., "exchange QVV")
    pass                   - Pass your turn
    quit                   - End game
"""

import sys
import random
sys.path.insert(0, '/home/claude')

from crossplay_v9.board import Board
from crossplay_v9.move_finder_gaddag import GADDAGMoveFinder
from crossplay_v9.gaddag import get_gaddag
from crossplay_v9.scoring import calculate_move_score
from crossplay_v9.dictionary import Dictionary
from crossplay_v9.config import TILE_DISTRIBUTION, TILE_VALUES

class CrossplayGame:
    def __init__(self):
        self.board = Board()
        self.gaddag = get_gaddag()
        self.dictionary = Dictionary.load('/home/claude/crossplay_v9/crossplay_dict.pkl')
        
        # Initialize bag
        self.bag = []
        for letter, count in TILE_DISTRIBUTION.items():
            self.bag.extend([letter] * count)
        random.shuffle(self.bag)
        
        # Player racks and scores
        self.human_rack = self._draw_tiles(7)
        self.claude_rack = self._draw_tiles(7)
        self.human_score = 0
        self.claude_score = 0
        
        # Game state
        self.turn = 'human'  # human goes first
        self.consecutive_passes = 0
        self.moves_played = []
        self.final_turns_remaining = None  # Counts down once bag empties
        
    def _draw_tiles(self, n: int) -> list:
        """Draw n tiles from bag."""
        drawn = []
        for _ in range(min(n, len(self.bag))):
            drawn.append(self.bag.pop())
        return drawn
    
    def _refill_rack(self, rack: list) -> list:
        """Refill rack to 7 tiles."""
        needed = 7 - len(rack)
        rack.extend(self._draw_tiles(needed))
        return rack
    
    def show_board(self):
        """Display the board."""
        print("\n    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15")
        print("   " + "-" * 46)
        
        for row in range(1, 16):
            row_str = f"{row:2} |"
            for col in range(1, 16):
                tile = self.board.get_tile(row, col)
                row_str += f" {tile} " if tile else " . "
            print(row_str + "|")
        print("   " + "-" * 46)
    
    def show_scores(self):
        """Display current scores."""
        spread = self.human_score - self.claude_score
        spread_str = f"+{spread}" if spread >= 0 else str(spread)
        print(f"\n📊 SCORE: You {self.human_score} - Claude {self.claude_score} ({spread_str})")
        print(f"📦 Tiles in bag: {len(self.bag)}")
    
    def show_rack(self):
        """Show human's rack."""
        rack_str = ' '.join(self.human_rack)
        values = sum(TILE_VALUES.get(t, 0) for t in self.human_rack)
        print(f"\n🎯 Your rack: [{rack_str}] (value: {values})")
    
    def analyze_human_moves(self):
        """Show best moves for human's rack."""
        rack_str = ''.join(self.human_rack)
        finder = GADDAGMoveFinder(self.board, self.gaddag)
        moves = finder.find_all_moves(rack_str)
        
        if not moves:
            print("\n❌ No valid moves found for your rack!")
            return
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS FOR YOUR RACK: {rack_str}")
        print(f"{'='*60}")
        print(f"\n{'#':<3} {'Word':<12} {'Position':<10} {'Pts':>4} {'Tiles Used'}")
        print("-" * 50)
        
        for i, move in enumerate(moves[:15], 1):
            word = move['word']
            pos = f"R{move['row']}C{move['col']} {move['direction']}"
            pts = move['score']
            
            # Figure out which tiles from rack are used
            used = self._tiles_used(move)
            
            print(f"{i:<3} {word:<12} {pos:<10} {pts:>4} {used}")
    
    def _tiles_used(self, move: dict) -> str:
        """Determine which rack tiles a move uses."""
        word = move['word']
        row, col = move['row'], move['col']
        horiz = move['direction'] == 'H'
        
        used = []
        for i, letter in enumerate(word):
            r = row if horiz else row + i
            c = col + i if horiz else col
            if not self.board.get_tile(r, c):
                used.append(letter)
        return ''.join(used)
    
    def human_play(self, word: str, row: int, col: int, horizontal: bool) -> bool:
        """Process human's move."""
        word = word.upper()
        
        # Check word is valid
        if not self.dictionary.is_valid(word) and len(word) > 2:
            print(f"❌ '{word}' is not a valid word!")
            return False
        
        # Check tiles available
        tiles_needed = self._tiles_used({
            'word': word, 'row': row, 'col': col, 
            'direction': 'H' if horizontal else 'V'
        })
        
        rack_copy = self.human_rack.copy()
        for tile in tiles_needed:
            if tile in rack_copy:
                rack_copy.remove(tile)
            elif '?' in rack_copy:
                rack_copy.remove('?')
            else:
                print(f"❌ You don't have the tiles for '{word}'! Need: {tiles_needed}")
                return False
        
        # Try to place
        try:
            score, crosswords = calculate_move_score(self.board, word, row, col, horizontal)
        except Exception as e:
            print(f"❌ Invalid placement: {e}")
            return False
        
        # Place the word
        self.board.place_word(word, row, col, horizontal)
        
        # Update rack
        for tile in tiles_needed:
            if tile in self.human_rack:
                self.human_rack.remove(tile)
            elif '?' in self.human_rack:
                self.human_rack.remove('?')
        
        # Bingo bonus (Crossplay uses 40)
        if len(tiles_needed) == 7:
            score += 40
            print("🎉 BINGO! +40 bonus!")
        
        self.human_score += score
        self._refill_rack(self.human_rack)
        self._check_bag_empty()
        if self.final_turns_remaining is not None:
            self.final_turns_remaining -= 1

        self.moves_played.append(('human', word, row, col, horizontal, score))
        self.consecutive_passes = 0
        
        d = 'H' if horizontal else 'V'
        print(f"\n✅ You played {word} at R{row}C{col} {d} for {score} points!")
        if crosswords:
            cw_str = ', '.join(f"{c['word']}({c['score']})" for c in crosswords)
            print(f"   Cross-words: {cw_str}")
        
        return True
    
    def claude_play(self):
        """Claude makes a move."""
        rack_str = ''.join(self.claude_rack)
        finder = GADDAGMoveFinder(self.board, self.gaddag)
        moves = finder.find_all_moves(rack_str)
        
        if not moves:
            print("\n🤖 Claude passes (no valid moves).")
            self.consecutive_passes += 1
            if self.final_turns_remaining is not None:
                self.final_turns_remaining -= 1
            return
        
        # Claude picks the highest-scoring move (simple strategy)
        # Could add risk evaluation here for smarter play
        move = moves[0]
        
        word = move['word']
        row, col = move['row'], move['col']
        horizontal = move['direction'] == 'H'
        score = move['score']
        
        # Determine tiles used
        tiles_needed = self._tiles_used(move)
        
        # Place word
        self.board.place_word(word, row, col, horizontal)
        
        # Update Claude's rack
        for tile in tiles_needed:
            if tile in self.claude_rack:
                self.claude_rack.remove(tile)
            elif '?' in self.claude_rack:
                self.claude_rack.remove('?')
        
        # Bingo bonus (Crossplay uses 40)
        if len(tiles_needed) == 7:
            score += 40
            print("🤖 Claude got a BINGO! +40 bonus!")
        
        self.claude_score += score
        self._refill_rack(self.claude_rack)
        self._check_bag_empty()
        if self.final_turns_remaining is not None:
            self.final_turns_remaining -= 1

        self.moves_played.append(('claude', word, row, col, horizontal, score))
        self.consecutive_passes = 0
        
        d = 'H' if horizontal else 'V'
        print(f"\n🤖 Claude plays {word} at R{row}C{col} {d} for {score} points!")
    
    def human_exchange(self, tiles: str):
        """Human exchanges tiles."""
        tiles = tiles.upper()
        
        if len(self.bag) < 7:
            print("❌ Not enough tiles in bag to exchange!")
            return False
        
        # Check tiles available
        rack_copy = self.human_rack.copy()
        for tile in tiles:
            if tile in rack_copy:
                rack_copy.remove(tile)
            else:
                print(f"❌ You don't have '{tile}' to exchange!")
                return False
        
        # Do exchange
        for tile in tiles:
            self.human_rack.remove(tile)
            self.bag.append(tile)
        
        random.shuffle(self.bag)
        self._refill_rack(self.human_rack)
        
        print(f"✅ Exchanged {tiles}. Drew {len(tiles)} new tiles.")
        self.consecutive_passes = 0
        return True
    
    def _check_bag_empty(self):
        """Start final turn countdown when bag empties.

        Crossplay rule: when the bag empties, both players get one final
        turn. We track this with final_turns_remaining (2 = both still
        need a final turn, 1 = one player has taken theirs, 0 = done).
        """
        if len(self.bag) == 0 and self.final_turns_remaining is None:
            self.final_turns_remaining = 2  # Both players get one more turn

    def is_game_over(self) -> bool:
        """Check if game is over.

        Crossplay rules:
        - When bag empties, both players get one final turn, then game ends
        - 6 consecutive passes ends the game immediately
        - No more valid moves ends the game
        """
        if self.consecutive_passes >= 6:
            return True
        if self.final_turns_remaining is not None and self.final_turns_remaining <= 0:
            return True
        return False

    def final_scores(self):
        """Calculate final scores.

        Crossplay rule: leftover tiles do NOT count against the player.
        Final scores are simply the accumulated scores from played moves.
        """
        print(f"\n{'='*60}")
        print("GAME OVER!")
        print(f"{'='*60}")
        print(f"Final Score: You {self.human_score} - Claude {self.claude_score}")

        if self.human_score > self.claude_score:
            print("YOU WIN!")
        elif self.claude_score > self.human_score:
            print("Claude wins!")
        else:
            print("It's a tie!")
    
    def play(self):
        """Main game loop."""
        print("\n" + "="*60)
        print("CROSSPLAY V7 - PLAY AGAINST CLAUDE")
        print("="*60)
        print("\nCommands: play WORD ROW COL H/V | analyze | rack | board | score | exchange | pass | quit")
        print("\nExample: play HELLO 8 4 H")
        
        self.show_board()
        self.show_scores()
        self.show_rack()
        
        while not self.is_game_over():
            print(f"\n--- YOUR TURN ---")
            self.show_rack()
            
            try:
                cmd = input("\n> ").strip().lower()
            except EOFError:
                break
            
            if not cmd:
                continue
            
            parts = cmd.split()
            action = parts[0]
            
            if action == 'quit':
                print("Game ended.")
                break
            
            elif action == 'board':
                self.show_board()
            
            elif action == 'score':
                self.show_scores()
            
            elif action == 'rack':
                self.show_rack()
            
            elif action == 'analyze':
                self.analyze_human_moves()
            
            elif action == 'pass':
                self.consecutive_passes += 1
                if self.final_turns_remaining is not None:
                    self.final_turns_remaining -= 1
                print("You passed.")
                if not self.is_game_over():
                    # Claude's turn
                    self.claude_play()
                    self.show_board()
                    self.show_scores()
            
            elif action == 'exchange' and len(parts) >= 2:
                if self.human_exchange(parts[1]):
                    # Claude's turn
                    self.claude_play()
                    self.show_board()
                    self.show_scores()
            
            elif action == 'play' and len(parts) >= 5:
                word = parts[1].upper()
                try:
                    row = int(parts[2])
                    col = int(parts[3])
                    horizontal = parts[4].upper() == 'H'
                except:
                    print("Usage: play WORD ROW COL H/V")
                    continue
                
                if self.human_play(word, row, col, horizontal):
                    self.show_board()
                    self.show_scores()
                    
                    if not self.is_game_over():
                        # Claude's turn
                        self.claude_play()
                        self.show_board()
                        self.show_scores()
            
            else:
                print("Unknown command. Try: play, analyze, rack, board, score, exchange, pass, quit")
        
        if self.is_game_over():
            self.final_scores()


if __name__ == '__main__':
    game = CrossplayGame()
    game.play()
