"""Game move execution mixin -- play_move, record_opponent, exchange.

Extracted from game_manager.py to reduce file size. Mixed into Game
via multiple inheritance. All methods access shared Game instance
state through self.
"""

import time
from datetime import datetime
from typing import List, Tuple

from .board import Board, tiles_used as _board_tiles_used
from .scoring import calculate_move_score, find_crosswords
from .config import TILE_DISTRIBUTION, TILE_VALUES, BONUS_SQUARES, RACK_SIZE, BOARD_SIZE, CENTER_ROW, CENTER_COL
from .tile_tracker import TileTracker
from .nyt_filter import is_nyt_curated
from . import __version__


class GameMovesMixin:
    """Move execution methods for the Game class.
    
    Mixed into Game via multiple inheritance. All methods access
    shared Game instance state through self.
    """
    
    def play_move(self, word: str, row: int, col: int, horizontal: bool,
                  is_opponent: bool = False, rack: str = None,
                  new_rack: str = None) -> Tuple[bool, int]:
        """Play a move. Returns (success, score).

        Args:
            rack: Pre-move rack for tile validation. If None, skips validation.
            new_rack: Post-move rack (after drawing). If provided, the rack is
                set to this value after the move instead of simulating a draw.
                This supports the assisted-play workflow where the user reports
                their new rack after playing on the real board.
        """
        word = word.upper()

        # Capture rack BEFORE the move (for enriched record)
        rack_before = self.state.your_rack if not is_opponent else None

        # Validate word
        if len(word) > 2 and not self.dictionary.is_valid(word):
            print(f"[X] '{word}' is not a valid word!")
            return False, 0

        # Get tiles used
        move_dict = {'word': word, 'row': row, 'col': col, 'direction': 'H' if horizontal else 'V'}
        tiles_used = self._tiles_used(move_dict)

        # Check rack and detect blanks
        blank_word_indices = []
        if rack:
            rack_list = list(rack)
            new_indices = self._get_new_tile_word_indices(move_dict)
            used_idx = 0
            for t in tiles_used:
                if t in rack_list:
                    rack_list.remove(t)
                elif '?' in rack_list:
                    rack_list.remove('?')
                    # Map this used tile to its word index
                    if used_idx < len(new_indices):
                        blank_word_indices.append(new_indices[used_idx])
                else:
                    print(f"[X] Missing tile '{t}' for word '{word}'!")
                    return False, 0
                used_idx += 1

        # Calculate score
        try:
            score, crosswords = calculate_move_score(
                self.board, word, row, col, horizontal,
                blanks_used=blank_word_indices,
                board_blanks=self.state.blank_positions
            )
        except Exception as e:
            print(f"[X] Invalid placement: {e}")
            return False, 0

        # Bingo bonus -- already included in score from calculate_move_score()
        if len(tiles_used) == 7:
            print("[WIN] BINGO! (40 bonus included in score)")

        # Compute bag count BEFORE placing the move (for final_turns tracking)
        _trk_pre = TileTracker()
        _trk_pre.sync_with_board(self.board, your_rack=self.state.your_rack or "",
                                 blanks_in_rack=(self.state.your_rack or "").count('?'),
                                 blank_positions=self.state.blank_positions)
        bag_before_play = _trk_pre.get_bag_count()

        # Place word on board
        self.board.place_word(word, row, col, horizontal)

        # Update score and track drawn tiles
        drawn_tiles = None
        if is_opponent:
            self.state.opp_score += score
        else:
            self.state.your_score += score
            # Update rack
            if new_rack is not None:
                # User provided post-draw rack directly (assisted play workflow)
                # Infer drawn tiles by comparing old rack (minus used) to new rack
                if self.state.your_rack:
                    old_remaining = list(self.state.your_rack)
                    for t in tiles_used:
                        if t in old_remaining:
                            old_remaining.remove(t)
                        elif '?' in old_remaining:
                            old_remaining.remove('?')
                    # drawn = new_rack minus old_remaining
                    new_list = list(new_rack)
                    for t in old_remaining:
                        if t in new_list:
                            new_list.remove(t)
                    drawn_tiles = ''.join(new_list)
                self.state.your_rack = new_rack
            elif self.state.your_rack:
                rack_list = list(self.state.your_rack)
                for t in tiles_used:
                    if t in rack_list:
                        rack_list.remove(t)
                    elif '?' in rack_list:
                        rack_list.remove('?')
                # Draw new tiles
                new_tiles = self.draw_tiles(len(tiles_used))
                drawn_tiles = ''.join(new_tiles)
                rack_list.extend(new_tiles)
                self.state.your_rack = ''.join(rack_list)

        # Record blank positions
        for wi in blank_word_indices:
            if horizontal:
                br, bc = row, col + wi
            else:
                br, bc = row + wi, col
            self.state.blank_positions.append((br, bc, word[wi]))

        # Compute bag count from board state (tiles in bag, not counting opp rack)
        _trk_bag = TileTracker()
        _trk_bag.sync_with_board(self.board, your_rack=self.state.your_rack or "",
                                 blanks_in_rack=(self.state.your_rack or "").count('?'),
                                 blank_positions=self.state.blank_positions)
        bag_count = _trk_bag.get_bag_count()

        # Build engine recommendation record (from last analyze())
        engine_rec = None
        if not is_opponent and self.last_analysis:
            top_word = self.last_analysis['top3'][0]['word'] if self.last_analysis.get('top3') else None
            engine_rec = {
                'top3': self.last_analysis.get('top3', []),
                'followed': (word.upper() == top_word.upper()) if top_word else None,
            }
            self.last_analysis = None  # Clear after use

        # Build enriched move record
        player = self.state.opponent_name if is_opponent else 'me'
        enriched = {
            'word': word,
            'row': row,
            'col': col,
            'dir': 'H' if horizontal else 'V',
            'player': player,
            'score': score,
            'bag': bag_count,
            'blanks': blank_word_indices,
            'cumulative': [self.state.your_score, self.state.opp_score],
            'rack': rack_before,
            'drawn': drawn_tiles,
            'note': 'bingo' if len(tiles_used) >= RACK_SIZE else '',
            'timestamp': datetime.now().isoformat(),
            'engine': engine_rec,
            'engine_version': __version__,
            'win_pct': None,
            'nyt': None,
        }
        self.state.board_moves.append(enriched)

        # Track final_turns_remaining
        _trk_post = TileTracker()
        _trk_post.sync_with_board(self.board, your_rack=self.state.your_rack or "",
                                  blanks_in_rack=(self.state.your_rack or "").count('?'),
                                  blank_positions=self.state.blank_positions)
        bag_after_play = _trk_post.get_bag_count()
        if bag_before_play > 0 and bag_after_play == 0:
            # Bag just emptied: both players get one final turn each = 2 turns remain
            self.state.final_turns_remaining = 2
        elif bag_before_play == 0 and self.state.final_turns_remaining is not None:
            self.state.final_turns_remaining = max(0, self.state.final_turns_remaining - 1)

        # Toggle turn: after your move it's opponent's turn and vice versa
        self.state.is_your_turn = is_opponent  # If opponent played, now it's your turn (True); if you played, their turn (False)

        self.state.updated_at = datetime.now().isoformat()

        # Update blocked square cache and invalidate threats cache
        self.blocked_cache.update_after_move(self.board, word, row, col, horizontal)
        self._threats_cache = None

        d = 'H' if horizontal else 'V'
        who = self.state.opponent_name if is_opponent else "You"
        print(f"[OK] {who} played {word} at R{row}C{col} {d} for {score} points!")

        # Show existing threats (skip when opponent has no turns left)
        # Note: is_your_turn hasn't been toggled yet, so use is_opponent param
        ftr = self.state.final_turns_remaining
        if ftr is not None and ftr <= 0:
            print("\n[ENDGAME] Game over -- no turns remaining.")
        elif ftr == 2 and not is_opponent:
            # User just played and bag emptied -- both players get final turns
            print("\n[ENDGAME] Bag just emptied -- opponent plays next, then you get final move.")
            self.show_existing_threats(top_n=5)
        elif ftr == 2 and is_opponent:
            # Opponent just played and bag emptied -- both players get final turns
            print("\n[ENDGAME] Bag just emptied -- you play next, then opponent gets final move.")
            self.show_existing_threats(top_n=5)
        elif ftr == 1 and not is_opponent:
            # User just played their final turn -- opponent gets last move
            print("\n[ENDGAME] Your final turn done -- opponent gets last move.")
            self.show_existing_threats(top_n=5)
        elif ftr == 1 and is_opponent:
            # Opponent just played -- user's final move, no opp response
            print("\n[ENDGAME] Bag empty -- your final move (no opponent response). Use 'analyze' for 1-ply ranking.")
        elif bag_after_play == 0:
            print("\n[ENDGAME] Bag empty -- use 'analyze' for endgame solver.")
        else:
            self.show_existing_threats(top_n=5)

        # Auto-save if enabled
        self._auto_save()

        # Check if game should be archived
        self._check_archive()

        return True, score

    def record_exchange(self, tiles_dumped: str, new_rack: str):
        """Record a tile exchange in assisted play mode.

        Args:
            tiles_dumped: Letters exchanged away (e.g. 'UYI')
            new_rack: Full rack after drawing replacement tiles (e.g. 'DTPAISE')
        """
        tiles_dumped = tiles_dumped.upper()
        new_rack = new_rack.upper()

        old_rack = self.state.your_rack or ''

        # Validate: tiles_dumped must exist in current rack
        rack_list = list(old_rack)
        for tile in tiles_dumped:
            if tile in rack_list:
                rack_list.remove(tile)
            elif '?' in rack_list:
                rack_list.remove('?')
            else:
                print(f"[X] You don't have '{tile}' to exchange!")
                return False

        # Compute bag count before exchange
        _trk = TileTracker()
        _trk.sync_with_board(self.board, your_rack=old_rack,
                             blanks_in_rack=old_rack.count('?'),
                             blank_positions=self.state.blank_positions)
        bag_count = _trk.get_bag_count()

        if bag_count < 7:
            print(f"[X] Not enough tiles in bag to exchange! (bag={bag_count})")
            return False

        # Compute kept tiles and drawn tiles
        kept = ''.join(rack_list)  # old rack minus dumped
        # drawn = new_rack minus kept
        new_list = list(new_rack)
        for t in kept:
            if t in new_list:
                new_list.remove(t)
        drawn_tiles = ''.join(new_list)

        # Build engine recommendation record (from last analyze())
        engine_rec = None
        if self.last_analysis:
            engine_rec = {
                'top3': self.last_analysis.get('top3', []),
                'followed': 'exchange',
            }
            self.last_analysis = None

        # Build enriched move record
        enriched = {
            'word': 'EXCHANGE',
            'row': 0, 'col': 0, 'dir': '-',
            'player': 'me',
            'score': 0,
            'is_exchange': True,
            'exchange_dump': tiles_dumped,
            'exchange_keep': kept,
            'bag': bag_count,
            'blanks': [],
            'cumulative': [self.state.your_score, self.state.opp_score],
            'rack': old_rack,
            'drawn': drawn_tiles,
            'note': f'exchanged {len(tiles_dumped)} tiles',
            'timestamp': datetime.now().isoformat(),
            'engine': engine_rec,
            'engine_version': __version__,
            'win_pct': None,
            'nyt': None,
        }
        self.state.board_moves.append(enriched)

        # Update rack and toggle turn
        self.state.your_rack = new_rack
        self.state.is_your_turn = False
        self.state.updated_at = datetime.now().isoformat()

        print(f"[OK] Exchanged {tiles_dumped} (kept {kept}), drew {drawn_tiles}")
        print(f"   New rack: {new_rack}")
        print(f"   Score: You {self.state.your_score} - "
              f"{self.state.opponent_name} {self.state.opp_score}")

        self._auto_save()
        return True

    # -----------------------------------------------------------------
    # Opponent move validation (V17.2)
    # -----------------------------------------------------------------
    def _validate_opponent_move(self, word: str, row: int, col: int,
                                horizontal: bool):
        """Validate an opponent move BEFORE placing it on the board.

        Checks bounds, tile conflicts, connectivity, main-word validity,
        and cross-word validity.  Returns (is_valid, errors, warnings).
        All checks are read-only — board state is never modified.
        """
        errors = []
        warnings = []
        new_tile_positions = []
        connects = False
        covers_center = False

        # --- Check 1 & 2: bounds + tile conflicts ---
        for i, letter in enumerate(word):
            r = row + (0 if horizontal else i)
            c = col + (i if horizontal else 0)

            # Bounds
            if r < 1 or r > BOARD_SIZE or c < 1 or c > BOARD_SIZE:
                errors.append(f"Out of bounds at R{r}C{c}")
                continue

            if r == CENTER_ROW and c == CENTER_COL:
                covers_center = True

            existing = self.board.get_tile(r, c)
            if existing is not None:
                if existing != letter:
                    errors.append(
                        f"Tile conflict at R{r}C{c}: board has '{existing}', "
                        f"word has '{letter}'")
                else:
                    connects = True          # overlapping existing tile
            else:
                new_tile_positions.append((r, c))
                if self.board.has_adjacent_tile(r, c):
                    connects = True

        if errors:
            return False, errors, warnings

        # --- Check 3: must place at least one new tile ---
        if not new_tile_positions:
            errors.append("No new tiles placed — word is already on the board")
            return False, errors, warnings

        # --- Check 4: connectivity ---
        if self.board.is_board_empty():
            if not covers_center:
                errors.append(
                    f"First move must cover the center square "
                    f"(R{CENTER_ROW}C{CENTER_COL})")
        else:
            if not connects:
                pos_str = ", ".join(f"R{r}C{c}" for r, c in new_tile_positions)
                errors.append(
                    f"Move does not connect to existing tiles\n"
                    f"        New tile positions: {pos_str}")

        if errors:
            return False, errors, warnings

        # --- Check 5: main word validity ---
        if not self.gaddag.is_word(word):
            errors.append(f"'{word}' is not a valid dictionary word")

        # --- Check 6: cross-word validity ---
        crosswords = find_crosswords(
            self.board, word, row, col, horizontal, new_tile_positions)
        for cw in crosswords:
            cw_word = cw['word']
            cw_dir = 'H' if cw['horizontal'] else 'V'
            if not self.gaddag.is_word(cw_word):
                errors.append(
                    f"Invalid cross-word '{cw_word}' at "
                    f"R{cw['row']}C{cw['col']} {cw_dir}")

        # --- NYT warnings (non-blocking) ---
        if is_nyt_curated(word):
            warnings.append(f"'{word}' is NYT-curated (may not be in WWF)")
        for cw in crosswords:
            if is_nyt_curated(cw['word']):
                warnings.append(
                    f"Cross-word '{cw['word']}' is NYT-curated")

        return len(errors) == 0, errors, warnings

    def record_opponent_move(self, word: str, row: int, col: int, horizontal: bool,
                             score: int, new_opp_score: int = None, new_bag_count: int = None,
                             blanks: List[str] = None, force: bool = False):
        """Record opponent's move with validation.

        Args:
            word: Word played
            row, col: Starting position (1-indexed)
            horizontal: True if horizontal, False if vertical
            score: Points scored for this move
            new_opp_score: Opponent's total score after move (for validation)
            new_bag_count: Tiles remaining in bag after move (for validation)
            blanks: List of letters in the word that are blanks, e.g. ['M'] if M is blank
            force: If True, accept even if validation fails (opp! command)

        Validation checks:
            - Board connectivity, word validity, cross-words (pre-placement)
            - Calculated score vs reported score (catches unreported blanks)
            - Expected opp total vs reported total
            - Expected bag count vs reported count
        """
        word = word.upper()
        blanks = [b.upper() for b in (blanks or [])]

        # --- Move validation gate (V17.2) ---
        direction = 'H' if horizontal else 'V'
        valid, errors, warnings = self._validate_opponent_move(
            word, row, col, horizontal)

        for w in warnings:
            print(f"  [NOTE] {w}")

        if not valid:
            if force:
                print(f"  [FORCED] Accepting {word} at R{row}C{col} "
                      f"{direction} despite errors:")
                for e in errors:
                    print(f"    [X] {e}")
            else:
                print(f"\n  [INVALID MOVE] {word} at R{row}C{col} {direction}")
                for e in errors:
                    print(f"    [X] {e}")
                print(f"\n    To force-accept anyway, use:")
                print(f"    opp! {word.lower()} {row} {col} {direction} {score}")
                print()
                return

        from .scoring import calculate_move_score
        from .config import TILE_VALUES

        # Identify which tile positions are new (placed from rack)
        new_tile_indices = []
        for i, letter in enumerate(word):
            if horizontal:
                check_row, check_col = row, col + i
            else:
                check_row, check_col = row + i, col
            if self.board.is_empty(check_row, check_col):
                new_tile_indices.append(i)
        
        # --- Score calculation helper ---
        def _calc_score_with_blanks(blank_indices):
            """Calculate score with specific indices treated as blanks."""
            try:
                s, _ = calculate_move_score(
                    self.board, word, row, col, horizontal,
                    blanks_used=blank_indices,
                    board_blanks=self.state.blank_positions
                )
                return s
            except Exception:
                return None
        
        # --- Blank detection logic ---
        detected_blanks = []  # list of word indices detected as blanks
        
        if blanks:
            # User specified blanks explicitly — convert letters to indices
            blanks_remaining = list(blanks)
            for i in new_tile_indices:
                if word[i] in blanks_remaining:
                    detected_blanks.append(i)
                    blanks_remaining.remove(word[i])
            
            expected_score = _calc_score_with_blanks(detected_blanks)
            if expected_score is not None and expected_score != score:
                print(f"\nWARNING:  SCORE MISMATCH (with specified blanks)")
                print(f"    Reported: {score} pts, Calculated: {expected_score} pts")
        else:
            # No blanks specified — check if score matches without blanks
            no_blank_score = _calc_score_with_blanks([])
            
            if no_blank_score is not None and no_blank_score != score:
                # Score doesn't match — try to auto-detect which tile is blank
                # Only new tiles from rack can be blanks, and only tiles with
                # point value > 0 would cause a score difference
                candidates = []
                for i in new_tile_indices:
                    if TILE_VALUES.get(word[i], 0) > 0:
                        test_score = _calc_score_with_blanks([i])
                        if test_score == score:
                            candidates.append((i, word[i]))
                
                # Also try 2-blank combinations if single didn't match
                if not candidates and len(new_tile_indices) >= 2:
                    for j in range(len(new_tile_indices)):
                        for k in range(j + 1, len(new_tile_indices)):
                            i1, i2 = new_tile_indices[j], new_tile_indices[k]
                            if TILE_VALUES.get(word[i1], 0) > 0 or TILE_VALUES.get(word[i2], 0) > 0:
                                test_score = _calc_score_with_blanks([i1, i2])
                                if test_score == score:
                                    candidates.append(((i1, i2), f"{word[i1]},{word[i2]}"))
                
                if len(candidates) == 1:
                    # Exactly one blank assignment makes the score work
                    idx_or_pair, letter_info = candidates[0]
                    if isinstance(idx_or_pair, tuple):
                        detected_blanks = list(idx_or_pair)
                        print(f"\n[?] AUTO-DETECTED BLANKS: {word[idx_or_pair[0]]} and {word[idx_or_pair[1]]}")
                        print(f"    Score {no_blank_score} -> {score} (only valid blank assignment)")
                    else:
                        detected_blanks = [idx_or_pair]
                        print(f"\n[?] AUTO-DETECTED BLANK: {word[idx_or_pair]} (position {idx_or_pair + 1} in word)")
                        print(f"    Score {no_blank_score} -> {score} (only valid blank assignment)")
                elif len(candidates) > 1:
                    # Multiple possible blank assignments
                    print(f"\nWARNING:  SCORE MISMATCH -- possible blank detected!")
                    print(f"    Reported: {score} pts, Without blanks: {no_blank_score} pts")
                    print(f"    Possible blanks (each makes score match):")
                    for idx_or_pair, letter_info in candidates:
                        if isinstance(idx_or_pair, tuple):
                            print(f"      blanks=['{word[idx_or_pair[0]]}', '{word[idx_or_pair[1]]}']")
                        else:
                            pos = idx_or_pair
                            if horizontal:
                                print(f"      blanks=['{word[pos]}']  (R{row}C{col + pos})")
                            else:
                                print(f"      blanks=['{word[pos]}']  (R{row + pos}C{col})")
                    print(f"    TIP: Re-record with blanks=['X'] to confirm")
                else:
                    # No single/double blank makes the score match
                    print(f"\nWARNING:  SCORE MISMATCH!")
                    print(f"    Reported: {score} pts, Calculated: {no_blank_score} pts")
                    print(f"    No blank assignment found -- check move details")
            else:
                expected_score = no_blank_score
        
        # Compute bag count BEFORE placing the move (for final_turns tracking)
        _trk_pre = TileTracker()
        _trk_pre.sync_with_board(self.board, your_rack=self.state.your_rack or "",
                                 blanks_in_rack=(self.state.your_rack or "").count('?'),
                                 blank_positions=self.state.blank_positions)
        bag_before = _trk_pre.get_bag_count()

        # Place the word on board
        self.board.place_word(word, row, col, horizontal)

        # Record blank positions (from explicit blanks or auto-detected)
        for i in detected_blanks:
            if horizontal:
                blank_row, blank_col = row, col + i
            else:
                blank_row, blank_col = row + i, col
            self.state.blank_positions.append((blank_row, blank_col, word[i]))
            print(f"    [>] Blank recorded at R{blank_row}C{blank_col} = {word[i]}")

        # Update opponent score
        old_opp_score = self.state.opp_score
        self.state.opp_score += score

        # Validate opponent total score
        if new_opp_score is not None:
            if self.state.opp_score != new_opp_score:
                print(f"\nWARNING:  OPP SCORE MISMATCH!")
                print(f"    Expected: {old_opp_score} + {score} = {self.state.opp_score}")
                print(f"    Reported: {new_opp_score}")
                print(f"    Using reported score.")
                self.state.opp_score = new_opp_score

        # Validate bag count
        if new_bag_count is not None:
            # Sync tracker to get current count
            tracker = TileTracker()
            tracker.sync_with_board(self.board, self.state.your_rack, 0, self.state.blank_positions)

            # Expected: unseen - 7 (opp rack) = bag
            unseen = tracker.get_unseen_count()
            expected_bag = unseen - 7  # Assuming opp has 7 tiles

            if expected_bag != new_bag_count:
                diff = expected_bag - new_bag_count
                print(f"\nWARNING:  BAG COUNT MISMATCH!")
                print(f"    Expected: ~{expected_bag} tiles")
                print(f"    Reported: {new_bag_count} tiles")
                print(f"    Difference: {diff} tiles")
                if diff > 0:
                    print(f"    TIP: Missing tiles - check if blanks were reported")

        # Compute bag count from board state for enriched record
        _trk_bag = TileTracker()
        _trk_bag.sync_with_board(self.board, your_rack=self.state.your_rack or "",
                                 blanks_in_rack=(self.state.your_rack or "").count('?'),
                                 blank_positions=self.state.blank_positions)
        bag_count = _trk_bag.get_bag_count()

        # Build enriched move record (after score/bag updates for accurate cumulative)
        enriched = {
            'word': word,
            'row': row,
            'col': col,
            'dir': 'H' if horizontal else 'V',
            'player': 'opp',
            'score': score,
            'bag': new_bag_count if new_bag_count is not None else bag_count,
            'blanks': detected_blanks,
            'cumulative': [self.state.your_score, self.state.opp_score],
            'rack': None,       # Unknown for opponent
            'drawn': None,      # Unknown for opponent
            'note': 'bingo' if len(new_tile_indices) >= RACK_SIZE else '',
            'timestamp': datetime.now().isoformat(),
            'engine': None,     # No engine analysis for opponent moves
            'engine_version': __version__,
            'win_pct': None,
            'nyt': None,
        }
        self.state.board_moves.append(enriched)

        # Track final_turns_remaining
        _trk_post = TileTracker()
        _trk_post.sync_with_board(self.board, your_rack=self.state.your_rack or "",
                                  blanks_in_rack=(self.state.your_rack or "").count('?'),
                                  blank_positions=self.state.blank_positions)
        bag_after = _trk_post.get_bag_count()
        if bag_before > 0 and bag_after == 0:
            # Bag just emptied: both players get one final turn each = 2 turns remain
            self.state.final_turns_remaining = 2
        elif bag_before == 0 and self.state.final_turns_remaining is not None:
            self.state.final_turns_remaining = max(0, self.state.final_turns_remaining - 1)

        self.state.is_your_turn = True
        self.state.updated_at = datetime.now().isoformat()

        # Update blocked square cache and invalidate threats cache
        self.blocked_cache.update_after_move(self.board, word, row, col, horizontal)
        self._threats_cache = None

        print(f"\n[NOTE] Recorded: {self.state.opponent_name} played {word} for {score} pts")
        print(f"   Score: You {self.state.your_score} - {self.state.opponent_name} {self.state.opp_score}")

        # Auto-analyze after opponent move: always run full analysis so the
        # engine recommendation is available before any move is played.
        # The analyze() method already shows threats as its first step.
        ftr = self.state.final_turns_remaining
        if ftr is not None and ftr <= 0:
            print("\n[ENDGAME] Game over -- no turns remaining.")
        elif self.state.your_rack:
            # We have a rack — run full analysis automatically
            print("\n[AUTO-ANALYZE] Running engine analysis...")
            try:
                self.analyze()
            except Exception as e:
                print(f"[AUTO-ANALYZE] Analysis failed: {e}")
                # Fall back to showing threats only
                self.show_existing_threats(top_n=5)
        else:
            # No rack set yet — just show threats
            if ftr == 2 and self.state.is_your_turn:
                print("\n[ENDGAME] Bag just emptied -- you play next, then opponent gets final move.")
                self.show_existing_threats(top_n=5)
            elif ftr == 1 and self.state.is_your_turn:
                print("\n[ENDGAME] Bag empty -- your final move (no opponent response).")
            elif ftr == 1 and not self.state.is_your_turn:
                print("\n[ENDGAME] Your final turn done -- opponent gets last move.")
                self.show_existing_threats(top_n=5)
            elif bag_after == 0:
                print("\n[ENDGAME] Bag empty.")
            else:
                self.show_existing_threats(top_n=5)
            print("\n[TIP] Set your rack with 'rack LETTERS' then 'analyze' for full engine analysis.")

        # Auto-save if enabled
        self._auto_save()

        # Check if game should be archived
        self._check_archive()
    
