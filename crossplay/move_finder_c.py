"""
CROSSPLAY V15 - C-accelerated Move Finder

Uses gaddag_accel.find_moves_c() for GADDAG traversal (the hot path),
then scores moves in Python (only ~3% of total time).

Drop-in replacement for find_all_moves_opt().
"""

import os
import sys
from typing import List, Dict, Tuple, Optional

from .config import (
    BOARD_SIZE, CENTER_ROW, CENTER_COL, VALID_TWO_LETTER,
    TILE_VALUES, BONUS_SQUARES, BINGO_BONUS, RACK_SIZE,
)

# Try to import C extension
_accel = None
_ACCEL_DIR = os.path.dirname(os.path.abspath(__file__))
if _ACCEL_DIR not in sys.path:
    sys.path.insert(0, _ACCEL_DIR)
try:
    import gaddag_accel as _accel
except ImportError:
    pass

# Tile values
_ORD_A = 65
_TV = [0] * 26
for _c, _v in TILE_VALUES.items():
    if _c != '?':
        _TV[ord(_c) - _ORD_A] = _v

# Bonus grid
_BONUS = [[(1, 1)] * 15 for _ in range(15)]
for (r1, c1), btype in BONUS_SQUARES.items():
    r0, c0 = r1 - 1, c1 - 1
    if btype == '2L':
        _BONUS[r0][c0] = (2, 1)
    elif btype == '3L':
        _BONUS[r0][c0] = (3, 1)
    elif btype == '2W':
        _BONUS[r0][c0] = (1, 2)
    elif btype == '3W':
        _BONUS[r0][c0] = (1, 3)

_dictionary = None
def _get_dict():
    global _dictionary
    if _dictionary is None:
        from .dictionary import get_dictionary
        _dictionary = get_dictionary()
    return _dictionary

# Cache bytes(gaddag._data) to avoid 28MB copy per find_all_moves_c call.
# The GADDAG is immutable after construction, so this is safe.
_gdata_bytes_cache = None
_gdata_source_id = None  # id() of the bytearray we cached from

def _get_gdata_bytes(gdata):
    """Return cached bytes view of GADDAG data (avoids 10ms copy per call)."""
    global _gdata_bytes_cache, _gdata_source_id
    if _gdata_source_id != id(gdata):
        _gdata_bytes_cache = bytes(gdata)
        _gdata_source_id = id(gdata)
    return _gdata_bytes_cache


def is_available():
    """Check if C acceleration is available."""
    return _accel is not None


def is_mc_fast_available():
    """Check if Cython MC fast path (BoardContext + find_best_score_c) is available."""
    return _accel is not None and hasattr(_accel, 'prepare_board_context')


def find_all_moves_c(board, gaddag, rack_str: str,
                     board_blanks: List[Tuple[int, int, str]] = None) -> List[Dict]:
    """
    Find all valid moves using C-accelerated GADDAG traversal.
    Falls back to Python if C extension not available.
    """
    if _accel is None:
        from .move_finder_opt import find_all_moves_opt
        return find_all_moves_opt(board, gaddag, rack_str, board_blanks=board_blanks)

    rack_str = rack_str.upper()
    grid = board._grid
    gdata = gaddag._data
    dictionary = _get_dict()
    board_blank_set = {(r - 1, c - 1) for r, c, _ in (board_blanks or [])}

    # Call C extension - returns list of (word, row_1idx, col_1idx, is_horiz, blanks_list)
    raw_moves = _accel.find_moves_c(
        _get_gdata_bytes(gdata),  # cached bytes (avoids 28MB copy per call)
        grid,          # list of lists
        rack_str,
        dictionary,
        VALID_TWO_LETTER,
    )

    # Score moves in C (replaces Python scoring loop)
    moves = _accel.score_moves_c(
        raw_moves, grid, board_blank_set,
        _TV, _BONUS, BINGO_BONUS, RACK_SIZE,
    )

    # ── Post-validation: reject any move whose main word or cross-words
    #    are invalid.  Catches edge cases missed by the C cross-check cache.
    dictionary = _get_dict()
    validated = []
    rejected = []
    for m in moves:
        ok = True
        word_str = m['word']
        r1, c1 = m['row'], m['col']
        r0, c0 = r1 - 1, c1 - 1
        is_h = m['direction'] == 'H'
        wlen = len(word_str)

        # 1) Build the FULL main-axis word (may extend beyond placed word)
        if is_h:
            # scan left from start
            full_start_c = c0
            fc = c0 - 1
            while fc >= 0 and grid[r0][fc] is not None:
                full_start_c = fc
                fc -= 1
            # scan right from end
            full_end_c = c0 + wlen - 1
            fc = c0 + wlen
            while fc < 15 and grid[r0][fc] is not None:
                full_end_c = fc
                fc += 1
            # Build by merging board + placed letters
            full_chars = []
            for fc in range(full_start_c, full_end_c + 1):
                if c0 <= fc < c0 + wlen:
                    full_chars.append(word_str[fc - c0])
                elif grid[r0][fc] is not None:
                    full_chars.append(grid[r0][fc])
                else:
                    ok = False
                    break
        else:
            full_start_r = r0
            fr = r0 - 1
            while fr >= 0 and grid[fr][c0] is not None:
                full_start_r = fr
                fr -= 1
            full_end_r = r0 + wlen - 1
            fr = r0 + wlen
            while fr < 15 and grid[fr][c0] is not None:
                full_end_r = fr
                fr += 1
            full_chars = []
            for fr in range(full_start_r, full_end_r + 1):
                if r0 <= fr < r0 + wlen:
                    full_chars.append(word_str[fr - r0])
                elif grid[fr][c0] is not None:
                    full_chars.append(grid[fr][c0])
                else:
                    ok = False
                    break

        if ok:
            full_word = ''.join(full_chars)
            if len(full_word) >= 2:
                if len(full_word) == 2:
                    if full_word not in VALID_TWO_LETTER:
                        ok = False
                elif not dictionary.is_valid(full_word):
                    ok = False
            if full_word != word_str and ok:
                # The placed word extends existing tiles — re-check the full word
                if len(full_word) == 2:
                    ok = full_word in VALID_TWO_LETTER
                else:
                    ok = dictionary.is_valid(full_word)

        # 2) Validate every cross-word
        if ok:
            for cw in m.get('crosswords', []):
                cw_word = cw['word']
                if len(cw_word) == 2:
                    if cw_word not in VALID_TWO_LETTER:
                        ok = False
                        break
                elif not dictionary.is_valid(cw_word):
                    ok = False
                    break

        # 3) Check for unintended main-axis extension making invalid word
        if ok and full_word != word_str:
            # The move creates a longer word than intended
            if len(full_word) == 2:
                ok = full_word in VALID_TWO_LETTER
            else:
                ok = dictionary.is_valid(full_word)
            if not ok:
                rejected.append((m['word'], f"R{r1}C{c1}", f"extends to '{full_word}' (invalid)"))

        if ok:
            validated.append(m)
        elif m['word'] not in [r[0] for r in rejected]:
            rejected.append((m['word'], f"R{r1}C{c1} {m['direction']}", "invalid cross-word"))

    if rejected:
        import sys
        print(f"  ! Post-validation rejected {len(moves)-len(validated)} moves "
              f"(showing first 5):", file=sys.stderr)
        for word, pos, reason in rejected[:5]:
            print(f"    {word} @ {pos}: {reason}", file=sys.stderr)

    return validated


class CMoveFinder:
    """Drop-in replacement for GADDAGMoveFinder using Cython acceleration.

    Same interface: __init__(board, gaddag) + find_all_moves(rack_str).
    Used by SuperLeaves trainer for ~5-8x faster self-play.
    """

    def __init__(self, board, gaddag=None):
        self.board = board
        if gaddag is None:
            from .gaddag import get_gaddag
            gaddag = gaddag or get_gaddag()
        self.gaddag = gaddag

    def find_all_moves(self, rack_str: str) -> List[Dict]:
        return find_all_moves_c(self.board, self.gaddag, rack_str)
