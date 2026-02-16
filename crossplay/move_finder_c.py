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
        bytes(gdata),  # bytearray -> bytes for buffer protocol
        grid,          # list of lists
        rack_str,
        dictionary,
        VALID_TWO_LETTER,
    )

    # Score each move in Python (only ~3% of total time)
    bonus_grid = _BONUS
    tv = _TV
    OA = _ORD_A
    moves = []

    for word_str, row1, col1, is_horiz, blanks_list in raw_moves:
        r0 = row1 - 1
        c0 = col1 - 1
        wlen = len(word_str)
        blanks_set = set(blanks_list)

        # Score main word
        main_score = 0
        word_mult = 1
        new_positions = []

        for i in range(wlen):
            if is_horiz:
                cr, cc = r0, c0 + i
            else:
                cr, cc = r0 + i, c0

            is_new = grid[cr][cc] is None

            if i in blanks_set:
                lv = 0
            elif not is_new and (cr, cc) in board_blank_set:
                lv = 0
            else:
                lv = tv[ord(word_str[i]) - OA]

            if is_new:
                new_positions.append((cr, cc, i))
                lm, wm = bonus_grid[cr][cc]
                lv *= lm
                word_mult *= wm

            main_score += lv

        main_score *= word_mult

        # Score crosswords
        cw_total = 0
        crosswords = []
        for cr, cc, wi in new_positions:
            placed_letter = word_str[wi]
            is_blank = wi in blanks_set

            if is_horiz:
                has_perp = (cr > 0 and grid[cr-1][cc] is not None) or \
                           (cr < 14 and grid[cr+1][cc] is not None)
            else:
                has_perp = (cc > 0 and grid[cr][cc-1] is not None) or \
                           (cc < 14 and grid[cr][cc+1] is not None)

            if not has_perp:
                continue

            cw_horiz = not is_horiz
            cw_score = 0
            cw_wmult = 1
            cw_chars = []
            cw_start_r, cw_start_c = cr, cc

            if cw_horiz:
                c2 = cc - 1
                pre = []
                while c2 >= 0 and grid[cr][c2] is not None:
                    ch = grid[cr][c2]
                    pre.append(ch)
                    cw_score += 0 if (cr, c2) in board_blank_set else tv[ord(ch)-OA]
                    cw_start_c = c2
                    c2 -= 1
                pre.reverse()
                cw_chars.extend(pre)
                cw_chars.append(placed_letter)
                plv = 0 if is_blank else tv[ord(placed_letter)-OA]
                lm, wm = bonus_grid[cr][cc]
                plv *= lm
                cw_wmult *= wm
                cw_score += plv
                c2 = cc + 1
                while c2 < 15 and grid[cr][c2] is not None:
                    ch = grid[cr][c2]
                    cw_chars.append(ch)
                    cw_score += 0 if (cr, c2) in board_blank_set else tv[ord(ch)-OA]
                    c2 += 1
            else:
                r2 = cr - 1
                pre = []
                while r2 >= 0 and grid[r2][cc] is not None:
                    ch = grid[r2][cc]
                    pre.append(ch)
                    cw_score += 0 if (r2, cc) in board_blank_set else tv[ord(ch)-OA]
                    cw_start_r = r2
                    r2 -= 1
                pre.reverse()
                cw_chars.extend(pre)
                cw_chars.append(placed_letter)
                plv = 0 if is_blank else tv[ord(placed_letter)-OA]
                lm, wm = bonus_grid[cr][cc]
                plv *= lm
                cw_wmult *= wm
                cw_score += plv
                r2 = cr + 1
                while r2 < 15 and grid[r2][cc] is not None:
                    ch = grid[r2][cc]
                    cw_chars.append(ch)
                    cw_score += 0 if (r2, cc) in board_blank_set else tv[ord(ch)-OA]
                    r2 += 1

            cw_score *= cw_wmult
            cw_total += cw_score
            crosswords.append({
                'word': ''.join(cw_chars),
                'row': cw_start_r + 1,
                'col': cw_start_c + 1,
                'horizontal': cw_horiz,
                'score': cw_score
            })

        total = main_score + cw_total
        if len(new_positions) >= RACK_SIZE:
            total += BINGO_BONUS

        moves.append({
            'word': word_str,
            'row': row1,
            'col': col1,
            'direction': 'H' if is_horiz else 'V',
            'score': total,
            'crosswords': crosswords,
            'blanks_used': blanks_list,
        })

    moves.sort(key=lambda m: -m['score'])

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
