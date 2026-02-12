# cython: language_level=3, boundscheck=False, wraparound=False
"""
CROSSPLAY v12.1 - C-accelerated GADDAG move finder.

Hot-path data structures are C-level:
  - Word buffer: char[15] + int length (no Python list append/pop)
  - Rack counts: int[27] array (no Python dict get/set)
  - Blanks buffer: int[7] + int count (no Python list allocation)

Cross-checks and dictionary validation remain Python (only ~58 calls/rack).

Build: python setup_accel.py build_ext --inplace
"""

from cpython.bytes cimport PyBytes_AsString
from libc.string cimport memcpy

# =========================================================================
# C-level node helpers (nogil)
# =========================================================================

cdef inline int _get_child(const unsigned char* data, int offset, int char_idx) nogil:
    cdef int count = data[offset] & 0x1F
    cdef int pos = offset + 1
    cdef int end = pos + count * 5
    cdef int ci
    while pos < end:
        ci = data[pos]
        if ci == char_idx:
            return (data[pos+1] | (data[pos+2] << 8) |
                    (data[pos+3] << 16) | (data[pos+4] << 24))
        if ci > char_idx:
            return -1
        pos += 5
    return -1

cdef inline bint _is_terminal(const unsigned char* data, int offset) nogil:
    return (data[offset] & 0x80) != 0

# Inline child iteration — returns count and fills arrays (no Python list)
cdef inline int _get_children(const unsigned char* data, int offset,
                              int* child_idxs, int* child_offsets) nogil:
    """Fill arrays with children. Returns child count."""
    cdef int count = data[offset] & 0x1F
    cdef int pos = offset + 1
    cdef int i
    for i in range(count):
        child_idxs[i] = data[pos]
        child_offsets[i] = (data[pos+1] | (data[pos+2] << 8) |
                            (data[pos+3] << 16) | (data[pos+4] << 24))
        pos += 5
    return count


# =========================================================================
# Python-visible helpers (used by move_finder_c.py)
# =========================================================================

def get_child(bytes gdata, int offset, int char_idx):
    cdef const unsigned char* data = <const unsigned char*>PyBytes_AsString(gdata)
    return _get_child(data, offset, char_idx)

def is_terminal(bytes gdata, int offset):
    cdef const unsigned char* data = <const unsigned char*>PyBytes_AsString(gdata)
    return _is_terminal(data, offset)


# =========================================================================
# Constants and state
# =========================================================================

DEF DELIM_IDX = 26
DEF BOARD_SIZE = 15
DEF MAX_WORD = 15
DEF MAX_BLANKS = 7

cdef class _SearchState:
    cdef const unsigned char* data
    cdef list grid                  # Python list-of-lists (kept for cross-check)
    cdef int rack[27]               # C array: counts per letter (0-25=A-Z, 26=blank)
    cdef int rack_indices[7]        # C array: which letter indices are in rack
    cdef int rack_nletters          # how many distinct letters in rack
    cdef int num_blanks
    cdef bint empty_board
    cdef dict cross_cache
    cdef object NOT_COMPUTED
    cdef object dictionary
    cdef set valid_2
    cdef list results
    cdef set seen
    # Cross-check cache as C-friendly: returns set or None
    # We keep this as Python because dictionary.is_valid is Python anyway


# =========================================================================
# Grid helper: get cell value as int (-1=empty, 0-25=A-Z)
# =========================================================================

cdef inline int _grid_get(list grid, int r, int c):
    """Get grid cell as int. Returns -1 for empty, 0-25 for A-Z."""
    cdef object val = (<list>(<list>grid)[r])[c]
    if val is None:
        return -1
    return <int>(ord(<str>val)) - 65


# =========================================================================
# Cross-check (perpendicular word validation) — stays Python
# =========================================================================

cdef object _cross_check(_SearchState st, int r0, int c0, bint horiz):
    """Returns set of valid letters, or None if unconstrained."""
    cdef tuple key = (r0, c0, horiz)
    cached = st.cross_cache.get(key, st.NOT_COMPUTED)
    if cached is not st.NOT_COMPUTED:
        return cached

    cdef list above = []
    cdef list below = []
    cdef int rr, cc
    cdef list grid = st.grid

    if horiz:
        rr = r0 - 1
        while rr >= 0 and (<list>(<list>grid)[rr])[c0] is not None:
            above.append((<list>(<list>grid)[rr])[c0])
            rr -= 1
        above.reverse()
        rr = r0 + 1
        while rr < BOARD_SIZE and (<list>(<list>grid)[rr])[c0] is not None:
            below.append((<list>(<list>grid)[rr])[c0])
            rr += 1
    else:
        cc = c0 - 1
        while cc >= 0 and (<list>(<list>grid)[r0])[cc] is not None:
            above.append((<list>(<list>grid)[r0])[cc])
            cc -= 1
        above.reverse()
        cc = c0 + 1
        while cc < BOARD_SIZE and (<list>(<list>grid)[r0])[cc] is not None:
            below.append((<list>(<list>grid)[r0])[cc])
            cc += 1

    if not above and not below:
        st.cross_cache[key] = None
        return None

    cdef str prefix = ''.join(above)
    cdef str suffix = ''.join(below)
    cdef set valid = set()
    cdef str letter, w
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        w = prefix + letter + suffix
        if len(w) == 2:
            if w in st.valid_2:
                valid.add(letter)
        elif st.dictionary.is_valid(w):
            valid.add(letter)
    st.cross_cache[key] = valid
    return valid

# Cross-check returning a C bitmask for the hot path
cdef inline unsigned int _cross_check_mask(_SearchState st, int r0, int c0, bint horiz):
    """Returns bitmask: bit i set = letter i valid. 0xFFFFFFFF = unconstrained."""
    cdef object cc = _cross_check(st, r0, c0, horiz)
    if cc is None:
        return 0xFFFFFFFF  # all valid
    cdef unsigned int mask = 0
    cdef str letter
    for letter in <set>cc:
        mask |= (1 << (<int>(ord(letter)) - 65))
    return mask


# =========================================================================
# Record a valid move — converts C buffers back to Python for output
# =========================================================================

cdef void _record_move(_SearchState st, int* wordbuf, int wlen,
                       bint horiz, int* blankbuf, int nblank,
                       int sr0, int sc0) except *:
    if sr0 < 0 or sc0 < 0:
        return
    if horiz and sc0 + wlen > BOARD_SIZE:
        return
    if not horiz and sr0 + wlen > BOARD_SIZE:
        return

    # Convert C word buffer to Python string
    cdef int i
    cdef list chars = []
    for i in range(wlen):
        chars.append(chr(wordbuf[i] + 65))
    cdef str word = ''.join(chars)

    if wlen == 2:
        if word not in st.valid_2:
            return
    elif not st.dictionary.is_valid(word):
        return

    cdef tuple key = (word, sr0, sc0, horiz)
    if key in st.seen:
        return
    st.seen.add(key)

    # Connectivity check
    cdef bint connects = False
    cdef bint uses_new = False
    cdef int rr, cc
    cdef list grid = st.grid
    for i in range(wlen):
        rr = sr0 + (0 if horiz else i)
        cc = sc0 + (i if horiz else 0)
        if (<list>(<list>grid)[rr])[cc] is not None:
            connects = True
        else:
            uses_new = True
            if (rr > 0 and (<list>(<list>grid)[rr-1])[cc] is not None) or \
               (rr < 14 and (<list>(<list>grid)[rr+1])[cc] is not None) or \
               (cc > 0 and (<list>(<list>grid)[rr])[cc-1] is not None) or \
               (cc < 14 and (<list>(<list>grid)[rr])[cc+1] is not None):
                connects = True

    if not uses_new:
        return
    if st.empty_board:
        found = False
        for i in range(wlen):
            rr = sr0 + (0 if horiz else i)
            cc = sc0 + (i if horiz else 0)
            if rr == 7 and cc == 7:
                found = True
                break
        if not found:
            return
    elif not connects:
        return

    # Convert blanks C buffer to Python list
    cdef list blanks_list = []
    for i in range(nblank):
        blanks_list.append(blankbuf[i])

    st.results.append((word, sr0 + 1, sc0 + 1, horiz, blanks_list))


# =========================================================================
# extend_right — hottest function, now with C buffers
# =========================================================================

cdef void _extend_right(_SearchState st,
                        int row0, int col0, bint horiz, int offset,
                        int* wordbuf, int wlen, int sr0, int sc0,
                        int blanks_rem, int* blankbuf, int nblank) except *:
    cdef const unsigned char* data = st.data
    cdef int existing, idx, child
    cdef int ci, child_off
    cdef unsigned int cc_mask
    cdef int n_children
    cdef int c_idxs[32]
    cdef int c_offs[32]
    cdef int j

    if row0 < 0 or row0 >= BOARD_SIZE or col0 < 0 or col0 >= BOARD_SIZE:
        if _is_terminal(data, offset) and wlen >= 2:
            _record_move(st, wordbuf, wlen, horiz, blankbuf, nblank, sr0, sc0)
        return

    existing = _grid_get(st.grid, row0, col0)

    if existing >= 0:
        # Square has a tile — must follow it
        child = _get_child(data, offset, existing)
        if child >= 0:
            wordbuf[wlen] = existing
            if horiz:
                _extend_right(st, row0, col0 + 1, True, child,
                              wordbuf, wlen + 1, sr0, sc0,
                              blanks_rem, blankbuf, nblank)
            else:
                _extend_right(st, row0 + 1, col0, False, child,
                              wordbuf, wlen + 1, sr0, sc0,
                              blanks_rem, blankbuf, nblank)
    else:
        # Empty square — try rack letters and blanks
        cc_mask = _cross_check_mask(st, row0, col0, horiz)

        if _is_terminal(data, offset) and wlen >= 2:
            _record_move(st, wordbuf, wlen, horiz, blankbuf, nblank, sr0, sc0)

        # Try each rack letter
        for j in range(st.rack_nletters):
            idx = st.rack_indices[j]
            if st.rack[idx] <= 0:
                continue
            if not (cc_mask & (1 << idx)):
                continue
            child = _get_child(data, offset, idx)
            if child < 0:
                continue

            st.rack[idx] -= 1
            wordbuf[wlen] = idx
            if horiz:
                _extend_right(st, row0, col0 + 1, True, child,
                              wordbuf, wlen + 1, sr0, sc0,
                              blanks_rem, blankbuf, nblank)
            else:
                _extend_right(st, row0 + 1, col0, False, child,
                              wordbuf, wlen + 1, sr0, sc0,
                              blanks_rem, blankbuf, nblank)
            st.rack[idx] += 1

        # Try blank tiles
        if blanks_rem > 0:
            n_children = _get_children(data, offset, c_idxs, c_offs)
            for j in range(n_children):
                ci = c_idxs[j]
                if ci == DELIM_IDX:
                    continue
                if not (cc_mask & (1 << ci)):
                    continue
                wordbuf[wlen] = ci
                blankbuf[nblank] = wlen
                if horiz:
                    _extend_right(st, row0, col0 + 1, True, c_offs[j],
                                  wordbuf, wlen + 1, sr0, sc0,
                                  blanks_rem - 1, blankbuf, nblank + 1)
                else:
                    _extend_right(st, row0 + 1, col0, False, c_offs[j],
                                  wordbuf, wlen + 1, sr0, sc0,
                                  blanks_rem - 1, blankbuf, nblank + 1)


# =========================================================================
# gen_left_part — prefix generation with C buffers
#
# The word buffer builds the prefix in REVERSE order (right-to-left),
# matching the GADDAG's reversed prefix representation. When crossing
# the delimiter to extend_right, we reverse the prefix into a new buffer.
# =========================================================================

cdef void _gen_left_part(_SearchState st,
                         int anchor_r0, int anchor_c0, bint horiz,
                         int offset, int* prefbuf, int plen, int limit,
                         int blanks_rem, int* blankbuf, int nblank) except *:
    cdef const unsigned char* data = st.data
    cdef int delim_child, sr0, sc0, pr0, pc0
    cdef int idx, child
    cdef int ci, child_off
    cdef int n_children
    cdef int c_idxs[32]
    cdef int c_offs[32]
    cdef int j, i
    cdef unsigned int cc_mask
    # Buffer for extend_right (reversed prefix)
    cdef int wordbuf[MAX_WORD]
    cdef int blankbuf2[MAX_BLANKS]

    # Try crossing delimiter to extend right
    delim_child = _get_child(data, offset, DELIM_IDX)
    if delim_child >= 0:
        if horiz:
            sr0 = anchor_r0
            sc0 = anchor_c0 - plen
        else:
            sr0 = anchor_r0 - plen
            sc0 = anchor_c0
        # Reverse prefix into wordbuf for extend_right
        for i in range(plen):
            wordbuf[i] = prefbuf[plen - 1 - i]
        # Copy blanks (blank indices need adjusting: they were stored
        # relative to reversed prefix, now need to be relative to word)
        for i in range(nblank):
            blankbuf2[i] = plen - 1 - blankbuf[i]
        _extend_right(st, anchor_r0, anchor_c0, horiz, delim_child,
                      wordbuf, plen, sr0, sc0,
                      blanks_rem, blankbuf2, nblank)

    # Try extending left if we have room
    if limit > 0:
        if horiz:
            pr0 = anchor_r0
            pc0 = anchor_c0 - plen - 1
        else:
            pr0 = anchor_r0 - plen - 1
            pc0 = anchor_c0
        if pr0 < 0 or pc0 < 0:
            return

        cc_mask = _cross_check_mask(st, pr0, pc0, horiz)

        for j in range(st.rack_nletters):
            idx = st.rack_indices[j]
            if st.rack[idx] <= 0:
                continue
            if not (cc_mask & (1 << idx)):
                continue
            child = _get_child(data, offset, idx)
            if child < 0:
                continue
            st.rack[idx] -= 1
            prefbuf[plen] = idx
            _gen_left_part(st, anchor_r0, anchor_c0, horiz, child,
                           prefbuf, plen + 1, limit - 1,
                           blanks_rem, blankbuf, nblank)
            st.rack[idx] += 1

        if blanks_rem > 0:
            n_children = _get_children(data, offset, c_idxs, c_offs)
            for j in range(n_children):
                ci = c_idxs[j]
                if ci == DELIM_IDX:
                    continue
                if not (cc_mask & (1 << ci)):
                    continue
                prefbuf[plen] = ci
                blankbuf[nblank] = plen
                _gen_left_part(st, anchor_r0, anchor_c0, horiz, c_offs[j],
                               prefbuf, plen + 1, limit - 1,
                               blanks_rem - 1, blankbuf, nblank + 1)

    # At root with limit=0: single letter at anchor then cross delimiter
    if plen == 0 and limit == 0:
        cc_mask = _cross_check_mask(st, anchor_r0, anchor_c0, horiz)

        for j in range(st.rack_nletters):
            idx = st.rack_indices[j]
            if st.rack[idx] <= 0:
                continue
            if not (cc_mask & (1 << idx)):
                continue
            lc = _get_child(data, 0, idx)
            if lc < 0:
                continue
            dc = _get_child(data, lc, DELIM_IDX)
            if dc < 0:
                continue
            st.rack[idx] -= 1
            wordbuf[0] = idx
            if horiz:
                _extend_right(st, anchor_r0, anchor_c0 + 1, True, dc,
                              wordbuf, 1, anchor_r0, anchor_c0,
                              blanks_rem, blankbuf, 0)
            else:
                _extend_right(st, anchor_r0 + 1, anchor_c0, False, dc,
                              wordbuf, 1, anchor_r0, anchor_c0,
                              blanks_rem, blankbuf, 0)
            st.rack[idx] += 1

        if blanks_rem > 0:
            n_children = _get_children(data, 0, c_idxs, c_offs)
            for j in range(n_children):
                ci = c_idxs[j]
                if ci == DELIM_IDX:
                    continue
                if not (cc_mask & (1 << ci)):
                    continue
                dc = _get_child(data, c_offs[j], DELIM_IDX)
                if dc < 0:
                    continue
                wordbuf[0] = ci
                blankbuf[0] = 0
                if horiz:
                    _extend_right(st, anchor_r0, anchor_c0 + 1, True, dc,
                                  wordbuf, 1, anchor_r0, anchor_c0,
                                  blanks_rem - 1, blankbuf, 1)
                else:
                    _extend_right(st, anchor_r0 + 1, anchor_c0, False, dc,
                                  wordbuf, 1, anchor_r0, anchor_c0,
                                  blanks_rem - 1, blankbuf, 1)


# =========================================================================
# extend_from_existing — walk existing tiles into GADDAG, then extend right
# =========================================================================

cdef void _extend_from_existing(_SearchState st,
                                int anchor_r0, int anchor_c0,
                                bint horiz, int blanks_rem) except *:
    cdef const unsigned char* data = st.data
    cdef list grid = st.grid
    cdef int prefix[MAX_WORD]
    cdef int plen = 0
    cdef int c, r, sr0, sc0, offset, idx, child, dc
    cdef int wordbuf[MAX_WORD]
    cdef int blankbuf[MAX_BLANKS]
    cdef int i, val

    # Collect existing tiles to the left/above (stored in forward order)
    if horiz:
        c = anchor_c0 - 1
        while c >= 0:
            val = _grid_get(grid, anchor_r0, c)
            if val < 0:
                break
            prefix[plen] = val
            plen += 1
            c -= 1
        sc0 = c + 1
        sr0 = anchor_r0
    else:
        r = anchor_r0 - 1
        while r >= 0:
            val = _grid_get(grid, r, anchor_c0)
            if val < 0:
                break
            prefix[plen] = val
            plen += 1
            r -= 1
        sr0 = r + 1
        sc0 = anchor_c0

    # prefix is in reverse order (closest to anchor first) — which is
    # exactly what GADDAG wants (reversed prefix traversal)
    offset = 0
    for i in range(plen):
        idx = prefix[i]
        if idx < 0 or idx >= 26:
            return
        child = _get_child(data, offset, idx)
        if child < 0:
            return
        offset = child

    # Cross delimiter
    dc = _get_child(data, offset, DELIM_IDX)
    if dc < 0:
        return

    # Build forward-order prefix in wordbuf for extend_right
    for i in range(plen):
        wordbuf[i] = prefix[plen - 1 - i]

    _extend_right(st, anchor_r0, anchor_c0, horiz, dc,
                  wordbuf, plen, sr0, sc0, blanks_rem, blankbuf, 0)


# =========================================================================
# calc_left_limit
# =========================================================================

cdef int _calc_left_limit(list grid, int anchor_r0, int anchor_c0,
                          bint horiz):
    cdef int limit = 0
    cdef int r0, c, c0, r
    if horiz:
        r0 = anchor_r0
        c = anchor_c0 - 1
        while c >= 0:
            if (<list>(<list>grid)[r0])[c] is not None:
                break
            if (r0 > 0 and (<list>(<list>grid)[r0-1])[c] is not None) or \
               (r0 < 14 and (<list>(<list>grid)[r0+1])[c] is not None) or \
               (c > 0 and (<list>(<list>grid)[r0])[c-1] is not None) or \
               (c < 14 and (<list>(<list>grid)[r0])[c+1] is not None):
                break
            limit += 1
            c -= 1
    else:
        c0 = anchor_c0
        r = anchor_r0 - 1
        while r >= 0:
            if (<list>(<list>grid)[r])[c0] is not None:
                break
            if (r > 0 and (<list>(<list>grid)[r-1])[c0] is not None) or \
               (r < 14 and (<list>(<list>grid)[r+1])[c0] is not None) or \
               (c0 > 0 and (<list>(<list>grid)[r])[c0-1] is not None) or \
               (c0 < 14 and (<list>(<list>grid)[r])[c0+1] is not None):
                break
            limit += 1
            r -= 1
    return limit


# =========================================================================
# Public API: find_moves_c
# =========================================================================

def find_moves_c(bytes gdata, list grid, str rack_str,
                 object dictionary, set valid_2):
    """
    Find all valid moves using GADDAG traversal.
    Returns list of (word_str, row_1idx, col_1idx, is_horiz, blanks_list).
    """
    cdef _SearchState st = _SearchState.__new__(_SearchState)
    st.data = <const unsigned char*>PyBytes_AsString(gdata)
    st.grid = grid
    st.dictionary = dictionary
    st.valid_2 = valid_2
    st.results = []
    st.seen = set()
    st.cross_cache = {}
    st.NOT_COMPUTED = object()

    # Parse rack into C array
    cdef int i
    for i in range(27):
        st.rack[i] = 0
    cdef int num_blanks = 0
    cdef int ci
    cdef str ch
    for ch in rack_str:
        if ch == '?':
            num_blanks += 1
        else:
            ci = <int>(ord(ch)) - 65
            if 0 <= ci < 26:
                st.rack[ci] += 1
    st.num_blanks = num_blanks
    st.rack[26] = num_blanks  # store blank count too

    # Build rack_indices: which letter indices have count > 0
    cdef int n = 0
    for i in range(26):
        if st.rack[i] > 0:
            st.rack_indices[n] = i
            n += 1
    st.rack_nletters = n

    # Count rack letters for left_limit clamping
    cdef int rack_letter_count = 0
    for i in range(26):
        rack_letter_count += st.rack[i]

    # Detect empty board
    cdef bint empty_board = True
    cdef int r, c
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if (<list>(<list>grid)[r])[c] is not None:
                empty_board = False
                break
        if not empty_board:
            break
    st.empty_board = empty_board

    # Find anchor squares
    cdef list anchors
    if empty_board:
        anchors = [(7, 7)]
    else:
        anchors = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (<list>(<list>grid)[r])[c] is not None:
                    continue
                if (r > 0 and (<list>(<list>grid)[r-1])[c] is not None) or \
                   (r < 14 and (<list>(<list>grid)[r+1])[c] is not None) or \
                   (c > 0 and (<list>(<list>grid)[r])[c-1] is not None) or \
                   (c < 14 and (<list>(<list>grid)[r])[c+1] is not None):
                    anchors.append((r, c))

    # Main loop
    cdef int ar, ac, ll, max_limit
    cdef bint has_left
    cdef int prefbuf[MAX_WORD]
    cdef int blankbuf[MAX_BLANKS]
    max_limit = rack_letter_count + num_blanks - 1

    for ar, ac in anchors:
        for horiz_pass in [True, False]:
            if horiz_pass:
                has_left = ac > 0 and (<list>(<list>grid)[ar])[ac - 1] is not None
            else:
                has_left = ar > 0 and (<list>(<list>grid)[ar - 1])[ac] is not None

            if has_left:
                _extend_from_existing(st, ar, ac, horiz_pass, num_blanks)
            else:
                ll = _calc_left_limit(grid, ar, ac, horiz_pass)
                if ll > max_limit:
                    ll = max_limit
                if ll < 0:
                    ll = 0
                _gen_left_part(st, ar, ac, horiz_pass, 0,
                               prefbuf, 0, ll,
                               num_blanks, blankbuf, 0)

    return st.results


# =========================================================================
# BoardContext — pre-computed board state for fast MC simulation
#
# Usage: ctx = prepare_board_context(...) once per board state,
#        then find_best_score_c(ctx, rack) for each of K simulations.
#
# All cross-checks, perpendicular scores, anchors, and GADDAG caches
# are pre-computed once. The K-sim loop only does GADDAG traversal +
# scoring using C arrays — zero Python callbacks except word validation.
# =========================================================================

cdef class BoardContext:
    # Board state
    cdef int grid[15][15]               # -1=empty, 0-25=A-Z
    cdef int board_blanks[15][15]       # 1=blank tile at position
    cdef bint empty_board

    # Pre-computed cross-checks (uint32 bitmasks, bit i = letter i valid)
    cdef unsigned int cross_h[15][15]   # for horizontal play (checks vertical perp)
    cdef unsigned int cross_v[15][15]   # for vertical play (checks horizontal perp)

    # Pre-computed perpendicular tile sums (for crossword scoring in C)
    cdef int perp_sum_h[15][15]         # sum of tile values above+below
    cdef int perp_sum_v[15][15]         # sum of tile values left+right
    cdef bint has_perp_h[15][15]        # has perpendicular neighbors (horiz play)
    cdef bint has_perp_v[15][15]        # has perpendicular neighbors (vert play)

    # Scoring constants
    cdef int tile_values[26]
    cdef int bonus_lm[15][15]           # letter multiplier
    cdef int bonus_wm[15][15]           # word multiplier
    cdef int bingo_bonus
    cdef int rack_size

    # GADDAG data pointer + caches
    cdef const unsigned char* gdata
    cdef object _gdata_ref              # prevent GC of bytes object
    cdef int root_children[27]          # direct-indexed root child offsets
    cdef int level2[27][27]             # level2[parent_ci][child_ci] = offset

    # Anchors
    cdef int anchor_count
    cdef int anchor_r[225]
    cdef int anchor_c[225]
    cdef int left_limit_h[15][15]
    cdef int left_limit_v[15][15]
    cdef bint has_left_h[15][15]        # tile exists to left of anchor (horiz)
    cdef bint has_left_v[15][15]        # tile exists above anchor (vert)

    # Word validation (Python sets, called only at terminal nodes)
    cdef object word_set                # Python set of valid words
    cdef object valid_2                 # set of valid 2-letter words

    # Best move tracking (reset per find_best_score_c call)
    cdef int best_score
    cdef int best_word[15]
    cdef int best_wlen
    cdef int best_row0
    cdef int best_col0
    cdef bint best_horiz


# =========================================================================
# BoardContext helpers: get_child with root cache
# =========================================================================

cdef inline int _ctx_get_child(BoardContext ctx, int offset, int char_idx):
    """Get child offset, using root cache when offset==0."""
    if offset == 0:
        return ctx.root_children[char_idx]
    return _get_child(ctx.gdata, offset, char_idx)


# =========================================================================
# Pre-compute cross-checks as bitmasks
# =========================================================================

cdef void _compute_cross_checks(BoardContext ctx, object word_set,
                                 set valid_2) except *:
    """Pre-compute cross-check bitmasks for all empty squares."""
    cdef int r, c, rr, cc, i
    cdef unsigned int mask
    cdef list above, below
    cdef str prefix, suffix, word, letter

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if ctx.grid[r][c] >= 0:
                # Non-empty: not used in cross-checks, set all-valid
                ctx.cross_h[r][c] = 0xFFFFFFFF
                ctx.cross_v[r][c] = 0xFFFFFFFF
                continue

            # --- Horizontal play: check vertical perpendicular ---
            above = []
            rr = r - 1
            while rr >= 0 and ctx.grid[rr][c] >= 0:
                above.append(chr(ctx.grid[rr][c] + 65))
                rr -= 1
            above.reverse()
            below = []
            rr = r + 1
            while rr < BOARD_SIZE and ctx.grid[rr][c] >= 0:
                below.append(chr(ctx.grid[rr][c] + 65))
                rr += 1

            if not above and not below:
                ctx.cross_h[r][c] = 0xFFFFFFFF
            else:
                prefix = ''.join(above)
                suffix = ''.join(below)
                mask = 0
                for i in range(26):
                    letter = chr(i + 65)
                    word = prefix + letter + suffix
                    if len(word) == 2:
                        if word in valid_2:
                            mask |= (1 << i)
                    else:
                        if word in word_set:
                            mask |= (1 << i)
                ctx.cross_h[r][c] = mask

            # --- Vertical play: check horizontal perpendicular ---
            above = []
            cc = c - 1
            while cc >= 0 and ctx.grid[r][cc] >= 0:
                above.append(chr(ctx.grid[r][cc] + 65))
                cc -= 1
            above.reverse()
            below = []
            cc = c + 1
            while cc < BOARD_SIZE and ctx.grid[r][cc] >= 0:
                below.append(chr(ctx.grid[r][cc] + 65))
                cc += 1

            if not above and not below:
                ctx.cross_v[r][c] = 0xFFFFFFFF
            else:
                prefix = ''.join(above)
                suffix = ''.join(below)
                mask = 0
                for i in range(26):
                    letter = chr(i + 65)
                    word = prefix + letter + suffix
                    if len(word) == 2:
                        if word in valid_2:
                            mask |= (1 << i)
                    else:
                        if word in word_set:
                            mask |= (1 << i)
                ctx.cross_v[r][c] = mask


# =========================================================================
# Pre-compute perpendicular tile sums for crossword scoring
# =========================================================================

cdef void _compute_perp_scores(BoardContext ctx):
    """Pre-compute sum of perpendicular tile values at each empty square."""
    cdef int r, c, rr, cc, s
    cdef bint has_p

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if ctx.grid[r][c] >= 0:
                ctx.perp_sum_h[r][c] = 0
                ctx.perp_sum_v[r][c] = 0
                ctx.has_perp_h[r][c] = False
                ctx.has_perp_v[r][c] = False
                continue

            # Horiz play: perpendicular is vertical (above + below)
            s = 0
            has_p = False
            rr = r - 1
            while rr >= 0 and ctx.grid[rr][c] >= 0:
                if not ctx.board_blanks[rr][c]:
                    s += ctx.tile_values[ctx.grid[rr][c]]
                has_p = True
                rr -= 1
            rr = r + 1
            while rr < BOARD_SIZE and ctx.grid[rr][c] >= 0:
                if not ctx.board_blanks[rr][c]:
                    s += ctx.tile_values[ctx.grid[rr][c]]
                has_p = True
                rr += 1
            ctx.perp_sum_h[r][c] = s
            ctx.has_perp_h[r][c] = has_p

            # Vert play: perpendicular is horizontal (left + right)
            s = 0
            has_p = False
            cc = c - 1
            while cc >= 0 and ctx.grid[r][cc] >= 0:
                if not ctx.board_blanks[r][cc]:
                    s += ctx.tile_values[ctx.grid[r][cc]]
                has_p = True
                cc -= 1
            cc = c + 1
            while cc < BOARD_SIZE and ctx.grid[r][cc] >= 0:
                if not ctx.board_blanks[r][cc]:
                    s += ctx.tile_values[ctx.grid[r][cc]]
                has_p = True
                cc += 1
            ctx.perp_sum_v[r][c] = s
            ctx.has_perp_v[r][c] = has_p


# =========================================================================
# Pre-compute anchors, left limits, has-left flags
# =========================================================================

cdef void _compute_anchors(BoardContext ctx):
    """Find anchors and pre-compute left limits and has-left flags."""
    cdef int r, c, rr, cc, limit

    # Anchors
    if ctx.empty_board:
        ctx.anchor_count = 1
        ctx.anchor_r[0] = 7
        ctx.anchor_c[0] = 7
    else:
        ctx.anchor_count = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if ctx.grid[r][c] >= 0:
                    continue
                if (r > 0 and ctx.grid[r-1][c] >= 0) or \
                   (r < 14 and ctx.grid[r+1][c] >= 0) or \
                   (c > 0 and ctx.grid[r][c-1] >= 0) or \
                   (c < 14 and ctx.grid[r][c+1] >= 0):
                    ctx.anchor_r[ctx.anchor_count] = r
                    ctx.anchor_c[ctx.anchor_count] = c
                    ctx.anchor_count += 1

    # Has-left flags and left limits for all squares
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            # Has-left for horizontal (tile to the left)
            ctx.has_left_h[r][c] = (c > 0 and ctx.grid[r][c-1] >= 0)
            # Has-left for vertical (tile above)
            ctx.has_left_v[r][c] = (r > 0 and ctx.grid[r-1][c] >= 0)

            # Left limit for horizontal play
            limit = 0
            cc = c - 1
            while cc >= 0:
                if ctx.grid[r][cc] >= 0:
                    break
                if (r > 0 and ctx.grid[r-1][cc] >= 0) or \
                   (r < 14 and ctx.grid[r+1][cc] >= 0) or \
                   (cc > 0 and ctx.grid[r][cc-1] >= 0) or \
                   (cc < 14 and ctx.grid[r][cc+1] >= 0):
                    break
                limit += 1
                cc -= 1
            ctx.left_limit_h[r][c] = limit

            # Left limit for vertical play
            limit = 0
            rr = r - 1
            while rr >= 0:
                if ctx.grid[rr][c] >= 0:
                    break
                if (rr > 0 and ctx.grid[rr-1][c] >= 0) or \
                   (rr < 14 and ctx.grid[rr+1][c] >= 0) or \
                   (c > 0 and ctx.grid[rr][c-1] >= 0) or \
                   (c < 14 and ctx.grid[rr][c+1] >= 0):
                    break
                limit += 1
                rr -= 1
            ctx.left_limit_v[r][c] = limit


# =========================================================================
# Cache GADDAG root and level-2 children
# =========================================================================

cdef void _cache_gaddag_root(BoardContext ctx):
    """Cache root children and level-2 children for fast lookup."""
    cdef const unsigned char* data = ctx.gdata
    cdef int i, j, count, pos, ci, child, parent_off, child_ci, child_off

    # Initialize to -1
    for i in range(27):
        ctx.root_children[i] = -1
    for i in range(27):
        for j in range(27):
            ctx.level2[i][j] = -1

    # Root children
    count = data[0] & 0x1F
    pos = 1
    for i in range(count):
        ci = data[pos]
        child = (data[pos+1] | (data[pos+2] << 8) |
                 (data[pos+3] << 16) | (data[pos+4] << 24))
        if ci < 27:
            ctx.root_children[ci] = child
        pos += 5

    # Level-2 children (one level below root)
    for ci in range(27):
        parent_off = ctx.root_children[ci]
        if parent_off < 0:
            continue
        count = data[parent_off] & 0x1F
        pos = parent_off + 1
        for j in range(count):
            child_ci = data[pos]
            child_off = (data[pos+1] | (data[pos+2] << 8) |
                         (data[pos+3] << 16) | (data[pos+4] << 24))
            if child_ci < 27:
                ctx.level2[ci][child_ci] = child_off
            pos += 5


# =========================================================================
# try_record_best — validate word, check connectivity, score, compare
# =========================================================================

cdef void _ctx_try_record_best(BoardContext ctx,
                                int* wordbuf, int wlen,
                                int sr0, int sc0, bint horiz,
                                int* blankbuf, int nblank) except *:
    """Validate, score, and compare to current best. Only creates Python
    string if the word reaches terminal-node validation."""
    cdef int i, r0, c0
    cdef bint connects, uses_new, found_center, sq_is_new
    cdef int main_score, word_mult, new_count, cw_total, total
    cdef int lv, lm, wm, plv
    cdef bint blank_at[15]

    # Bounds check
    if sr0 < 0 or sc0 < 0:
        return
    if horiz and sc0 + wlen > BOARD_SIZE:
        return
    if not horiz and sr0 + wlen > BOARD_SIZE:
        return

    # Build word string for validation (only Python step in hot path)
    cdef list chars = []
    for i in range(wlen):
        chars.append(chr(wordbuf[i] + 65))
    cdef str word = ''.join(chars)

    # Validate word
    if wlen == 2:
        if word not in ctx.valid_2:
            return
    else:
        if word not in ctx.word_set:
            return

    # Connectivity check (all C)
    connects = False
    uses_new = False
    for i in range(wlen):
        if horiz:
            r0 = sr0
            c0 = sc0 + i
        else:
            r0 = sr0 + i
            c0 = sc0
        if ctx.grid[r0][c0] >= 0:
            connects = True
            if uses_new:
                break
        else:
            uses_new = True
            if connects:
                break
            if (r0 > 0 and ctx.grid[r0-1][c0] >= 0) or \
               (r0 < 14 and ctx.grid[r0+1][c0] >= 0) or \
               (c0 > 0 and ctx.grid[r0][c0-1] >= 0) or \
               (c0 < 14 and ctx.grid[r0][c0+1] >= 0):
                connects = True
                break

    if not uses_new:
        return
    if ctx.empty_board:
        found_center = False
        for i in range(wlen):
            if horiz:
                r0 = sr0
                c0 = sc0 + i
            else:
                r0 = sr0 + i
                c0 = sc0
            if r0 == 7 and c0 == 7:
                found_center = True
                break
        if not found_center:
            return
    elif not connects:
        return

    # Build blank position flags
    for i in range(MAX_WORD):
        blank_at[i] = False
    for i in range(nblank):
        if 0 <= blankbuf[i] < MAX_WORD:
            blank_at[blankbuf[i]] = True

    # Score the move (all C, using pre-computed perp sums)
    main_score = 0
    word_mult = 1
    new_count = 0
    cw_total = 0

    for i in range(wlen):
        if horiz:
            r0 = sr0
            c0 = sc0 + i
        else:
            r0 = sr0 + i
            c0 = sc0

        sq_is_new = ctx.grid[r0][c0] < 0

        if blank_at[i]:
            lv = 0
        elif not sq_is_new and ctx.board_blanks[r0][c0]:
            lv = 0
        else:
            lv = ctx.tile_values[wordbuf[i]]

        if sq_is_new:
            new_count += 1
            lm = ctx.bonus_lm[r0][c0]
            wm = ctx.bonus_wm[r0][c0]
            lv = lv * lm
            word_mult = word_mult * wm

            # Crossword scoring using pre-computed perp sums
            if horiz:
                if ctx.has_perp_h[r0][c0]:
                    plv = 0 if blank_at[i] else ctx.tile_values[wordbuf[i]]
                    cw_total += (ctx.perp_sum_h[r0][c0] + plv * lm) * wm
            else:
                if ctx.has_perp_v[r0][c0]:
                    plv = 0 if blank_at[i] else ctx.tile_values[wordbuf[i]]
                    cw_total += (ctx.perp_sum_v[r0][c0] + plv * lm) * wm

        main_score += lv

    total = main_score * word_mult + cw_total
    if new_count >= ctx.rack_size:
        total += ctx.bingo_bonus

    if total > ctx.best_score:
        ctx.best_score = total
        for i in range(wlen):
            ctx.best_word[i] = wordbuf[i]
        ctx.best_wlen = wlen
        ctx.best_row0 = sr0
        ctx.best_col0 = sc0
        ctx.best_horiz = horiz


# =========================================================================
# extend_right for BoardContext — C arrays + pre-computed cross-checks
# =========================================================================

cdef void _ctx_extend_right(BoardContext ctx,
                             int row0, int col0, bint horiz, int offset,
                             int* wordbuf, int wlen, int sr0, int sc0,
                             int blanks_rem, int* blankbuf, int nblank,
                             int* rack, int rack_nletters,
                             int* rack_indices) except *:
    cdef const unsigned char* data = ctx.gdata
    cdef int existing, idx, child
    cdef unsigned int cc_mask
    cdef int ci
    cdef int n_children
    cdef int c_idxs[32]
    cdef int c_offs[32]
    cdef int j

    if row0 < 0 or row0 >= BOARD_SIZE or col0 < 0 or col0 >= BOARD_SIZE:
        if _is_terminal(data, offset) and wlen >= 2:
            _ctx_try_record_best(ctx, wordbuf, wlen, sr0, sc0,
                                  horiz, blankbuf, nblank)
        return

    existing = ctx.grid[row0][col0]

    if existing >= 0:
        # Square has a tile -- must follow it
        child = _ctx_get_child(ctx, offset, existing)
        if child >= 0:
            wordbuf[wlen] = existing
            if horiz:
                _ctx_extend_right(ctx, row0, col0 + 1, True, child,
                                   wordbuf, wlen + 1, sr0, sc0,
                                   blanks_rem, blankbuf, nblank,
                                   rack, rack_nletters, rack_indices)
            else:
                _ctx_extend_right(ctx, row0 + 1, col0, False, child,
                                   wordbuf, wlen + 1, sr0, sc0,
                                   blanks_rem, blankbuf, nblank,
                                   rack, rack_nletters, rack_indices)
    else:
        # Empty square -- use pre-computed cross-check bitmask
        if horiz:
            cc_mask = ctx.cross_h[row0][col0]
        else:
            cc_mask = ctx.cross_v[row0][col0]

        if _is_terminal(data, offset) and wlen >= 2:
            _ctx_try_record_best(ctx, wordbuf, wlen, sr0, sc0,
                                  horiz, blankbuf, nblank)

        # Try each rack letter
        for j in range(rack_nletters):
            idx = rack_indices[j]
            if rack[idx] <= 0:
                continue
            if not (cc_mask & (1 << idx)):
                continue
            child = _ctx_get_child(ctx, offset, idx)
            if child < 0:
                continue

            rack[idx] -= 1
            wordbuf[wlen] = idx
            if horiz:
                _ctx_extend_right(ctx, row0, col0 + 1, True, child,
                                   wordbuf, wlen + 1, sr0, sc0,
                                   blanks_rem, blankbuf, nblank,
                                   rack, rack_nletters, rack_indices)
            else:
                _ctx_extend_right(ctx, row0 + 1, col0, False, child,
                                   wordbuf, wlen + 1, sr0, sc0,
                                   blanks_rem, blankbuf, nblank,
                                   rack, rack_nletters, rack_indices)
            rack[idx] += 1

        # Try blank tiles
        if blanks_rem > 0:
            n_children = _get_children(data, offset, c_idxs, c_offs)
            for j in range(n_children):
                ci = c_idxs[j]
                if ci == DELIM_IDX:
                    continue
                if not (cc_mask & (1 << ci)):
                    continue
                wordbuf[wlen] = ci
                blankbuf[nblank] = wlen
                if horiz:
                    _ctx_extend_right(ctx, row0, col0 + 1, True, c_offs[j],
                                       wordbuf, wlen + 1, sr0, sc0,
                                       blanks_rem - 1, blankbuf, nblank + 1,
                                       rack, rack_nletters, rack_indices)
                else:
                    _ctx_extend_right(ctx, row0 + 1, col0, False, c_offs[j],
                                       wordbuf, wlen + 1, sr0, sc0,
                                       blanks_rem - 1, blankbuf, nblank + 1,
                                       rack, rack_nletters, rack_indices)


# =========================================================================
# gen_left_part for BoardContext
# =========================================================================

cdef void _ctx_gen_left_part(BoardContext ctx,
                              int anchor_r0, int anchor_c0, bint horiz,
                              int offset, int* prefbuf, int plen, int limit,
                              int blanks_rem, int* blankbuf, int nblank,
                              int* rack, int rack_nletters,
                              int* rack_indices) except *:
    cdef const unsigned char* data = ctx.gdata
    cdef int delim_child, sr0, sc0, pr0, pc0
    cdef int idx, child, dc
    cdef int ci
    cdef int n_children
    cdef int c_idxs[32]
    cdef int c_offs[32]
    cdef int j, i
    cdef unsigned int cc_mask
    cdef int wordbuf[MAX_WORD]
    cdef int blankbuf2[MAX_BLANKS]

    # Try crossing delimiter to extend right
    delim_child = _ctx_get_child(ctx, offset, DELIM_IDX)
    if delim_child >= 0:
        if horiz:
            sr0 = anchor_r0
            sc0 = anchor_c0 - plen
        else:
            sr0 = anchor_r0 - plen
            sc0 = anchor_c0
        # Reverse prefix into wordbuf
        for i in range(plen):
            wordbuf[i] = prefbuf[plen - 1 - i]
        # Adjust blank positions (stored relative to reversed prefix)
        for i in range(nblank):
            blankbuf2[i] = plen - 1 - blankbuf[i]
        _ctx_extend_right(ctx, anchor_r0, anchor_c0, horiz, delim_child,
                           wordbuf, plen, sr0, sc0,
                           blanks_rem, blankbuf2, nblank,
                           rack, rack_nletters, rack_indices)

    # Try extending left if we have room
    if limit > 0:
        if horiz:
            pr0 = anchor_r0
            pc0 = anchor_c0 - plen - 1
        else:
            pr0 = anchor_r0 - plen - 1
            pc0 = anchor_c0
        if pr0 < 0 or pc0 < 0:
            return

        if horiz:
            cc_mask = ctx.cross_h[pr0][pc0]
        else:
            cc_mask = ctx.cross_v[pr0][pc0]

        for j in range(rack_nletters):
            idx = rack_indices[j]
            if rack[idx] <= 0:
                continue
            if not (cc_mask & (1 << idx)):
                continue
            child = _ctx_get_child(ctx, offset, idx)
            if child < 0:
                continue
            rack[idx] -= 1
            prefbuf[plen] = idx
            _ctx_gen_left_part(ctx, anchor_r0, anchor_c0, horiz, child,
                                prefbuf, plen + 1, limit - 1,
                                blanks_rem, blankbuf, nblank,
                                rack, rack_nletters, rack_indices)
            rack[idx] += 1

        if blanks_rem > 0:
            n_children = _get_children(data, offset, c_idxs, c_offs)
            for j in range(n_children):
                ci = c_idxs[j]
                if ci == DELIM_IDX:
                    continue
                if not (cc_mask & (1 << ci)):
                    continue
                prefbuf[plen] = ci
                blankbuf[nblank] = plen
                _ctx_gen_left_part(ctx, anchor_r0, anchor_c0, horiz, c_offs[j],
                                    prefbuf, plen + 1, limit - 1,
                                    blanks_rem - 1, blankbuf, nblank + 1,
                                    rack, rack_nletters, rack_indices)

    # At root with limit=0: single letter at anchor then cross delimiter
    if plen == 0 and limit == 0:
        if horiz:
            cc_mask = ctx.cross_h[anchor_r0][anchor_c0]
        else:
            cc_mask = ctx.cross_v[anchor_r0][anchor_c0]

        for j in range(rack_nletters):
            idx = rack_indices[j]
            if rack[idx] <= 0:
                continue
            if not (cc_mask & (1 << idx)):
                continue
            # Use root cache
            child = ctx.root_children[idx]
            if child < 0:
                continue
            # Use level2 cache for delimiter
            dc = ctx.level2[idx][DELIM_IDX]
            if dc < 0:
                continue

            rack[idx] -= 1
            wordbuf[0] = idx
            if horiz:
                _ctx_extend_right(ctx, anchor_r0, anchor_c0 + 1, True, dc,
                                   wordbuf, 1, anchor_r0, anchor_c0,
                                   blanks_rem, blankbuf, 0,
                                   rack, rack_nletters, rack_indices)
            else:
                _ctx_extend_right(ctx, anchor_r0 + 1, anchor_c0, False, dc,
                                   wordbuf, 1, anchor_r0, anchor_c0,
                                   blanks_rem, blankbuf, 0,
                                   rack, rack_nletters, rack_indices)
            rack[idx] += 1

        if blanks_rem > 0:
            # Try all 26 letters as blank at anchor
            for ci in range(26):
                if not (cc_mask & (1 << ci)):
                    continue
                child = ctx.root_children[ci]
                if child < 0:
                    continue
                dc = ctx.level2[ci][DELIM_IDX]
                if dc < 0:
                    continue
                wordbuf[0] = ci
                blankbuf[0] = 0
                if horiz:
                    _ctx_extend_right(ctx, anchor_r0, anchor_c0 + 1, True, dc,
                                       wordbuf, 1, anchor_r0, anchor_c0,
                                       blanks_rem - 1, blankbuf, 1,
                                       rack, rack_nletters, rack_indices)
                else:
                    _ctx_extend_right(ctx, anchor_r0 + 1, anchor_c0, False, dc,
                                       wordbuf, 1, anchor_r0, anchor_c0,
                                       blanks_rem - 1, blankbuf, 1,
                                       rack, rack_nletters, rack_indices)


# =========================================================================
# extend_from_existing for BoardContext
# =========================================================================

cdef void _ctx_extend_from_existing(BoardContext ctx,
                                     int anchor_r0, int anchor_c0,
                                     bint horiz, int blanks_rem,
                                     int* rack, int rack_nletters,
                                     int* rack_indices) except *:
    cdef int prefix[MAX_WORD]
    cdef int plen = 0
    cdef int c, r, sr0, sc0, offset, idx, child, dc
    cdef int wordbuf[MAX_WORD]
    cdef int blankbuf[MAX_BLANKS]
    cdef int i, val

    # Collect existing tiles to the left/above (closest to anchor first)
    if horiz:
        c = anchor_c0 - 1
        while c >= 0 and ctx.grid[anchor_r0][c] >= 0:
            prefix[plen] = ctx.grid[anchor_r0][c]
            plen += 1
            c -= 1
        sc0 = c + 1
        sr0 = anchor_r0
    else:
        r = anchor_r0 - 1
        while r >= 0 and ctx.grid[r][anchor_c0] >= 0:
            prefix[plen] = ctx.grid[r][anchor_c0]
            plen += 1
            r -= 1
        sr0 = r + 1
        sc0 = anchor_c0

    # Navigate GADDAG with reversed prefix (closest first = GADDAG order)
    offset = 0
    for i in range(plen):
        idx = prefix[i]
        if idx < 0 or idx >= 26:
            return
        child = _ctx_get_child(ctx, offset, idx)
        if child < 0:
            return
        offset = child

    # Cross delimiter
    dc = _get_child(ctx.gdata, offset, DELIM_IDX)
    if dc < 0:
        return

    # Build forward-order prefix in wordbuf
    for i in range(plen):
        wordbuf[i] = prefix[plen - 1 - i]

    _ctx_extend_right(ctx, anchor_r0, anchor_c0, horiz, dc,
                       wordbuf, plen, sr0, sc0,
                       blanks_rem, blankbuf, 0,
                       rack, rack_nletters, rack_indices)


# =========================================================================
# Public API: prepare_board_context
# =========================================================================

def prepare_board_context(list grid, bytes gdata_bytes, set board_blank_set,
                          object word_set, set valid_2,
                          list tile_values, list bonus_data,
                          int bingo_bonus, int rack_size):
    """Create a pre-computed BoardContext for fast MC simulation.

    Call once per board state, then call find_best_score_c(ctx, rack)
    for each of K simulations.

    Args:
        grid:            board._grid (0-indexed 15x15 list of lists)
        gdata_bytes:     bytes(gaddag._data)
        board_blank_set: set of (r0, c0) for blanks on board (0-indexed)
        word_set:        set of valid words (dictionary._words)
        valid_2:         set of valid 2-letter words (VALID_TWO_LETTER)
        tile_values:     list of 26 ints (tile values by letter index)
        bonus_data:      list of 15 lists of 15 (lm, wm) tuples
        bingo_bonus:     bonus for using all rack tiles (40)
        rack_size:       rack size (7)

    Returns:
        BoardContext object
    """
    cdef BoardContext ctx = BoardContext.__new__(BoardContext)
    cdef const unsigned char* gdata = <const unsigned char*>PyBytes_AsString(gdata_bytes)
    ctx.gdata = gdata
    ctx._gdata_ref = gdata_bytes  # prevent GC

    ctx.word_set = word_set
    ctx.valid_2 = valid_2
    ctx.bingo_bonus = bingo_bonus
    ctx.rack_size = rack_size

    # Fill tile values
    cdef int i, j
    for i in range(26):
        ctx.tile_values[i] = <int>tile_values[i]

    # Fill bonus grid
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            lm_wm = (<list>(<list>bonus_data)[i])[j]
            ctx.bonus_lm[i][j] = <int>(lm_wm[0])
            ctx.bonus_wm[i][j] = <int>(lm_wm[1])

    # Fill grid and board_blanks
    ctx.empty_board = True
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            val = (<list>(<list>grid)[i])[j]
            if val is None:
                ctx.grid[i][j] = -1
            else:
                ctx.grid[i][j] = <int>(ord(<str>val)) - 65
                ctx.empty_board = False
            ctx.board_blanks[i][j] = 1 if (i, j) in board_blank_set else 0

    # Pre-compute everything
    _compute_cross_checks(ctx, word_set, valid_2)
    _compute_perp_scores(ctx)
    _compute_anchors(ctx)
    _cache_gaddag_root(ctx)

    return ctx


# =========================================================================
# Public API: find_best_score_c
# =========================================================================

def find_best_score_c(BoardContext ctx, str rack_str):
    """Find the best-scoring move using pre-computed BoardContext.

    Call K times with different rack strings for MC simulation.
    Returns (score, word, row1, col1, dir_str) or (0, None, 0, 0, None).
    """
    cdef int rack[27]
    cdef int rack_indices[26]
    cdef int rack_nletters = 0
    cdef int num_blanks = 0
    cdef int i, ci
    cdef str ch
    cdef int rack_letter_count, max_limit
    cdef int ar, ac, ll
    cdef int prefbuf[MAX_WORD]
    cdef int blankbuf[MAX_BLANKS]

    # Reset best
    ctx.best_score = 0
    ctx.best_wlen = 0

    # Parse rack into C array
    for i in range(27):
        rack[i] = 0
    for ch in rack_str:
        if ch == '?':
            num_blanks += 1
        else:
            ci = <int>(ord(ch)) - 65
            if 0 <= ci < 26:
                rack[ci] += 1

    # Build rack_indices (which letter indices have count > 0)
    for i in range(26):
        if rack[i] > 0:
            rack_indices[rack_nletters] = i
            rack_nletters += 1

    # Count rack letters for left_limit clamping
    rack_letter_count = 0
    for i in range(26):
        rack_letter_count += rack[i]
    max_limit = rack_letter_count + num_blanks - 1

    # Search all anchors x both directions
    for i in range(ctx.anchor_count):
        ar = ctx.anchor_r[i]
        ac = ctx.anchor_c[i]

        # Horizontal
        if ctx.has_left_h[ar][ac]:
            _ctx_extend_from_existing(ctx, ar, ac, True, num_blanks,
                                       rack, rack_nletters, rack_indices)
        else:
            ll = ctx.left_limit_h[ar][ac]
            if ll > max_limit:
                ll = max_limit
            if ll < 0:
                ll = 0
            _ctx_gen_left_part(ctx, ar, ac, True, 0,
                                prefbuf, 0, ll,
                                num_blanks, blankbuf, 0,
                                rack, rack_nletters, rack_indices)

        # Vertical
        if ctx.has_left_v[ar][ac]:
            _ctx_extend_from_existing(ctx, ar, ac, False, num_blanks,
                                       rack, rack_nletters, rack_indices)
        else:
            ll = ctx.left_limit_v[ar][ac]
            if ll > max_limit:
                ll = max_limit
            if ll < 0:
                ll = 0
            _ctx_gen_left_part(ctx, ar, ac, False, 0,
                                prefbuf, 0, ll,
                                num_blanks, blankbuf, 0,
                                rack, rack_nletters, rack_indices)

    # Return result
    if ctx.best_score > 0:
        chars = []
        for i in range(ctx.best_wlen):
            chars.append(chr(ctx.best_word[i] + 65))
        word = ''.join(chars)
        return (ctx.best_score, word, ctx.best_row0 + 1, ctx.best_col0 + 1,
                'H' if ctx.best_horiz else 'V')
    return (0, None, 0, 0, None)
