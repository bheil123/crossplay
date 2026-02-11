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
