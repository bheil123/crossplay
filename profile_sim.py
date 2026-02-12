"""Profile a single find_best_score_opt call to find the hotspot."""
import sys, os, time, random, cProfile, pstats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from crossplay_v9.board import Board
from crossplay_v9.gaddag import get_gaddag
from crossplay_v9.move_finder_opt import find_best_score_opt
from crossplay_v9.dictionary import get_dictionary

BOARD_MOVES = [
    ("DIZEN", 4, 8, False), ("BUSKER", 9, 6, True), ("TOEA", 8, 3, True),
    ("GAIT", 2, 7, False), ("YA", 10, 5, True), ("NAH", 5, 9, False),
    ("DUTY", 2, 10, False), ("CHICS", 5, 12, False), ("AWE", 10, 9, True),
    ("OBE", 11, 8, True), ("QI", 12, 7, True), ("PIGEON", 10, 4, False),
    ("LODGE", 12, 1, True), ("EX", 13, 4, True), ("JET", 7, 2, True),
    ("FEH", 7, 7, True), ("LOUVERS", 13, 8, True), ("PEAL", 12, 12, True),
    ("PERM", 12, 12, False), ("GEE", 6, 5, False),
    ("HA", 14, 14, True), ("FIDO", 14, 1, True),
]
BOARD_BLANKS = [(13, 10, 'U')]

print("Loading...", flush=True)
gaddag = get_gaddag()
dictionary = get_dictionary()

board = Board()
for word, row, col, horiz in BOARD_MOVES:
    board.place_word(word, row, col, horiz)
# Place a candidate move (IF at R13C1 H)
placed = board.place_move('IF', 13, 1, True)

grid = board._grid
gdata = gaddag._data
bb_set = {(r-1, c-1) for r, c, _ in BOARD_BLANKS}
cross_cache = {}

# Count occupied squares
occupied = sum(1 for r in range(15) for c in range(15) if grid[r][c])
print(f"Board has {occupied} occupied squares", flush=True)

# Count anchors (empty squares adjacent to occupied)
anchors = 0
for r in range(15):
    for c in range(15):
        if grid[r][c] is None:
            has_adj = False
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < 15 and 0 <= nc < 15 and grid[nr][nc] is not None:
                    has_adj = True
                    break
            if has_adj:
                anchors += 1
print(f"Anchor squares: {anchors}", flush=True)

random.seed(42)
# Build unseen pool
from crossplay_v9.config import TILE_DISTRIBUTION
board_tiles = Counter()
for r in range(15):
    for c in range(15):
        ch = grid[r][c]
        if ch:
            board_tiles[ch] += 1
rack = "SRERIOE"
rack_tiles = Counter(rack)
pool_tiles = []
blank_letters = Counter(letter for _, _, letter in BOARD_BLANKS)
for tile, count in TILE_DISTRIBUTION.items():
    if tile == '?':
        remaining = 3 - len(BOARD_BLANKS) - rack.count('?')
    else:
        board_count = board_tiles.get(tile, 0) - blank_letters.get(tile, 0)
        remaining = count - board_count - rack_tiles.get(tile, 0)
    for _ in range(max(0, remaining)):
        pool_tiles.append(tile)

opp_rack = ''.join(random.sample(pool_tiles, min(7, len(pool_tiles))))
print(f"Opp rack: {opp_rack}", flush=True)

# Warm up cross_cache
print("\nWarm-up sim (builds cross_cache)...", flush=True)
t0 = time.perf_counter()
find_best_score_opt(grid, gdata, opp_rack, bb_set,
                    cross_cache=cross_cache, dictionary=dictionary)
t1 = time.perf_counter()
print(f"  Warm-up: {(t1-t0)*1000:.1f}ms, cache={len(cross_cache)} entries", flush=True)

# Profile 3 sims with warm cache
print("\nProfiling 3 sims with warm cache...", flush=True)
profiler = cProfile.Profile()
profiler.enable()
for _ in range(3):
    opp_rack = ''.join(random.sample(pool_tiles, min(7, len(pool_tiles))))
    find_best_score_opt(grid, gdata, opp_rack, bb_set,
                        cross_cache=cross_cache, dictionary=dictionary)
profiler.disable()

print("\n--- TOP 30 BY CUMULATIVE TIME ---", flush=True)
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(30)

print("\n--- TOP 30 BY TOTAL (SELF) TIME ---", flush=True)
stats.sort_stats('tottime')
stats.print_stats(30)
