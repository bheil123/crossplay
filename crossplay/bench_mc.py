"""MC Worker Performance Benchmark.

Profiles the actual time spent in each phase of MC evaluation:
1. Worker initialization (GADDAG load)
2. Board reconstruction + move placement (per-candidate setup)
3. Per-simulation cost (find_best_score_opt)
4. Cross-cache hit rate
5. Worker scaling (1, 4, 6, 8, 10, 12 workers)
"""
import sys, os, time, random, statistics

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support

def P(msg):
    """Print with flush."""
    print(msg, flush=True)

# Board from Game 5 turn 23 (after FIDO played)
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
CANDIDATE_MOVE = {'word': 'IF', 'row': 13, 'col': 1, 'direction': 'H', 'score': 13}
YOUR_RACK = "SRERIOE"
BAG_SIZE = 10


def build_unseen_pool():
    from .config import TILE_DISTRIBUTION
    from .board import Board
    board = Board()
    for word, row, col, horiz in BOARD_MOVES:
        board.place_word(word, row, col, horiz)
    board_tiles = Counter()
    for r in range(15):
        for c in range(15):
            ch = board._grid[r][c]
            if ch:
                board_tiles[ch] += 1
    blank_letters = Counter(letter for _, _, letter in BOARD_BLANKS)
    rack_tiles = Counter(YOUR_RACK.upper())
    pool = []
    for tile, count in TILE_DISTRIBUTION.items():
        if tile == '?':
            remaining = 3 - len(BOARD_BLANKS) - YOUR_RACK.count('?')
        else:
            board_count = board_tiles.get(tile, 0) - blank_letters.get(tile, 0)
            remaining = count - board_count - rack_tiles.get(tile, 0)
        for _ in range(max(0, remaining)):
            pool.append(tile)
    return pool


def _worker_task(args):
    """Simulate a real MC worker: init + board setup + K sims with timing."""
    board_moves, move, unseen_pool, board_blanks, k_sims, seed = args
    import sys, os, time, random

    t_start = time.perf_counter()

    from .gaddag import get_gaddag
    from .move_finder_opt import find_best_score_opt
    from .dictionary import get_dictionary
    gaddag = get_gaddag()
    dictionary = get_dictionary()
    t_init = time.perf_counter()

    from .board import Board
    board = Board()
    for word, row, col, horiz in board_moves:
        board.place_word(word, row, col, horiz)
    horizontal = move['direction'] == 'H'
    placed = board.place_move(move['word'], move['row'], move['col'], horizontal)
    t_board = time.perf_counter()

    grid = board._grid
    gdata = gaddag._data
    bb_set = {(r-1, c-1) for r, c, _ in (board_blanks or [])}
    cross_cache = {}
    random.seed(seed)

    sim_times = []
    for i in range(k_sims):
        opp_rack = ''.join(random.sample(unseen_pool, min(7, len(unseen_pool))))
        t0 = time.perf_counter()
        find_best_score_opt(grid, gdata, opp_rack, bb_set,
                            cross_cache=cross_cache, dictionary=dictionary)
        t1 = time.perf_counter()
        sim_times.append(t1 - t0)

    board.undo_move(placed)
    t_done = time.perf_counter()

    return {
        'total': t_done - t_start,
        'init': t_init - t_start,
        'board': t_board - t_init,
        'sims_total': t_done - t_board,
        'sim_times': sim_times,
        'k': k_sims,
        'cache_size': len(cross_cache),
        'pid': os.getpid(),
    }


def main():
    P("=" * 70)
    P("MC WORKER PERFORMANCE BENCHMARK")
    P("=" * 70)
    P(f"CPU cores: {os.cpu_count()}")
    P(f"Board: {len(BOARD_MOVES)} moves, bag {BAG_SIZE}")

    pool = build_unseen_pool()
    P(f"Unseen pool: {len(pool)} tiles")

    # === Test 1: Single-process baseline ===
    P("\n--- 1. SINGLE SIM COST (main process, 20 sims) ---")
    from .gaddag import get_gaddag
    from .move_finder_opt import find_best_score_opt
    from .dictionary import get_dictionary
    from .board import Board

    gaddag = get_gaddag()
    dictionary = get_dictionary()

    board = Board()
    for word, row, col, horiz in BOARD_MOVES:
        board.place_word(word, row, col, horiz)
    placed = board.place_move(CANDIDATE_MOVE['word'], CANDIDATE_MOVE['row'],
                              CANDIDATE_MOVE['col'],
                              CANDIDATE_MOVE['direction'] == 'H')
    grid = board._grid
    gdata = gaddag._data
    bb_set = {(r-1, c-1) for r, c, _ in BOARD_BLANKS}

    # Cold cache
    random.seed(42)
    cross_cache = {}
    opp_rack = ''.join(random.sample(pool, min(7, len(pool))))
    t0 = time.perf_counter()
    find_best_score_opt(grid, gdata, opp_rack, bb_set,
                        cross_cache=cross_cache, dictionary=dictionary)
    t_cold = time.perf_counter() - t0
    P(f"  Sim 1 (cold cross_cache): {t_cold*1000:.1f}ms  cache_entries={len(cross_cache)}")

    # Warm cache
    warm_times = []
    for _ in range(19):
        opp_rack = ''.join(random.sample(pool, min(7, len(pool))))
        t0 = time.perf_counter()
        find_best_score_opt(grid, gdata, opp_rack, bb_set,
                            cross_cache=cross_cache, dictionary=dictionary)
        warm_times.append(time.perf_counter() - t0)
    P(f"  Sims 2-20 (warm cache):   avg={statistics.mean(warm_times)*1000:.1f}ms "
      f"med={statistics.median(warm_times)*1000:.1f}ms "
      f"p95={sorted(warm_times)[int(len(warm_times)*0.95)]*1000:.1f}ms")
    P(f"  Throughput (1 core):      {1/statistics.mean(warm_times):.0f} sims/s")

    board.undo_move(placed)

    # === Test 2: Worker startup cost ===
    P("\n--- 2. WORKER STARTUP COST (1 worker, 1 sim) ---")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=1, initializer=None) as exe:
        fut = exe.submit(_worker_task,
                         (BOARD_MOVES, CANDIDATE_MOVE, pool, BOARD_BLANKS, 1, 42))
        r = fut.result()
    t_total = time.perf_counter() - t0
    P(f"  Wall time (spawn+init+1sim): {t_total*1000:.0f}ms")
    P(f"    Worker init (GADDAG+dict): {r['init']*1000:.0f}ms")
    P(f"    Board reconstruct:         {r['board']*1000:.1f}ms")
    P(f"    1 sim:                     {r['sim_times'][0]*1000:.1f}ms")
    P(f"    Overhead (pickle/IPC):     {(t_total - r['total'])*1000:.0f}ms")

    # === Test 3: Per-candidate cost at K=150 (matches real MC config) ===
    P("\n--- 3. PER-CANDIDATE COST (1 worker, K=150) ---")
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=1, initializer=None) as exe:
        fut = exe.submit(_worker_task,
                         (BOARD_MOVES, CANDIDATE_MOVE, pool, BOARD_BLANKS, 150, 42))
        r = fut.result()
    t_total = time.perf_counter() - t0
    st = r['sim_times']
    P(f"  Wall time (1 cand, K=150):   {t_total:.2f}s")
    P(f"    Init:    {r['init']*1000:.0f}ms")
    P(f"    Board:   {r['board']*1000:.1f}ms")
    P(f"    150 sims: {r['sims_total']:.2f}s ({r['sims_total']/150*1000:.1f}ms/sim)")
    P(f"    Sim breakdown: cold={st[0]*1000:.1f}ms warm_avg={statistics.mean(st[1:])*1000:.1f}ms")
    P(f"    Cross-cache: {r['cache_size']} entries")

    # === Test 4: Scaling across worker counts ===
    P("\n--- 4. SCALING: 15 candidates x 150 sims (matches real MC) ---")
    n_candidates = 15
    k_sims = 150
    total_work = n_candidates * k_sims

    scaling = {}
    for n_workers in [1, 2, 4, 6, 8, 10, 12]:
        tasks = [(BOARD_MOVES, CANDIDATE_MOVE, pool, BOARD_BLANKS, k_sims, 42 + i)
                 for i in range(n_candidates)]

        t0 = time.perf_counter()
        if n_workers == 1:
            # Sequential — reuse same process (no pool overhead)
            worker_results = [_worker_task(t) for t in tasks]
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(_worker_task, t) for t in tasks]
                worker_results = [f.result() for f in futures]
        wall = time.perf_counter() - t0

        # Analyze
        avg_init = statistics.mean(w['init'] for w in worker_results)
        avg_sim = statistics.mean(statistics.mean(w['sim_times']) for w in worker_results)
        throughput = total_work / wall
        pids = set(w['pid'] for w in worker_results)

        scaling[n_workers] = wall
        P(f"  {n_workers:2d}w: {wall:6.1f}s | {throughput:5.0f} sims/s | "
          f"sim={avg_sim*1000:.1f}ms/ea | init={avg_init*1000:.0f}ms | "
          f"pids={len(pids)}")

    # Summary
    P("\n--- SCALING SUMMARY ---")
    base = scaling[1]
    P(f"  {'Workers':>7} {'Wall':>7} {'Speedup':>8} {'Sims/s':>7} {'Efficiency':>10}")
    for nw in sorted(scaling.keys()):
        w = scaling[nw]
        sp = base / w
        eff = sp / nw * 100
        P(f"  {nw:>7} {w:>7.1f}s {sp:>7.2f}x {int(n_candidates*k_sims/w):>7} {eff:>9.0f}%")

    P("\n" + "=" * 70)


if __name__ == '__main__':
    freeze_support()
    main()
