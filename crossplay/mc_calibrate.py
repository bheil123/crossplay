"""
CROSSPLAY V15.0 - MC Calibration Module

Auto-measures MC throughput on the current hardware and computes optimal
N×K parameters for each game phase. Replaces hardcoded N×K tables.

Usage:
    from .mc_calibrate import get_mc_params
    n, k = get_mc_params(bag_size, n_candidates, mc_budget_secs)
"""

import time
import os
import json
import platform
import multiprocessing

# Calibration cache file (persists across sessions)
_CACHE_PATH = os.path.join(os.path.dirname(__file__), '.mc_calibration.json')

# Calibration result (module-level singleton)
_calibration = None
_env_reported = False

# --- Constants ---
MIN_K = 100           # Below this, MC variance too high to be useful
MAX_K = 2000          # Diminishing returns above this (Maven/Quackle rarely exceed ~1000)
MIN_N_PYTHON = 15     # With Python MC path (~80-150 sims/sec), 15 candidates
                      # gives K~200-400 (better per-candidate confidence) while
                      # still covering the competitive equity window
MIN_N_CYTHON = 40     # With Cython fast path + early stopping, N=40 costs
                      # only 3-7s more than N=25 in most cases. Better equity
                      # spread coverage catches more non-obvious strong moves.
EQUITY_SPREAD = 15.0  # Include candidates within this equity of best
CALIBRATION_SECS = 3  # Time budget for benchmark


def _get_min_n():
    """Return MIN_N based on detected engine."""
    return MIN_N_CYTHON if _detect_engine() == 'cython' else MIN_N_PYTHON


def _detect_engine():
    """Detect which MC engine is available: 'cython', 'c', or 'python'."""
    try:
        from .move_finder_c import is_mc_fast_available
        if is_mc_fast_available():
            return 'cython'
    except ImportError:
        pass
    try:
        from .move_finder_c import is_available
        if is_available():
            return 'c'
    except ImportError:
        pass
    return 'python'


def _run_benchmark():
    """Run a quick MC benchmark to measure sims/sec on this hardware.

    Uses a reference board position (sparse + dense) to get representative
    throughput for different game phases.

    Auto-detects the fastest available path:
    - Cython fast path (BoardContext + find_best_score_c): all-C traversal+scoring
    - Python find_best_score_opt: pure Python fallback
    """
    from .board import Board
    from .gaddag import get_gaddag
    from .config import VALID_TWO_LETTER, BINGO_BONUS, RACK_SIZE
    import random

    gaddag = get_gaddag()

    # Detect which path to benchmark
    engine = _detect_engine()
    use_cython = (engine == 'cython')
    use_c = (engine == 'c')

    gdata_bytes = bytes(gaddag._data)

    if use_cython:
        import gaddag_accel as _accel
        from .dictionary import get_dictionary
        _dict = get_dictionary()
        _word_set = _dict._words
        from .mc_eval import _MC_TV, _MC_BONUS
    elif use_c:
        from .move_finder_c import _get_dict
        from .mc_eval import _mc_find_best_score
        d = _get_dict()
        class SD:
            __slots__ = ('is_valid',)
            def __init__(self, s): self.is_valid = s.__contains__
        sd = SD(d._words)
    else:
        from .move_finder_opt import find_best_score_opt
        from .dictionary import get_dictionary
        _dict = get_dictionary()

    # Phase-specific tile pools.  Blanks have a massive impact on sim cost
    # (~14x slower per blank in rack) because each blank explores all 26
    # GADDAG children at every traversal step.  The old single pool had zero
    # blanks, causing calibration to overestimate late-game throughput by >10x.
    #
    # Pool sizes and blank counts are representative of real game conditions:
    #   Sparse (opening): large pool, 1 blank (~16% of racks get a blank)
    #   Dense  (mid):     moderate pool, 1 blank (~20% of racks get a blank)
    #   VDense (late):    small pool, 2 blanks (~60% of racks get a blank)
    pool_sparse = list("AAABCDDEEEEEFGHIIIJKLMNOOOOPQRRSTTTUUVWXYZ?")    # 43 tiles, 1 blank
    pool_dense  = list("AABCDEEEEFGHIIJKLMNOOOPRRSTTUUVWX?")             # 33 tiles, 1 blank
    pool_vdense = list("AALMNNOORRSTTUUVW??")                            # 19 tiles, 2 blanks
    random.seed(42)

    # --- Sparse board (opening-like) ---
    b_sparse = Board()
    b_sparse.place_word("HELLO", 8, 4, True)

    cross_cache_sparse = {}
    n_sparse = 0
    t0 = time.perf_counter()
    deadline = t0 + CALIBRATION_SECS / 2
    if use_cython:
        ctx_sparse = _accel.prepare_board_context(
            b_sparse._grid, gdata_bytes, set(),
            _word_set, VALID_TWO_LETTER, _MC_TV, _MC_BONUS, BINGO_BONUS, RACK_SIZE)
    while time.perf_counter() < deadline:
        rack = ''.join(random.sample(pool_sparse, 7))
        if use_cython:
            _accel.find_best_score_c(ctx_sparse, rack)
        elif use_c:
            _mc_find_best_score(b_sparse._grid, gdata_bytes, rack, sd, set())
        else:
            find_best_score_opt(b_sparse._grid, gaddag._data, rack, set(),
                                cross_cache=cross_cache_sparse, dictionary=_dict)
        n_sparse += 1
    t_sparse = time.perf_counter() - t0
    sps_sparse = n_sparse / t_sparse

    # --- Dense board (mid-game, ~6 words, ~25 tiles) ---
    b_dense = Board()
    for word, r, c, h in [
        ("HELLO", 8, 4, True), ("OAK", 8, 8, False),
        ("PLANE", 10, 4, True), ("BRAVE", 12, 5, True),
        ("TIGER", 6, 10, False), ("JAM", 4, 10, False),
    ]:
        try:
            b_dense.place_word(word, r, c, h)
        except ValueError:
            pass  # skip conflicting words

    cross_cache_dense = {}
    n_dense = 0
    t0 = time.perf_counter()
    deadline = t0 + CALIBRATION_SECS / 2
    if use_cython:
        ctx_dense = _accel.prepare_board_context(
            b_dense._grid, gdata_bytes, set(),
            _word_set, VALID_TWO_LETTER, _MC_TV, _MC_BONUS, BINGO_BONUS, RACK_SIZE)
    while time.perf_counter() < deadline:
        rack = ''.join(random.sample(pool_dense, 7))
        if use_cython:
            _accel.find_best_score_c(ctx_dense, rack)
        elif use_c:
            _mc_find_best_score(b_dense._grid, gdata_bytes, rack, sd, set())
        else:
            find_best_score_opt(b_dense._grid, gaddag._data, rack, set(),
                                cross_cache=cross_cache_dense, dictionary=_dict)
        n_dense += 1
    t_dense = time.perf_counter() - t0
    sps_dense = n_dense / t_dense

    # --- Very dense board (late-game, ~18 words, ~52 tiles, ~81 anchors) ---
    # Representative of turns 18-24 in a real game.
    # Uses a small pool with 2 blanks to match real late-game conditions
    # where ~60% of opponent racks contain at least one blank.
    b_vdense = Board()
    for word, r, c, h in [
        ("HELLO", 8, 4, True), ("OAK", 8, 8, False),
        ("PLANE", 10, 4, True), ("BRAVE", 12, 5, True),
        ("TIGER", 6, 10, False), ("JAM", 4, 10, False),
        ("DONE", 5, 3, True), ("QUIT", 2, 8, False),
        ("WAXEN", 14, 3, True), ("SLY", 4, 10, True),
        ("PIE", 12, 9, True), ("RUN", 3, 12, False),
        ("MOB", 9, 3, True), ("FIG", 7, 5, True),
        ("VEND", 9, 10, False), ("CUT", 11, 13, False),
        ("HEW", 13, 1, True), ("DIN", 1, 8, False),
    ]:
        try:
            b_vdense.place_word(word, r, c, h)
        except ValueError:
            pass

    cross_cache_vdense = {}
    n_vdense = 0
    t0 = time.perf_counter()
    # Allow more time: sims with blanks are much slower (~300-500ms each),
    # so we need 6+ seconds to get 15-20 samples for statistical stability
    deadline = t0 + max(CALIBRATION_SECS * 2, 6.0)
    if use_cython:
        ctx_vdense = _accel.prepare_board_context(
            b_vdense._grid, gdata_bytes, set(),
            _word_set, VALID_TWO_LETTER, _MC_TV, _MC_BONUS, BINGO_BONUS, RACK_SIZE)
    while time.perf_counter() < deadline:
        rack = ''.join(random.sample(pool_vdense, min(7, len(pool_vdense))))
        if use_cython:
            _accel.find_best_score_c(ctx_vdense, rack)
        elif use_c:
            _mc_find_best_score(b_vdense._grid, gdata_bytes, rack, sd, set())
        else:
            find_best_score_opt(b_vdense._grid, gaddag._data, rack, set(),
                                cross_cache=cross_cache_vdense, dictionary=_dict)
        n_vdense += 1
    t_vdense = time.perf_counter() - t0
    sps_vdense = n_vdense / t_vdense

    n_workers = min(10, max(2, multiprocessing.cpu_count() - 2))

    return {
        'sps_sparse_1w': round(sps_sparse, 1),
        'sps_dense_1w': round(sps_dense, 1),
        'sps_vdense_1w': round(sps_vdense, 1),
        'sps_sparse_nw': round(sps_sparse * n_workers, 1),
        'sps_dense_nw': round(sps_dense * n_workers, 1),
        'sps_vdense_nw': round(sps_vdense * n_workers, 1),
        'n_workers': n_workers,
        'timestamp': time.time(),
        'calibration_secs': CALIBRATION_SECS,
        'engine': engine,
        'env': get_env_info(),
    }


def calibrate(force=False, quiet=False):
    """Run calibration benchmark and cache results.
    
    Args:
        force: Re-run even if cache exists
        quiet: Suppress output
        
    Returns:
        Calibration dict with throughput measurements
    """
    global _calibration, _env_reported
    
    # Detect current engine
    current_engine = _detect_engine()

    # Check cache
    if not force and os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, 'r') as f:
                cached = json.load(f)
            # Cache valid for 48 hours AND engine must match
            cache_engine = cached.get('engine', 'python')
            if time.time() - cached.get('timestamp', 0) < 172800:
                if cache_engine != current_engine:
                    # Engine changed (e.g. Cython extension was built/removed)
                    if not quiet:
                        print(f"  MC engine changed: {cache_engine} -> {current_engine}, recalibrating...")
                else:
                    _calibration = cached
                    if not quiet:
                        if not _env_reported:
                            print(f"  Environment: {env_summary_line()}")
                            _env_reported = True
                        vd = cached.get('sps_vdense_nw', '?')
                        vd_str = f"{vd:.0f}" if isinstance(vd, (int, float)) else vd
                        print(f"  MC calibration: dense={cached['sps_dense_nw']:.0f} "
                              f"vdense={vd_str} sims/s ({cached['n_workers']}w, {cache_engine}, cached)")
                    return cached
        except (json.JSONDecodeError, KeyError):
            pass

    if not quiet:
        print(f"  Calibrating MC throughput ({CALIBRATION_SECS}s benchmark, {current_engine} path)...")
        if not _env_reported:
            print(f"  Environment: {env_summary_line()}")
            _env_reported = True
    
    result = _run_benchmark()
    _calibration = result
    
    # Cache to disk
    try:
        with open(_CACHE_PATH, 'w') as f:
            json.dump(result, f)
    except OSError:
        pass
    
    if not quiet:
        print(f"  MC calibration: sparse={result['sps_sparse_nw']:.0f} "
              f"dense={result['sps_dense_nw']:.0f} "
              f"vdense={result['sps_vdense_nw']:.0f} sims/s "
              f"({result['n_workers']}w, {result.get('engine', 'python')})")
    
    return result


def get_calibration():
    """Get current calibration (runs benchmark if needed)."""
    global _calibration
    if _calibration is None:
        return calibrate(quiet=True)
    return _calibration


def estimate_throughput(bag_size):
    """Estimate N-worker throughput for a given game phase.

    Interpolates between sparse (opening), dense (mid), and very-dense
    (late game) measurements. Late-game boards are much denser with more
    anchors, making each sim significantly slower.
    """
    cal = get_calibration()
    sps_sparse = cal['sps_sparse_nw']
    sps_dense = cal['sps_dense_nw']
    # Very-dense: fall back to dense * 0.3 if not measured (old cache)
    sps_vdense = cal.get('sps_vdense_nw', sps_dense * 0.3)

    if bag_size > 70:
        return sps_sparse  # Opening: sparse board
    elif bag_size > 40:
        # Linear interpolation sparse -> dense
        frac = (70 - bag_size) / 30.0
        return sps_sparse * (1 - frac) + sps_dense * frac
    elif bag_size > 15:
        return sps_dense  # Mid-game: moderately dense board
    else:
        # Late game / endgame: board is very dense, sims are much slower.
        # Interpolate dense -> very-dense as bag shrinks from 15 to 0.
        frac = (15 - bag_size) / 15.0
        return sps_dense * (1 - frac) + sps_vdense * frac


def get_mc_params(bag_size, n_candidates, mc_budget_secs,
                  equity_spread=None, top_n_display=15):
    """Compute optimal N and K for this turn.
    
    Adaptive N: includes all candidates within equity_spread of best,
    with MIN_N floor. K fills remaining budget.
    
    Args:
        bag_size: tiles remaining in bag
        n_candidates: total analyzed candidates available
        mc_budget_secs: seconds available for MC evaluation
        equity_spread: include candidates within this equity of best
                      (None = use default EQUITY_SPREAD)
        top_n_display: minimum N to ensure display has enough results
        
    Returns:
        (N, K) tuple
    """
    if equity_spread is None:
        equity_spread = EQUITY_SPREAD
    
    sps = estimate_throughput(bag_size)
    total_budget = sps * mc_budget_secs  # total sims we can afford
    
    # N: all candidates within spread, clamped to available
    # The actual spread-based N is determined by the caller who knows
    # the equity values. Here we compute the K given N.
    # We return a "max N" that fits the budget with MIN_K sims each.
    min_n = _get_min_n()
    max_n_for_budget = max(min_n, int(total_budget / MIN_K))

    # Clamp to available candidates
    n = min(max_n_for_budget, n_candidates)
    n = max(n, min(top_n_display, n_candidates))  # at least display count
    
    # K: fill remaining budget
    if n > 0:
        k = int(total_budget / n)
        k = max(MIN_K, min(MAX_K, k))
    else:
        k = MIN_K
    
    return n, k


def compute_adaptive_n(candidates, mc_budget_secs, bag_size,
                       top_n_display=15, equity_spread=None):
    """Given actual candidates with prelim_equity, compute N and K.

    With MC early stopping (convergence-based), K is effectively a ceiling
    that most candidates never reach. N=40 is flat across all game phases
    (early stopping makes per-candidate cost self-limiting, so N=40 only
    costs 3-7s more than N=25). K=MAX_K since early stopping controls
    actual sim count per candidate.

    Args:
        candidates: list of move dicts with 'equity' or 'prelim_equity' key
        mc_budget_secs: seconds available for MC (less relevant with early stopping)
        bag_size: tiles in bag
        top_n_display: minimum N for display
        equity_spread: equity window (default EQUITY_SPREAD)

    Returns:
        (N, K, reason_str) tuple
    """
    if equity_spread is None:
        equity_spread = EQUITY_SPREAD

    min_n = _get_min_n()

    if not candidates:
        return min_n, MIN_K, "no candidates"

    # Find best equity
    best_eq = max(m.get('equity', m.get('prelim_equity', 0)) for m in candidates)

    # Count candidates within spread
    n_in_spread = sum(1 for m in candidates
                      if (best_eq - m.get('equity', m.get('prelim_equity', 0)))
                      <= equity_spread)

    # N: at least min_n (40 for Cython, 15 for Python), capped at min_n
    # to avoid evaluating hundreds of candidates on boards with tight spreads.
    # Early stopping makes per-candidate cost self-limiting (~150-530 sims).
    n = min(min_n, len(candidates))
    n = max(n, min(top_n_display, len(candidates)))

    # K=MAX_K -- early stopping handles actual convergence
    k = MAX_K

    # Build reason string
    reason = f"N={n} ({n_in_spread} within {equity_spread:.0f}eq), K={k} (early-stop)"

    return n, k, reason


def get_env_info() -> dict:
    """Gather environment info for reporting."""
    import sys
    total_cores = os.cpu_count() or 1
    n_workers = min(10, max(2, total_cores - 2))
    
    # Try to get CPU model name
    cpu_model = platform.processor() or "unknown"
    if cpu_model in ("", "unknown", "x86_64", "aarch64", "arm"):
        # platform.processor() is often unhelpful; try /proc/cpuinfo on Linux
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        cpu_model = line.split(":", 1)[1].strip()
                        break
        except (FileNotFoundError, PermissionError):
            pass
    
    return {
        'cpu_model': cpu_model,
        'total_cores': total_cores,
        'mc_workers': n_workers,
        'python': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'platform': platform.platform(),
        'arch': platform.machine(),
    }


def env_summary_line() -> str:
    """One-line environment summary for startup banner."""
    info = get_env_info()
    cpu_short = info['cpu_model']
    # Trim common Intel/AMD prefixes for brevity
    for prefix in ("Intel(R) Core(TM) ", "Intel(R) ", "AMD ", "Apple "):
        if cpu_short.startswith(prefix):
            cpu_short = cpu_short[len(prefix):]
            break
    if len(cpu_short) > 50:
        cpu_short = cpu_short[:47] + "..."
    return (f"{cpu_short} | {info['total_cores']} cores "
            f"({info['mc_workers']}w MC) | Python {info['python']} | {info['arch']}")


def show_calibration():
    """Print a detailed calibration report."""
    cal = calibrate(force=True, quiet=True)
    info = get_env_info()
    
    print(f"\n{'='*60}")
    print(f"MC CALIBRATION REPORT")
    print(f"{'='*60}")
    print(f"CPU:          {info['cpu_model']}")
    print(f"Cores:        {info['total_cores']} total, {info['mc_workers']} MC workers")
    print(f"Platform:     {info['platform']}")
    print(f"Python:       {info['python']} ({info['arch']})")
    print(f"{'-'*60}")
    print(f"Sparse (1w):    {cal['sps_sparse_1w']:.0f} sims/s")
    print(f"Dense (1w):     {cal['sps_dense_1w']:.0f} sims/s")
    print(f"VDense (1w):    {cal.get('sps_vdense_1w', '?')} sims/s")
    print(f"Sparse ({cal['n_workers']}w):   {cal['sps_sparse_nw']:.0f} sims/s")
    print(f"Dense ({cal['n_workers']}w):    {cal['sps_dense_nw']:.0f} sims/s")
    print(f"VDense ({cal['n_workers']}w):   {cal.get('sps_vdense_nw', '?')} sims/s")
    
    print(f"\n{'Phase':<20} {'sims/s':>8} {'Budget':>7} {'Max N':>6} {'K@N=80':>7} {'NxK':>10}")
    print("-" * 60)
    
    for label, bag_size, mc_budget in [
        ("Opening (bag>70)", 80, 27),
        ("Early-Mid (50-70)", 60, 27),
        ("Mid Game (30-49)", 40, 27),
        ("Late (20-29)", 25, 27),
        ("Late+ (13-19)", 16, 24),
        ("Pre-Endgame (6-12)", 9, 21),
        ("Endgame (0-5)", 3, 26),
    ]:
        sps = estimate_throughput(bag_size)
        total = sps * mc_budget
        max_n = int(total / MIN_K)
        k_at_80 = min(MAX_K, int(total / 80))
        print(f"{label:<20} {sps:>8.0f} {mc_budget:>5}s {max_n:>6} {k_at_80:>7} {80*k_at_80:>10,}")
    
    print(f"{'='*60}")
