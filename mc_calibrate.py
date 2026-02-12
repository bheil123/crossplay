"""
CROSSPLAY v13.1 - MC Calibration Module

Auto-measures MC throughput on the current hardware and computes optimal
N×K parameters for each game phase. Replaces hardcoded N×K tables.

Usage:
    from crossplay_v9.mc_calibrate import get_mc_params
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
MIN_N = 15            # With Python MC path (~80-150 sims/sec), 15 candidates
                      # gives K~200-400 (better per-candidate confidence) while
                      # still covering the competitive equity window
EQUITY_SPREAD = 15.0  # Include candidates within this equity of best
CALIBRATION_SECS = 3  # Time budget for benchmark


def _run_benchmark():
    """Run a quick MC benchmark to measure sims/sec on this hardware.

    Uses a reference board position (sparse + dense) to get representative
    throughput for different game phases.

    When C accel is unavailable (e.g. Windows), benchmarks find_best_score_opt
    instead of the C fast path — gives realistic throughput for calibration.
    """
    from crossplay_v9.board import Board
    from crossplay_v9.gaddag import get_gaddag
    from crossplay_v9.config import VALID_TWO_LETTER
    import random

    gaddag = get_gaddag()

    # Detect which path to benchmark
    from crossplay_v9.move_finder_c import is_available as _c_avail
    use_c = _c_avail()

    if use_c:
        from crossplay_v9.move_finder_c import _get_dict
        from crossplay_v9.mc_eval import _mc_find_best_score
        gdata_bytes = bytes(gaddag._data)
        d = _get_dict()
        class SD:
            __slots__ = ('is_valid',)
            def __init__(self, s): self.is_valid = s.__contains__
        sd = SD(d._words)
    else:
        from crossplay_v9.move_finder_opt import find_best_score_opt
        from crossplay_v9.dictionary import get_dictionary
        _dict = get_dictionary()

    pool = list("AAABCDDEEEEEFGHIIIJKLMNOOOOPQRRSTTTUUVWXYZ")
    random.seed(42)

    # --- Sparse board (opening-like) ---
    b_sparse = Board()
    b_sparse.place_word("HELLO", 8, 4, True)

    cross_cache_sparse = {}
    n_sparse = 0
    t0 = time.perf_counter()
    deadline = t0 + CALIBRATION_SECS / 2
    while time.perf_counter() < deadline:
        rack = ''.join(random.sample(pool, 7))
        if use_c:
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
    while time.perf_counter() < deadline:
        rack = ''.join(random.sample(pool, 7))
        if use_c:
            _mc_find_best_score(b_dense._grid, gdata_bytes, rack, sd, set())
        else:
            find_best_score_opt(b_dense._grid, gaddag._data, rack, set(),
                                cross_cache=cross_cache_dense, dictionary=_dict)
        n_dense += 1
    t_dense = time.perf_counter() - t0
    sps_dense = n_dense / t_dense

    # --- Very dense board (late-game, ~18 words, ~65+ tiles, ~70 anchors) ---
    # Representative of turns 18-24 in a real game.
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
    # Allow a bit more time for very dense since sims are slow
    deadline = t0 + max(CALIBRATION_SECS / 2, 2.0)
    while time.perf_counter() < deadline:
        rack = ''.join(random.sample(pool, 7))
        if use_c:
            _mc_find_best_score(b_vdense._grid, gdata_bytes, rack, sd, set())
        else:
            find_best_score_opt(b_vdense._grid, gaddag._data, rack, set(),
                                cross_cache=cross_cache_vdense, dictionary=_dict)
        n_vdense += 1
    t_vdense = time.perf_counter() - t0
    sps_vdense = n_vdense / t_vdense

    n_workers = min(10, multiprocessing.cpu_count())

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
    
    # Check cache
    if not force and os.path.exists(_CACHE_PATH):
        try:
            with open(_CACHE_PATH, 'r') as f:
                cached = json.load(f)
            # Cache valid for 48 hours
            if time.time() - cached.get('timestamp', 0) < 172800:
                _calibration = cached
                if not quiet:
                    if not _env_reported:
                        print(f"  Environment: {env_summary_line()}")
                        _env_reported = True
                    vd = cached.get('sps_vdense_nw', '?')
                    vd_str = f"{vd:.0f}" if isinstance(vd, (int, float)) else vd
                    print(f"  MC calibration: dense={cached['sps_dense_nw']:.0f} "
                          f"vdense={vd_str} sims/s ({cached['n_workers']}w, cached)")
                return cached
        except (json.JSONDecodeError, KeyError):
            pass
    
    if not quiet:
        print(f"  Calibrating MC throughput ({CALIBRATION_SECS}s benchmark)...")
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
              f"vdense={result['sps_vdense_nw']:.0f} sims/s ({result['n_workers']}w)")
    
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
    max_n_for_budget = max(MIN_N, int(total_budget / MIN_K))
    
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
    
    This is the main entry point for the adaptive system. It examines
    the equity spread of candidates and decides how many to evaluate.
    
    Args:
        candidates: list of move dicts with 'equity' or 'prelim_equity' key
        mc_budget_secs: seconds available for MC
        bag_size: tiles in bag
        top_n_display: minimum N for display
        equity_spread: equity window (default EQUITY_SPREAD)
        
    Returns:
        (N, K, reason_str) tuple
    """
    if equity_spread is None:
        equity_spread = EQUITY_SPREAD
    
    if not candidates:
        return MIN_N, MIN_K, "no candidates"
    
    sps = estimate_throughput(bag_size)
    total_budget = sps * mc_budget_secs
    
    # Find best equity
    best_eq = max(m.get('equity', m.get('prelim_equity', 0)) for m in candidates)
    
    # Count candidates within spread
    n_in_spread = sum(1 for m in candidates
                      if (best_eq - m.get('equity', m.get('prelim_equity', 0))) 
                      <= equity_spread)
    
    # N: at least top_n_display, at most what budget allows with MIN_K
    max_n = int(total_budget / MIN_K)
    n = max(MIN_N, min(n_in_spread, max_n, len(candidates)))
    n = max(n, min(top_n_display, len(candidates)))
    
    # K: fill budget
    k = int(total_budget / n) if n > 0 else MIN_K
    k = max(MIN_K, min(MAX_K, k))
    
    # Build reason string
    reason = f"{n_in_spread} within {equity_spread:.0f}eq"
    if n < n_in_spread:
        reason += f", capped to {n} by budget"
    if k == MIN_K:
        reason += ", K at floor"
    elif k == MAX_K:
        reason += ", K at cap"
    
    return n, k, reason


def get_env_info() -> dict:
    """Gather environment info for reporting."""
    import sys
    total_cores = os.cpu_count() or 1
    n_workers = min(10, total_cores)
    
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
