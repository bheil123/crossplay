"""
SuperLeaves Hardware Benchmark

Measures training throughput at various worker counts to calibrate
optimal settings for a given machine. Designed to run from a fresh
clone with zero configuration.

Usage:
    python -m crossplay.superleaves.benchmark
    python -m crossplay.superleaves.benchmark --push
    python -m crossplay.superleaves.benchmark --games 500

Output: benchmark_results.json in the superleaves directory.
"""

import os
import sys
import json
import time
import platform
import subprocess
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed


def _superleaves_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _get_cpu_info():
    """Gather CPU info using platform and OS-level queries."""
    info = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
        'processor': platform.processor(),
    }

    # Try to get CPU name and cache info on Windows
    if platform.system() == 'Windows':
        try:
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'Name', '/value'],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.strip().split('\n'):
                if line.startswith('Name='):
                    info['cpu_name'] = line.split('=', 1)[1].strip()
        except Exception:
            pass
        try:
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'L3CacheSize', '/value'],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.strip().split('\n'):
                if line.startswith('L3CacheSize='):
                    val = line.split('=', 1)[1].strip()
                    if val:
                        info['l3_cache_kb'] = int(val)
                        info['l3_cache_mb'] = round(int(val) / 1024, 1)
        except Exception:
            pass
    elif platform.system() == 'Linux':
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if line.startswith('model name'):
                        info['cpu_name'] = line.split(':', 1)[1].strip()
                        break
        except Exception:
            pass
        try:
            result = subprocess.run(
                ['lscpu'], capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.split('\n'):
                if 'L3 cache' in line:
                    info['l3_cache_str'] = line.split(':', 1)[1].strip()
        except Exception:
            pass

    return info


# ---------------------------------------------------------------------------
# Worker init and batch play (mirrors trainer.py)
# ---------------------------------------------------------------------------

_worker_gaddag = None
_worker_move_finder_cls = None
_worker_leave_table = None


def _init_worker(leave_table_path):
    """Load GADDAG and leave table once per worker."""
    global _worker_gaddag, _worker_move_finder_cls, _worker_leave_table
    pid = os.getpid()

    from ..gaddag import get_gaddag
    _worker_gaddag = get_gaddag()

    from .leave_table import LeaveTable
    _worker_leave_table = LeaveTable.load(leave_table_path)

    from ..move_finder_c import is_available as c_available
    if c_available():
        from ..move_finder_c import CMoveFinder
        _worker_move_finder_cls = CMoveFinder
    else:
        from ..move_finder_gaddag import GADDAGMoveFinder
        _worker_move_finder_cls = GADDAGMoveFinder

    print(f"  [Worker {pid}] Ready", flush=True)


def _play_batch(batch_size):
    """Play a batch of games, return (games_played,)."""
    from .self_play import play_one_game

    for _ in range(batch_size):
        play_one_game(
            _worker_gaddag, _worker_move_finder_cls, _worker_leave_table,
            td_gamma=0.97
        )
    return batch_size


def _run_benchmark_at_workers(n_workers, n_games, leave_table_path):
    """Run n_games with n_workers and return games/sec."""
    # Import here to get correct pickle paths
    from crossplay.superleaves.benchmark import _init_worker as init_fn
    from crossplay.superleaves.benchmark import _play_batch as batch_fn

    batch_size = min(50, max(10, n_games // (n_workers * 4)))
    games_done = 0

    print(f"\n  Testing {n_workers} worker(s), {n_games} games "
          f"(batch={batch_size})...", flush=True)

    t0 = time.time()
    init_done = False

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_fn,
        initargs=(leave_table_path,)
    ) as executor:
        futures = []
        submitted = 0

        # Submit all batches
        while submitted < n_games:
            batch = min(batch_size, n_games - submitted)
            futures.append(executor.submit(batch_fn, batch))
            submitted += batch

        # Collect results
        for future in as_completed(futures):
            if not init_done:
                init_time = time.time() - t0
                init_done = True
            games_done += future.result()

    elapsed = time.time() - t0
    game_time = elapsed - (init_time if init_done else 0)
    gps_total = games_done / elapsed if elapsed > 0 else 0
    gps_game_only = games_done / game_time if game_time > 0 else 0

    print(f"    {n_workers}w: {games_done} games in {elapsed:.1f}s "
          f"({gps_total:.1f} g/s total, {gps_game_only:.1f} g/s game-only, "
          f"init={init_time:.1f}s)", flush=True)

    return {
        'workers': n_workers,
        'games': games_done,
        'elapsed_sec': round(elapsed, 2),
        'init_sec': round(init_time, 2) if init_done else 0,
        'game_sec': round(game_time, 2),
        'gps_total': round(gps_total, 1),
        'gps_game_only': round(gps_game_only, 1),
    }


def main():
    parser = argparse.ArgumentParser(description='SuperLeaves Hardware Benchmark')
    parser.add_argument('--games', type=int, default=1000,
                        help='Games per test point (default: 1000)')
    parser.add_argument('--push', action='store_true',
                        help='Auto-push results to GitHub after benchmark')
    args = parser.parse_args()

    cpu_info = _get_cpu_info()
    cpus = cpu_info.get('cpu_count', 4)

    print("=" * 60)
    print("  SUPERLEAVES HARDWARE BENCHMARK")
    print("=" * 60)
    print(f"  CPU: {cpu_info.get('cpu_name', cpu_info.get('processor', '?'))}")
    print(f"  Cores: {cpus}")
    if 'l3_cache_mb' in cpu_info:
        print(f"  L3 Cache: {cpu_info['l3_cache_mb']} MB")
    print(f"  Platform: {cpu_info['platform']} {cpu_info['machine']}")
    print(f"  Python: {cpu_info['python_version']}")
    print(f"  Games per test: {args.games}")
    print("=" * 60)

    # Ensure GADDAG is built
    print("\nStep 1: Building GADDAG (cached after first run)...")
    t0 = time.time()
    from ..gaddag import get_gaddag
    gaddag = get_gaddag()
    gaddag_time = time.time() - t0
    print(f"  GADDAG ready ({gaddag_time:.1f}s)")

    # Bootstrap a leave table for benchmarking
    print("\nStep 2: Bootstrapping leave table...")
    from .leave_table import LeaveTable
    table = LeaveTable()
    count = table.bootstrap_from_formula()
    table_path = os.path.join(_superleaves_dir(), '_benchmark_table.pkl')
    table.save(table_path)
    print(f"  {count:,} entries")

    # Check for C-accelerated move finder
    from ..move_finder_c import is_available as c_available
    c_accel = c_available()
    print(f"  Move finder: {'C-accelerated' if c_accel else 'Python fallback'}")

    # Determine worker counts to test
    # Standard: 1, 8, 16, 24, 28
    # But cap to what the machine can handle
    worker_counts = [1]
    for w in [8, 16, 24, 28]:
        if w <= cpus + 4:  # Allow slight oversubscription
            worker_counts.append(w)
    # Also add machine's default (cpu_count - 3)
    default_w = max(1, cpus - 3)
    if default_w not in worker_counts:
        worker_counts.append(default_w)
        worker_counts.sort()

    print(f"\nStep 3: Running benchmarks at worker counts: {worker_counts}")
    print(f"  ({args.games} games each, ~7-8 minutes total)")

    results = []
    total_t0 = time.time()

    for n_w in worker_counts:
        result = _run_benchmark_at_workers(n_w, args.games, table_path)
        results.append(result)

    total_elapsed = time.time() - total_t0

    # Clean up benchmark table
    try:
        os.unlink(table_path)
    except OSError:
        pass

    # Find optimal config
    best = max(results, key=lambda r: r['gps_game_only'])

    # Projections
    gps = best['gps_game_only']
    proj_100h = int(gps * 3600 * 100)

    # Build output
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'cpu_info': cpu_info,
        'c_accelerated': c_accel,
        'games_per_test': args.games,
        'total_benchmark_sec': round(total_elapsed, 1),
        'gaddag_build_sec': round(gaddag_time, 1),
        'results': results,
        'optimal': {
            'workers': best['workers'],
            'gps_game_only': best['gps_game_only'],
            'gps_total': best['gps_total'],
        },
        'projections': {
            '100h_games': proj_100h,
            '100h_games_M': round(proj_100h / 1_000_000, 1),
            'obs_per_6tile_entry': round(proj_100h * 17 / 921_000),
        },
    }

    # Save results
    out_path = os.path.join(_superleaves_dir(), 'benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("  BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\n  {'Workers':>8}  {'G/S (total)':>12}  {'G/S (game)':>12}  {'Init (s)':>9}")
    print("  " + "-" * 45)
    for r in results:
        marker = " <-- best" if r['workers'] == best['workers'] else ""
        print(f"  {r['workers']:>8}  {r['gps_total']:>12.1f}  {r['gps_game_only']:>12.1f}  {r['init_sec']:>9.1f}{marker}")

    print(f"\n  Optimal: {best['workers']} workers @ {best['gps_game_only']:.1f} g/s")
    print(f"\n  -- Projections for 100 hours --")
    print(f"  Total games: {proj_100h:,} ({proj_100h/1_000_000:.1f}M)")
    print(f"  Obs per 6-tile entry: ~{output['projections']['obs_per_6tile_entry']:,}")
    print(f"\n  Results saved to: {out_path}")
    print("=" * 60)

    # Auto-push if requested
    if args.push:
        print("\nPushing results to GitHub...")
        repo_root = os.path.dirname(os.path.dirname(_superleaves_dir()))
        rel_path = os.path.relpath(out_path, repo_root)
        try:
            subprocess.run(
                ['git', 'add', rel_path],
                cwd=repo_root, check=True, capture_output=True
            )
            subprocess.run(
                ['git', 'commit', '-m',
                 f'Benchmark: {cpu_info.get("cpu_name", "unknown")} '
                 f'({best["gps_game_only"]:.0f} g/s @ {best["workers"]}w)'],
                cwd=repo_root, check=True, capture_output=True
            )
            subprocess.run(
                ['git', 'push'],
                cwd=repo_root, check=True, capture_output=True
            )
            print("  [OK] Results pushed to GitHub")
        except subprocess.CalledProcessError as e:
            print(f"  [!] Git push failed: {e}")
            print(f"      stderr: {e.stderr.decode() if e.stderr else 'none'}")
            print("      You can manually push with:")
            print(f"      git add {rel_path} && git commit -m 'Benchmark results' && git push")


if __name__ == '__main__':
    main()
