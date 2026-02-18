"""
CROSSPLAY V15 - SuperLeaves Trainer

Parallel self-play training with checkpointing and resume.

Usage:
    python -m crossplay.superleaves.trainer --smoke-test --workers 4
    python -m crossplay.superleaves.trainer --workers 6 --games 1000000
    python -m crossplay.superleaves.trainer --resume --workers 6
"""

import os
import sys
import json
import time
import math
import glob
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Directory and path helpers
# ---------------------------------------------------------------------------

def _superleaves_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _status_path():
    return os.path.join(_superleaves_dir(), 'status.json')


def _checkpoint_path(generation, games):
    return os.path.join(_superleaves_dir(), f'gen{generation}_{games}.pkl')


# ---------------------------------------------------------------------------
# Worker init and batch play
# ---------------------------------------------------------------------------

_worker_gaddag = None
_worker_move_finder_cls = None
_worker_leave_table = None


def _init_worker(leave_table_path):
    """Load GADDAG and leave table once per worker process."""
    global _worker_gaddag, _worker_move_finder_cls, _worker_leave_table
    from ..gaddag import get_gaddag
    from .leave_table import LeaveTable

    _worker_gaddag = get_gaddag()
    _worker_leave_table = LeaveTable.load(leave_table_path)

    # Prefer Cython-accelerated move finder (~5-8x faster)
    from ..move_finder_c import is_available as c_available
    if c_available():
        from ..move_finder_c import CMoveFinder
        _worker_move_finder_cls = CMoveFinder
    else:
        from ..move_finder_gaddag import GADDAGMoveFinder
        _worker_move_finder_cls = GADDAGMoveFinder


def _play_batch(batch_size):
    """Play a batch of games and return observations.

    Returns:
        (observations, total_score1, total_score2, games_played)
    """
    from .self_play import play_one_game

    all_obs = []
    total_s1, total_s2 = 0, 0

    for _ in range(batch_size):
        obs, s1, s2 = play_one_game(
            _worker_gaddag, _worker_move_finder_cls, _worker_leave_table
        )
        all_obs.extend(obs)
        total_s1 += s1
        total_s2 += s2

    return all_obs, total_s1, total_s2, batch_size


# ---------------------------------------------------------------------------
# Status file
# ---------------------------------------------------------------------------

def write_status(status, generation, games_completed, games_total,
                 games_per_sec, checkpoint_file, leave_table_size,
                 single_tile_vals=None):
    """Write training status to JSON file."""
    if games_per_sec > 0 and games_completed < games_total:
        eta_minutes = (games_total - games_completed) / games_per_sec / 60
    else:
        eta_minutes = 0.0

    data = {
        'status': status,
        'generation': generation,
        'games_completed': games_completed,
        'games_total': games_total,
        'pct_complete': round(100 * games_completed / max(games_total, 1), 1),
        'games_per_sec': round(games_per_sec, 1),
        'eta_minutes': round(eta_minutes, 1),
        'checkpoint_file': checkpoint_file,
        'leave_table_size': leave_table_size,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    if single_tile_vals:
        data['single_tile_values'] = {
            k: round(v, 2) for k, v in sorted(
                single_tile_vals.items(), key=lambda x: -x[1]
            )
        }

    path = _status_path()
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def read_status():
    """Read training status. Returns dict or None."""
    path = _status_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Find latest checkpoint for resume
# ---------------------------------------------------------------------------

def find_latest_checkpoint(generation=None):
    """Find the most recent checkpoint file.

    Returns:
        (path, generation, games) or (None, None, None)
    """
    pattern = os.path.join(_superleaves_dir(), 'gen*_*.pkl')
    files = glob.glob(pattern)
    if not files:
        return None, None, None

    best = None
    best_gen = 0
    best_games = 0
    for f in files:
        base = os.path.basename(f)
        # Parse gen{N}_{games}.pkl
        try:
            parts = base.replace('.pkl', '').split('_')
            gen = int(parts[0].replace('gen', ''))
            games = int(parts[1])
        except (ValueError, IndexError):
            continue
        if generation is not None and gen != generation:
            continue
        if gen > best_gen or (gen == best_gen and games > best_games):
            best = f
            best_gen = gen
            best_games = games

    return best, best_gen, best_games


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(num_games, workers, generation=1, resume_from=None,
          resume_games=0, batch_size=100, checkpoint_every=10000,
          report_every=1000, is_smoke_test=False, init_from=None):
    """Run training for one generation.

    Args:
        num_games: total games for this generation
        workers: number of parallel worker processes
        generation: generation number
        resume_from: path to checkpoint to resume from (or None)
        resume_games: games already completed (for resume)
        batch_size: games per worker submission
        checkpoint_every: save checkpoint every N games
        report_every: print progress every N games
        is_smoke_test: if True, status shows "smoke_test"
        init_from: path to a previous generation's checkpoint to use as
                   starting point (e.g., gen1 table for gen2 training).
                   Unlike resume, this starts at 0 games completed.
    """
    from .leave_table import LeaveTable

    # Load or bootstrap leave table
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        print(f"  Games already completed: {resume_games:,}")
        table = LeaveTable.load(resume_from)
    elif init_from:
        print(f"Initializing from previous generation: {init_from}")
        table = LeaveTable.load(init_from)
        print(f"  Loaded {len(table):,} leave entries as starting point")
    else:
        print("Bootstrapping leave table from formula...")
        table = LeaveTable()
        count = table.bootstrap_from_formula()
        print(f"  Initialized {count:,} leave entries")

    table_size = len(table)

    # Save initial table for workers to load
    worker_table_path = os.path.join(_superleaves_dir(), '_worker_table.pkl')
    table.save(worker_table_path)

    # Alpha schedule: exponential decay from 0.1 to 0.001
    alpha_start = 0.1
    alpha_end = 0.001

    games_done = resume_games
    games_remaining = num_games - games_done
    if games_remaining <= 0:
        print(f"Already completed {games_done:,}/{num_games:,} games.")
        return table

    print(f"\nStarting generation {generation} training:")
    print(f"  Games: {games_done:,} -> {num_games:,} ({games_remaining:,} remaining)")
    print(f"  Workers: {workers}")
    print(f"  Batch size: {batch_size}")
    print(f"  Checkpoint every: {checkpoint_every:,}")
    print(f"  Report every: {report_every:,}")
    print()

    status_label = 'smoke_test' if is_smoke_test else 'running'
    start_time = time.time()
    last_report_games = games_done
    total_s1, total_s2 = 0, 0
    total_obs_count = 0

    # Write initial status
    write_status(
        status_label, generation, games_done, num_games,
        0.0, '', table_size
    )

    # IMPORTANT: On Windows, ProcessPoolExecutor uses 'spawn' which pickles
    # functions by module path. If this file runs as __main__, the module-level
    # _init_worker/_play_batch get __module__='__main__' and workers can't find
    # them. Fix: import from the actual module to ensure correct pickle path.
    from crossplay.superleaves.trainer import _init_worker as init_fn
    from crossplay.superleaves.trainer import _play_batch as batch_fn

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_fn,
            initargs=(worker_table_path,)
        ) as executor:
            # Submit initial batches to fill the pipeline
            futures = {}
            batches_to_submit = min(
                workers * 4,
                math.ceil(games_remaining / batch_size)
            )
            submitted_games = 0
            for _ in range(batches_to_submit):
                bs = min(batch_size, games_remaining - submitted_games)
                if bs <= 0:
                    break
                fut = executor.submit(batch_fn, bs)
                futures[fut] = bs
                submitted_games += bs

            while futures:
                # Wait for any future to complete
                done_iter = as_completed(futures)
                for fut in done_iter:
                    bs = futures.pop(fut)
                    try:
                        obs, s1, s2, played = fut.result()
                    except Exception as e:
                        print(f"  [!] Worker error: {e}")
                        games_done += bs
                        break

                    games_done += played
                    total_s1 += s1
                    total_s2 += s2
                    total_obs_count += len(obs)

                    # Compute alpha for current progress
                    progress = games_done / max(num_games, 1)
                    alpha = alpha_start * (alpha_end / alpha_start) ** progress

                    # Apply EMA update
                    if obs:
                        table.batch_update_ema(obs, alpha)

                    # Progress report
                    if games_done - last_report_games >= report_every or games_done >= num_games:
                        elapsed = time.time() - start_time
                        gps = (games_done - resume_games) / max(elapsed, 0.01)
                        pct = 100 * games_done / num_games
                        stv = table.single_tile_values()
                        top3 = sorted(stv.items(), key=lambda x: -x[1])[:3]
                        top3_str = ' '.join(f"{k}={v:+.1f}" for k, v in top3)

                        print(
                            f"  [{pct:5.1f}%] {games_done:,}/{num_games:,}  "
                            f"{gps:.1f} g/s  alpha={alpha:.4f}  "
                            f"obs={total_obs_count:,}  top: {top3_str}"
                        )
                        last_report_games = games_done

                        # Update status file
                        write_status(
                            status_label, generation, games_done, num_games,
                            gps,
                            _checkpoint_path(generation, games_done),
                            table_size,
                            stv
                        )

                    # Checkpoint
                    if games_done % checkpoint_every < batch_size or games_done >= num_games:
                        cp_games = (games_done // checkpoint_every) * checkpoint_every
                        if games_done >= num_games:
                            cp_games = num_games
                        cp_path = _checkpoint_path(generation, cp_games)
                        if not os.path.exists(cp_path):
                            table.save(cp_path)
                            # Also update worker table for fresh submits
                            table.save(worker_table_path)

                    # Submit next batch if needed
                    if submitted_games < games_remaining:
                        bs = min(batch_size, games_remaining - submitted_games)
                        if bs > 0:
                            new_fut = executor.submit(batch_fn, bs)
                            futures[new_fut] = bs
                            submitted_games += bs

                    if games_done >= num_games:
                        break
                if games_done >= num_games:
                    break

    except KeyboardInterrupt:
        print(f"\n  [!] Interrupted at {games_done:,}/{num_games:,} games")
        cp_path = _checkpoint_path(generation, games_done)
        table.save(cp_path)
        print(f"  Checkpoint saved: {cp_path}")
        elapsed = time.time() - start_time
        gps = (games_done - resume_games) / max(elapsed, 0.01)
        write_status(
            'paused', generation, games_done, num_games,
            gps, cp_path, table_size, table.single_tile_values()
        )
        print("  Status: paused. Resume with --resume flag.")
        return table

    # Save final table
    final_path = _checkpoint_path(generation, num_games)
    table.save(final_path)

    elapsed = time.time() - start_time
    gps = (games_done - resume_games) / max(elapsed, 0.01)

    print(f"\n{'='*60}")
    print(f"Generation {generation} complete!")
    print(f"  Games: {num_games:,}")
    print(f"  Time: {elapsed/60:.1f} minutes ({gps:.1f} games/sec)")
    print(f"  Observations: {total_obs_count:,}")
    print(f"  Table size: {len(table):,}")
    print(f"  Saved: {final_path}")

    # Print single-tile values
    stv = table.single_tile_values()
    print(f"\nSingle-tile leave values (trained):")
    for tile, val in sorted(stv.items(), key=lambda x: -x[1]):
        print(f"  {tile}: {val:+.2f}")

    avg_score = (total_s1 + total_s2) / max(2 * num_games, 1)
    print(f"\nAvg game score: {avg_score:.1f}")

    write_status(
        'completed', generation, num_games, num_games,
        gps, final_path, table_size, stv
    )

    # Clean up worker table
    try:
        os.unlink(worker_table_path)
    except OSError:
        pass

    return table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='SuperLeaves trainer -- self-play leave value training'
    )
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--games', type=int, default=1_000_000,
                        help='Total games per generation (default: 1M)')
    parser.add_argument('--generation', type=int, default=1,
                        help='Generation number (default: 1)')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Quick 100K game smoke test')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--init-from', type=str, default=None,
                        help='Initialize from a previous generation checkpoint '
                             '(e.g., gen1_350000.pkl for gen2 training)')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Games per worker batch (default: 100)')
    parser.add_argument('--checkpoint-every', type=int, default=10000,
                        help='Checkpoint interval (default: 10K)')
    parser.add_argument('--report-every', type=int, default=1000,
                        help='Progress report interval (default: 1K)')
    args = parser.parse_args()

    if args.smoke_test:
        num_games = 100_000
        is_smoke = True
        print("="*60)
        print("SUPERLEAVES SMOKE TEST (100K games)")
        print("="*60)
    else:
        num_games = args.games
        is_smoke = False
        print("="*60)
        print(f"SUPERLEAVES TRAINING (gen{args.generation}, {num_games:,} games)")
        print("="*60)

    resume_from = None
    resume_games = 0
    gen = args.generation

    if args.resume:
        cp_path, cp_gen, cp_games = find_latest_checkpoint(args.generation)
        if cp_path:
            resume_from = cp_path
            resume_games = cp_games
            gen = cp_gen
            print(f"Found checkpoint: gen{cp_gen} at {cp_games:,} games")
        else:
            print("No checkpoint found, starting fresh.")

    # Resolve --init-from path (support bare filenames in superleaves dir)
    init_from = args.init_from
    if init_from and not os.path.isabs(init_from):
        candidate = os.path.join(_superleaves_dir(), init_from)
        if os.path.exists(candidate):
            init_from = candidate

    train(
        num_games=num_games,
        workers=args.workers,
        generation=gen,
        resume_from=resume_from,
        resume_games=resume_games,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        report_every=args.report_every,
        is_smoke_test=is_smoke,
        init_from=init_from
    )


if __name__ == '__main__':
    main()
