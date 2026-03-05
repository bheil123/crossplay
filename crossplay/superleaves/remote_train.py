"""
SuperLeaves Remote Training Wrapper

Runs the trainer and auto-pushes milestone checkpoints to GitHub.
Designed for unattended training on a remote machine (e.g., Bill3's 9950X3D).

Usage:
    python -m crossplay.superleaves.remote_train \
        --generation 5 --games 36000000 --workers 24 --push-every 1000000

    # Resume interrupted training
    python -m crossplay.superleaves.remote_train \
        --generation 5 --games 36000000 --workers 24 --push-every 1000000 --resume

Features:
    - Monitors for new checkpoint files during training
    - Auto-pushes milestone checkpoints to GitHub (every --push-every games)
    - Handles git conflicts with pull --rebase before push
    - Logs all push activity to remote_train.log
    - Training runs as a subprocess so Ctrl+C works cleanly
"""

import os
import sys
import time
import json
import subprocess
import argparse
import threading
import signal


def _superleaves_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _repo_root():
    """Find the git repo root."""
    sl_dir = _superleaves_dir()
    # superleaves is at crossplay/superleaves/, repo root is two levels up
    return os.path.dirname(os.path.dirname(sl_dir))


def _log(msg, log_file=None):
    """Print and optionally log a message."""
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if log_file:
        try:
            with open(log_file, 'a') as f:
                f.write(line + '\n')
        except OSError:
            pass


def _git_cmd(args, cwd):
    """Run a git command, return (success, stdout, stderr)."""
    try:
        result = subprocess.run(
            ['git'] + args,
            cwd=cwd, capture_output=True, text=True, timeout=120
        )
        return result.returncode == 0, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, '', 'git command timed out'
    except Exception as e:
        return False, '', str(e)


def _git_push_checkpoint(checkpoint_path, generation, games, repo_root, log_file):
    """Push a checkpoint file to GitHub with conflict handling."""
    rel_path = os.path.relpath(checkpoint_path, repo_root)
    status_path = os.path.join(_superleaves_dir(), 'status.json')
    rel_status = os.path.relpath(status_path, repo_root)

    _log(f"Pushing checkpoint: gen{generation}_{games:,} games", log_file)

    # Stage the checkpoint and status file
    files_to_add = [rel_path]
    if os.path.exists(status_path):
        files_to_add.append(rel_status)

    ok, out, err = _git_cmd(['add', '-f'] + files_to_add, repo_root)
    if not ok:
        _log(f"  [!] git add failed: {err}", log_file)
        return False

    # Commit
    msg = f"Training: gen{generation} @ {games:,} games (auto-push)"
    ok, out, err = _git_cmd(['commit', '-m', msg], repo_root)
    if not ok:
        if 'nothing to commit' in err or 'nothing to commit' in out:
            _log("  [skip] Nothing new to commit", log_file)
            return True
        _log(f"  [!] git commit failed: {err}", log_file)
        return False

    # Pull --rebase to handle upstream changes
    ok, out, err = _git_cmd(['pull', '--rebase'], repo_root)
    if not ok:
        _log(f"  [!] git pull --rebase failed: {err}", log_file)
        _log("  Attempting to abort rebase and retry...", log_file)
        _git_cmd(['rebase', '--abort'], repo_root)
        # Try a regular pull + merge
        ok, out, err = _git_cmd(['pull'], repo_root)
        if not ok:
            _log(f"  [!] git pull also failed: {err}", log_file)
            return False

    # Push
    ok, out, err = _git_cmd(['push'], repo_root)
    if not ok:
        _log(f"  [!] git push failed: {err}", log_file)
        _log(f"      stderr: {err}", log_file)
        return False

    _log(f"  [OK] Pushed gen{generation}_{games:,}.pkl", log_file)
    return True


def _monitor_and_push(generation, push_every, stop_event, log_file):
    """Background thread: watch for milestone checkpoints and push them."""
    repo_root = _repo_root()
    sl_dir = _superleaves_dir()
    pushed = set()

    while not stop_event.is_set():
        # Check for new milestone checkpoints
        status_path = os.path.join(sl_dir, 'status.json')
        if os.path.exists(status_path):
            try:
                with open(status_path) as f:
                    status = json.load(f)
                games_done = status.get('games_completed', 0)
            except (json.JSONDecodeError, OSError):
                games_done = 0

            # Check if we've hit a push milestone
            milestone = (games_done // push_every) * push_every
            if milestone > 0 and milestone not in pushed:
                # Look for the checkpoint file
                cp_path = os.path.join(
                    sl_dir, f'gen{generation}_{milestone}.pkl'
                )
                if os.path.exists(cp_path):
                    success = _git_push_checkpoint(
                        cp_path, generation, milestone, repo_root, log_file
                    )
                    if success:
                        pushed.add(milestone)
                    else:
                        _log(f"  [!] Push failed for {milestone:,}, "
                             f"will retry next cycle", log_file)

        # Sleep 30 seconds between checks
        stop_event.wait(30)

    # Final push on exit: push whatever the latest checkpoint is
    _log("Monitor thread stopping, doing final push...", log_file)
    status_path = os.path.join(sl_dir, 'status.json')
    if os.path.exists(status_path):
        try:
            with open(status_path) as f:
                status = json.load(f)
            games_done = status.get('games_completed', 0)
            cp_file = status.get('checkpoint_file', '')
            if cp_file and os.path.exists(cp_file) and games_done not in pushed:
                _git_push_checkpoint(
                    cp_file, generation, games_done, repo_root, log_file
                )
        except (json.JSONDecodeError, OSError):
            pass


def main():
    parser = argparse.ArgumentParser(
        description='SuperLeaves Remote Training (with auto-push to GitHub)'
    )
    parser.add_argument('--generation', type=int, required=True,
                        help='Generation number (e.g., 5)')
    parser.add_argument('--games', type=int, required=True,
                        help='Total games for this generation')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of workers (default: cpu_count - 3)')
    parser.add_argument('--push-every', type=int, default=1_000_000,
                        help='Push checkpoint every N games (default: 1M)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--init-from', type=str, default=None,
                        help='Initialize from a previous generation checkpoint')
    parser.add_argument('--td-gamma', type=float, default=0.97,
                        help='TD discount factor (default: 0.97)')
    parser.add_argument('--alpha-start', type=float, default=0.1,
                        help='Initial learning rate (default: 0.1)')
    parser.add_argument('--alpha-end', type=float, default=0.001,
                        help='Final learning rate (default: 0.001)')
    parser.add_argument('--checkpoint-every', type=int, default=10000,
                        help='Checkpoint interval (default: 10K)')
    args = parser.parse_args()

    log_file = os.path.join(_superleaves_dir(), 'remote_train.log')
    repo_root = _repo_root()

    _log("=" * 60, log_file)
    _log("SUPERLEAVES REMOTE TRAINING", log_file)
    _log("=" * 60, log_file)
    _log(f"Generation: {args.generation}", log_file)
    _log(f"Games: {args.games:,}", log_file)
    _log(f"Workers: {args.workers or 'auto'}", log_file)
    _log(f"Push every: {args.push_every:,} games", log_file)
    _log(f"Alpha: {args.alpha_start} -> {args.alpha_end}", log_file)
    _log(f"TD gamma: {args.td_gamma}", log_file)
    _log(f"Resume: {args.resume}", log_file)
    if args.init_from:
        _log(f"Init from: {args.init_from}", log_file)
    _log("=" * 60, log_file)

    # Build trainer command
    trainer_cmd = [
        sys.executable, '-m', 'crossplay.superleaves.trainer',
        '--generation', str(args.generation),
        '--games', str(args.games),
        '--td-gamma', str(args.td_gamma),
        '--alpha-start', str(args.alpha_start),
        '--alpha-end', str(args.alpha_end),
        '--checkpoint-every', str(args.checkpoint_every),
    ]
    if args.workers:
        trainer_cmd += ['--workers', str(args.workers)]
    if args.resume:
        trainer_cmd += ['--resume']
    if args.init_from:
        trainer_cmd += ['--init-from', args.init_from]

    _log(f"Trainer command: {' '.join(trainer_cmd)}", log_file)

    # Start the push monitor thread
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_monitor_and_push,
        args=(args.generation, args.push_every, stop_event, log_file),
        daemon=True
    )
    monitor.start()
    _log("Push monitor thread started", log_file)

    # Run the trainer as a subprocess
    _log("Starting trainer...", log_file)
    try:
        proc = subprocess.Popen(
            trainer_cmd,
            cwd=repo_root,
            stdout=sys.stdout,
            stderr=sys.stderr
        )

        # Wait for trainer to finish
        returncode = proc.wait()

        if returncode == 0:
            _log("Trainer completed successfully!", log_file)
        else:
            _log(f"Trainer exited with code {returncode}", log_file)

    except KeyboardInterrupt:
        _log("Interrupted! Sending SIGTERM to trainer...", log_file)
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _log("Trainer didn't stop, killing...", log_file)
            proc.kill()
            proc.wait()
        _log("Trainer stopped.", log_file)

    finally:
        # Stop the monitor thread and let it do a final push
        _log("Stopping push monitor...", log_file)
        stop_event.set()
        monitor.join(timeout=120)
        _log("Done.", log_file)


if __name__ == '__main__':
    main()
