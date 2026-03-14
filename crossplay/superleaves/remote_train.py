"""
SuperLeaves Remote Training Wrapper (V2 - Enhanced Remote Control)

Runs the trainer and auto-pushes milestone checkpoints to GitHub.
Designed for unattended training on a remote machine (e.g., Bill3's 7950X3D).

Usage:
    python -m crossplay.superleaves.remote_train \
        --generation 8 --games 100000000 --workers 28 --push-every 250000 --resume

Remote control via restart_config.json pushed through git:
    pause       - Stop training, save checkpoint, push, exit
    resume      - Resume from latest checkpoint (optionally with new params)
    restart     - Fresh start with new parameters
    validate    - Pause, run validation, push results, resume training
    tournament  - Pause, run tournament, push results, resume training
    recalibrate - Pause, recalibrate MC, resume training
    status      - Force immediate status push (no training interruption)
    update_code - Pull latest code from git, optionally rebuild Cython, resume

All remote actions are Crossplay-specific operations -- no arbitrary command
execution. Signals are detected via periodic git pull (every 60s).
"""

import os
import sys
import time
import json
import subprocess
import argparse
import threading
import signal
import platform


def _machine_name():
    """Return a short machine identifier for commit messages."""
    return platform.node() or 'unknown'


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


def _has_unpushed_commits(repo_root):
    """Check if there are local commits not yet pushed to origin."""
    ok, out, _ = _git_cmd(['rev-list', '--count', 'origin/main..HEAD'], repo_root)
    if ok and out.strip().isdigit():
        return int(out.strip()) > 0
    return False


def _squash_unpushed(repo_root, log_file):
    """Squash all unpushed commits into one to prevent pileup."""
    ok, out, _ = _git_cmd(['rev-list', '--count', 'origin/main..HEAD'], repo_root)
    if ok and out.strip().isdigit():
        count = int(out.strip())
        if count > 1:
            _log(f"  Squashing {count} unpushed commits into 1...", log_file)
            _git_cmd(['reset', '--soft', 'origin/main'], repo_root)
            return True
    return False


def _git_push_checkpoint(checkpoint_path, generation, games, repo_root, log_file):
    """Push a checkpoint file to GitHub with conflict handling.

    Prevents commit pileup: if previous push failed, squashes all unpushed
    commits into one before adding the new checkpoint. Only one unpushed
    commit exists at any time.
    """
    rel_path = os.path.relpath(checkpoint_path, repo_root)
    status_path = os.path.join(_superleaves_dir(), 'status.json')
    rel_status = os.path.relpath(status_path, repo_root)

    _log(f"Pushing checkpoint: gen{generation}_{games:,} games", log_file)

    # Squash any piled-up unpushed commits before adding more
    _squash_unpushed(repo_root, log_file)

    # Stage the checkpoint and status file
    files_to_add = [rel_path]
    if os.path.exists(status_path):
        files_to_add.append(rel_status)

    ok, out, err = _git_cmd(['add', '-f'] + files_to_add, repo_root)
    if not ok:
        _log(f"  [!] git add failed: {err}", log_file)
        return False

    # Commit (single commit replaces any squashed ones)
    msg = f"Gen{generation} training progress from {_machine_name()} ({games:,} games)"
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
        ok, out, err = _git_cmd(['pull'], repo_root)
        if not ok:
            _log(f"  [!] git pull also failed: {err}", log_file)
            return False

    # Push
    ok, out, err = _git_cmd(['push'], repo_root)
    if not ok:
        _log(f"  [!] git push failed: {err}", log_file)
        return False

    _log(f"  [OK] Pushed gen{generation}_{games:,}.pkl", log_file)
    return True


def _git_push_files(file_paths, message, repo_root, log_file):
    """Push one or more files to GitHub."""
    rel_paths = []
    for fp in file_paths:
        if os.path.exists(fp):
            rel_paths.append(os.path.relpath(fp, repo_root))

    if not rel_paths:
        return False

    ok, out, err = _git_cmd(['add', '-f'] + rel_paths, repo_root)
    if not ok:
        _log(f"  [!] git add failed: {err}", log_file)
        return False

    ok, out, err = _git_cmd(['commit', '-m', message], repo_root)
    if not ok:
        if 'nothing to commit' in err or 'nothing to commit' in out:
            return True
        _log(f"  [!] git commit failed: {err}", log_file)
        return False

    ok, out, err = _git_cmd(['pull', '--rebase'], repo_root)
    if not ok:
        _git_cmd(['rebase', '--abort'], repo_root)
        ok, _, _ = _git_cmd(['pull'], repo_root)
        if not ok:
            return False

    ok, out, err = _git_cmd(['push'], repo_root)
    if not ok:
        _log(f"  [!] git push failed: {err}", log_file)
        return False

    _log(f"  [OK] Pushed {', '.join(rel_paths)}", log_file)
    return True


# ---------------------------------------------------------------------------
# Remote control signals
# ---------------------------------------------------------------------------

VALID_ACTIONS = {'restart', 'resume', 'pause', 'validate', 'tournament',
                 'recalibrate', 'status', 'update_code'}

# Actions that require stopping the trainer subprocess
STOPPING_ACTIONS = {'restart', 'resume', 'pause', 'validate', 'tournament',
                    'recalibrate', 'update_code'}


def _restart_config_path():
    """Path to remote config file (pushed via git for remote control)."""
    return os.path.join(_superleaves_dir(), 'restart_config.json')


def _restart_request_path():
    """Path to local flag file (signals trainer subprocess to exit)."""
    return os.path.join(os.path.dirname(_superleaves_dir()), '.restart_request')


def _check_for_remote_signal(log_file=None):
    """Check for remote control signals in restart_config.json.

    Returns (action, config_dict) or (None, None).
    """
    cfg_path = _restart_config_path()
    if not os.path.exists(cfg_path):
        return None, None

    try:
        with open(cfg_path) as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        _log(f"  [!] Bad restart_config.json: {e}", log_file)
        return None, None

    action = config.get('action')
    if action not in VALID_ACTIONS:
        return None, None

    _log(f"  [>>] REMOTE SIGNAL: action={action}", log_file)
    _log(f"       Message: {config.get('message', 'no message')}", log_file)

    return action, config


def _create_restart_request(log_file=None):
    """Create .restart_request file to signal trainer to exit."""
    try:
        with open(_restart_request_path(), 'w') as f:
            f.write('restart')
    except OSError as e:
        _log(f"  [!] Failed to create .restart_request: {e}", log_file)


def _cleanup_signal_files(log_file=None):
    """Remove signal files after processing and push cleanup to git."""
    for path in [_restart_config_path(), _restart_request_path()]:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass
    # Remove from git tracking so it doesn't persist
    repo_root = _repo_root()
    rel_cfg = os.path.relpath(_restart_config_path(), repo_root)
    _git_cmd(['rm', '-f', '--cached', rel_cfg], repo_root)
    _git_cmd(['commit', '-m', 'Clear remote signal (processed)'], repo_root)
    _git_cmd(['push'], repo_root)
    _log("  Remote signal cleared from git", log_file)


def _handle_status_signal(config, repo_root, log_file):
    """Handle 'status' action: force push status.json immediately."""
    _log("  Forcing status push (remote request)...", log_file)
    status_path = os.path.join(_superleaves_dir(), 'status.json')
    if os.path.exists(status_path):
        _git_push_files([status_path], 'Forced status push (remote request)',
                        repo_root, log_file)
    _cleanup_signal_files(log_file)


# ---------------------------------------------------------------------------
# Crossplay-specific remote actions
# ---------------------------------------------------------------------------

def _run_validation(config, repo_root, log_file):
    """Run SuperLeaves validation and return output text.

    Config fields:
        table: checkpoint filename (e.g., 'gen8_2000000.pkl') or 'latest'
        opponent: 'formula', 'research', 'quackle', 'deployed', or .pkl path
        validate_games: number of games (default: 5000)
        generation: generation number (used to find 'latest' checkpoint)
    """
    table = config.get('table', 'latest')
    opponent = config.get('opponent', 'formula')
    games = config.get('validate_games', 5000)

    # Resolve 'latest' table from status.json
    if table == 'latest':
        status_path = os.path.join(_superleaves_dir(), 'status.json')
        if os.path.exists(status_path):
            try:
                with open(status_path) as f:
                    status = json.load(f)
                table = status.get('checkpoint_file', '')
            except (json.JSONDecodeError, OSError):
                pass
        if not table or not os.path.exists(table):
            return "ERROR: Could not find latest checkpoint from status.json"
    elif not os.path.isabs(table):
        table = os.path.join(_superleaves_dir(), table)

    if not os.path.exists(table):
        return f"ERROR: Table not found: {table}"

    # Build validate command
    cmd = [sys.executable, '-m', 'crossplay.superleaves.validate',
           '--table', table, '--games', str(games)]

    # Set CROSSPLAY_LEAVES env var for opponent type
    env = os.environ.copy()
    if opponent in ('formula', 'research', 'quackle'):
        env['CROSSPLAY_LEAVES'] = opponent
    elif opponent == 'deployed':
        env['CROSSPLAY_LEAVES'] = 'superleaves'
    elif opponent and opponent not in ('formula', 'research', 'quackle', 'deployed'):
        # Path to a .pkl file for table-vs-table comparison
        opp_path = opponent
        if not os.path.isabs(opp_path):
            opp_path = os.path.join(_superleaves_dir(), opp_path)
        if os.path.exists(opp_path):
            cmd += ['--opponent', opp_path]
        else:
            return f"ERROR: Opponent table not found: {opp_path}"

    _log(f"Running validation: {games} games", log_file)
    _log(f"  Table: {os.path.basename(table)}", log_file)
    _log(f"  Opponent: {opponent}", log_file)
    _log(f"  Command: {' '.join(cmd)}", log_file)

    try:
        result = subprocess.run(cmd, cwd=repo_root, capture_output=True,
                                text=True, timeout=3600, env=env)
        output = result.stdout
        if result.stderr:
            output += '\n--- STDERR ---\n' + result.stderr
        _log(f"  Validation complete (exit code {result.returncode})", log_file)
        return output
    except subprocess.TimeoutExpired:
        return "ERROR: Validation timed out after 1 hour"
    except Exception as e:
        return f"ERROR: Validation failed: {e}"


def _run_tournament(config, repo_root, log_file):
    """Run a Crossplay bot tournament and return output text.

    Config fields:
        bot1: bot name (default: 'dadbot_v6')
        bot2: bot name (default: 'my_bot')
        tourney_games: number of games (default: 20)
        tier: speed tier (default: 'fast')
        env_vars: dict of environment variables (optional, e.g. DADBOT_LEAVES)
    """
    bot1 = config.get('bot1', 'dadbot_v6')
    bot2 = config.get('bot2', 'my_bot')
    games = config.get('tourney_games', 20)
    tier = config.get('tier', 'fast')

    # Tournament dir is sibling to repo root
    tourney_dir = os.path.join(os.path.dirname(repo_root), 'crossplay-tournament')
    if not os.path.isdir(tourney_dir):
        return f"ERROR: Tournament directory not found: {tourney_dir}"

    play_match = os.path.join(tourney_dir, 'play_match.py')
    if not os.path.exists(play_match):
        return f"ERROR: play_match.py not found in {tourney_dir}"

    cmd = [sys.executable, play_match, bot1, bot2,
           '--games', str(games), '--tier', tier]

    env = os.environ.copy()
    for k, v in config.get('env_vars', {}).items():
        env[k] = str(v)

    _log(f"Running tournament: {bot1} vs {bot2}, {games} games, tier={tier}",
         log_file)
    _log(f"  Command: {' '.join(cmd)}", log_file)

    try:
        result = subprocess.run(cmd, cwd=tourney_dir, capture_output=True,
                                text=True, timeout=7200, env=env)
        output = result.stdout
        if result.stderr:
            output += '\n--- STDERR ---\n' + result.stderr
        _log(f"  Tournament complete (exit code {result.returncode})", log_file)
        return output
    except subprocess.TimeoutExpired:
        return "ERROR: Tournament timed out after 2 hours"
    except Exception as e:
        return f"ERROR: Tournament failed: {e}"


def _run_recalibration(repo_root, log_file):
    """Run MC speed recalibration and return output text."""
    cmd = [sys.executable, '-m', 'crossplay.mc_calibrate', 'calibrate', '--force']

    _log("Running MC recalibration...", log_file)
    _log(f"  Command: {' '.join(cmd)}", log_file)

    try:
        result = subprocess.run(cmd, cwd=repo_root, capture_output=True,
                                text=True, timeout=300)
        output = result.stdout
        if result.stderr:
            output += '\n--- STDERR ---\n' + result.stderr
        _log(f"  Recalibration complete (exit code {result.returncode})", log_file)
        return output
    except subprocess.TimeoutExpired:
        return "ERROR: Recalibration timed out after 5 minutes"
    except Exception as e:
        return f"ERROR: Recalibration failed: {e}"


def _run_update_code(config, repo_root, log_file):
    """Pull latest code from all Crossplay repos and optionally rebuild Cython.

    Pulls both the main crossplay repo and crossplay-tournament (if it exists).
    All repos are hardcoded Crossplay paths -- no arbitrary repo access.

    Config fields:
        rebuild_cython: if True, rebuild gaddag_accel.pyd (default: False)
        git_lfs_pull: if True, also run git lfs pull (default: False)
    """
    output_lines = []

    # Known Crossplay repos (hardcoded, no arbitrary repos)
    repos = [
        ('crossplay', repo_root),
        ('crossplay-tournament', os.path.join(os.path.dirname(repo_root),
                                              'crossplay-tournament')),
    ]

    for name, path in repos:
        if not os.path.isdir(os.path.join(path, '.git')):
            output_lines.append(f"{name}: skipped (not a git repo)")
            continue

        _log(f"  Pulling {name}...", log_file)
        ok, out, err = _git_cmd(['pull', '--rebase'], path)
        if not ok:
            _git_cmd(['rebase', '--abort'], path)
            ok, out, err = _git_cmd(['pull'], path)

        if ok:
            output_lines.append(f"{name}: {out or 'up to date'}")
        else:
            output_lines.append(f"{name}: FAILED - {err}")

        if config.get('git_lfs_pull'):
            ok, out, err = _git_cmd(['lfs', 'pull'], path)
            output_lines.append(f"{name} lfs: {out if ok else err}")

    if config.get('rebuild_cython'):
        _log("  Rebuilding Cython extension...", log_file)
        try:
            result = subprocess.run(
                [sys.executable, 'setup_accel.py', 'build_ext', '--inplace'],
                cwd=os.path.join(repo_root, 'crossplay'),
                capture_output=True, text=True, timeout=120
            )
            output_lines.append(
                f"Cython rebuild: {'OK' if result.returncode == 0 else 'FAILED'}")
            if result.stderr:
                output_lines.append(result.stderr[-500:])
        except Exception as e:
            output_lines.append(f"Cython rebuild error: {e}")

    _log("  Code update complete.", log_file)
    return '\n'.join(output_lines)


def _push_remote_results(action, output, repo_root, log_file):
    """Save action results to file and push to git."""
    results_path = os.path.join(_superleaves_dir(), 'remote_results.txt')
    ts = time.strftime('%Y-%m-%d %H:%M:%S')

    header = (f"{'='*60}\n"
              f"Remote {action} results -- {ts}\n"
              f"Machine: {_machine_name()}\n"
              f"{'='*60}\n\n")

    try:
        with open(results_path, 'w') as f:
            f.write(header + output)
    except OSError as e:
        _log(f"  [!] Failed to write results: {e}", log_file)
        return

    _git_push_files([results_path],
                    f'Remote {action} results from Bill3',
                    repo_root, log_file)


# ---------------------------------------------------------------------------
# Trainer command builders
# ---------------------------------------------------------------------------

def _build_trainer_cmd(config, log_file=None):
    """Build a trainer command list from remote config dict."""
    cmd = [
        sys.executable, '-m', 'crossplay.superleaves.trainer',
        '--generation', str(config['generation']),
        '--games', str(config['games']),
        '--td-gamma', str(config.get('td_gamma', 0.97)),
        '--alpha-start', str(config.get('alpha_start', 0.1)),
        '--alpha-end', str(config.get('alpha_end', 0.001)),
        '--checkpoint-every', str(config.get('checkpoint_every', 10000)),
    ]
    if config.get('workers'):
        cmd += ['--workers', str(config['workers'])]
    if config.get('resume'):
        cmd += ['--resume']
    if config.get('init_from'):
        cmd += ['--init-from', config['init_from']]
    return cmd


def _build_trainer_cmd_from_args(args):
    """Build trainer command from argparse args."""
    cmd = [
        sys.executable, '-m', 'crossplay.superleaves.trainer',
        '--generation', str(args.generation),
        '--games', str(args.games),
        '--td-gamma', str(args.td_gamma),
        '--alpha-start', str(args.alpha_start),
        '--alpha-end', str(args.alpha_end),
        '--checkpoint-every', str(args.checkpoint_every),
    ]
    if args.workers:
        cmd += ['--workers', str(args.workers)]
    if args.resume:
        cmd += ['--resume']
    if args.init_from:
        cmd += ['--init-from', args.init_from]
    return cmd


# ---------------------------------------------------------------------------
# Monitor thread (checkpoint push + remote signal detection)
# ---------------------------------------------------------------------------

_GIT_POLL_INTERVAL = 60  # seconds between git pulls for signal detection


def _monitor_and_push(generation, push_every, stop_event, restart_event,
                      signal_holder, log_file):
    """Background thread: push milestone checkpoints and poll for remote signals.

    signal_holder is a mutable list for passing (action, config) back to main.
    """
    repo_root = _repo_root()
    sl_dir = _superleaves_dir()
    pushed = set()
    last_git_poll = 0

    while not stop_event.is_set():
        now = time.time()
        did_git_pull = False

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
                cp_path = os.path.join(
                    sl_dir, f'gen{generation}_{milestone}.pkl'
                )
                if os.path.exists(cp_path):
                    success = _git_push_checkpoint(
                        cp_path, generation, milestone, repo_root, log_file
                    )
                    if success:
                        pushed.add(milestone)
                        did_git_pull = True  # push includes git pull
                        last_git_poll = now
                    else:
                        _log(f"  [!] Push failed for {milestone:,}, "
                             f"will retry next cycle", log_file)

        # Periodic git pull for faster signal detection (even without push)
        if not did_git_pull and now - last_git_poll >= _GIT_POLL_INTERVAL:
            ok, _, _ = _git_cmd(['pull', '--rebase'], repo_root)
            if not ok:
                _git_cmd(['rebase', '--abort'], repo_root)
                _git_cmd(['pull'], repo_root)
            did_git_pull = True
            last_git_poll = now

        # Check for remote signal after any git pull
        if did_git_pull:
            action, config = _check_for_remote_signal(log_file)
            if action == 'status':
                _handle_status_signal(config, repo_root, log_file)
            elif action in STOPPING_ACTIONS:
                _create_restart_request(log_file)
                signal_holder.clear()
                signal_holder.append((action, config))
                restart_event.set()
                return  # Exit monitor thread

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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='SuperLeaves Remote Training (with auto-push and remote control)'
    )
    parser.add_argument('--generation', type=int, required=True,
                        help='Generation number (e.g., 8)')
    parser.add_argument('--games', type=int, required=True,
                        help='Total games for this generation')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of workers (default: cpu_count - 3)')
    parser.add_argument('--push-every', type=int, default=250_000,
                        help='Push checkpoint every N games (default: 250K)')
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

    # Build initial trainer command
    trainer_cmd = _build_trainer_cmd_from_args(args)
    generation = args.generation
    push_every = args.push_every

    # Track current training config for auto-resume after actions
    current_config = {
        'generation': args.generation,
        'games': args.games,
        'workers': args.workers,
        'td_gamma': args.td_gamma,
        'alpha_start': args.alpha_start,
        'alpha_end': args.alpha_end,
        'checkpoint_every': args.checkpoint_every,
        'push_every': args.push_every,
    }

    _log_config(args, log_file)
    _log(f"Trainer command: {' '.join(trainer_cmd)}", log_file)

    # Clear any stale signal files from previous runs to prevent
    # the monitor thread from immediately triggering a restart
    for stale_path in [_restart_config_path(), _restart_request_path()]:
        if os.path.exists(stale_path):
            _log(f"  Clearing stale signal file: {os.path.basename(stale_path)}", log_file)
            try:
                os.unlink(stale_path)
            except OSError:
                pass

    # Main training loop (supports remote control via restart_config.json)
    while True:
        stop_event = threading.Event()
        restart_event = threading.Event()
        signal_holder = []  # mutable container for thread -> main communication

        # Start the push monitor thread
        monitor = threading.Thread(
            target=_monitor_and_push,
            args=(generation, push_every, stop_event, restart_event,
                  signal_holder, log_file),
            daemon=True
        )
        monitor.start()
        _log("Push monitor started (polling git every 60s for signals)", log_file)

        # Run the trainer as a subprocess
        _log("Starting trainer...", log_file)
        interrupted = False
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
                _log("Trainer completed/exited successfully.", log_file)
            else:
                _log(f"Trainer exited with code {returncode}", log_file)

        except KeyboardInterrupt:
            interrupted = True
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

        if interrupted:
            _log("User interrupted. Exiting.", log_file)
            break

        # ---------------------------------------------------------------
        # Handle remote action (from monitor thread or config file)
        # ---------------------------------------------------------------
        action, config = None, None

        # Check what the monitor thread found
        if signal_holder:
            action, config = signal_holder[0]
        else:
            # Read directly from file (trainer may have exited on its own)
            cfg_path = _restart_config_path()
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path) as f:
                        config = json.load(f)
                    action = config.get('action')
                    if action not in VALID_ACTIONS:
                        action, config = None, None
                except (json.JSONDecodeError, OSError):
                    pass

        if action:
            _log("=" * 60, log_file)
            _log(f"HANDLING REMOTE ACTION: {action}", log_file)
            _log(f"  Message: {config.get('message', 'N/A')}", log_file)
            _log("=" * 60, log_file)

            if action == 'pause':
                _cleanup_signal_files(log_file)
                _log("Training PAUSED. Push 'resume' signal to continue.", log_file)
                _log("Exiting remote_train. Bill3 must restart the process", log_file)
                _log("after a 'resume' signal is pushed and pulled.", log_file)
                break

            elif action == 'restart':
                # Fresh start with new parameters (no --resume)
                trainer_cmd = _build_trainer_cmd(config, log_file)
                generation = config['generation']
                push_every = config.get('push_every', push_every)
                # Update tracked config
                for k in current_config:
                    if k in config and config[k] is not None:
                        current_config[k] = config[k]
                _log(f"New trainer command: {' '.join(trainer_cmd)}", log_file)
                _cleanup_signal_files(log_file)
                time.sleep(5)
                continue

            elif action == 'resume':
                # Resume from latest checkpoint, optionally with new params
                resume_cfg = dict(current_config)
                for k in ('generation', 'games', 'workers', 'td_gamma',
                          'alpha_start', 'alpha_end', 'checkpoint_every',
                          'push_every'):
                    if k in config and config[k] is not None:
                        resume_cfg[k] = config[k]
                resume_cfg['resume'] = True
                trainer_cmd = _build_trainer_cmd(resume_cfg, log_file)
                generation = resume_cfg['generation']
                push_every = resume_cfg.get('push_every', push_every)
                current_config.update(resume_cfg)
                _log(f"Resume command: {' '.join(trainer_cmd)}", log_file)
                _cleanup_signal_files(log_file)
                time.sleep(5)
                continue

            elif action == 'validate':
                # Run validation, push results, then resume training
                config.setdefault('generation', generation)
                output = _run_validation(config, repo_root, log_file)
                _push_remote_results('validate', output, repo_root, log_file)
                _log("Validation complete. Resuming training...", log_file)
                resume_cfg = dict(current_config)
                resume_cfg['resume'] = True
                trainer_cmd = _build_trainer_cmd(resume_cfg, log_file)
                _cleanup_signal_files(log_file)
                time.sleep(5)
                continue

            elif action == 'tournament':
                # Run tournament, push results, then resume training
                output = _run_tournament(config, repo_root, log_file)
                _push_remote_results('tournament', output, repo_root, log_file)
                _log("Tournament complete. Resuming training...", log_file)
                resume_cfg = dict(current_config)
                resume_cfg['resume'] = True
                trainer_cmd = _build_trainer_cmd(resume_cfg, log_file)
                _cleanup_signal_files(log_file)
                time.sleep(5)
                continue

            elif action == 'recalibrate':
                # Run MC recalibration, push results, then resume training
                output = _run_recalibration(repo_root, log_file)
                _push_remote_results('recalibrate', output, repo_root, log_file)
                _log("Recalibration complete. Resuming training...", log_file)
                resume_cfg = dict(current_config)
                resume_cfg['resume'] = True
                trainer_cmd = _build_trainer_cmd(resume_cfg, log_file)
                _cleanup_signal_files(log_file)
                time.sleep(5)
                continue

            elif action == 'update_code':
                # Pull latest code, optionally rebuild, then resume training
                output = _run_update_code(config, repo_root, log_file)
                _push_remote_results('update_code', output, repo_root, log_file)
                _log("Code update complete. Resuming training...", log_file)
                resume_cfg = dict(current_config)
                resume_cfg['resume'] = True
                trainer_cmd = _build_trainer_cmd(resume_cfg, log_file)
                _cleanup_signal_files(log_file)
                time.sleep(5)
                continue

        # Normal exit (no remote action pending)
        _log("Training complete. Exiting.", log_file)
        break


def _log_config(args, log_file):
    """Log training configuration."""
    _log("=" * 60, log_file)
    _log("SUPERLEAVES REMOTE TRAINING (V2)", log_file)
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
    _log("Remote control actions (via restart_config.json in git):", log_file)
    _log("  pause | resume | restart | validate | tournament", log_file)
    _log("  recalibrate | status | update_code", log_file)
    _log("=" * 60, log_file)


if __name__ == '__main__':
    main()
