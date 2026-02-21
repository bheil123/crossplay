"""
CROSSPLAY V17 - SuperLeaves Validation (Multi-Worker)

Head-to-head: SuperLeaves bot vs formula bot.
Alternates first player. Reports win rate, avg score, spread.
Parallel workers for fast validation with visible progress.

Usage:
    python -m crossplay.superleaves.validate --table gen2_1000000.pkl --games 1000
    python -m crossplay.superleaves.validate --table gen2_1000000.pkl --games 1000 --workers 5
"""

import os
import sys
import time
import random
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from ..board import Board
from ..config import TILE_DISTRIBUTION, RACK_SIZE
from .fast_bot import select_best_move
from .leave_table import LeaveTable


def _default_workers():
    """Return sensible default worker count: cpu_count - 3 (min 1)."""
    try:
        cpus = os.cpu_count() or 4
    except Exception:
        cpus = 4
    return max(1, cpus - 3)


class FormulaLeaveAdapter:
    """Wraps the formula-based evaluate_leave() behind LeaveTable interface."""

    def __init__(self):
        from ..leave_eval import evaluate_leave
        self._eval = evaluate_leave

    def get(self, leave_key, default=0.0):
        if not leave_key:
            return 0.0
        leave_str = ''.join(leave_key)
        return self._eval(leave_str)


def play_validation_game(gaddag, move_finder_cls, table_a, table_b):
    """Play one game: table_a is P1, table_b is P2.

    Returns:
        (score_a, score_b)
    """
    board = Board()
    bag = []
    for letter, count in TILE_DISTRIBUTION.items():
        bag.extend([letter] * count)
    random.shuffle(bag)

    rack1 = [bag.pop() for _ in range(RACK_SIZE)]
    rack2 = [bag.pop() for _ in range(RACK_SIZE)]
    score1, score2 = 0, 0

    turn = 1
    consecutive_passes = 0
    max_turns = 100
    final_turns = None

    while consecutive_passes < 4 and turn <= max_turns:
        if final_turns is not None and final_turns <= 0:
            break

        is_p1 = (turn % 2 == 1)
        current_rack = rack1 if is_p1 else rack2
        current_table = table_a if is_p1 else table_b
        bag_size = len(bag)

        rack_str = ''.join(current_rack)
        finder = move_finder_cls(board, gaddag)
        moves = finder.find_all_moves(rack_str)

        if not moves:
            consecutive_passes += 1
            if final_turns is not None:
                final_turns -= 1
            turn += 1
            continue

        best_move, best_leave, _ = select_best_move(
            board, moves, current_rack, current_table, bag_size
        )

        if best_move is None:
            consecutive_passes += 1
            if final_turns is not None:
                final_turns -= 1
            turn += 1
            continue

        word = best_move['word']
        row, col = best_move['row'], best_move['col']
        horiz = best_move['direction'] == 'H'
        points = best_move['score']

        board.place_word(word, row, col, horiz)

        if is_p1:
            score1 += points
        else:
            score2 += points

        current_rack.clear()
        current_rack.extend(best_leave)

        draw_count = min(RACK_SIZE - len(current_rack), len(bag))
        for _ in range(draw_count):
            current_rack.append(bag.pop())

        if len(bag) == 0 and final_turns is None:
            final_turns = 2
        if final_turns is not None:
            final_turns -= 1

        consecutive_passes = 0
        turn += 1

    return score1, score2


# ---------------------------------------------------------------------------
# Worker process init and batch play
# ---------------------------------------------------------------------------

_worker_gaddag = None
_worker_move_finder_cls = None
_worker_super_table = None
_worker_formula_table = None


def _init_worker(table_path):
    """Load GADDAG and tables once per worker process."""
    global _worker_gaddag, _worker_move_finder_cls
    global _worker_super_table, _worker_formula_table
    pid = os.getpid()

    try:
        t0 = time.time()
        print(f"  [Worker {pid}] Loading GADDAG...", flush=True)
        from ..gaddag import get_gaddag
        _worker_gaddag = get_gaddag()
        print(f"  [Worker {pid}] GADDAG loaded ({time.time()-t0:.1f}s)", flush=True)

        t1 = time.time()
        _worker_super_table = LeaveTable.load(table_path)
        print(f"  [Worker {pid}] SuperLeaves table: {len(_worker_super_table):,} entries ({time.time()-t1:.1f}s)", flush=True)

        _worker_formula_table = FormulaLeaveAdapter()

        # Prefer C-accelerated move finder
        from ..move_finder_c import is_available as c_available
        if c_available():
            from ..move_finder_c import CMoveFinder
            _worker_move_finder_cls = CMoveFinder
            print(f"  [Worker {pid}] Move finder: C-accelerated", flush=True)
        else:
            from ..move_finder_gaddag import GADDAGMoveFinder
            _worker_move_finder_cls = GADDAGMoveFinder
            print(f"  [Worker {pid}] Move finder: Python", flush=True)

        print(f"  [Worker {pid}] Ready! ({time.time()-t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"  [Worker {pid}] INIT FAILED: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


def _play_batch(batch):
    """Play a batch of validation games in a worker.

    Args:
        batch: list of game_num ints (used for alternating first player)

    Returns:
        list of (game_num, super_score, formula_score)
    """
    results = []
    for game_num in batch:
        if game_num % 2 == 1:
            s_a, s_b = play_validation_game(
                _worker_gaddag, _worker_move_finder_cls,
                _worker_super_table, _worker_formula_table
            )
            super_score, formula_score = s_a, s_b
        else:
            s_a, s_b = play_validation_game(
                _worker_gaddag, _worker_move_finder_cls,
                _worker_formula_table, _worker_super_table
            )
            formula_score, super_score = s_a, s_b
        results.append((game_num, super_score, formula_score))
    return results


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def validate(table_path, num_games, workers=None):
    """Run head-to-head validation with parallel workers.

    Args:
        table_path: path to trained SuperLeaves table
        num_games: number of games to play
        workers: number of parallel workers (default: cpu_count - 3)
    """
    if workers is None:
        workers = _default_workers()

    print(f"Loading SuperLeaves table: {table_path}")
    super_table = LeaveTable.load(table_path)
    print(f"  Table size: {len(super_table):,}")

    print(f"\nValidation: SuperLeaves vs Formula ({num_games:,} games, {workers} workers)")
    print(f"  Alternating first player each game")
    print("=" * 60)

    # Divide games into batches for workers
    batch_size = 10  # small batches for frequent progress updates
    game_nums = list(range(1, num_games + 1))
    batches = []
    for i in range(0, len(game_nums), batch_size):
        batches.append(game_nums[i:i + batch_size])

    super_wins = 0
    formula_wins = 0
    ties = 0
    super_total = 0
    formula_total = 0
    spreads = []
    games_done = 0
    t_start = time.time()
    last_checkpoint = 0

    print(f"\nStarting {workers} workers (30-90s init)...", flush=True)

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(table_path,)
    ) as executor:
        futures = {}
        for batch in batches:
            fut = executor.submit(_play_batch, batch)
            futures[fut] = batch

        for fut in as_completed(futures):
            try:
                results = fut.result()
            except Exception as e:
                print(f"  [ERROR] Batch failed: {e}", flush=True)
                continue

            for game_num, super_score, formula_score in results:
                super_total += super_score
                formula_total += formula_score
                spread = super_score - formula_score
                spreads.append(spread)

                if super_score > formula_score:
                    super_wins += 1
                elif formula_score > super_score:
                    formula_wins += 1
                else:
                    ties += 1

                games_done += 1

            # Print progress every 50 games
            if games_done >= last_checkpoint + 50 or games_done == num_games:
                elapsed = time.time() - t_start
                gps = games_done / elapsed if elapsed > 0 else 0
                win_pct = 100 * super_wins / games_done
                avg_spread = sum(spreads) / len(spreads)
                remaining = num_games - games_done
                eta = remaining / gps if gps > 0 else 0
                print(
                    f"  [{games_done:,}/{num_games:,}] "
                    f"Super {super_wins}-{formula_wins}-{ties}  "
                    f"({win_pct:.1f}%)  "
                    f"spread: {avg_spread:+.1f}  "
                    f"({gps:.1f} g/s, ETA {eta:.0f}s)",
                    flush=True
                )
                last_checkpoint = games_done

    elapsed = time.time() - t_start
    gps = num_games / elapsed if elapsed > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({num_games:,} games in {elapsed:.0f}s, {gps:.1f} games/sec)")
    print(f"  SuperLeaves wins: {super_wins} ({100*super_wins/num_games:.1f}%)")
    print(f"  Formula wins:     {formula_wins} ({100*formula_wins/num_games:.1f}%)")
    print(f"  Ties:             {ties}")
    print(f"  Avg SuperLeaves score: {super_total/num_games:.1f}")
    print(f"  Avg Formula score:     {formula_total/num_games:.1f}")
    avg_spread = sum(spreads) / len(spreads)
    print(f"  Avg spread: {avg_spread:+.1f}")

    if super_wins > formula_wins:
        print(f"\n  --> SuperLeaves WINS by {super_wins - formula_wins} games")
    elif formula_wins > super_wins:
        print(f"\n  --> Formula WINS by {formula_wins - super_wins} games")
    else:
        print(f"\n  --> TIE")


def main():
    parser = argparse.ArgumentParser(
        description='Validate SuperLeaves table vs formula'
    )
    parser.add_argument('--table', type=str, required=True,
                        help='Path to trained SuperLeaves .pkl file')
    parser.add_argument('--games', type=int, default=1000,
                        help='Number of validation games (default: 1000)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of workers (default: cpu_count - 3)')
    args = parser.parse_args()

    # Resolve table path relative to superleaves dir
    table_path = args.table
    if not os.path.isabs(table_path):
        table_path = os.path.join(_superleaves_dir(), table_path)
        if not os.path.exists(table_path):
            # Also try relative to cwd
            table_path = args.table

    if not os.path.exists(table_path):
        print(f"Error: table not found: {table_path}")
        sys.exit(1)

    validate(table_path, args.games, workers=args.workers)


def _superleaves_dir():
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    main()
