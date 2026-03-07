"""
CROSSPLAY V18 - SuperLeaves Validation (Multi-Worker)

Head-to-head: SuperLeaves bot vs formula bot (or vs another table).
Alternates first player. Reports win rate, avg score, spread.
Parallel workers for fast validation with visible progress.

Usage:
    # Table vs formula
    python -m crossplay.superleaves.validate --table gen3_1000000.pkl --games 1000

    # Table vs table (gen3 vs gen2)
    python -m crossplay.superleaves.validate --table gen3_1000000.pkl --opponent gen2_1000000.pkl --games 1000
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


# Quackle per-tile values (from Scrabble, used as baseline)
QUACKLE_VALUES = {
    '?': 20.0, 'S': 7.0, 'Z': 5.12, 'X': 3.31, 'R': 1.10,
    'C': 0.85, 'H': 0.60, 'M': 0.58, 'D': 0.45, 'E': 0.35,
    'N': 0.22, 'L': 0.20, 'T': -0.10, 'P': -0.46, 'K': -0.20,
    'Y': -0.63, 'A': -0.63, 'J': -0.80, 'B': -1.50, 'I': -2.07,
    'F': -2.21, 'O': -2.50, 'G': -1.80, 'W': -4.50, 'U': -4.00,
    'V': -6.50, 'Q': -6.79,
}


class QuackleLeaveAdapter:
    """Wraps Quackle per-tile values behind LeaveTable interface."""

    def __init__(self):
        self._values = QUACKLE_VALUES

    def get(self, leave_key, default=0.0):
        if not leave_key:
            return 0.0
        return sum(self._values.get(t, -1.0) for t in leave_key)


class ResearchLeaveAdapter:
    """Wraps research-derived Crossplay values behind LeaveTable interface."""

    def __init__(self):
        from ..leave_eval import _research_evaluate
        self._eval = _research_evaluate

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


def _load_table(path_or_sentinel, label, pid):
    """Load a leave table from path or create an adapter from sentinel."""
    if path_or_sentinel == '__formula__':
        print(f"  [Worker {pid}] {label}: Formula adapter", flush=True)
        return FormulaLeaveAdapter()
    elif path_or_sentinel == '__quackle__':
        print(f"  [Worker {pid}] {label}: Quackle per-tile adapter", flush=True)
        return QuackleLeaveAdapter()
    elif path_or_sentinel == '__research__':
        print(f"  [Worker {pid}] {label}: Research-derived adapter", flush=True)
        return ResearchLeaveAdapter()
    else:
        t = time.time()
        table = LeaveTable.load(path_or_sentinel)
        print(f"  [Worker {pid}] {label}: {len(table):,} entries ({time.time()-t:.1f}s)", flush=True)
        return table


def _init_worker(table_path, opponent_path=None):
    """Load GADDAG and tables once per worker process.

    table_path/opponent_path can be file paths or sentinels:
      '__formula__' -> FormulaLeaveAdapter
      '__quackle__' -> QuackleLeaveAdapter
    """
    global _worker_gaddag, _worker_move_finder_cls
    global _worker_super_table, _worker_formula_table
    pid = os.getpid()

    try:
        t0 = time.time()
        print(f"  [Worker {pid}] Loading GADDAG...", flush=True)
        from ..gaddag import get_gaddag
        _worker_gaddag = get_gaddag()
        print(f"  [Worker {pid}] GADDAG loaded ({time.time()-t0:.1f}s)", flush=True)

        _worker_super_table = _load_table(table_path, "Table A", pid)

        if opponent_path:
            _worker_formula_table = _load_table(opponent_path, "Table B", pid)
        else:
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

def _label_for(path_or_sentinel):
    """Human-readable label for a table path or sentinel."""
    if path_or_sentinel == '__formula__':
        return 'Formula'
    elif path_or_sentinel == '__quackle__':
        return 'Quackle'
    elif path_or_sentinel == '__research__':
        return 'Research'
    else:
        return os.path.basename(path_or_sentinel).replace('.pkl', '')


def validate(table_path, num_games, workers=None, opponent_path=None):
    """Run head-to-head validation with parallel workers.

    Args:
        table_path: path to SuperLeaves .pkl, or sentinel
                    ('__formula__', '__quackle__')
        num_games: number of games to play
        workers: number of parallel workers (default: cpu_count - 3)
        opponent_path: path to opponent .pkl, or sentinel, or None for formula
    """
    if workers is None:
        workers = _default_workers()

    if opponent_path is None:
        opponent_path = '__formula__'

    table_label = _label_for(table_path)
    opp_label = _label_for(opponent_path)

    print(f"\nValidation: {table_label} vs {opp_label} ({num_games:,} games, {workers} workers)")
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
        initargs=(table_path, opponent_path)
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
                    f"A {super_wins}-{formula_wins}-{ties}  "
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
    print(f"  {table_label} wins: {super_wins} ({100*super_wins/num_games:.1f}%)")
    print(f"  {opp_label} wins:   {formula_wins} ({100*formula_wins/num_games:.1f}%)")
    print(f"  Ties:             {ties}")
    print(f"  Avg {table_label} score: {super_total/num_games:.1f}")
    print(f"  Avg {opp_label} score:   {formula_total/num_games:.1f}")
    avg_spread = sum(spreads) / len(spreads)
    print(f"  Avg spread ({table_label} - {opp_label}): {avg_spread:+.1f}")

    if super_wins > formula_wins:
        print(f"\n  --> {table_label} WINS by {super_wins - formula_wins} games")
    elif formula_wins > super_wins:
        print(f"\n  --> {opp_label} WINS by {formula_wins - super_wins} games")
    else:
        print(f"\n  --> TIE")

    return {
        'a_label': table_label, 'b_label': opp_label,
        'a_wins': super_wins, 'b_wins': formula_wins, 'ties': ties,
        'avg_spread': avg_spread, 'games': num_games,
        'elapsed': elapsed, 'gps': gps,
    }


def _resolve_path(path_str):
    """Resolve a table path: sentinels pass through, files resolved."""
    if path_str in ('__formula__', '__quackle__', '__research__',
                     'formula', 'quackle', 'research'):
        # Normalize short names to sentinel format
        if path_str == 'formula':
            return '__formula__'
        if path_str == 'quackle':
            return '__quackle__'
        if path_str == 'research':
            return '__research__'
        return path_str

    # Try as-is, then relative to superleaves dir
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str
    candidate = os.path.join(_superleaves_dir(), path_str)
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(path_str):
        return path_str
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Validate SuperLeaves table vs formula/quackle/other table'
    )
    parser.add_argument('--table', type=str, required=True,
                        help='Table A: .pkl path, "formula", or "quackle"')
    parser.add_argument('--games', type=int, default=1000,
                        help='Number of validation games (default: 1000)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of workers (default: cpu_count - 3)')
    parser.add_argument('--opponent', type=str, default=None,
                        help='Table B: .pkl path, "formula", or "quackle" '
                             '(default: formula)')
    args = parser.parse_args()

    table_path = _resolve_path(args.table)
    if table_path is None:
        print(f"Error: table not found: {args.table}")
        sys.exit(1)

    opponent_path = None
    if args.opponent:
        opponent_path = _resolve_path(args.opponent)
        if opponent_path is None:
            print(f"Error: opponent not found: {args.opponent}")
            sys.exit(1)

    validate(table_path, args.games, workers=args.workers,
             opponent_path=opponent_path)


def _superleaves_dir():
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    main()
