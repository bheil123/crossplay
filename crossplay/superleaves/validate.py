"""
CROSSPLAY V15 - SuperLeaves Validation

Head-to-head: SuperLeaves bot vs formula bot.
Alternates first player. Reports win rate, avg score, spread.

Usage:
    python -m crossplay.superleaves.validate --table gen1_100000.pkl --games 1000
"""

import os
import sys
import random
import argparse
from ..board import Board
from ..config import TILE_DISTRIBUTION, RACK_SIZE
from .fast_bot import select_best_move
from .leave_table import LeaveTable


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


def validate(table_path, num_games, workers=1):
    """Run head-to-head validation.

    Args:
        table_path: path to trained SuperLeaves table
        num_games: number of games to play
        workers: (unused, runs single-threaded for simplicity)
    """
    from ..gaddag import get_gaddag
    from ..move_finder_gaddag import GADDAGMoveFinder

    print(f"Loading GADDAG...")
    gaddag = get_gaddag()

    print(f"Loading SuperLeaves table: {table_path}")
    super_table = LeaveTable.load(table_path)
    print(f"  Table size: {len(super_table):,}")

    formula_table = FormulaLeaveAdapter()

    print(f"\nValidation: SuperLeaves vs Formula ({num_games:,} games)")
    print(f"  Alternating first player each game")
    print("="*60)

    super_wins = 0
    formula_wins = 0
    ties = 0
    super_total = 0
    formula_total = 0
    spreads = []

    for game_num in range(1, num_games + 1):
        # Alternate who goes first
        if game_num % 2 == 1:
            s_a, s_b = play_validation_game(
                gaddag, GADDAGMoveFinder, super_table, formula_table
            )
            super_score, formula_score = s_a, s_b
        else:
            s_a, s_b = play_validation_game(
                gaddag, GADDAGMoveFinder, formula_table, super_table
            )
            formula_score, super_score = s_a, s_b

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

        if game_num % 100 == 0 or game_num == num_games:
            win_pct = 100 * super_wins / game_num
            avg_spread = sum(spreads) / len(spreads)
            print(
                f"  [{game_num:,}/{num_games:,}] "
                f"SuperLeaves {super_wins}-{formula_wins}-{ties}  "
                f"({win_pct:.1f}%)  "
                f"avg spread: {avg_spread:+.1f}"
            )

    print(f"\n{'='*60}")
    print(f"RESULTS ({num_games:,} games)")
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

    validate(table_path, args.games)


def _superleaves_dir():
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    main()
