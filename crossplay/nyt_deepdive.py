"""Deep dive diagnostic: Run full MC analysis on positions where engine disagrees with NYT.

Reconstructs board positions from archive data and runs the complete
analyze() pipeline (1-ply + risk + MC 2-ply) to compare engine vs NYT
recommendations.

Usage:
    python -m crossplay.nyt_deepdive [--position N]

    Without args: runs all positions sequentially
    --position N: run only position N (1-indexed)
"""

import json
import sys
import io
import os
import time
from datetime import datetime


def load_archive():
    """Load all archive records keyed by game_id."""
    archive_path = os.path.join(os.path.dirname(__file__), 'games', 'archive.jsonl')
    records = {}
    with open(archive_path) as f:
        for line in f:
            rec = json.loads(line)
            records[rec['game_id']] = rec
    return records


def reconstruct_position(record, turn_num):
    """Reconstruct game state just before turn_num (1-indexed).

    Returns (GameState, move_dict) where move_dict is the move that was played.
    """
    from .game_manager import GameState

    moves = record['moves']
    move = moves[turn_num - 1]
    board_moves = moves[:turn_num - 1]

    if turn_num >= 2:
        prev = moves[turn_num - 2]
        your_score = prev['cumulative'][0]
        opp_score = prev['cumulative'][1]
    else:
        your_score = 0
        opp_score = 0

    rack = move.get('rack')
    all_blanks = record.get('blank_positions', [])

    state = GameState(
        name="Replay %s T%d" % (record['game_id'], turn_num),
        board_moves=board_moves,
        blank_positions=all_blanks,
        your_score=your_score,
        opp_score=opp_score,
        your_rack=rack or "",
        is_your_turn=(move.get('player') == 'me'),
        opponent_name=record.get('opponent', ''),
        created_at=record.get('created_at', ''),
        updated_at='',
    )
    return state, move


# Positions to investigate
POSITIONS = [
    # Leave overvaluation: engine picks lower score + better leave
    ('piph_001', 14, 'Engine JO(26/34eq) vs NYT EMOJI(30)', 'leave'),
    ('sophie_001', 15, 'Engine ALL(16/30.6eq) vs NYT LITS(27)', 'leave'),
    ('pkp8_002', 9, 'Engine FER(6/24.7eq) vs NYT SYN(24)', 'leave'),
    ('pkp8_002', 7, 'Engine NEW(7/20.3eq) vs NYT SYN(24)', 'leave'),
    ('sophie_001', 24, 'Engine JO(25/23.5eq) vs NYT TWO(27)', 'leave'),
    ('katie_001', 19, 'Engine WOODIE(32/32eq) vs NYT WED(39)', 'leave'),
    # Exchange positions
    ('piph_001', 8, 'Swapped IUY vs NYT SAIYID(32)', 'exchange'),
    ('katie_001', 11, 'Swapped vs NYT PE(20)', 'exchange'),
]


def run_position(pos_idx, records):
    """Run full analyze() on a single position."""
    from .game_manager import Game

    gid, turn, desc, cat = POSITIONS[pos_idx]
    rec = records[gid]
    state, move = reconstruct_position(rec, turn)

    played_word = move.get('word', '?')
    played_score = move.get('score', 0)
    nyt = move.get('nyt', {})
    nyt_word = nyt.get('word', '?') if nyt else '?'
    nyt_score = nyt.get('score', 0) if nyt else 0

    print()
    print("=" * 70)
    print("POSITION %d/%d: %s T%d [%s]" % (pos_idx + 1, len(POSITIONS), gid, turn, cat))
    print("  %s" % desc)
    print("  Played: %s(%d)  NYT best: %s(%d)  Rack: %s" % (
        played_word, played_score, nyt_word, nyt_score, state.your_rack))
    print("  Scores: You %d - Opp %d  Bag: ~%d" % (
        state.your_score, state.opp_score, move.get('bag', 0)))
    if move.get('is_exchange'):
        print("  Exchange: dumped %s, kept %s" % (
            move.get('exchange_dump', '?'), move.get('exchange_keep', '?')))
    print("=" * 70)

    # Create game (suppress init output)
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    game = Game(state)
    sys.stdout = old_stdout

    print("  Board tiles: %d  Bag: %d tiles" % (
        sum(1 for r in range(15) for c in range(15) if game.board.get_tile(r+1, c+1)),
        len(game.bag)))
    print()

    # Run full analysis
    t0 = time.time()
    result = game.analyze(rack=state.your_rack, top_n=10)
    elapsed = time.time() - t0

    print()
    print("  [Position %d complete in %.1fs]" % (pos_idx + 1, elapsed))
    print()

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Deep dive on engine vs NYT disagreements')
    parser.add_argument('--position', type=int, default=None,
                        help='Run only position N (1-indexed)')
    args = parser.parse_args()

    records = load_archive()

    if args.position:
        idx = args.position - 1
        if idx < 0 or idx >= len(POSITIONS):
            print("Position must be 1-%d" % len(POSITIONS))
            return
        run_position(idx, records)
    else:
        print("NYT DEEP DIVE ANALYSIS")
        print("Running full MC 2-ply on %d positions" % len(POSITIONS))
        print("This will take several minutes...")
        print()

        t_total = time.time()
        for i in range(len(POSITIONS)):
            run_position(i, records)

        elapsed = time.time() - t_total
        print()
        print("=" * 70)
        print("ALL %d POSITIONS COMPLETE in %.1fs (%.1f min)" % (
            len(POSITIONS), elapsed, elapsed / 60))
        print("=" * 70)


if __name__ == '__main__':
    main()
