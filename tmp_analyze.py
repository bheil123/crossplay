"""Quick board analysis for Game 5 vs garnetgirl."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['PYTHONIOENCODING'] = 'utf-8'

from crossplay_v9.board import Board
from crossplay_v9.gaddag import get_gaddag
from crossplay_v9.move_finder_opt import find_all_moves_opt
from crossplay_v9.leave_eval import evaluate_leave
from crossplay_v9.config import TILE_DISTRIBUTION, VALID_TWO_LETTER
from crossplay_v9.dictionary import get_dictionary
from crossplay_v9.mc_eval import mc_evaluate_2ply
from crossplay_v9.mc_calibrate import calibrate, compute_adaptive_n
from collections import Counter
import time

# All moves in order (1-indexed coordinates, horizontal=True/False)
BOARD_MOVES = [
    ("DIZEN", 4, 8, False),
    ("BUSKER", 9, 6, True),
    ("TOEA", 8, 3, True),
    ("GAIT", 2, 7, False),
    ("YA", 10, 5, True),
    ("NAH", 5, 9, False),
    ("DUTY", 2, 10, False),
    ("CHICS", 5, 12, False),
    ("AWE", 10, 9, True),
    ("OBE", 11, 8, True),
    ("QI", 12, 7, True),
    ("PIGEON", 10, 4, False),
    ("LODGE", 12, 1, True),
    ("EX", 13, 4, True),
    ("JET", 7, 2, True),
    ("FEH", 7, 7, True),
    ("LOUVERS", 13, 8, True),  # Opp bingo (U is blank)
    ("PEAL", 12, 12, True),    # You played
    ("PERM", 12, 12, False),   # Opp played
    ("GEE", 6, 5, False),      # You played
]

BOARD_BLANKS = [(13, 10, 'U')]  # Blank U in LOUVERS
YOUR_RACK = "SREIFDI"
YOUR_SCORE = 342
OPP_SCORE = 287
BAG_SIZE = 15
N_MC = 15
K_MC = 150
MAX_WORKERS = 6


def main():
    print("Loading GADDAG...")
    gaddag = get_gaddag()
    dictionary = get_dictionary()
    print("GADDAG loaded.")

    board = Board()

    print("\nPlacing moves on board...")
    for word, row, col, horiz in BOARD_MOVES:
        try:
            board.place_word(word, row, col, horiz)
            dir_str = "H" if horiz else "V"
            print(f"  Placed {word} at R{row}C{col} {dir_str}")
        except Exception as e:
            print(f"  ERROR placing {word} at R{row}C{col}: {e}")

    # Print the board
    print("\nCurrent board:")
    grid = board._grid
    for r in range(15):
        row_str = ""
        for c in range(15):
            ch = grid[r][c]
            row_str += (ch if ch else ".") + " "
        print(f"  R{r+1:2d}: {row_str}")

    print(f"\nYour rack: {YOUR_RACK}")
    print(f"Score: You {YOUR_SCORE} - Opp {OPP_SCORE} (+{YOUR_SCORE - OPP_SCORE})")
    print(f"Bag: {BAG_SIZE}")

    # Compute unseen tiles
    full_dist = Counter()
    for tile, count in TILE_DISTRIBUTION.items():
        full_dist[tile] = count

    board_tiles = Counter()
    for r in range(15):
        for c in range(15):
            ch = grid[r][c]
            if ch:
                board_tiles[ch] += 1

    rack_tiles = Counter(YOUR_RACK.upper())

    unseen = Counter()
    for tile, count in full_dist.items():
        if tile == '?':
            unseen['?'] = 3 - 1  # 1 blank used on board
        else:
            remaining = count - board_tiles.get(tile, 0) - rack_tiles.get(tile, 0)
            if remaining > 0:
                unseen[tile] = remaining

    unseen_str = ''.join(tile * count for tile, count in sorted(unseen.items()))
    print(f"Unseen tiles ({len(unseen_str)}): {unseen_str}")
    expected_unseen = BAG_SIZE + 7
    print(f"Expected unseen: {expected_unseen}, Actual: {len(unseen_str)}")

    # Find all moves
    print("\n--- Finding all moves ---")
    t0 = time.time()
    all_moves = find_all_moves_opt(board, gaddag, YOUR_RACK, board_blanks=[])
    t1 = time.time()
    print(f"Found {len(all_moves)} moves in {t1-t0:.2f}s")

    # Add leave evaluation
    def get_leave(rack, move):
        rack_counter = Counter(rack.upper())
        word = move['word']
        r0 = move['row'] - 1
        c0 = move['col'] - 1
        is_h = move['direction'] == 'H'
        blanks_set = set(move.get('blanks_used', []))
        for i, ch in enumerate(word):
            cr, cc = (r0, c0 + i) if is_h else (r0 + i, c0)
            if grid[cr][cc] is None:
                if i in blanks_set:
                    if '?' in rack_counter and rack_counter['?'] > 0:
                        rack_counter['?'] -= 1
                    else:
                        rack_counter[ch] -= 1
                else:
                    rack_counter[ch] -= 1
        return ''.join(ch * cnt for ch, cnt in sorted(rack_counter.items()) if cnt > 0)

    for m in all_moves:
        leave = get_leave(YOUR_RACK, m)
        m['leave'] = leave
        m['leave_value'] = evaluate_leave(leave)
        m['equity'] = m['score'] + m['leave_value']
        used = []
        word = m['word']
        r0 = m['row'] - 1
        c0 = m['col'] - 1
        is_h = m['direction'] == 'H'
        for i, ch in enumerate(word):
            cr, cc = (r0, c0 + i) if is_h else (r0 + i, c0)
            if grid[cr][cc] is None:
                used.append(ch)
        m['tiles_used'] = ''.join(used)

    all_moves.sort(key=lambda m: -m['equity'])

    # Print top 25
    print(f"\n{'Rank':<5} {'Word':<12} {'Pos':<10} {'Score':>6} {'Leave':>8} {'LvVal':>7} {'Equity':>8}")
    print("-" * 60)
    for i, m in enumerate(all_moves[:25]):
        pos = f"R{m['row']}C{m['col']}{m['direction']}"
        print(f"{i+1:<5} {m['word']:<12} {pos:<10} {m['score']:>6} {m['leave']:>8} {m['leave_value']:>7.1f} {m['equity']:>8.1f}")

    # Run MC 2-ply
    print(f"\n--- MC 2-ply Evaluation ({MAX_WORKERS} workers) ---")
    candidates = all_moves[:N_MC]

    print(f"Running MC 2-ply: {len(candidates)} candidates x {K_MC} sims...")
    sys.stdout.flush()
    t0 = time.time()
    mc_results = mc_evaluate_2ply(
        board, YOUR_RACK, unseen_str,
        board_moves=BOARD_MOVES,
        gaddag=gaddag,
        top_n=N_MC,
        k_sims=K_MC,
        board_blanks=BOARD_BLANKS,
        max_workers=MAX_WORKERS,
        pre_ranked_candidates=candidates,
    )
    t1 = time.time()
    print(f"MC completed in {t1-t0:.1f}s")

    # Print MC results
    print(f"\n{'Rank':<5} {'Word':<12} {'Pos':<10} {'Score':>6} {'AvgOpp':>7} {'MCEq':>7} {'Leave':>7} {'TotEq':>8}")
    print("-" * 70)
    for i, m in enumerate(mc_results[:20]):
        pos = f"R{m['row']}C{m['col']}{m['direction']}"
        print(f"{i+1:<5} {m['word']:<12} {pos:<10} {m['score']:>6} {m['mc_avg_opp']:>7.1f} {m['mc_equity']:>7.1f} {m['leave_value']:>7.1f} {m['total_equity']:>8.1f}")

    # Show top 3 detailed
    print("\n=== TOP 3 RECOMMENDATIONS ===")
    for i, m in enumerate(mc_results[:3]):
        pos = f"R{m['row']}C{m['col']} {m['direction']}"
        print(f"\n#{i+1}: {m['word']} at {pos}")
        print(f"  Score: {m['score']}  |  Avg Opp Response: {m['mc_avg_opp']:.1f}")
        print(f"  MC Equity: {m['mc_equity']:.1f}  |  Leave: {m['leave']} ({m['leave_value']:.1f})")
        print(f"  Total Equity: {m['total_equity']:.1f}")
        if m.get('top_opp_responses'):
            print(f"  Top opp threats:")
            for resp in m['top_opp_responses'][:3]:
                print(f"    {resp['word']} R{resp['row']}C{resp['col']}{resp['direction']} = {resp['score']}")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
