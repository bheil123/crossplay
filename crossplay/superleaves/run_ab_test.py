r"""
SuperLeaves A/B Test Suite

Runs multiple head-to-head matchups and collects results in a summary table.

Usage:
    cd C:\Users\billh\crossplay
    python -m crossplay.superleaves.run_ab_test --games 10000
    python -m crossplay.superleaves.run_ab_test --games 10000 --workers 9
"""

import os
import sys
import time
import argparse
import math


def _superleaves_dir():
    return os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description='SuperLeaves A/B Test Suite')
    parser.add_argument('--games', type=int, default=10000,
                        help='Games per matchup (default: 10000)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Workers (default: cpu_count - 3)')
    args = parser.parse_args()

    sl_dir = _superleaves_dir()
    gen4_path = os.path.join(sl_dir, 'gen4_3000000.pkl')
    gen3_path = os.path.join(sl_dir, 'gen3_1000000.pkl')

    # Verify files exist
    for path, name in [(gen4_path, 'gen4'), (gen3_path, 'gen3')]:
        if not os.path.exists(path):
            print(f"Error: {name} not found at {path}")
            sys.exit(1)

    matchups = [
        ('Gen4 vs Formula',  gen4_path,      '__formula__'),
        ('Gen4 vs Gen3',     gen4_path,      gen3_path),
        ('Gen4 vs Quackle',  gen4_path,      '__quackle__'),
        ('Quackle vs Formula', '__quackle__', '__formula__'),
        ('Gen3 vs Formula',  gen3_path,      '__formula__'),
        ('Gen3 vs Quackle',  gen3_path,      '__quackle__'),
    ]

    print("=" * 70)
    print(f"  SuperLeaves A/B Test Suite")
    print(f"  {len(matchups)} matchups x {args.games:,} games each")
    print(f"  Workers: {args.workers or 'auto'}")
    print("=" * 70)

    from .validate import validate

    results = []
    t_total = time.time()

    for i, (name, table_a, table_b) in enumerate(matchups, 1):
        print(f"\n{'#' * 70}")
        print(f"  MATCHUP {i}/{len(matchups)}: {name}")
        print(f"{'#' * 70}")

        result = validate(table_a, args.games, workers=args.workers,
                          opponent_path=table_b)
        result['name'] = name
        results.append(result)

    # Summary table
    total_elapsed = time.time() - t_total
    print(f"\n\n{'=' * 70}")
    print(f"  SUMMARY  ({args.games:,} games each, {total_elapsed:.0f}s total)")
    print(f"{'=' * 70}")
    print(f"{'Matchup':<25} {'W-L-T':>12} {'Spread':>8} {'Winner':>12}")
    print("-" * 60)

    for r in results:
        wlt = f"{r['a_wins']}-{r['b_wins']}-{r['ties']}"
        spread = f"{r['avg_spread']:+.1f}"
        # Determine winner with significance
        n = r['games']
        se = math.sqrt(sum((s - r['avg_spread'])**2 for s in [r['avg_spread']])) if n > 0 else 0
        if r['a_wins'] > r['b_wins']:
            winner = r['a_label']
        elif r['b_wins'] > r['a_wins']:
            winner = r['b_label']
        else:
            winner = 'TIE'
        sig = '*' if abs(r['avg_spread']) > 3.0 else ''
        print(f"{r['name']:<25} {wlt:>12} {spread:>8} {winner + sig:>12}")

    print(f"\n  * = avg spread > 3.0 pts/game")
    print(f"  Total time: {total_elapsed/60:.1f} min")


if __name__ == '__main__':
    main()
