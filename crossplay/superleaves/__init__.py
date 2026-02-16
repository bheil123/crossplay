"""
CROSSPLAY V14 - SuperLeaves Training Pipeline

Empirical leave value table trained via self-play.
Replaces hand-tuned formula with ~921K lookup entries.

Usage:
    python -m crossplay.superleaves.trainer --smoke-test --workers 4
    python -m crossplay.superleaves.trainer --workers 6
    python -m crossplay.superleaves.trainer --resume --workers 6
    python -m crossplay.superleaves.validate --table gen1_100000.pkl --games 1000
"""
