"""
CROSSPLAY V21.4 - Leave Evaluation Module
Evaluate the quality of tiles remaining after a play.

V21.4: Denoised SuperLeaves deployed as default. Rebased gen4 interactions
onto Quackle per-tile scale with James-Stein shrinkage (medium). Validated
at 10K games: +5.9 vs Quackle, +7.8 vs formula (greedy bots).

V21.3: Added 'quackle' leave mode -- Quackle-derived per-tile values
adapted for Crossplay tile values, with bag decay factor.

V21.2: Env-var-controlled leave mode (CROSSPLAY_LEAVES):
  superleaves -- trained table -> formula fallback -> bingo bonus (V21.4 default)
  formula     -- hand-tuned per-tile formula (V21 default)
  blend       -- alpha * formula + (1-alpha) * superleaves
  quackle     -- Quackle-derived per-tile values with bag decay

  CROSSPLAY_BLEND_ALPHA -- blend weight (0.0=pure SL, 1.0=pure formula,
  default 0.5). Only used when CROSSPLAY_LEAVES=blend.

V21: Formula-only mode validated via DadBot v5/v6 tournament testing.
Formula dominates SuperLeaves and bingo at all compute levels.

V9 upgrade: Integrates precomputed bingo probability database.
Each leave value = base formula + bingo equity bonus.
"""

import os
import pickle
from typing import List, Tuple
from collections import Counter
from .config import (
    TILE_VALUES, PREMIUM_TILES, GOOD_TILES, AWKWARD_TILES,
    IDEAL_VOWEL_MIN, IDEAL_VOWEL_MAX, IDEAL_CONSONANT_MIN, IDEAL_CONSONANT_MAX,
    DUPLICATE_PENALTY, BINGO_STEMS
)


# Leave mode: 'superleaves' (default), 'formula', 'blend', or 'quackle'
LEAVE_MODE = os.environ.get('CROSSPLAY_LEAVES', 'superleaves')
BLEND_ALPHA = float(os.environ.get('CROSSPLAY_BLEND_ALPHA', '0.5'))

# Legacy compatibility
USE_FORMULA_ONLY = (LEAVE_MODE == 'formula')

VOWELS = set('AEIOU')
CONSONANTS = set('BCDFGHJKLMNPQRSTVWXYZ')

# Quackle-derived per-tile leave values, adapted for Crossplay tile values.
# Source: Quackle/Maven research, adjusted for Crossplay differences:
#   - Sweep=40 (not 50): blank 25.57->20.0, S 8.04->7.0
#   - Different tile values: L=2, K=6, G=4, J=10, B=4 -> less negative
#   - V=6, W=5 -> more negative (high face value, hard to play)
QUACKLE_TILE_VALUES = {
    '?': 20.0, 'S': 7.0, 'Z': 5.12, 'X': 3.31, 'R': 1.10,
    'C': 0.85, 'H': 0.60, 'M': 0.58, 'D': 0.45, 'E': 0.35,
    'N': 0.22, 'L': 0.20, 'T': -0.10, 'P': -0.46, 'K': -0.20,
    'Y': -0.63, 'A': -0.63, 'J': -0.80, 'B': -1.50, 'I': -2.07,
    'F': -2.21, 'O': -2.50, 'G': -1.80, 'W': -4.50, 'U': -4.00,
    'V': -6.50, 'Q': -6.79,
}


def _leave_decay(bag_size):
    """Scale leave value down as bag empties (Crossplay has no tile penalty)."""
    if bag_size is None or bag_size >= 30:
        return 1.0
    elif bag_size >= 15:
        return 0.70
    elif bag_size >= 7:
        return 0.40
    else:
        return 0.10


def _quackle_evaluate(leave, bag_size=None):
    """Quackle-derived per-tile evaluation with bag decay."""
    if not leave:
        return 0.0
    leave = leave.upper()
    value = sum(QUACKLE_TILE_VALUES.get(t, -1.0) for t in leave)
    # Vowel/consonant balance bonus
    vowels = sum(1 for t in leave if t in VOWELS)
    consonants = sum(1 for t in leave if t.isalpha() and t not in VOWELS and t != '?')
    if len(leave) >= 2:
        if vowels == 1 and consonants >= 1:
            value += 2.0
        elif vowels >= 2 and consonants == 0:
            value -= 5.0
    # Q without U penalty (can't check unseen here, just penalize Q in leave)
    if 'Q' in leave:
        value -= 4.0  # Partial Q-without-U penalty
    # Bag decay
    value *= _leave_decay(bag_size)
    return value

# --- SuperLeaves table (lazy-loaded) ---
_SUPERLEAVES = None
_SUPERLEAVES_CHECKED = False


def _load_superleaves():
    """Try to load trained SuperLeaves table (lazy, once)."""
    global _SUPERLEAVES, _SUPERLEAVES_CHECKED
    if _SUPERLEAVES_CHECKED:
        return
    _SUPERLEAVES_CHECKED = True
    sl_path = os.path.join(os.path.dirname(__file__), 'superleaves',
                           'deployed_leaves.pkl')
    if os.path.exists(sl_path):
        try:
            from .superleaves.leave_table import LeaveTable
            _SUPERLEAVES = LeaveTable.load(sl_path)
        except Exception:
            _SUPERLEAVES = None

# --- Bingo probability database ---
# Expected bingo score: 40pt bonus + ~37pt average word = ~77pts (Heuri research)
EXPECTED_BINGO_SCORE = 77.0
# Weight for bingo equity contribution (tune via self-play; 0.5 = conservative start)
BINGO_WEIGHT = 0.5

_BINGO_DB = None

def _load_bingo_db():
    """Load bingo probability database (lazy, once)."""
    global _BINGO_DB
    if _BINGO_DB is not None:
        return
    db_path = os.path.join(os.path.dirname(__file__), 'leave_bingo_prod.pkl')
    if os.path.exists(db_path):
        with open(db_path, 'rb') as f:
            _BINGO_DB = pickle.load(f)
    else:
        _BINGO_DB = {}  # Graceful fallback — formula only


def get_bingo_prob(leave_key):
    """
    Look up bingo probability for a sorted leave tuple.
    Returns 0.0 for missing keys (zero-probability leaves were stripped).
    """
    _load_bingo_db()
    return _BINGO_DB.get(leave_key, 0.0)


def count_vowels(tiles: str) -> int:
    """Count vowels in a tile string."""
    return sum(1 for t in tiles.upper() if t in VOWELS)


def count_consonants(tiles: str) -> int:
    """Count consonants in a tile string."""
    return sum(1 for t in tiles.upper() if t in CONSONANTS)


def _formula_evaluate(leave: str, bag_empty: bool = False) -> float:
    """
    Original hand-tuned leave evaluation formula.
    Retained as the base component of the composite evaluation.
    """
    if not leave:
        return 0.0

    leave = leave.upper()
    tiles = list(leave)
    counter = Counter(tiles)

    value = 0.0

    # 1. Premium tile bonus
    for tile in tiles:
        if tile in PREMIUM_TILES:
            value += 8.0 if tile == '?' else 6.0
        elif tile in GOOD_TILES:
            value += 1.5
        elif tile in AWKWARD_TILES:
            value -= 3.0

    # 2. Vowel/consonant balance
    vowels = count_vowels(leave)
    consonants = count_consonants(leave)
    total = len(tiles)

    if total >= 3:
        if vowels == 0:
            value -= 8.0
        elif vowels > IDEAL_VOWEL_MAX:
            value -= (vowels - IDEAL_VOWEL_MAX) * 2.0
        elif vowels < IDEAL_VOWEL_MIN and total >= 4:
            value -= (IDEAL_VOWEL_MIN - vowels) * 2.0

        if consonants == 0 and total >= 3:
            value -= 5.0

    # 3. Duplicate penalty
    for tile, count in counter.items():
        if count > 1 and tile != '?':
            value -= DUPLICATE_PENALTY * (count - 1)

    # 4. Synergy bonus (partial bingo stems)
    leave_set = set(tiles)
    for stem, bonus in BINGO_STEMS.items():
        matching = sum(1 for c in stem if c in leave_set)
        if matching >= 4:
            value += bonus * (matching / len(stem))

    # 5. Endgame adjustments
    if bag_empty:
        vowel_count = sum(1 for t in tiles if t in 'AEIOU')
        consonant_count = len(tiles) - vowel_count - tiles.count('?')
        if len(tiles) >= 2:
            if vowel_count == 0 and consonant_count >= 2:
                value -= 1.0
            if consonant_count == 0 and vowel_count >= 2:
                value -= 1.0

    return value


def evaluate_leave(leave: str, bag_empty: bool = False,
                   bag_size: int = None) -> float:
    """
    Evaluate the quality of a leave (tiles remaining after a play).

    Args:
        leave: Tiles remaining in rack (e.g., 'AERT')
        bag_empty: True if bag is empty (endgame -- bingo bonus disabled)
        bag_size: Tiles remaining in bag (optional, used by quackle mode
                  for decay calculation)

    Returns:
        Float value representing leave quality (higher = better)
    """
    if not leave:
        return 0.0

    # Quackle mode: per-tile values with bag decay
    if LEAVE_MODE == 'quackle':
        if bag_size is None and bag_empty:
            bag_size = 0
        return _quackle_evaluate(leave, bag_size)

    # Formula mode (V21 default, proven best in tournament testing)
    if LEAVE_MODE == 'formula':
        return _formula_evaluate(leave, bag_empty)

    # --- SuperLeaves value (used by both 'superleaves' and 'blend' modes) ---
    sl_val = None
    if not bag_empty:
        _load_superleaves()
        if _SUPERLEAVES is not None:
            leave_key = tuple(sorted(leave.upper()))
            if leave_key in _SUPERLEAVES:
                sl_val = _SUPERLEAVES.get(leave_key)

    # Fallback: formula + bingo bonus
    if sl_val is None:
        sl_val = _formula_evaluate(leave, bag_empty)
        if not bag_empty and len(leave) < 7:
            leave_key = tuple(sorted(leave.upper()))
            bingo_prob = get_bingo_prob(leave_key)
            sl_val += BINGO_WEIGHT * bingo_prob * EXPECTED_BINGO_SCORE

    if LEAVE_MODE == 'superleaves':
        return sl_val

    # Blend mode: alpha * formula + (1-alpha) * superleaves
    formula_val = _formula_evaluate(leave, bag_empty)
    return BLEND_ALPHA * formula_val + (1.0 - BLEND_ALPHA) * sl_val


def evaluate_leave_detailed(leave: str, bag_empty: bool = False) -> Tuple[float, List[str]]:
    """
    Evaluate leave with detailed breakdown.

    Returns:
        Tuple of (value, list of reasons)
    """
    if not leave:
        return (0.0, ["Empty leave"])

    leave = leave.upper()
    tiles = list(leave)
    counter = Counter(tiles)

    value = 0.0
    reasons = []

    # Check composition
    vowels = count_vowels(leave)
    consonants = count_consonants(leave)
    blanks = counter.get('?', 0)

    # Premium tiles
    if blanks > 0:
        value += blanks * 8.0
        reasons.append(f"+{blanks} blank(s)")

    s_count = counter.get('S', 0)
    if s_count > 0:
        value += s_count * 6.0
        reasons.append(f"+{s_count} S")

    # Good tiles
    good_count = sum(1 for t in tiles if t in GOOD_TILES and t != 'S')
    if good_count > 0:
        value += good_count * 1.5
        reasons.append(f"+{good_count} good tiles ({','.join(t for t in tiles if t in GOOD_TILES and t != 'S')})")

    # Awkward tiles
    awkward = [t for t in tiles if t in AWKWARD_TILES]
    if awkward:
        value -= len(awkward) * 3.0
        reasons.append(f"-{len(awkward)} awkward ({','.join(awkward)})")

    # Balance
    if vowels == 0 and len(tiles) >= 2:
        value -= 8.0
        reasons.append("No vowels!")
    elif vowels > IDEAL_VOWEL_MAX:
        penalty = (vowels - IDEAL_VOWEL_MAX) * 2.0
        value -= penalty
        reasons.append(f"Too many vowels ({vowels})")

    # Duplicates
    for tile, count in counter.items():
        if count > 1 and tile != '?':
            penalty = DUPLICATE_PENALTY * (count - 1)
            value -= penalty
            reasons.append(f"Duplicate {tile}x{count}")

    # Bingo bonus
    if not bag_empty and len(leave) < 7:
        leave_key = tuple(sorted(leave))
        bingo_prob = get_bingo_prob(leave_key)
        if bingo_prob > 0:
            bingo_bonus = BINGO_WEIGHT * bingo_prob * EXPECTED_BINGO_SCORE
            value += bingo_bonus
            reasons.append(f"Bingo {bingo_prob*100:.1f}% -> +{bingo_bonus:.1f}")

    return (value, reasons)


def rate_rack_quality(rack: str) -> Tuple[int, str, List[str]]:
    """
    Rate overall rack quality on a 1-5 scale.

    Returns:
        Tuple of (rating 1-5, label, reasons)
    """
    rack = rack.upper()

    vowels = count_vowels(rack)
    consonants = count_consonants(rack)
    counter = Counter(rack)
    blanks = counter.get('?', 0)

    reasons = []
    score = 3  # Start at average

    # Vowel/consonant balance
    if len(rack) >= 5:
        if vowels == 0:
            score -= 2
            reasons.append("No vowels")
        elif vowels == 1:
            score -= 1
            reasons.append("Only 1 vowel")
        elif vowels >= 5:
            score -= 1
            reasons.append("Too many vowels")

        if consonants <= 1:
            score -= 1
            reasons.append("Too few consonants")

    # Duplicates
    max_dup = max(counter.values())
    if max_dup >= 3:
        score -= 1
        dup_tile = [t for t, c in counter.items() if c >= 3][0]
        reasons.append(f"Triple {dup_tile}")
    elif max_dup == 2:
        dup_count = sum(1 for c in counter.values() if c == 2)
        if dup_count >= 2:
            score -= 1
            reasons.append("Multiple duplicates")

    # Premium tiles
    if blanks > 0:
        score += 1
        reasons.append(f"Has blank")
    if 'S' in rack:
        score += 0.5
        reasons.append("Has S")

    # Awkward tiles
    awkward = sum(1 for t in rack if t in AWKWARD_TILES)
    if awkward >= 2:
        score -= 1
        reasons.append(f"{awkward} awkward tiles")

    # Bingo potential (enhanced with DB)
    leave_key = tuple(sorted(rack))
    bingo_prob = get_bingo_prob(leave_key)
    if bingo_prob > 0.3:
        score += 1
        reasons.append(f"Bingo potential ({bingo_prob*100:.0f}%)")
    elif bingo_prob > 0.15:
        score += 0.5
        reasons.append(f"Some bingo chance ({bingo_prob*100:.0f}%)")
    else:
        # Fallback to old heuristic if no DB data
        good_tiles = sum(1 for t in rack if t in GOOD_TILES)
        if good_tiles >= 5 and 2 <= vowels <= 3:
            score += 1
            reasons.append("Bingo potential")

    # Clamp and label
    score = max(1, min(5, int(score)))
    labels = {1: 'TERRIBLE', 2: 'WEAK', 3: 'MEH', 4: 'GOOD', 5: 'AMAZING'}

    return (score, labels[score], reasons)


if __name__ == "__main__":
    # Test leave evaluation with bingo DB
    test_leaves = ['AEINST', 'RR', 'QVJ', 'S?', 'AEIO', 'BCDKM', '', 'ERST', 'ER', 'Q']

    print("LEAVE EVALUATION TESTS (v9 -- Bingo-Aware)")
    print("=" * 60)

    for leave in test_leaves:
        value_new = evaluate_leave(leave)
        value_old = _formula_evaluate(leave)
        _, reasons = evaluate_leave_detailed(leave)
        delta = value_new - value_old
        print(f"\n{leave or '(empty)'}:")
        print(f"  Formula: {value_old:+.1f}  |  With bingo: {value_new:+.1f}  |  Delta = {delta:+.1f}")
        for r in reasons:
            print(f"    - {r}")

    print("\n" + "=" * 60)
    print("RACK QUALITY TESTS")
    print("=" * 60)

    test_racks = ['AEINRST', 'QVJXZZK', 'AAAAAEI', 'BCDFGHJ', 'RETINAS']
    for rack in test_racks:
        rating, label, reasons = rate_rack_quality(rack)
        print(f"\n{rack}: {rating}/5 ({label})")
        for r in reasons:
            print(f"    - {r}")
