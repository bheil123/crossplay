"""
CROSSPLAY V15 - Configuration Constants
All magic numbers and configuration in one place.
"""

from typing import Dict, List, Tuple

# =============================================================================
# TILE VALUES (Crossplay specific - differs from Scrabble!)
# =============================================================================

TILE_VALUES: Dict[str, int] = {
    'A': 1, 'B': 4, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 4, 'H': 3,
    'I': 1, 'J': 10, 'K': 6, 'L': 2, 'M': 3, 'N': 1, 'O': 1, 'P': 3,
    'Q': 10, 'R': 1, 'S': 1, 'T': 1, 'U': 2, 'V': 6, 'W': 5, 'X': 8,
    'Y': 4, 'Z': 10, '?': 0  # Blank
}

# =============================================================================
# TILE DISTRIBUTION (Crossplay has 3 blanks, different from Scrabble's 2)
# =============================================================================

TILE_DISTRIBUTION: Dict[str, int] = {
    'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12, 'F': 2, 'G': 3, 'H': 3,
    'I': 8, 'J': 1, 'K': 1, 'L': 4, 'M': 2, 'N': 5, 'O': 8, 'P': 2,
    'Q': 1, 'R': 6, 'S': 5, 'T': 6, 'U': 3, 'V': 2, 'W': 2, 'X': 1,
    'Y': 2, 'Z': 1, '?': 3  # 100 tiles total (Crossplay distribution)
}

TOTAL_TILES: int = sum(TILE_DISTRIBUTION.values())  # 100

# =============================================================================
# BOARD CONFIGURATION
# =============================================================================

BOARD_SIZE: int = 15
CENTER_ROW: int = 8  # 1-indexed
CENTER_COL: int = 8  # 1-indexed

# Bonus squares (all 1-indexed for consistency)
# Format: (row, col) -> bonus_type

BONUS_SQUARES: Dict[Tuple[int, int], str] = {
    # Triple Letter (3L) - corners and scattered
    (1, 1): '3L', (1, 15): '3L', (15, 1): '3L', (15, 15): '3L',
    (2, 7): '3L', (2, 9): '3L', (14, 7): '3L', (14, 9): '3L',
    (7, 2): '3L', (9, 2): '3L', (7, 14): '3L', (9, 14): '3L',
    (5, 6): '3L', (5, 10): '3L', (11, 6): '3L', (11, 10): '3L',
    (6, 5): '3L', (6, 11): '3L', (10, 5): '3L', (10, 11): '3L',
    
    # Double Letter (2L)
    (1, 8): '2L', (15, 8): '2L', (8, 1): '2L', (8, 15): '2L',
    (3, 5): '2L', (3, 11): '2L', (13, 5): '2L', (13, 11): '2L',
    (5, 3): '2L', (5, 13): '2L', (11, 3): '2L', (11, 13): '2L',
    (4, 4): '2L', (4, 12): '2L', (12, 4): '2L', (12, 12): '2L',
    (6, 8): '2L', (10, 8): '2L', (8, 6): '2L', (8, 10): '2L',
    
    # Triple Word (3W)
    (1, 4): '3W', (1, 12): '3W', (15, 4): '3W', (15, 12): '3W',
    (4, 1): '3W', (4, 15): '3W', (12, 1): '3W', (12, 15): '3W',
    
    # Double Word (2W)
    (2, 2): '2W', (2, 14): '2W', (14, 2): '2W', (14, 14): '2W',
    (4, 8): '2W', (12, 8): '2W', (8, 4): '2W', (8, 12): '2W',
    
    # Center square - NO BONUS in Crossplay (confirmed by HIKE=17, FRAUD=22)
    # (8, 8): None  # Explicitly no bonus
}

# Extract specific bonus types for quick lookup
TRIPLE_WORD_SQUARES: List[Tuple[int, int]] = [
    pos for pos, bonus in BONUS_SQUARES.items() if bonus == '3W'
]

DOUBLE_WORD_SQUARES: List[Tuple[int, int]] = [
    pos for pos, bonus in BONUS_SQUARES.items() if bonus == '2W'
]

TRIPLE_LETTER_SQUARES: List[Tuple[int, int]] = [
    pos for pos, bonus in BONUS_SQUARES.items() if bonus == '3L'
]

DOUBLE_LETTER_SQUARES: List[Tuple[int, int]] = [
    pos for pos, bonus in BONUS_SQUARES.items() if bonus == '2L'
]

# =============================================================================
# SCORING CONFIGURATION
# =============================================================================

BINGO_BONUS: int = 40  # Bonus for using all 7 tiles (Crossplay uses 40, not 50)
RACK_SIZE: int = 7


# V21.1: Risk assessment, threat analysis, opening heuristics, and blocking
# bonus constants removed (proven unhelpful in DadBot v5 tournament testing).

# =============================================================================
# LEAVE EVALUATION CONFIGURATION
# =============================================================================

# Vowel/consonant balance targets
IDEAL_VOWEL_MIN: int = 2
IDEAL_VOWEL_MAX: int = 3
IDEAL_CONSONANT_MIN: int = 3
IDEAL_CONSONANT_MAX: int = 4

# Tile quality ratings for leave evaluation
PREMIUM_TILES: str = 'S?'  # S and blank are most valuable
GOOD_TILES: str = 'ERATIN'  # Common bingo-friendly tiles
AWKWARD_TILES: str = 'QVJXZ'  # Hard to use
DUPLICATE_PENALTY: float = 3.0  # Penalty per duplicate tile

# Bingo stems (high-probability 7-letter combinations)
BINGO_STEMS: Dict[str, float] = {
    'AEINST': 10.0,
    'AEIRST': 9.0,
    'AEINRS': 8.0,
    'EINRST': 8.0,
    'AEIRST': 7.0,
}

# =============================================================================
# VALID 2-LETTER WORDS (for hook validation)
# =============================================================================

VALID_TWO_LETTER: set = {
    'AA', 'AB', 'AD', 'AE', 'AG', 'AH', 'AI', 'AL', 'AM', 'AN', 'AR', 'AS', 'AT', 'AW', 'AX', 'AY',
    'BA', 'BE', 'BI', 'BO', 'BY',
    'DA', 'DE', 'DO',
    'ED', 'EF', 'EH', 'EL', 'EM', 'EN', 'ER', 'ES', 'ET', 'EW', 'EX',
    'FA', 'FE',
    'GI', 'GO',
    'HA', 'HE', 'HI', 'HM', 'HO',
    'ID', 'IF', 'IN', 'IS', 'IT',
    'JO',
    'KA', 'KI',
    'LA', 'LI', 'LO',
    'MA', 'ME', 'MI', 'MM', 'MO', 'MU', 'MY',
    'NA', 'NE', 'NO', 'NU',
    'OD', 'OE', 'OF', 'OH', 'OI', 'OK', 'OM', 'ON', 'OP', 'OR', 'OS', 'OW', 'OX', 'OY',
    'PA', 'PE', 'PI', 'PO',
    'QI',
    'RE',
    'SH', 'SI', 'SO',
    'TA', 'TE', 'TI', 'TO',
    'UH', 'UM', 'UN', 'UP', 'US', 'UT',
    'WE', 'WO',
    'XI', 'XU',
    'YA', 'YE', 'YO',
    'ZA',
}

# Pre-compute valid hooks for each letter
VALID_FRONT_HOOKS: Dict[str, List[str]] = {}  # Letters that can come BEFORE
VALID_BACK_HOOKS: Dict[str, List[str]] = {}   # Letters that can come AFTER

for word in VALID_TWO_LETTER:
    first, second = word[0], word[1]
    if second not in VALID_FRONT_HOOKS:
        VALID_FRONT_HOOKS[second] = []
    VALID_FRONT_HOOKS[second].append(first)
    
    if first not in VALID_BACK_HOOKS:
        VALID_BACK_HOOKS[first] = []
    VALID_BACK_HOOKS[first].append(second)

# V21.1: Exchange configuration removed (exchange evaluation disabled).


# =============================================================================
# ANALYSIS TIME BUDGETS (seconds)
# =============================================================================

MC_TIME_BUDGET: float = 27.0                # main MC 2-ply evaluation budget
THREE_PLY_TIME_BUDGET: float = 20.0         # 3-ply endgame analysis budget

# =============================================================================
# ANALYSIS CANDIDATE SELECTION
# =============================================================================

ANALYSIS_MIN_CANDIDATES_MC: int = 130       # min candidates to pass to MC pipeline
ANALYSIS_DEFAULT_TOP_N: int = 15            # default number of moves to display

# =============================================================================
# MC EARLY STOPPING
# =============================================================================

MC_ES_MIN_SIMS: int = 100          # min sims before convergence check
MC_ES_CHECK_EVERY: int = 10        # check convergence every N sims
MC_ES_SE_THRESHOLD: float = 1.0    # std error target (95% CI +/- 2 pts)
MC_CEILING_K: int = 2000           # max simulations per candidate
MC_PROBE_COUNT: int = 3            # initial probes before adaptive K
MC_SLOW_BOARD_MS: int = 20         # threshold (ms) for slow board detection
MC_TOTAL_TIMEOUT: int = 90         # max seconds for entire MC phase (cancels stuck candidates)

# =============================================================================
# BLANK STRATEGY 3-PLY (triggered when rack has 2+ blanks)
# =============================================================================

BLANK_3PLY_MIN_BLANKS: int = 2       # blanks needed to trigger
BLANK_3PLY_MIN_BAG: int = 9          # skip if bag <= 8 (near-endgame handles it)
BLANK_3PLY_K_SIMS: int = 200         # MC sims per candidate
BLANK_3PLY_TOP_N: int = 8            # top MC candidates to include
BLANK_3PLY_FORCE_SAVERS: int = 4     # additional blank-saving candidates to force-include
BLANK_3PLY_TIME_BUDGET: float = 25.0 # max seconds for entire blank 3-ply pass

# =============================================================================
# ENDGAME
# =============================================================================

ENDGAME_FINAL_TURNS: int = 2        # turns remaining when bag empties (both get 1)

# =============================================================================
# STRATEGIC THRESHOLDS
# =============================================================================

CATCHUP_DEFICIT_THRESHOLD: int = 50         # show catch-up advice if down by 50+
HIGH_VARIANCE_DEFICIT_THRESHOLD: int = 100  # "down 100+" advice trigger
POWER_TILE_PROB_THRESHOLD: float = 0.05     # show power tile chance only if > 5%
LATE_GAME_UNSEEN_THRESHOLD: int = 21        # unseen <= 21 = late game


if __name__ == "__main__":
    print("CROSSPLAY V15 CONFIGURATION")
    print("=" * 50)
    print(f"Board size: {BOARD_SIZE}x{BOARD_SIZE}")
    print(f"Total tiles: {TOTAL_TILES}")
    print(f"Bingo bonus: {BINGO_BONUS}")
    print(f"3W squares: {len(TRIPLE_WORD_SQUARES)}")
    print(f"2W squares: {len(DOUBLE_WORD_SQUARES)}")
    print(f"3L squares: {len(TRIPLE_LETTER_SQUARES)}")
    print(f"2L squares: {len(DOUBLE_LETTER_SQUARES)}")
    print(f"Valid 2-letter words: {len(VALID_TWO_LETTER)}")
