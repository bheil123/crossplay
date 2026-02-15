"""
Opening heuristics for Scrabble move evaluation.

Implements:
- #2: DLS Exposure penalty (parallel play vulnerability)
- #4: Double-Double Lane detection
- #5: Tile Turnover bonus

These apply primarily to opening moves and early game,
but DLS exposure is relevant throughout.
"""

from .config import BONUS_SQUARES, TILE_VALUES

# =====================================================================
# HOOKABILITY DATA
# For each letter, how many 2-letter words can it form?
# Higher = more hookable = more dangerous when exposed near premium squares
# =====================================================================
HOOKABILITY = {
    'A': 28, 'B': 6, 'C': 0, 'D': 7, 'E': 24, 'F': 5, 'G': 3, 'H': 10,
    'I': 18, 'J': 1, 'K': 3, 'L': 5, 'M': 12, 'N': 9, 'O': 27, 'P': 6,
    'Q': 1, 'R': 4, 'S': 8, 'T': 8, 'U': 9, 'V': 0, 'W': 5, 'X': 5,
    'Y': 7, 'Z': 1
}

# High-value tiles that could be played on DLS via parallel play
# and their 2-letter words
HIGH_VALUE_2LETTER = {
    'J': ['JO'],           # 10 pts
    'Q': ['QI'],           # 10 pts  
    'Z': ['ZA'],           # 10 pts
    'X': ['AX','EX','OX','XI','XU'],  # 8 pts
    'K': ['KA','KI','OK'], # 6 pts
}

# DLS squares on the board
DLS_SQUARES = set((r,c) for (r,c), v in BONUS_SQUARES.items() if v == '2L')

# DWS squares on the board  
DWS_SQUARES = set((r,c) for (r,c), v in BONUS_SQUARES.items() if v == '2W')

# TWS squares
TWS_SQUARES = set((r,c) for (r,c), v in BONUS_SQUARES.items() if v == '3W')

# Center star
CENTER = (8, 8)


def evaluate_dls_exposure(board, move, unseen_tiles=None):
    """
    #2: DLS Exposure Penalty
    
    After placing a word, check if any newly placed tiles are adjacent to
    unused DLS squares. If so, estimate the damage an opponent could do
    with a high-value parallel play on that DLS.
    
    Returns:
        float: penalty (negative value, to subtract from equity)
        list: details of exposed DLS positions
    """
    word = move['word']
    row, col = move['row'], move['col']
    direction = move['direction']
    horizontal = (direction == 'H')
    
    total_penalty = 0.0
    exposed = []
    
    # Get positions of newly placed tiles
    new_positions = []
    for i, letter in enumerate(word):
        if horizontal:
            r, c = row, col + i
        else:
            r, c = row + i, col
        
        # Only count tiles we're placing (not already on board)
        if board.get_tile(r, c) is None:
            new_positions.append((r, c, letter))
    
    # For each new tile, check for DLS exposure in two patterns:
    # Pattern A: Same-direction DLS 1 square beyond word end
    #   (opponent extends our word to land on DLS)
    # Pattern B: Perpendicular DLS 1-2 squares away
    #   (opponent plays parallel/perpendicular word hitting DLS)
    
    for r, c, letter in new_positions:
        # Pattern A: Same direction - check 1 square beyond word ends
        # (handled by checking if adjacent same-direction square is DLS)
        adj_positions = []
        
        if horizontal:
            # Perpendicular: above and below (1 and 2 away)
            for dist in [1, 2]:
                adj_positions.append((r-dist, c, 'perp'))
                adj_positions.append((r+dist, c, 'perp'))
            # Same direction: left and right of this tile
            adj_positions.append((r, c-1, 'same'))
            adj_positions.append((r, c+1, 'same'))
        else:
            # Perpendicular: left and right (1 and 2 away)
            for dist in [1, 2]:
                adj_positions.append((r, c-dist, 'perp'))
                adj_positions.append((r, c+dist, 'perp'))
            # Same direction: above and below
            adj_positions.append((r-1, c, 'same'))
            adj_positions.append((r+1, c, 'same'))
        
        for ar, ac, pattern in adj_positions:
            if 0 < ar <= 15 and 0 < ac <= 15:
                # Is there a DLS here that's currently empty?
                if (ar, ac) in DLS_SQUARES and board.get_tile(ar, ac) is None:
                    # This letter is adjacent to an open DLS
                    # How hookable is this letter?
                    hookability = HOOKABILITY.get(letter, 0)
                    
                    # Estimate penalty based on:
                    # - How hookable the letter is (can opponent form 2-letter words?)
                    # - What high-value tiles could land on the DLS
                    
                    # Check if any high-value tile + this letter = valid 2-letter word
                    max_damage = 0
                    worst_tile = None
                    
                    for hv_tile, hv_words in HIGH_VALUE_2LETTER.items():
                        tile_val = TILE_VALUES.get(hv_tile, 0)
                        # Check if hv_tile + letter or letter + hv_tile is valid
                        for w in hv_words:
                            if letter in w and len(w) == 2:
                                # This high-value tile can play on the DLS
                                # Damage = tile_value * 2 (DLS) + letter_value
                                damage = tile_val * 2 + TILE_VALUES.get(letter, 0)
                                if damage > max_damage:
                                    max_damage = damage
                                    worst_tile = hv_tile
                    
                    if max_damage > 0:
                        # Probability opponent has this tile
                        prob = 0.15  # Default 15% chance
                        if unseen_tiles:
                            total_unseen = sum(unseen_tiles.values())
                            tile_count = unseen_tiles.get(worst_tile, 0)
                            if total_unseen > 0 and tile_count > 0:
                                prob = 1 - ((total_unseen - tile_count) / total_unseen) ** 7
                        
                        # Distance scaling: DLS 2 away is less threatening
                        dist = abs(ar - r) + abs(ac - c)
                        dist_factor = 1.0 if dist == 1 else 0.5
                        
                        penalty = max_damage * prob * dist_factor
                        total_penalty += penalty
                        exposed.append({
                            'tile_pos': (r, c, letter),
                            'dls_pos': (ar, ac),
                            'worst_tile': worst_tile,
                            'max_damage': max_damage,
                            'probability': prob,
                            'penalty': penalty
                        })
                    elif hookability > 10:
                        # Even without high-value tiles, very hookable letters
                        # near DLS are somewhat risky
                        penalty = hookability * 0.1
                        total_penalty += penalty
                        exposed.append({
                            'tile_pos': (r, c, letter),
                            'dls_pos': (ar, ac),
                            'worst_tile': 'general',
                            'max_damage': 0,
                            'probability': 0,
                            'penalty': penalty
                        })
    
    return -total_penalty, exposed


def evaluate_double_double(board, move):
    """
    #4: Double-Double Lane Detection
    
    Check if a move creates a path connecting two DWS squares,
    enabling a potential 4x word score (double-double).
    
    For opening: a 5+ letter word through center can reach a DWS.
    A word spanning two DWS = 4x multiplier opportunity.
    
    Returns:
        float: bonus (positive if we get the DD) or penalty (if we open it for opponent)
        str: description
    """
    word = move['word']
    row, col = move['row'], move['col']
    direction = move['direction']
    horizontal = (direction == 'H')
    word_len = len(word)
    
    # Get all positions this word occupies
    positions = set()
    for i in range(word_len):
        if horizontal:
            positions.add((row, col + i))
        else:
            positions.add((row + i, col))
    
    # Check if this word covers any DWS squares
    dws_covered = positions & DWS_SQUARES
    # Include center as DWS for opening
    if CENTER in positions:
        dws_covered.add(CENTER)
    
    # If we cover 2+ DWS, we're getting a double-double ourselves
    if len(dws_covered) >= 2:
        return 5.0, f"DOUBLE-DOUBLE! Covers {len(dws_covered)} DWS: {dws_covered}"
    
    # Check if we OPEN a double-double lane for opponent
    # After our word, can opponent extend perpendicular through two DWS?
    dd_risk = 0.0
    dd_desc = ""
    
    # For each tile we place, check if it creates a hook point
    # that connects to a DWS lane
    for i, letter in enumerate(word):
        if horizontal:
            r, c = row, col + i
        else:
            r, c = row + i, col
        
        # Only new tiles matter
        if board.get_tile(r, c) is not None:
            continue
        
        # Check perpendicular direction for DWS access
        if horizontal:
            # Check vertical: is there a DWS above or below?
            for dr in [-1, -2, -3, -4, -5, -6, -7]:
                nr = r + dr
                if 0 < nr <= 15 and (nr, c) in DWS_SQUARES:
                    # A DWS is reachable vertically from this new tile
                    dd_risk += 1.0
                    dd_desc += f"DWS@R{nr}C{c} reachable from {letter}@R{r}C{c}; "
                    break
            for dr in [1, 2, 3, 4, 5, 6, 7]:
                nr = r + dr
                if 0 < nr <= 15 and (nr, c) in DWS_SQUARES:
                    dd_risk += 1.0
                    dd_desc += f"DWS@R{nr}C{c} reachable from {letter}@R{r}C{c}; "
                    break
        else:
            for dc in [-1, -2, -3, -4, -5, -6, -7]:
                nc = c + dc
                if 0 < nc <= 15 and (r, nc) in DWS_SQUARES:
                    dd_risk += 1.0
                    dd_desc += f"DWS@R{r}C{nc} reachable from {letter}@R{r}C{c}; "
                    break
            for dc in [1, 2, 3, 4, 5, 6, 7]:
                nc = c + dc
                if 0 < nc <= 15 and (r, nc) in DWS_SQUARES:
                    dd_risk += 1.0
                    dd_desc += f"DWS@R{r}C{nc} reachable from {letter}@R{r}C{c}; "
                    break
    
    # Mild penalty for opening DD lanes (opponent might not have tiles)
    if dd_risk > 0:
        penalty = -dd_risk * 0.5
        return penalty, f"Opens DD lanes: {dd_desc}"
    
    return 0.0, "No DD impact"


def evaluate_tile_turnover(move, rack, bag_size):
    """
    #5: Tile Turnover Bonus
    
    Playing more tiles = drawing more tiles = better chance at blanks and S's.
    Small bonus for using more tiles from rack, scaled by bag size.
    
    When bag is large, turnover matters more (more tiles to see).
    When bag is small, turnover matters less.
    
    Returns:
        float: bonus value
    """
    if bag_size <= 7:
        return 0.0  # No turnover bonus when bag is nearly empty
    
    # Count tiles used from rack
    # 'used' field has tiles from rack if available, otherwise estimate from word length
    tiles_from_rack = len(move.get('used', move['word']))
    
    # Bonus: +0.3 per tile played beyond 2, capped at +2.0
    # Reasoning: each additional tile drawn gives ~1-2% chance of blank/S
    base_bonus = max(0, (tiles_from_rack - 2)) * 0.3
    
    # Scale by bag fullness (more bonus when bag is large)
    bag_factor = min(1.0, bag_size / 80.0)
    
    return min(2.0, base_bonus * bag_factor)


def evaluate_opening_heuristics(board, move, rack, unseen_tiles=None, bag_size=86):
    """
    Combined evaluation of all opening heuristics.
    
    Returns:
        dict with:
            'dls_penalty': float
            'dls_details': list
            'dd_bonus': float  
            'dd_desc': str
            'turnover_bonus': float
            'total_adjustment': float
    """
    dls_penalty, dls_details = evaluate_dls_exposure(board, move, unseen_tiles)
    dd_bonus, dd_desc = evaluate_double_double(board, move)
    turnover_bonus = evaluate_tile_turnover(move, rack, bag_size)
    
    total = dls_penalty + dd_bonus + turnover_bonus
    
    return {
        'dls_penalty': dls_penalty,
        'dls_details': dls_details,
        'dd_bonus': dd_bonus,
        'dd_desc': dd_desc,
        'turnover_bonus': turnover_bonus,
        'total_adjustment': total
    }
