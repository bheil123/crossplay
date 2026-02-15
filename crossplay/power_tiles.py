"""
CROSSPLAY V14 - Power Tile Probability Module
Calculate probability of drawing high-value tiles from a small bag.
"""

from typing import Dict, List, Tuple
from math import comb
from .config import TILE_VALUES


# High-value tiles worth tracking
POWER_TILES = {'X': 8, 'Z': 10, 'J': 10, 'Q': 10, 'Y': 5, 'K': 6}


def get_power_tiles_in_pool(unseen: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    """
    Get power tiles that are still unseen.
    
    Returns:
        Dict of {letter: (count, point_value)}
    """
    power = {}
    for tile, value in POWER_TILES.items():
        count = unseen.get(tile, 0)
        if count > 0:
            power[tile] = (count, value)
    return power


def prob_draw_at_least_one(pool_size: int, target_count: int, draw_count: int) -> float:
    """
    Calculate probability of drawing at least one target tile.
    
    Args:
        pool_size: Total tiles in the pool (bag)
        target_count: How many of the target tile exist in pool
        draw_count: How many tiles we're drawing
        
    Returns:
        Probability (0.0 to 1.0)
    """
    if pool_size <= 0 or draw_count <= 0 or target_count <= 0:
        return 0.0
    
    if target_count >= pool_size:
        return 1.0
    
    if draw_count > pool_size:
        draw_count = pool_size
    
    # P(at least one) = 1 - P(none)
    # P(none) = C(pool - target, draw) / C(pool, draw)
    non_target = pool_size - target_count
    
    if non_target < draw_count:
        return 1.0  # Must draw at least one target
    
    try:
        p_none = comb(non_target, draw_count) / comb(pool_size, draw_count)
        return 1.0 - p_none
    except:
        return 0.0


def prob_draw_any_power_tile(unseen: Dict[str, int], bag_size: int, draw_count: int) -> float:
    """
    Calculate probability of drawing at least one power tile.
    
    Args:
        unseen: Dict of unseen tiles {letter: count}
        bag_size: Tiles in bag (unseen - opponent rack)
        draw_count: How many tiles we're drawing
        
    Returns:
        Probability (0.0 to 1.0)
    """
    power_tiles = get_power_tiles_in_pool(unseen)
    if not power_tiles:
        return 0.0
    
    total_power = sum(count for count, _ in power_tiles.values())
    
    # Estimate how many power tiles are in the bag vs opponent rack
    total_unseen = sum(unseen.values())
    if total_unseen <= 0:
        return 0.0
    
    # Expected power tiles in bag (proportional distribution)
    expected_in_bag = total_power * (bag_size / total_unseen)
    
    # Use expected value for probability calculation
    return prob_draw_at_least_one(bag_size, int(round(expected_in_bag)), draw_count)


def calculate_power_tile_stats(unseen: Dict[str, int], bag_size: int, rack_size: int = 7) -> dict:
    """
    Calculate comprehensive power tile statistics.
    
    Args:
        unseen: Dict of unseen tiles
        bag_size: Tiles remaining in bag
        rack_size: Current rack size (to calc draw count for various plays)
        
    Returns:
        Dict with power tile info and probabilities
    """
    power_tiles = get_power_tiles_in_pool(unseen)
    total_unseen = sum(unseen.values())
    
    stats = {
        'power_tiles': power_tiles,
        'total_power_count': sum(count for count, _ in power_tiles.values()),
        'total_power_value': sum(count * value for count, value in power_tiles.values()),
        'bag_size': bag_size,
        'total_unseen': total_unseen,
        'draw_probabilities': {}
    }
    
    # Calculate probabilities for different draw counts (1-7 tiles)
    for draw in range(1, 8):
        prob = prob_draw_any_power_tile(unseen, bag_size, draw)
        stats['draw_probabilities'][draw] = prob
    
    return stats


def format_power_tile_display(unseen: Dict[str, int], bag_size: int) -> str:
    """
    Format power tile info for display in analysis.
    
    Returns:
        Formatted string like "* Power tiles: X(8), Y(5)x2 | Draw 2->35%"
    """
    power_tiles = get_power_tiles_in_pool(unseen)
    
    if not power_tiles:
        return "* No power tiles remaining"
    
    # Format each power tile
    parts = []
    for tile in sorted(power_tiles.keys(), key=lambda t: -power_tiles[t][1]):  # Sort by value desc
        count, value = power_tiles[tile]
        if count > 1:
            parts.append(f"{tile}({value})x{count}")
        else:
            parts.append(f"{tile}({value})")
    
    tile_str = ", ".join(parts)
    
    # Calculate draw probabilities for draws 1-6
    probs = []
    for draw in range(1, 7):
        prob = prob_draw_any_power_tile(unseen, bag_size, draw)
        if prob > 0.01:  # Only show if >1%
            probs.append(f"{draw}->{prob*100:.0f}%")
    
    prob_str = " | ".join(["Draw " + probs[0]] + probs[1:]) if probs else ""
    
    return f"* Power tiles: {tile_str}" + (f" | {prob_str}" if prob_str else "")


def calculate_leave_power_synergy(leave: str, unseen: Dict[str, int], bag_size: int, draw_count: int) -> float:
    """
    Calculate bonus equity for leaves that synergize with drawable power tiles.
    
    A leave like "AENRT" can make great plays with X (AX, EX), Y (YEN), etc.
    
    Returns:
        Bonus equity value (0 to ~5 points typically)
    """
    power_tiles = get_power_tiles_in_pool(unseen)
    if not power_tiles:
        return 0.0
    
    total_unseen = sum(unseen.values())
    if total_unseen <= 0:
        return 0.0
    
    leave_set = set(leave.upper())
    bonus = 0.0
    
    # Synergy patterns: vowels work great with X, certain combos with others
    synergy_map = {
        'X': {'A', 'E', 'I', 'O', 'U'},  # AX, EX, XI, OX, XU all valid
        'Y': {'A', 'E', 'O'},  # AY, YEA, YOW, etc.
        'Z': {'A', 'E', 'O'},  # ZA, ZE, ZO
        'J': {'A', 'O'},  # JA, JO
        'Q': {'I'},  # QI (only useful Q play without U)
        'K': {'A', 'E', 'I', 'O'},  # KA, KI, KO
    }
    
    for tile, (count, value) in power_tiles.items():
        # Probability this tile is in the bag
        prob_in_bag = bag_size / total_unseen
        
        # Probability of drawing it
        expected_in_bag = count * prob_in_bag
        draw_prob = prob_draw_at_least_one(bag_size, max(1, int(expected_in_bag)), draw_count)
        
        # Check synergy with leave
        synergy_letters = synergy_map.get(tile, set())
        synergy_count = len(leave_set & synergy_letters)
        
        # Bonus = probability * synergy * tile_value_factor
        # A rough heuristic: each synergy letter adds ~0.5 expected points
        tile_bonus = draw_prob * synergy_count * 0.5 * (value / 8)  # Normalize by avg power value
        bonus += tile_bonus
    
    return bonus


def test_power_tiles():
    """Test the power tile calculations."""
    # Example: late game scenario
    unseen = {
        'A': 2, 'D': 1, 'E': 1, 'F': 1, 'G': 1, 
        'I': 3, 'L': 1, 'N': 2, 'O': 2, 'T': 1,
        'X': 1, 'Y': 2
    }
    bag_size = 10  # 17 unseen - 7 opp rack
    
    print("Test: Late game with X and Yx2 in pool")
    print(f"Unseen: {unseen}")
    print(f"Bag size: {bag_size}")
    print()
    
    print(format_power_tile_display(unseen, bag_size))
    print()
    
    stats = calculate_power_tile_stats(unseen, bag_size)
    print(f"Power tiles: {stats['power_tiles']}")
    print(f"Total power value: {stats['total_power_value']}")
    print()
    
    print("Draw probabilities:")
    for draw, prob in stats['draw_probabilities'].items():
        print(f"  Draw {draw}: {prob*100:.1f}%")
    print()
    
    # Test synergy
    print("Leave synergy bonuses:")
    for leave in ['AENRT', 'ETN', 'N', 'II']:
        draw = 7 - len(leave)  # How many we'd draw
        synergy = calculate_leave_power_synergy(leave, unseen, bag_size, draw)
        print(f"  {leave}: +{synergy:.2f} (drawing {draw})")


if __name__ == "__main__":
    test_power_tiles()
