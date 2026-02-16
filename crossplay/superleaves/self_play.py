"""
CROSSPLAY V15 - Self-Play Game Loop for SuperLeaves Training

Two greedy bots play a full game. Records (leave_key, equity_signal, weight)
observations for every move where bag > 0.

Based on run_simulation() pattern in game_manager.py.
"""

import random
from ..board import Board
from ..config import TILE_DISTRIBUTION, RACK_SIZE
from .fast_bot import select_best_move


def play_one_game(gaddag, move_finder_cls, leave_table):
    """Play one complete self-play game and collect observations.

    Args:
        gaddag: GADDAG instance
        move_finder_cls: GADDAGMoveFinder class
        leave_table: LeaveTable instance (frozen for this generation)

    Returns:
        (observations, score1, score2)
        observations: list of (leave_key, equity_signal, weight)
    """
    board = Board()

    # Build and shuffle bag
    bag = []
    for letter, count in TILE_DISTRIBUTION.items():
        bag.extend([letter] * count)
    random.shuffle(bag)

    rack1 = [bag.pop() for _ in range(RACK_SIZE)]
    rack2 = [bag.pop() for _ in range(RACK_SIZE)]
    score1, score2 = 0, 0

    # Track all move scores for computing game mean
    all_scores = []
    # Track (player, leave_key, move_score, weight) for post-game signal
    raw_observations = []

    turn = 1
    consecutive_passes = 0
    max_turns = 100
    final_turns = None

    while consecutive_passes < 4 and turn <= max_turns:
        if final_turns is not None and final_turns <= 0:
            break

        is_p1 = (turn % 2 == 1)
        current_rack = rack1 if is_p1 else rack2
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

        # Greedy move selection
        best_move, best_leave, _ = select_best_move(
            board, moves, current_rack, leave_table, bag_size
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
        points = best_move['score']  # already includes bingo

        # Record observation (only when bag > 0)
        if bag_size > 0 and best_leave and len(best_leave) > 0:
            weight = min(bag_size / RACK_SIZE, 1.0)
            raw_observations.append((best_leave, points, weight))
        all_scores.append(points)

        # Place word on board
        board.place_word(word, row, col, horiz)

        # Update score
        if is_p1:
            score1 += points
        else:
            score2 += points

        # Rebuild rack: leave tiles + new draw
        # (best_leave was computed before place_word, so it's correct)
        current_rack.clear()
        current_rack.extend(best_leave)

        # Draw new tiles
        draw_count = min(RACK_SIZE - len(current_rack), len(bag))
        for _ in range(draw_count):
            current_rack.append(bag.pop())

        # Crossplay endgame: both players get one more turn after bag empties
        if len(bag) == 0 and final_turns is None:
            final_turns = 2
        if final_turns is not None:
            final_turns -= 1

        consecutive_passes = 0
        turn += 1

    # Compute game mean score for equity signal
    if all_scores:
        game_mean = sum(all_scores) / len(all_scores)
    else:
        game_mean = 0.0

    # Build final observations: signal = move_score - game_mean
    observations = []
    for leave_key, move_score, weight in raw_observations:
        signal = move_score - game_mean
        observations.append((leave_key, signal, weight))

    return observations, score1, score2
