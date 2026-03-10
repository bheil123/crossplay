"""
CROSSPLAY V21.1 - Self-Play Game Loop for SuperLeaves Training

Two greedy bots play a full game. Records (leave_key, td_target, weight)
observations for every move where bag > 0.

V21.1: skip_postvalidation=True for ~20% training speedup.
       Finder reused per game (Board mutated in place).

Gen3 improvements (TD-learning):
  - Per-player trajectory tracking through each game
  - TD(0) backward pass at game end: td_target = advantage + gamma * V(next_leave)
  - Bootstraps from future leave values, giving ~3-5x more signal per game
  - Outcome weighting removed (TD naturally propagates trajectory info)
  - Top-K expanded to 30 candidates in fast_bot (unchanged from gen2)
"""

import random
from ..board import Board
from ..config import TILE_DISTRIBUTION, RACK_SIZE
from .fast_bot import select_best_move, compute_leave


def play_one_game(gaddag, move_finder_cls, leave_table, td_gamma=0.97):
    """Play one complete self-play game and collect observations.

    Args:
        gaddag: GADDAG instance
        move_finder_cls: GADDAGMoveFinder class
        leave_table: LeaveTable instance (frozen for this generation)
        td_gamma: TD discount factor (0.0 = no bootstrapping, 1.0 = full)

    Returns:
        (observations, score1, score2)
        observations: list of (leave_key, td_target, weight, bag_size)
    """
    board = Board()

    # Create finder once per game -- Board is mutated in place by place_word,
    # and CMoveFinder reads board._grid each call, so reuse is safe.
    try:
        finder = move_finder_cls(board, gaddag, skip_postvalidation=True)
    except TypeError:
        # Fallback for move finders that don't support skip_postvalidation
        finder = move_finder_cls(board, gaddag)

    # Build and shuffle bag
    bag = []
    for letter, count in TILE_DISTRIBUTION.items():
        bag.extend([letter] * count)
    random.shuffle(bag)

    rack1 = [bag.pop() for _ in range(RACK_SIZE)]
    rack2 = [bag.pop() for _ in range(RACK_SIZE)]
    score1, score2 = 0, 0

    # Per-player trajectories for TD backward pass
    # Each entry: (leave_key, advantage, weight, bag_size)
    trajectory_p1 = []
    trajectory_p2 = []

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
        moves = finder.find_all_moves(rack_str)

        if not moves:
            consecutive_passes += 1
            if final_turns is not None:
                final_turns -= 1
            turn += 1
            continue

        # Greedy move selection (returns best move + leave + equity)
        best_move, best_leave, best_equity = select_best_move(
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

        # Record trajectory entry (only when bag > 0 and non-empty leave)
        if bag_size > 0 and best_leave and len(best_leave) > 0:
            weight = min(bag_size / RACK_SIZE, 1.0)

            # Advantage: how much equity did the leave add beyond raw score?
            top_score_move = moves[0]  # moves sorted by score desc
            top_score_equity = top_score_move['score']  # pure score, no leave
            advantage = best_equity - top_score_equity

            traj = trajectory_p1 if is_p1 else trajectory_p2
            traj.append((best_leave, advantage, weight, bag_size))

        # Place word on board
        board.place_word(word, row, col, horiz)

        # Update score
        if is_p1:
            score1 += points
        else:
            score2 += points

        # Rebuild rack: leave tiles + new draw
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

    # --- Post-game: TD(0) backward pass ---
    # Compute TD targets for each trajectory entry, working backwards.
    # td_target_t = advantage_t + gamma * V(next_leave_t+1)
    # Terminal moves (last in trajectory) get V_next = 0 (no tile penalty).
    observations = []

    for traj in (trajectory_p1, trajectory_p2):
        if not traj:
            continue
        # Backward pass: last move has V_next = 0
        v_next = 0.0
        for i in range(len(traj) - 1, -1, -1):
            leave_key, advantage, weight, bag_size = traj[i]
            td_target = advantage + td_gamma * v_next
            observations.append((leave_key, td_target, weight, bag_size))
            # V(this_leave) for the move before this one
            v_next = leave_table.get(leave_key, 0.0)

    return observations, score1, score2
