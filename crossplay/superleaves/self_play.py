"""
CROSSPLAY V16 - Self-Play Game Loop for SuperLeaves Training

Two greedy bots play a full game. Records (leave_key, equity_signal, weight)
observations for every move where bag > 0.

Gen2 improvements:
  - Equity-based signal: signal = move_equity - best_alternative_equity
    (directly measures leave quality contribution)
  - Outcome weighting: signals scaled by game spread (winning moves boosted)
  - Top-K expanded to 30 candidates in fast_bot
"""

import random
from ..board import Board
from ..config import TILE_DISTRIBUTION, RACK_SIZE
from .fast_bot import select_best_move, compute_leave


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

    # Track raw observations for post-game signal computation
    # Each entry: (leave_key, move_equity, best_equity, weight, player)
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

        # Record observation (only when bag > 0 and non-empty leave)
        if bag_size > 0 and best_leave and len(best_leave) > 0:
            weight = min(bag_size / RACK_SIZE, 1.0)

            # Compute equity of the best-scoring move (ignoring leave)
            # This is the "no-leave baseline" — what you'd pick if leave = 0
            top_score_move = moves[0]  # moves sorted by score desc
            top_score_equity = top_score_move['score']  # pure score, no leave

            # The chosen move's equity (score + leave_value)
            move_equity = best_equity

            # Signal: how much equity did the leave add beyond raw score?
            # = (score + leave_value) - score_of_best_scoring_move
            # Positive = leave choice improved over raw-score pick
            # Negative = leave cost points (chose worse score for better leave)
            signal = move_equity - top_score_equity

            raw_observations.append(
                (best_leave, signal, weight, 1 if is_p1 else 2)
            )

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

    # --- Post-game: apply outcome weighting ---
    # Scale signals by how well the player did overall.
    # Winning player's moves get boosted, losing player's moves get penalized.
    # This adds game-outcome context to the per-move leave signal.
    spread = score1 - score2  # positive = P1 won

    # Outcome multiplier: 1.0 +/- scale * normalized_spread
    # Cap at 0.5-1.5 to avoid extreme weights from blowouts
    OUTCOME_SCALE = 0.3  # How strongly outcome affects signal (tune this)
    MAX_SPREAD = 150.0   # Spread normalization range

    observations = []
    for leave_key, signal, weight, player in raw_observations:
        # Player 1 spread is positive when P1 wins
        player_spread = spread if player == 1 else -spread
        outcome_mult = 1.0 + OUTCOME_SCALE * max(-1.0, min(1.0,
            player_spread / MAX_SPREAD))

        weighted_signal = signal * outcome_mult
        observations.append((leave_key, weighted_signal, weight))

    return observations, score1, score2
