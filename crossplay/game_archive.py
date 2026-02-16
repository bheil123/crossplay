"""
CROSSPLAY - Game Archive Module
Enriched move history, persistent game archive (JSONL), and CSV export.

Provides:
  - enrich_move_history()    -- Replay moves through scoring engine, compute full metadata
  - archive_game()           -- Append completed game to game_archive.jsonl
  - load_archive()           -- Read all archived game records
  - export_archive_csv()     -- One-row-per-game summary CSV
  - export_archive_moves_csv() -- One-row-per-move detail CSV
  - backfill_saved_games()   -- Backfill enriched data for saved games 2, 3, 4
"""

import json
import csv
import os
from datetime import datetime
from typing import List, Tuple, Dict, Optional

from .board import Board
from .scoring import calculate_move_score
from .config import TILE_DISTRIBUTION, TILE_VALUES, RACK_SIZE, TOTAL_TILES

# Archive files live in games/ subdirectory (git-tracked)
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
GAMES_DIR = os.path.join(_MODULE_DIR, 'games')
ARCHIVE_PATH = os.path.join(GAMES_DIR, 'archive.jsonl')
SUMMARY_CSV_PATH = os.path.join(GAMES_DIR, 'archive_summary.csv')
MOVES_CSV_PATH = os.path.join(GAMES_DIR, 'archive_moves.csv')
# Legacy path (for migration)
LEGACY_ARCHIVE_PATH = os.path.join(_MODULE_DIR, 'game_archive.jsonl')


def _extract_move_tuple(move):
    """Extract (word, row, col, horizontal) from either a tuple or enriched dict."""
    if isinstance(move, dict):
        horiz = move.get('dir', 'H') == 'H' if 'dir' in move else move.get('horizontal', True)
        return move['word'], move['row'], move['col'], horiz
    else:
        return move[0], move[1], move[2], move[3]


def enrich_move_history(board_moves, blank_positions, first_player='opp'):
    """Replay moves through the scoring engine and build enriched move records.

    Args:
        board_moves: List of (word, row, col, horizontal) tuples or enriched dicts.
        blank_positions: List of (row, col, letter) tuples for all blanks in the game.
        first_player: 'me' or 'opp' -- who played the first move.

    Returns:
        List of enriched move dicts, one per move.
    """
    board = Board()
    # Starting bag: 100 tiles total, minus 2 racks of 7 = 86
    bag_count = TOTAL_TILES - 2 * RACK_SIZE  # 86
    my_score = 0
    opp_score = 0

    # Build blank lookup: (row, col) -> letter
    blank_lookup = {(r, c): let for r, c, let in blank_positions}

    # Track blanks already placed on board (accumulates as moves are played)
    board_blanks_so_far = []

    enriched = []

    for move_idx, move in enumerate(board_moves):
        word, row, col, horizontal = _extract_move_tuple(move)
        word = word.upper()

        # Determine player for this move
        if first_player == 'opp':
            player = 'opp' if move_idx % 2 == 0 else 'me'
        else:
            player = 'me' if move_idx % 2 == 0 else 'opp'

        # Find new tile positions (squares currently empty on board)
        new_tile_positions = []
        new_tile_indices = []
        for i in range(len(word)):
            if horizontal:
                r, c = row, col + i
            else:
                r, c = row + i, col
            if board.is_empty(r, c):
                new_tile_positions.append((r, c))
                new_tile_indices.append(i)

        # Identify which new tiles are blanks
        blanks_used = []  # word indices that are blanks
        blank_letters = []
        for i in new_tile_indices:
            if horizontal:
                r, c = row, col + i
            else:
                r, c = row + i, col
            if (r, c) in blank_lookup:
                blanks_used.append(i)
                blank_letters.append(word[i])

        # Calculate score using the scoring engine
        score, crosswords = calculate_move_score(
            board, word, row, col, horizontal,
            blanks_used=blanks_used,
            board_blanks=board_blanks_so_far
        )

        # Check for bingo annotation
        tiles_placed = len(new_tile_positions)
        note = 'bingo' if tiles_placed >= RACK_SIZE else ''

        # Update cumulative scores
        if player == 'me':
            my_score += score
        else:
            opp_score += score

        # Update bag count: tiles drawn to refill = min(tiles_placed, bag_count)
        drawn = min(tiles_placed, bag_count)
        bag_count -= drawn

        # Build enriched record
        enriched_move = {
            'word': word,
            'row': row,
            'col': col,
            'dir': 'H' if horizontal else 'V',
            'player': player,
            'score': score,
            'bag': bag_count,
            'blanks': blanks_used,
            'cumulative': [my_score, opp_score],
            'note': note,
        }
        enriched.append(enriched_move)

        # Place the word on the board (after scoring, since scoring needs pre-move board)
        board.place_word(word, row, col, horizontal)

        # Update board_blanks_so_far with any blanks from this move
        for i in blanks_used:
            if horizontal:
                br, bc = row, col + i
            else:
                br, bc = row + i, col
            board_blanks_so_far.append((br, bc, word[i]))

    return enriched


def archive_game(game_state, enriched_moves, archive_path=None):
    """Append a completed game record to the JSONL archive.

    Args:
        game_state: GameState dataclass instance.
        enriched_moves: List of enriched move dicts from enrich_move_history().
        archive_path: Optional override for archive file path.
    """
    path = archive_path or ARCHIVE_PATH
    spread = game_state.your_score - game_state.opp_score
    result = 'win' if spread > 0 else 'loss' if spread < 0 else 'tie'

    record = {
        'name': game_state.name,
        'opponent': game_state.opponent_name,
        'your_score': game_state.your_score,
        'opp_score': game_state.opp_score,
        'result': result,
        'spread': spread,
        'moves': enriched_moves,
        'move_count': len(enriched_moves),
        'blank_positions': [list(bp) for bp in game_state.blank_positions],
        'created_at': game_state.created_at,
        'completed_at': game_state.updated_at,
        'archived_at': datetime.now().isoformat(),
        'notes': game_state.notes,
    }

    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Archived: {game_state.name} vs {game_state.opponent_name} "
          f"({game_state.your_score}-{game_state.opp_score}, {result})")
    return record


def load_archive(archive_path=None):
    """Load all game records from the JSONL archive.

    Returns:
        List of game record dicts.
    """
    path = archive_path or ARCHIVE_PATH
    if not os.path.exists(path):
        return []

    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: skipping malformed line {line_num}: {e}")
    return records


def export_archive_csv(archive_path=None, output_path=None):
    """Export one-row-per-game summary CSV.

    Args:
        archive_path: Override for JSONL source.
        output_path: Override for CSV output.

    Returns:
        Number of games exported.
    """
    records = load_archive(archive_path)
    if not records:
        print("No archived games to export.")
        return 0

    path = output_path or SUMMARY_CSV_PATH
    fieldnames = [
        'name', 'opponent', 'your_score', 'opp_score', 'result', 'spread',
        'move_count', 'created_at', 'completed_at', 'archived_at', 'notes',
    ]

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    print(f"Exported {len(records)} games to {path}")
    return len(records)


def export_archive_moves_csv(archive_path=None, output_path=None):
    """Export one-row-per-move detail CSV (for 1000+ game analysis).

    Args:
        archive_path: Override for JSONL source.
        output_path: Override for CSV output.

    Returns:
        Number of move rows exported.
    """
    records = load_archive(archive_path)
    if not records:
        print("No archived games to export.")
        return 0

    path = output_path or MOVES_CSV_PATH
    fieldnames = [
        'game_name', 'opponent', 'result', 'move_num',
        'word', 'row', 'col', 'dir', 'player', 'score', 'bag',
        'blanks', 'my_cumulative', 'opp_cumulative', 'note',
    ]

    total_rows = 0
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            for i, move in enumerate(rec.get('moves', [])):
                row = {
                    'game_name': rec['name'],
                    'opponent': rec['opponent'],
                    'result': rec['result'],
                    'move_num': i + 1,
                    'word': move['word'],
                    'row': move['row'],
                    'col': move['col'],
                    'dir': move['dir'],
                    'player': move['player'],
                    'score': move['score'],
                    'bag': move['bag'],
                    'blanks': ','.join(str(b) for b in move.get('blanks', [])),
                    'my_cumulative': move['cumulative'][0],
                    'opp_cumulative': move['cumulative'][1],
                    'note': move.get('note', ''),
                }
                writer.writerow(row)
                total_rows += 1

    print(f"Exported {total_rows} moves across {len(records)} games to {path}")
    return total_rows


def backfill_saved_games():
    """Legacy backfill function -- no longer used.

    The factory functions this relied on were removed in V15.
    Game data now lives in crossplay/games/ (see game_library.py).
    """
    print("[WARN] backfill_saved_games() is deprecated.")
    print("  Factory functions were removed in V15.")
    print("  Use game_library.py for game persistence.")
    return {}


if __name__ == '__main__':
    results = backfill_saved_games()
