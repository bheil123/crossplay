"""
CROSSPLAY V15 - Game Library Manager

Two-tier persistence:
  - Active games:  Individual JSON files in games/active/{opponent}_{NNN}.json
  - Completed games: Append-only JSONL in games/archive.jsonl
  - Index:          games/index.json (slot assignments + per-opponent counters)

Usage:
    from .game_library import (
        save_active, load_active, list_active,
        archive_completed, search_archive,
        load_index, save_index,
        get_game_id, migrate_factories,
    )
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Paths relative to this module
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
GAMES_DIR = os.path.join(_MODULE_DIR, 'games')
ACTIVE_DIR = os.path.join(GAMES_DIR, 'active')
ARCHIVE_PATH = os.path.join(GAMES_DIR, 'archive.jsonl')
INDEX_PATH = os.path.join(GAMES_DIR, 'index.json')


def _ensure_dirs():
    """Create games/ and games/active/ if they don't exist."""
    os.makedirs(ACTIVE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Index management (slot assignments + per-opponent game counters)
# ---------------------------------------------------------------------------

def load_index() -> dict:
    """Load games/index.json. Returns default if missing."""
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, 'r') as f:
            return json.load(f)
    return {'slots': {}, 'counters': {}, 'opponent_notes': {}}


def save_index(index: dict):
    """Write games/index.json (atomic via temp file)."""
    _ensure_dirs()
    tmp = INDEX_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(index, f, indent=2)
    os.replace(tmp, INDEX_PATH)


def get_game_id(opponent: str, index: dict = None) -> str:
    """Allocate next game ID for an opponent (e.g., 'canjam_002').

    Increments the counter in the index. Caller must save_index() after.
    """
    if index is None:
        index = load_index()
    opp_key = opponent.lower().strip()
    counters = index.setdefault('counters', {})
    current = counters.get(opp_key, 0)
    new_num = current + 1
    counters[opp_key] = new_num
    return f"{opp_key}_{new_num:03d}"


# ---------------------------------------------------------------------------
# Active game CRUD
# ---------------------------------------------------------------------------

def _active_path(game_id: str) -> str:
    """Path to an active game JSON file."""
    return os.path.join(ACTIVE_DIR, f"{game_id}.json")


def save_active(game_id: str, game) -> str:
    """Save a Game to games/active/{game_id}.json.

    Args:
        game_id: e.g., 'canjam_002'
        game: Game instance (has .state, .bag attributes)

    Returns:
        Path to saved file.
    """
    _ensure_dirs()

    # Sync bag into state before saving
    game.state.bag = game.bag
    game.state.updated_at = datetime.now().isoformat()

    data = game.state.to_dict()
    data['game_id'] = game_id
    data['status'] = 'active'
    data['spread'] = game.state.your_score - game.state.opp_score
    data['move_count'] = len(game.state.board_moves)

    filepath = _active_path(game_id)
    tmp = filepath + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, filepath)
    return filepath


def load_active(game_id: str):
    """Load a Game from games/active/{game_id}.json.

    Returns:
        Game instance, or None if file doesn't exist.
    """
    filepath = _active_path(game_id)
    if not os.path.exists(filepath):
        return None

    # Import here to avoid circular imports
    from .game_manager import Game, GameState

    with open(filepath, 'r') as f:
        data = json.load(f)

    # Remove library-specific keys before constructing GameState
    data.pop('game_id', None)
    data.pop('status', None)
    data.pop('spread', None)
    data.pop('move_count', None)
    data.pop('result', None)
    data.pop('completed_at', None)

    state = GameState.from_dict(data)
    game = Game(state)
    game.game_id = game_id
    return game


def list_active() -> List[dict]:
    """List all active games (id, opponent, scores, turn, filename)."""
    _ensure_dirs()
    results = []
    for fname in sorted(os.listdir(ACTIVE_DIR)):
        if not fname.endswith('.json'):
            continue
        filepath = os.path.join(ACTIVE_DIR, fname)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            results.append({
                'game_id': data.get('game_id', fname.replace('.json', '')),
                'opponent': data.get('opponent_name', data.get('opponent', '?')),
                'name': data.get('name', '?'),
                'your_score': data.get('your_score', 0),
                'opp_score': data.get('opp_score', 0),
                'is_your_turn': data.get('is_your_turn', False),
                'move_count': data.get('move_count', len(data.get('board_moves', []))),
            })
        except Exception as e:
            results.append({'game_id': fname, 'error': str(e)})
    return results


def delete_active(game_id: str) -> bool:
    """Delete an active game file. Returns True if deleted."""
    filepath = _active_path(game_id)
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False


# ---------------------------------------------------------------------------
# Archive (completed games)
# ---------------------------------------------------------------------------

def archive_completed(game_id: str, game) -> bool:
    """Archive a completed game: append to archive.jsonl, delete active JSON.

    Args:
        game_id: e.g., 'canjam_002'
        game: Game instance

    Returns:
        True if archived successfully.
    """
    _ensure_dirs()

    # Sync state
    game.state.bag = game.bag
    game.state.updated_at = datetime.now().isoformat()

    # Build archive record
    spread = game.state.your_score - game.state.opp_score
    if spread > 0:
        result = 'win'
    elif spread < 0:
        result = 'loss'
    else:
        result = 'tie'

    record = {
        'game_id': game_id,
        'name': game.state.name,
        'opponent': game.state.opponent_name,
        'your_score': game.state.your_score,
        'opp_score': game.state.opp_score,
        'result': result,
        'spread': spread,
        'move_count': len(game.state.board_moves),
        'moves': game.state.board_moves,
        'blank_positions': [list(bp) for bp in game.state.blank_positions],
        'created_at': game.state.created_at,
        'completed_at': datetime.now().isoformat(),
        'notes': game.state.notes,
    }

    # Append to archive
    with open(ARCHIVE_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')

    # Delete active file
    delete_active(game_id)

    print(f"[ARCHIVE] {game_id}: {game.state.name} vs {game.state.opponent_name} "
          f"({game.state.your_score}-{game.state.opp_score}, {result.upper()})")
    return True


def load_archive() -> List[dict]:
    """Load all archived (completed) games from archive.jsonl."""
    if not os.path.exists(ARCHIVE_PATH):
        return []
    records = []
    with open(ARCHIVE_PATH, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: bad JSON on line {line_num} of archive: {e}")
    return records


def search_archive(opponent: str = None, move_word: str = None) -> List[dict]:
    """Search completed games by opponent name or a word played.

    Args:
        opponent: Filter by opponent name (case-insensitive substring match)
        move_word: Filter by a word that was played in the game

    Returns:
        List of matching archive records.
    """
    records = load_archive()
    results = []
    for rec in records:
        if opponent:
            rec_opp = rec.get('opponent', '').lower()
            if opponent.lower() not in rec_opp:
                continue
        if move_word:
            moves = rec.get('moves', [])
            words_played = set()
            for m in moves:
                if isinstance(m, dict):
                    words_played.add(m.get('word', '').upper())
                elif isinstance(m, (list, tuple)) and len(m) >= 1:
                    words_played.add(str(m[0]).upper())
            if move_word.upper() not in words_played:
                continue
        results.append(rec)
    return results


def get_opponent_stats() -> Dict[str, dict]:
    """Compute per-opponent statistics from the archive.

    Returns:
        Dict mapping opponent name -> stats dict with:
        games_played, wins, losses, ties, avg_spread, best_spread,
        worst_spread, first_played, last_played
    """
    records = load_archive()
    stats = {}
    for rec in records:
        opp = rec.get('opponent', 'unknown').lower()
        if opp not in stats:
            stats[opp] = {
                'name': opp,
                'games_played': 0, 'wins': 0, 'losses': 0, 'ties': 0,
                'spreads': [], 'first_played': None, 'last_played': None,
            }
        s = stats[opp]
        s['games_played'] += 1
        result = rec.get('result', '')
        if result == 'win':
            s['wins'] += 1
        elif result == 'loss':
            s['losses'] += 1
        else:
            s['ties'] += 1
        s['spreads'].append(rec.get('spread', 0))
        dt = rec.get('completed_at') or rec.get('created_at', '')
        if dt:
            if s['first_played'] is None or dt < s['first_played']:
                s['first_played'] = dt
            if s['last_played'] is None or dt > s['last_played']:
                s['last_played'] = dt

    # Compute aggregates
    for s in stats.values():
        spreads = s.pop('spreads')
        s['avg_spread'] = round(sum(spreads) / len(spreads), 1) if spreads else 0
        s['best_spread'] = max(spreads) if spreads else 0
        s['worst_spread'] = min(spreads) if spreads else 0

    return stats


# ---------------------------------------------------------------------------
# Migration: factory functions + old archive -> new library
# ---------------------------------------------------------------------------

def migrate_factories() -> dict:
    """One-time migration: create JSON files from factory functions.

    Reads the factory registry in game_manager.py, saves each game
    to games/active/, and creates games/index.json.

    Returns:
        The new index dict.
    """
    from .game_manager import (
        _create_saved_game_5,
        _create_saved_game_6,
        _create_saved_game_7,
        _create_saved_game_8,
        _create_saved_game_9,
    )

    _ensure_dirs()
    index = load_index()

    # Map: factory -> (opponent, slot, is_completed)
    factories = [
        (_create_saved_game_9, 1),   # canjam - slot 1
        (_create_saved_game_6, 2),   # mallenmelon - slot 2
        (_create_saved_game_7, 3),   # eggsbenny - slot 3
        (_create_saved_game_8, 4),   # sophie - slot 4
        (_create_saved_game_5, None),  # garnetgirl - completed, no slot
    ]

    print("\n[MIGRATE] Migrating factory functions to game library...")

    for factory, slot in factories:
        try:
            game = factory()
            opponent = game.state.opponent_name.lower()
            game_id = get_game_id(opponent, index)
            game.game_id = game_id

            if game.is_complete():
                # Archive completed games directly
                archive_completed(game_id, game)
                print(f"  [ARCHIVE] {game_id}: {game.state.name} vs {opponent} (completed)")
            else:
                # Save active games
                save_active(game_id, game)
                if slot is not None:
                    index.setdefault('slots', {})[str(slot)] = game_id
                print(f"  [ACTIVE] {game_id} -> Slot {slot}: {game.state.name} vs {opponent}")

        except Exception as e:
            print(f"  [ERROR] Factory failed: {e}")

    save_index(index)
    print(f"[MIGRATE] Done. {len(factories)} games migrated.")
    print(f"  Index saved to {INDEX_PATH}")
    return index


def migrate_old_archive():
    """Import records from the old game_archive.jsonl into the new archive.

    Reads crossplay/game_archive.jsonl and appends to games/archive.jsonl,
    adding game_id fields where missing.
    """
    old_path = os.path.join(_MODULE_DIR, 'game_archive.jsonl')
    if not os.path.exists(old_path):
        print("[MIGRATE] No old archive found, skipping.")
        return

    _ensure_dirs()
    index = load_index()

    with open(old_path, 'r') as f:
        lines = f.readlines()

    migrated = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            # Add game_id if missing
            if 'game_id' not in record:
                opponent = record.get('opponent', 'unknown').lower()
                record['game_id'] = get_game_id(opponent, index)
            # Append to new archive
            with open(ARCHIVE_PATH, 'a') as f:
                f.write(json.dumps(record) + '\n')
            migrated += 1
        except Exception as e:
            print(f"  [WARN] Skipping bad record: {e}")

    if migrated > 0:
        save_index(index)
        print(f"[MIGRATE] Imported {migrated} records from old archive.")


def ensure_library_initialized() -> dict:
    """Ensure the game library is initialized. Run migration if needed.

    Returns:
        The index dict.
    """
    if os.path.exists(INDEX_PATH):
        return load_index()

    print("\n[LIBRARY] First run — initializing game library...")
    migrate_old_archive()
    index = migrate_factories()
    return index
