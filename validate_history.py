"""Replay all games and validate opponent moves through the new validator.
Read-only — does not modify any game files."""

import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crossplay.board import Board
from crossplay.gaddag import get_gaddag
from crossplay.scoring import find_crosswords
from crossplay.config import BOARD_SIZE, CENTER_ROW, CENTER_COL

print("Loading GADDAG...")
gaddag = get_gaddag()
print("GADDAG loaded.\n")


def validate_move(board, word, row, col, horizontal):
    """Replay version of _validate_opponent_move."""
    errors = []
    new_tile_positions = []
    connects = False
    covers_center = False

    for i, letter in enumerate(word):
        r = row + (0 if horizontal else i)
        c = col + (i if horizontal else 0)

        if r < 1 or r > BOARD_SIZE or c < 1 or c > BOARD_SIZE:
            errors.append(f"Out of bounds at R{r}C{c}")
            continue

        if r == CENTER_ROW and c == CENTER_COL:
            covers_center = True

        existing = board.get_tile(r, c)
        if existing is not None:
            if existing != letter:
                errors.append(
                    f"Tile conflict at R{r}C{c}: board has '{existing}', "
                    f"word has '{letter}'")
            else:
                connects = True
        else:
            new_tile_positions.append((r, c))
            if board.has_adjacent_tile(r, c):
                connects = True

    if errors:
        return False, errors

    if not new_tile_positions:
        errors.append("No new tiles placed")
        return False, errors

    if board.is_board_empty():
        if not covers_center:
            errors.append(
                f"First move must cover center (R{CENTER_ROW}C{CENTER_COL})")
    else:
        if not connects:
            pos_str = ", ".join(f"R{r}C{c}" for r, c in new_tile_positions)
            errors.append(
                f"Move does not connect to existing tiles. "
                f"New tiles: {pos_str}")

    if errors:
        return False, errors

    if not gaddag.is_word(word):
        errors.append(f"'{word}' is not a valid dictionary word")

    crosswords = find_crosswords(
        board, word, row, col, horizontal, new_tile_positions)
    for cw in crosswords:
        cw_word = cw['word']
        cw_dir = 'H' if cw['horizontal'] else 'V'
        if not gaddag.is_word(cw_word):
            errors.append(
                f"Invalid cross-word '{cw_word}' at "
                f"R{cw['row']}C{cw['col']} {cw_dir}")

    return len(errors) == 0, errors


def replay_game(game_name, moves, source):
    """Replay all moves, validate opponent moves."""
    board = Board()
    issues = []
    move_num = 0

    for m in moves:
        # Skip compact array format (no direction info)
        if isinstance(m, list):
            if len(m) >= 4:
                move_num += 1
                word = m[0].upper()
                row, col = m[1], m[2]
                # Compact format lacks direction — try both
                placed = False
                for horiz in [True, False]:
                    try:
                        board.place_word(word, row, col, horiz)
                        placed = True
                        break
                    except Exception:
                        continue
                if not placed:
                    issues.append({
                        'game': game_name, 'source': source,
                        'move_num': move_num, 'word': word,
                        'pos': f"R{row}C{col} ?",
                        'score': None, 'player': 'compact',
                        'errors': ["Could not place (compact format)"]
                    })
            continue

        if not isinstance(m, dict):
            continue

        word = m.get('word', '').upper()
        row = m.get('row')
        col = m.get('col')
        direction = m.get('dir', 'H')
        player = m.get('player')
        score = m.get('score')
        horizontal = (direction == 'H')
        move_num += 1

        if not word or row is None or col is None:
            continue

        # Validate opponent moves BEFORE placing
        if player == 'opp':
            valid, errs = validate_move(board, word, row, col, horizontal)
            if not valid:
                issues.append({
                    'game': game_name, 'source': source,
                    'move_num': move_num, 'word': word,
                    'pos': f"R{row}C{col} {'H' if horizontal else 'V'}",
                    'score': score, 'player': 'opp',
                    'errors': errs
                })

        # Place on board for both players
        try:
            board.place_word(word, row, col, horizontal)
        except Exception as e:
            issues.append({
                'game': game_name, 'source': source,
                'move_num': move_num, 'word': word,
                'pos': f"R{row}C{col} {'H' if horizontal else 'V'}",
                'score': score, 'player': player or '?',
                'errors': [f"Board placement failed: {e}"]
            })

    return issues


# ---- Main ----
active_dir = os.path.join('crossplay', 'games', 'active')
archive_file = os.path.join('crossplay', 'games', 'archive.jsonl')
games = []

for fn in sorted(os.listdir(active_dir)):
    if fn.endswith('.json'):
        with open(os.path.join(active_dir, fn)) as f:
            data = json.load(f)
        games.append((fn.replace('.json', ''),
                       data.get('board_moves', []), 'active'))

if os.path.exists(archive_file):
    with open(archive_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            games.append((data.get('game_id', 'unknown'),
                          data.get('moves', []), 'archive'))

all_issues = []
total_opp_dict = 0
total_opp_compact = 0

for name, moves, source in games:
    dict_opp = sum(1 for m in moves
                   if isinstance(m, dict) and m.get('player') == 'opp')
    compact_opp = sum(1 for m in moves
                      if isinstance(m, list) and len(m) >= 4 and m[3] is True)
    total_opp_dict += dict_opp
    total_opp_compact += compact_opp

    issues = replay_game(name, moves, source)
    all_issues.extend(issues)

print("=" * 70)
print("OPPONENT MOVE VALIDATION REPORT")
print("=" * 70)
print(f"Games scanned:        {len(games)}")
print(f"Opp moves (dict fmt): {total_opp_dict}  (fully validated)")
print(f"Opp moves (compact):  {total_opp_compact}  (placement only)")
print(f"Issues found:         {len(all_issues)}")
print()

if all_issues:
    for issue in all_issues:
        tag = issue['player'].upper()
        print(f"  [{issue['source']:7s}] {issue['game']} "
              f"- Move #{issue['move_num']} ({tag})")
        print(f"           {issue['word']} at {issue['pos']} "
              f"(score: {issue['score']})")
        for err in issue['errors']:
            print(f"           [X] {err}")
        print()
else:
    print("  No issues found! All opponent moves validate cleanly.")
