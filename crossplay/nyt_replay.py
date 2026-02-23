"""
NYT Crossplay replay analysis tool.

Replays archived games through the current engine (1-ply) and produces
a three-way comparison: engine recommendation vs NYT recommendation vs
what was actually played.

Usage:
    python -m crossplay.nyt_replay              # all games with NYT data
    python -m crossplay.nyt_replay sophie_001   # specific game
    python -m crossplay.nyt_replay --detail     # show every move
    python -m crossplay.nyt_replay --save       # save JSON to games/analysis/
"""

import json
import sys
import os
import time
from typing import List, Dict, Optional, Tuple

from .board import Board, tiles_used
from .leave_eval import evaluate_leave
from .config import TOTAL_TILES, RACK_SIZE


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------

_GADDAG = None


def load_resources():
    """Load GADDAG (cached). Returns gaddag."""
    global _GADDAG
    if _GADDAG is None:
        from .gaddag import get_gaddag
        _GADDAG = get_gaddag()
    return _GADDAG


def _get_move_finder():
    """Return the best available move finder function."""
    try:
        from .move_finder_c import find_all_moves_c, is_available
        if is_available():
            return find_all_moves_c
    except ImportError:
        pass
    from .move_finder_opt import find_all_moves_opt
    return find_all_moves_opt


# ---------------------------------------------------------------------------
# Archive loading
# ---------------------------------------------------------------------------

def load_archive() -> List[Dict]:
    """Load all archived games."""
    archive_path = os.path.join(os.path.dirname(__file__), 'games', 'archive.jsonl')
    records = []
    with open(archive_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def has_replay_data(record: Dict) -> bool:
    """Check if game has NYT data AND at least one 'me' move with rack."""
    if record.get('nyt_strategy_you') is None:
        return False
    moves = record.get('moves', [])
    for m in moves:
        if isinstance(m, dict) and m.get('player') == 'me' and m.get('rack'):
            return True
    return False


# ---------------------------------------------------------------------------
# Leave and equity computation
# ---------------------------------------------------------------------------

def _get_new_tile_word_indices(board, word, row, col, horizontal):
    """Get list of word indices that correspond to new tiles placed."""
    indices = []
    for i, letter in enumerate(word):
        r = row if horizontal else row + i
        c = col + i if horizontal else col
        if not board.get_tile(r, c):
            indices.append(i)
    return indices


def compute_leave(rack, used, blanks_used, new_tile_indices):
    """Compute leave string after playing a move.

    Args:
        rack: Pre-move rack string (e.g. "AEINRST" or "AEI?ST")
        used: List of letters placed from rack (from tiles_used())
        blanks_used: List of word indices that used blanks (from move dict)
        new_tile_indices: List of word indices for newly placed tiles

    Returns:
        Sorted leave string.
    """
    leave_tiles = list(rack)
    for ui, letter in enumerate(used):
        word_idx = new_tile_indices[ui] if ui < len(new_tile_indices) else -1
        if word_idx in blanks_used:
            if '?' in leave_tiles:
                leave_tiles.remove('?')
        else:
            if letter in leave_tiles:
                leave_tiles.remove(letter)
            elif '?' in leave_tiles:
                leave_tiles.remove('?')
    return ''.join(sorted(leave_tiles)) if leave_tiles else '-'


def compute_1ply_equity(candidates, rack, board, bag_count):
    """Add 1-ply equity (score + leave value) to each candidate move.

    Modifies candidates in place. Adds: 'leave', 'leave_value', 'equity'.
    Returns candidates sorted by equity descending.
    """
    bag_empty = bag_count <= 0

    for move in candidates:
        word = move['word']
        row = move['row']
        col = move['col']
        horizontal = move['direction'] == 'H'
        score = move['score']

        used = tiles_used(board, word, row, col, horizontal)
        new_indices = _get_new_tile_word_indices(board, word, row, col, horizontal)
        blanks_used = move.get('blanks_used', [])

        leave_str = compute_leave(rack, used, blanks_used, new_indices)
        leave_value = evaluate_leave(leave_str, bag_empty=bag_empty)
        equity = score + leave_value

        move['leave'] = leave_str
        move['leave_value'] = round(leave_value, 1)
        move['equity'] = round(equity, 1)
        move['tiles_used'] = ''.join(used)

    candidates.sort(key=lambda m: -m['equity'])
    return candidates


# ---------------------------------------------------------------------------
# Agreement and matching
# ---------------------------------------------------------------------------

def moves_agree(a_word, a_row, a_col, a_dir, b_word, b_row, b_col, b_dir):
    """Check if two moves are the same word at the same position."""
    return (a_word == b_word and a_row == b_row
            and a_col == b_col and a_dir == b_dir)


def find_played_in_candidates(played_word, played_row, played_col, played_dir,
                               candidates):
    """Find actual played move among engine candidates.

    Returns (rank, candidate_dict) or (None, None).
    Rank is 1-indexed.
    """
    for i, c in enumerate(candidates):
        if moves_agree(played_word, played_row, played_col, played_dir,
                       c['word'], c['row'], c['col'], c['direction']):
            return i + 1, c
    return None, None


# ---------------------------------------------------------------------------
# Core replay
# ---------------------------------------------------------------------------

def replay_game(record, gaddag, find_moves_fn):
    """Replay a single game through the engine.

    Returns analysis dict or None if no analyzable moves.
    """
    game_id = record.get('game_id', '?')
    opponent = record.get('opponent', '?')
    result = record.get('result', '?')
    your_score = record.get('your_score', 0)
    opp_score = record.get('opp_score', 0)
    spread = record.get('spread', 0)
    moves = record.get('moves', [])

    # Build blank position lookup from game-level data
    game_blanks = {}
    for bp in record.get('blank_positions', []):
        if isinstance(bp, (list, tuple)) and len(bp) >= 3:
            game_blanks[(bp[0], bp[1])] = bp[2]

    board = Board()
    blanks_so_far = []
    tiles_placed_total = 0
    move_details = []
    analyzed_count = 0
    skipped_no_rack = 0
    skipped_exchange = 0
    skipped_opp = 0
    engine_nyt_agree = 0
    engine_played_agree = 0
    nyt_played_agree = 0
    all_three_agree = 0
    total_eq_diff_vs_played = 0.0
    total_eq_diff_vs_nyt = 0.0

    for move_idx, move in enumerate(moves):
        if not isinstance(move, dict):
            # Old tuple format -- place on board but skip analysis
            if isinstance(move, (list, tuple)) and len(move) >= 4:
                word, row, col, horiz = move[0], move[1], move[2], move[3]
                if isinstance(horiz, bool):
                    horizontal = horiz
                else:
                    horizontal = (str(horiz) == 'H')
                try:
                    board.place_word(word, row, col, horizontal)
                    # Update blanks
                    for i, letter in enumerate(word):
                        r = row if horizontal else row + i
                        c = col + i if horizontal else col
                        if (r, c) in game_blanks:
                            blanks_so_far.append((r, c, game_blanks[(r, c)]))
                except Exception:
                    pass
            continue

        player = move.get('player')
        word = move.get('word', '')
        row = move.get('row', 0)
        col = move.get('col', 0)
        dir_str = move.get('dir', 'H')
        horizontal = dir_str == 'H'
        score = move.get('score', 0)
        rack = move.get('rack')
        is_exchange = move.get('is_exchange', False)
        nyt = move.get('nyt')
        nyt_rating = move.get('nyt_rating')

        # Decide whether to analyze
        can_analyze = (player == 'me' and rack and not is_exchange
                       and word and row > 0 and col > 0)

        if player == 'opp':
            skipped_opp += 1
        elif is_exchange:
            skipped_exchange += 1
        elif player == 'me' and not rack:
            skipped_no_rack += 1

        detail = {
            'turn': move_idx + 1,
            'player': player,
            'is_exchange': is_exchange,
            'analyzed': False,
        }

        if can_analyze:
            # --- Run engine analysis ---
            candidates = find_moves_fn(board, gaddag, rack,
                                       board_blanks=blanks_so_far)

            # Estimate bag count
            bag_count = TOTAL_TILES - (2 * RACK_SIZE) - tiles_placed_total
            if bag_count < 0:
                bag_count = 0

            # Compute 1-ply equity for all candidates
            candidates = compute_1ply_equity(candidates, rack, board, bag_count)

            # Engine top 3
            engine_top3 = []
            for c in candidates[:3]:
                engine_top3.append({
                    'word': c['word'],
                    'row': c['row'],
                    'col': c['col'],
                    'dir': c['direction'],
                    'score': c['score'],
                    'equity': c['equity'],
                    'leave': c['leave'],
                    'leave_value': c['leave_value'],
                })

            engine_top = engine_top3[0] if engine_top3 else None

            # Find played move in candidates
            played_rank, played_candidate = find_played_in_candidates(
                word, row, col, dir_str, candidates)
            played_equity = played_candidate['equity'] if played_candidate else None

            # NYT best move info
            nyt_word = nyt.get('word') if nyt else None
            nyt_score = nyt.get('score') if nyt else None

            # Agreement checks
            eng_nyt = False
            eng_played = False
            nyt_pld = False

            if engine_top and nyt_word:
                # Engine agrees with NYT if same word
                # (position may differ -- NYT doesn't always report position)
                eng_nyt = (engine_top['word'] == nyt_word)

            if engine_top:
                eng_played = moves_agree(
                    engine_top['word'], engine_top['row'],
                    engine_top['col'], engine_top['dir'],
                    word, row, col, dir_str)

            if nyt_word:
                nyt_pld = (word == nyt_word)

            all3 = eng_nyt and eng_played and nyt_pld

            # Equity diffs
            eq_diff_vs_played = 0.0
            if engine_top and played_equity is not None:
                eq_diff_vs_played = engine_top['equity'] - played_equity

            # For engine vs NYT equity: find NYT's word in our candidates
            nyt_in_candidates = None
            if nyt_word:
                for c in candidates:
                    if c['word'] == nyt_word:
                        nyt_in_candidates = c
                        break

            eq_diff_vs_nyt = 0.0
            if engine_top and nyt_in_candidates:
                eq_diff_vs_nyt = engine_top['equity'] - nyt_in_candidates['equity']

            # Update counters
            analyzed_count += 1
            if eng_nyt:
                engine_nyt_agree += 1
            if eng_played:
                engine_played_agree += 1
            if nyt_pld:
                nyt_played_agree += 1
            if all3:
                all_three_agree += 1
            total_eq_diff_vs_played += eq_diff_vs_played
            total_eq_diff_vs_nyt += abs(eq_diff_vs_nyt)

            detail.update({
                'analyzed': True,
                'played': {
                    'word': word, 'row': row, 'col': col, 'dir': dir_str,
                    'score': score,
                    'equity': played_equity,
                    'rank': played_rank,
                },
                'engine_top': engine_top,
                'engine_top3': engine_top3,
                'nyt_best': {
                    'word': nyt_word, 'score': nyt_score,
                    'equity': nyt_in_candidates['equity'] if nyt_in_candidates else None,
                } if nyt else None,
                'nyt_rating': nyt_rating,
                'engine_nyt_agree': eng_nyt,
                'engine_played_agree': eng_played,
                'nyt_played_agree': nyt_pld,
                'all_three_agree': all3,
                'eq_diff_vs_played': round(eq_diff_vs_played, 1),
                'eq_diff_vs_nyt': round(eq_diff_vs_nyt, 1),
                'rack': rack,
                'bag_count': bag_count,
                'total_candidates': len(candidates),
            })
        else:
            detail.update({
                'played': {
                    'word': word if not is_exchange else 'SWAP',
                    'score': score,
                },
            })

        move_details.append(detail)

        # --- Place move on board ---
        if not is_exchange and word and row > 0 and col > 0:
            try:
                placed = board.place_word(word, row, col, horizontal)
                tiles_placed_total += len(placed)

                # Update blanks_so_far
                for pr, pc, _ in placed:
                    if (pr, pc) in game_blanks:
                        blanks_so_far.append((pr, pc, game_blanks[(pr, pc)]))
            except Exception as e:
                # Board placement failed -- continue anyway
                # Fallback: manually place tiles
                for i, letter in enumerate(word):
                    r = row if horizontal else row + i
                    c = col + i if horizontal else col
                    ri, ci = r - 1, c - 1
                    if 0 <= ri < 15 and 0 <= ci < 15:
                        if not board._grid[ri][ci]:
                            board._grid[ri][ci] = letter
                            tiles_placed_total += 1
                            if (r, c) in game_blanks:
                                blanks_so_far.append((r, c, game_blanks[(r, c)]))

    if analyzed_count == 0:
        return None

    avg_eq_vs_played = round(total_eq_diff_vs_played / analyzed_count, 1)
    avg_eq_vs_nyt = round(total_eq_diff_vs_nyt / analyzed_count, 1)

    return {
        'game_id': game_id,
        'opponent': opponent,
        'result': result,
        'score': '%d-%d' % (your_score, opp_score),
        'spread': spread,
        'engine_version': '18.0.0',
        'nyt_strategy_you': record.get('nyt_strategy_you'),
        'nyt_strategy_opp': record.get('nyt_strategy_opp'),
        'total_moves': len(moves),
        'analyzed_moves': analyzed_count,
        'skipped_no_rack': skipped_no_rack,
        'skipped_exchange': skipped_exchange,
        'skipped_opp': skipped_opp,
        'engine_nyt_agree': engine_nyt_agree,
        'engine_played_agree': engine_played_agree,
        'nyt_played_agree': nyt_played_agree,
        'all_three_agree': all_three_agree,
        'avg_eq_diff_vs_played': avg_eq_vs_played,
        'avg_eq_diff_vs_nyt': avg_eq_vs_nyt,
        'move_details': move_details,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_game_summary(stats):
    """Print replay analysis summary for one game."""
    gid = stats['game_id']
    opp = stats['opponent']
    res = stats['result'].upper()
    score = stats['score']
    spread = stats['spread']
    strat = stats.get('nyt_strategy_you') or '?'
    sign = '+' if spread > 0 else ''

    print('\n' + '=' * 65)
    print('  %s vs %s  --  %s %s (%s%d)' % (gid, opp, res, score, sign, spread))
    print('  NYT Strategy: %s/99 | Engine: %s' % (strat, stats['engine_version']))
    print('=' * 65)

    a = stats['analyzed_moves']
    t = stats['total_moves']
    print('\n  Analyzed: %d/%d moves (your moves with rack)' % (a, t))
    if stats['skipped_no_rack'] > 0:
        print('  Skipped: %d (no rack data), %d exchanges' % (
            stats['skipped_no_rack'], stats['skipped_exchange']))

    def pct(n, d):
        return '%.1f%%' % (100.0 * n / d) if d > 0 else 'N/A'

    print('  Engine agrees with NYT:   %2d/%d (%s)' % (
        stats['engine_nyt_agree'], a, pct(stats['engine_nyt_agree'], a)))
    print('  Engine agrees with play:  %2d/%d (%s)' % (
        stats['engine_played_agree'], a, pct(stats['engine_played_agree'], a)))
    print('  NYT agrees with play:     %2d/%d (%s)' % (
        stats['nyt_played_agree'], a, pct(stats['nyt_played_agree'], a)))
    print('  All three agree:          %2d/%d (%s)' % (
        stats['all_three_agree'], a, pct(stats['all_three_agree'], a)))
    print('  Avg engine equity over played: %+.1f/move' % stats['avg_eq_diff_vs_played'])
    print('  Avg engine-NYT equity gap:     %.1f/move' % stats['avg_eq_diff_vs_nyt'])


def print_move_detail(stats):
    """Print per-move detail."""
    print('\n  %4s %-12s %4s  %-16s %5s  %-12s %4s  %3s %3s %6s' % (
        'Turn', 'Played', 'Pts', 'Engine Top', 'Eq', 'NYT Best', 'Pts',
        'E=N', 'E=P', 'EqDif'))
    print('  %s %s %s  %s %s  %s %s  %s %s %s' % (
        '-' * 4, '-' * 12, '-' * 4, '-' * 16, '-' * 5, '-' * 12, '-' * 4,
        '-' * 3, '-' * 3, '-' * 6))

    for d in stats['move_details']:
        if not d.get('analyzed'):
            continue

        turn = d['turn']
        played = d['played']
        eng = d.get('engine_top')
        nyt = d.get('nyt_best')
        rating = d.get('nyt_rating', '')

        p_word = played['word']
        p_score = played['score']
        p_rank = played.get('rank')

        if eng:
            e_str = '%s' % eng['word']
            e_eq = eng['equity']
        else:
            e_str = '?'
            e_eq = 0

        if nyt and nyt.get('word'):
            n_word = nyt['word']
            n_score = nyt.get('score', 0) or 0
        else:
            n_word = 'N/A'
            n_score = 0

        en = 'Y' if d.get('engine_nyt_agree') else 'N'
        ep = 'Y' if d.get('engine_played_agree') else 'N'
        eq_diff = d.get('eq_diff_vs_played', 0)
        eq_str = '%+.1f' % eq_diff if eq_diff != 0 else '0.0'

        rank_str = ''
        if p_rank and p_rank > 1:
            rank_str = ' (#%d)' % p_rank

        marker = ''
        if eq_diff > 5:
            marker = ' **'

        print('  %4d %-12s %4d  %-16s %5.1f  %-12s %4d  %3s %3s %6s%s' % (
            turn, p_word + rank_str, p_score, e_str, e_eq,
            n_word, n_score, en, ep, eq_str, marker))


def print_aggregate(all_stats):
    """Print aggregate summary across all replayed games."""
    n = len(all_stats)
    if n == 0:
        return

    total_analyzed = sum(s['analyzed_moves'] for s in all_stats)
    total_eng_nyt = sum(s['engine_nyt_agree'] for s in all_stats)
    total_eng_played = sum(s['engine_played_agree'] for s in all_stats)
    total_nyt_played = sum(s['nyt_played_agree'] for s in all_stats)
    total_all3 = sum(s['all_three_agree'] for s in all_stats)

    def pct(n, d):
        return '%.1f%%' % (100.0 * n / d) if d > 0 else 'N/A'

    print('\n' + '#' * 65)
    print('  ENGINE REPLAY ANALYSIS -- %d games, %d positions' % (
        n, total_analyzed))
    print('#' * 65)

    print('\n  Engine agrees with NYT:   %2d/%d (%s)' % (
        total_eng_nyt, total_analyzed, pct(total_eng_nyt, total_analyzed)))
    print('  Engine agrees with play:  %2d/%d (%s)' % (
        total_eng_played, total_analyzed, pct(total_eng_played, total_analyzed)))
    print('  NYT agrees with play:     %2d/%d (%s)' % (
        total_nyt_played, total_analyzed, pct(total_nyt_played, total_analyzed)))
    print('  All three agree:          %2d/%d (%s)' % (
        total_all3, total_analyzed, pct(total_all3, total_analyzed)))

    # Collect all disagreements between engine and NYT
    disagreements = []
    for s in all_stats:
        for d in s['move_details']:
            if not d.get('analyzed'):
                continue
            if not d.get('engine_nyt_agree') and d.get('nyt_best'):
                eng = d.get('engine_top', {})
                nyt = d.get('nyt_best', {})
                disagreements.append({
                    'game_id': s['game_id'],
                    'turn': d['turn'],
                    'engine_word': eng.get('word', '?'),
                    'engine_score': eng.get('score', 0),
                    'engine_equity': eng.get('equity', 0),
                    'nyt_word': nyt.get('word', '?'),
                    'nyt_score': nyt.get('score', 0),
                    'nyt_equity': nyt.get('equity'),
                    'eq_diff': d.get('eq_diff_vs_nyt', 0),
                })

    if disagreements:
        # Sort by absolute equity difference
        disagreements.sort(key=lambda x: abs(x['eq_diff']), reverse=True)
        print('\n  Engine-vs-NYT disagreements (by equity gap):')
        for dg in disagreements[:15]:
            nyt_eq_str = ('%.1f' % dg['nyt_equity']) if dg['nyt_equity'] is not None else '?'
            print('    %s T%d: Engine=%s(%d/%.1feq) vs NYT=%s(%d/%seq) [gap %.1f]' % (
                dg['game_id'], dg['turn'],
                dg['engine_word'], dg['engine_score'], dg['engine_equity'],
                dg['nyt_word'], dg['nyt_score'], nyt_eq_str,
                abs(dg['eq_diff'])))

    # Per-game table
    print('\n  Per-game breakdown:')
    print('  %-16s %6s %5s %6s %6s %6s %7s' % (
        'Game', 'Result', 'Moves', 'E=NYT', 'E=Pld', 'All3', 'AvgEq'))
    print('  %s %s %s %s %s %s %s' % (
        '-' * 16, '-' * 6, '-' * 5, '-' * 6, '-' * 6, '-' * 6, '-' * 7))
    for s in all_stats:
        a = s['analyzed_moves']
        print('  %-16s %6s %5d %6s %6s %6s %+7.1f' % (
            s['game_id'][:16],
            s['result'].upper()[:3],
            a,
            pct(s['engine_nyt_agree'], a),
            pct(s['engine_played_agree'], a),
            pct(s['all_three_agree'], a),
            s['avg_eq_diff_vs_played']))


# ---------------------------------------------------------------------------
# Save to JSON
# ---------------------------------------------------------------------------

def save_analysis(stats, output_dir):
    """Save analysis JSON to games/analysis/."""
    os.makedirs(output_dir, exist_ok=True)
    game_id = stats['game_id']
    path = os.path.join(output_dir, '%s_replay.json' % game_id)
    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)
    return path


def save_summary(all_stats, output_dir):
    """Save aggregate summary JSON."""
    os.makedirs(output_dir, exist_ok=True)

    total_analyzed = sum(s['analyzed_moves'] for s in all_stats)
    total_eng_nyt = sum(s['engine_nyt_agree'] for s in all_stats)
    total_eng_played = sum(s['engine_played_agree'] for s in all_stats)
    total_all3 = sum(s['all_three_agree'] for s in all_stats)

    summary = {
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'engine_version': '18.0.0',
        'games_analyzed': len(all_stats),
        'total_positions': total_analyzed,
        'engine_nyt_agree': total_eng_nyt,
        'engine_played_agree': total_eng_played,
        'all_three_agree': total_all3,
        'engine_nyt_pct': round(100.0 * total_eng_nyt / total_analyzed, 1) if total_analyzed else 0,
        'per_game': [{
            'game_id': s['game_id'],
            'opponent': s['opponent'],
            'result': s['result'],
            'spread': s['spread'],
            'analyzed_moves': s['analyzed_moves'],
            'engine_nyt_agree': s['engine_nyt_agree'],
            'engine_played_agree': s['engine_played_agree'],
            'all_three_agree': s['all_three_agree'],
            'avg_eq_diff_vs_played': s['avg_eq_diff_vs_played'],
        } for s in all_stats],
    }

    path = os.path.join(output_dir, 'summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]
    show_detail = '--detail' in args
    do_save = '--save' in args
    args = [a for a in args if not a.startswith('--')]
    game_filter = args[0] if args else None

    print('Loading engine resources...')
    gaddag = load_resources()
    find_moves = _get_move_finder()
    print('Ready.\n')

    records = load_archive()
    all_stats = []

    for record in records:
        if game_filter and record.get('game_id') != game_filter:
            continue
        if not has_replay_data(record):
            continue

        gid = record.get('game_id', '?')
        t0 = time.time()
        stats = replay_game(record, gaddag, find_moves)
        elapsed = time.time() - t0

        if stats:
            all_stats.append(stats)
            print_game_summary(stats)
            if show_detail:
                print_move_detail(stats)
            print('  (%.1fs)' % elapsed)

    if not all_stats:
        if game_filter:
            print('No replay data found for game %s' % game_filter)
        else:
            print('No games with replay data found in archive.')
        return

    if len(all_stats) > 1:
        print_aggregate(all_stats)

    if do_save:
        output_dir = os.path.join(os.path.dirname(__file__), 'games', 'analysis')
        saved = []
        for stats in all_stats:
            path = save_analysis(stats, output_dir)
            saved.append(path)
        summary_path = save_summary(all_stats, output_dir)
        saved.append(summary_path)
        print('\n  Saved %d files to %s' % (len(saved), output_dir))
        for p in saved:
            print('    %s' % os.path.basename(p))


if __name__ == '__main__':
    main()
