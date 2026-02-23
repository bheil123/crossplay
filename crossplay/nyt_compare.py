"""
NYT Crossplay comparison tool.

Reads archived games with NYT data and produces summary statistics
comparing your moves against NYT's recommendations.

Usage:
    python -m crossplay.nyt_compare              # all games with NYT data
    python -m crossplay.nyt_compare sophie_001    # specific game
    python -m crossplay.nyt_compare --detail      # show every move
"""

import json
import sys
import os
from typing import List, Dict, Optional, Tuple


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


def has_nyt_data(record: Dict) -> bool:
    """Check if a game record has NYT analysis data."""
    return record.get('nyt_strategy_you') is not None


def analyze_game(record: Dict, show_detail: bool = False) -> Optional[Dict]:
    """Analyze a single game's NYT data. Returns stats dict or None."""
    if not has_nyt_data(record):
        return None

    game_id = record.get('game_id', '?')
    opponent = record.get('opponent', '?')
    result = record.get('result', '?')
    your_score = record.get('your_score', 0)
    opp_score = record.get('opp_score', 0)
    spread = record.get('spread', 0)
    moves = record.get('moves', [])

    # Collect per-move stats
    your_moves = []
    opp_moves = []
    your_best_count = 0
    your_excellent_count = 0
    your_great_count = 0
    your_good_count = 0
    your_fair_count = 0
    your_chance_count = 0
    your_weak_count = 0
    your_total_loss = 0  # total equity lost vs NYT best
    opp_total_loss = 0

    for i, move in enumerate(moves):
        nyt = move.get('nyt')
        nyt_rating = move.get('nyt_rating')
        player = move.get('player')
        word = move.get('word', '?')
        score = move.get('score', 0)
        is_exchange = move.get('is_exchange', False)

        if nyt is None:
            continue

        nyt_word = nyt.get('word', '?')
        nyt_score = nyt.get('score', 0)
        nyt_strat = nyt.get('strategy')

        if is_exchange:
            score = 0

        point_diff = (nyt_score or 0) - (score or 0)

        move_info = {
            'turn': i + 1,
            'word': word if not is_exchange else 'SWAP',
            'score': score,
            'nyt_word': nyt_word,
            'nyt_score': nyt_score,
            'nyt_rating': nyt_rating,
            'nyt_strategy': move.get('nyt_strategy'),
            'point_diff': point_diff,
            'player': player,
        }

        if player == 'me':
            your_moves.append(move_info)
            if nyt_rating == 'Best':
                your_best_count += 1
            elif nyt_rating == 'Excellent':
                your_excellent_count += 1
            elif nyt_rating == 'Great':
                your_great_count += 1
            elif nyt_rating == 'Good':
                your_good_count += 1
            elif nyt_rating == 'Fair':
                your_fair_count += 1
            elif nyt_rating == 'Chance to learn':
                your_chance_count += 1
            elif nyt_rating == 'Weak':
                your_weak_count += 1

            if point_diff > 0:
                your_total_loss += point_diff
        else:
            opp_moves.append(move_info)
            if point_diff > 0:
                opp_total_loss += point_diff

    your_rated = len([m for m in your_moves if m['nyt_rating'] is not None])
    opp_rated = len([m for m in opp_moves if m['nyt_rating'] is not None])

    stats = {
        'game_id': game_id,
        'opponent': opponent,
        'result': result,
        'score': f'{your_score}-{opp_score}',
        'spread': spread,
        'strategy_you': record.get('nyt_strategy_you'),
        'strategy_opp': record.get('nyt_strategy_opp'),
        'luck_you': record.get('nyt_luck_you'),
        'luck_opp': record.get('nyt_luck_opp'),
        'your_moves': len(your_moves),
        'your_rated': your_rated,
        'best': your_best_count,
        'excellent': your_excellent_count,
        'great': your_great_count,
        'good': your_good_count,
        'fair': your_fair_count,
        'chance': your_chance_count,
        'weak': your_weak_count,
        'your_pts_lost': your_total_loss,
        'opp_pts_lost': opp_total_loss,
        'your_move_details': your_moves,
        'opp_move_details': opp_moves,
    }

    # Best% = (Best + Excellent) / rated
    if your_rated > 0:
        stats['best_pct'] = round(100.0 * your_best_count / your_rated, 1)
        stats['top2_pct'] = round(100.0 * (your_best_count + your_excellent_count) / your_rated, 1)
        stats['avg_loss_per_move'] = round(your_total_loss / your_rated, 1)
    else:
        stats['best_pct'] = 0
        stats['top2_pct'] = 0
        stats['avg_loss_per_move'] = 0

    return stats


def print_game_summary(stats: Dict):
    """Print summary for one game."""
    gid = stats['game_id']
    opp = stats['opponent']
    res = stats['result'].upper()
    score = stats['score']
    spread = stats['spread']
    strat_y = stats['strategy_you'] or '?'
    strat_o = stats['strategy_opp'] or '?'
    luck_y = stats['luck_you'] or '?'
    luck_o = stats['luck_opp'] or '?'

    sign = '+' if spread > 0 else ''
    print(f'\n{"="*65}')
    print(f'  {gid} vs {opp}  --  {res} {score} ({sign}{spread})')
    print(f'  Strategy: You {strat_y}/99, Opp {strat_o}/99')
    print(f'  Luck: You {luck_y}, Opp {luck_o}')
    print(f'{"="*65}')

    rated = stats['your_rated']
    if rated == 0:
        print('  No rated moves found.')
        return

    print(f'\n  Your moves: {stats["your_moves"]} total, {rated} rated by NYT')
    print(f'  Rating breakdown:')
    print(f'    Best:      {stats["best"]:2d} ({stats["best_pct"]:5.1f}%)')
    print(f'    Excellent: {stats["excellent"]:2d}')
    print(f'    Great:     {stats["great"]:2d}')
    print(f'    Good:      {stats["good"]:2d}')
    print(f'    Fair:      {stats["fair"]:2d}')
    print(f'    Chance:    {stats["chance"]:2d}')
    print(f'    Weak:      {stats["weak"]:2d}')
    print(f'  Top 2 (Best+Excellent): {stats["top2_pct"]:.1f}%')
    print(f'  Points left on table: {stats["your_pts_lost"]} total'
          f' ({stats["avg_loss_per_move"]:.1f}/move)')
    print(f'  Opponent pts left on table: {stats["opp_pts_lost"]}')


def print_move_detail(stats: Dict):
    """Print per-move detail for one game."""
    print(f'\n  {"Turn":>4s} {"Played":>12s} {"Pts":>4s} {"NYT Best":>12s}'
          f' {"Pts":>4s} {"Diff":>5s} {"Rating":>16s} {"Strat":>5s}')
    print(f'  {"-"*4} {"-"*12} {"-"*4} {"-"*12} {"-"*4} {"-"*5}'
          f' {"-"*16} {"-"*5}')

    # Interleave your and opp moves by turn number
    all_moves = stats['your_move_details'] + stats['opp_move_details']
    all_moves.sort(key=lambda m: m['turn'])

    for m in all_moves:
        turn = m['turn']
        player = 'You' if m['player'] == 'me' else 'Opp'
        word = m['word']
        score = m['score'] or 0
        nyt_word = m['nyt_word']
        nyt_score = m['nyt_score'] or 0
        diff = m['point_diff']
        rating = m['nyt_rating'] or ''
        strat = m['nyt_strategy']
        strat_str = str(strat) if strat is not None else ''

        # Highlight misses
        marker = ''
        if diff > 10 and m['player'] == 'me':
            marker = ' **'
        elif diff > 20 and m['player'] == 'opp':
            marker = ' !!'

        diff_str = f'{diff:+d}' if diff != 0 else '0'

        print(f'  {turn:4d} {player:>3s} {word:>8s} {score:4d}'
              f' {nyt_word:>12s} {nyt_score:4d} {diff_str:>5s}'
              f' {rating:>16s} {strat_str:>5s}{marker}')


def print_aggregate(all_stats: List[Dict]):
    """Print aggregate summary across all games."""
    n = len(all_stats)
    if n == 0:
        print('\nNo games with NYT data found.')
        return

    total_best = sum(s['best'] for s in all_stats)
    total_excellent = sum(s['excellent'] for s in all_stats)
    total_great = sum(s['great'] for s in all_stats)
    total_good = sum(s['good'] for s in all_stats)
    total_fair = sum(s['fair'] for s in all_stats)
    total_chance = sum(s['chance'] for s in all_stats)
    total_weak = sum(s['weak'] for s in all_stats)
    total_rated = sum(s['your_rated'] for s in all_stats)
    total_your_loss = sum(s['your_pts_lost'] for s in all_stats)
    total_opp_loss = sum(s['opp_pts_lost'] for s in all_stats)

    wins = sum(1 for s in all_stats if s['result'] == 'win')
    losses = sum(1 for s in all_stats if s['result'] == 'loss')
    ties = n - wins - losses
    avg_spread = sum(s['spread'] for s in all_stats) / n

    strats_you = [s['strategy_you'] for s in all_stats if s['strategy_you']]
    strats_opp = [s['strategy_opp'] for s in all_stats if s['strategy_opp']]
    lucks_you = [s['luck_you'] for s in all_stats if s['luck_you']]
    lucks_opp = [s['luck_opp'] for s in all_stats if s['luck_opp']]

    avg_strat_you = sum(strats_you) / len(strats_you) if strats_you else 0
    avg_strat_opp = sum(strats_opp) / len(strats_opp) if strats_opp else 0
    avg_luck_you = sum(lucks_you) / len(lucks_you) if lucks_you else 0
    avg_luck_opp = sum(lucks_opp) / len(lucks_opp) if lucks_opp else 0

    best_pct = round(100.0 * total_best / total_rated, 1) if total_rated else 0
    top2_pct = round(100.0 * (total_best + total_excellent) / total_rated, 1) if total_rated else 0
    avg_loss = round(total_your_loss / total_rated, 1) if total_rated else 0

    print(f'\n{"#"*65}')
    print(f'  AGGREGATE NYT COMPARISON -- {n} games')
    print(f'{"#"*65}')
    print(f'\n  Record: {wins}W-{losses}L-{ties}T  Avg spread: {avg_spread:+.1f}')
    print(f'  Avg Strategy: You {avg_strat_you:.1f}/99, Opp {avg_strat_opp:.1f}/99')
    print(f'  Avg Luck: You {avg_luck_you:.1f}, Opp {avg_luck_opp:.1f}')
    print(f'\n  Your moves: {total_rated} rated across {n} games')
    print(f'  Rating breakdown:')
    print(f'    Best:      {total_best:3d} ({best_pct:5.1f}%)')
    print(f'    Excellent: {total_excellent:3d}')
    print(f'    Great:     {total_great:3d}')
    print(f'    Good:      {total_good:3d}')
    print(f'    Fair:      {total_fair:3d}')
    print(f'    Chance:    {total_chance:3d}')
    print(f'    Weak:      {total_weak:3d}')
    print(f'  Top 2 (Best+Excellent): {top2_pct:.1f}%')
    print(f'  Total pts left on table: {total_your_loss}'
          f' ({avg_loss:.1f}/move)')
    print(f'  Opponent total pts lost: {total_opp_loss}')

    # Biggest misses across all games
    all_misses = []
    for s in all_stats:
        for m in s['your_move_details']:
            if m['point_diff'] > 0:
                all_misses.append((s['game_id'], m))
    all_misses.sort(key=lambda x: x[1]['point_diff'], reverse=True)

    if all_misses:
        print(f'\n  Your biggest misses (vs NYT best):')
        for game_id, m in all_misses[:10]:
            turn = m['turn']
            word = m['word']
            score = m['score'] or 0
            nyt_word = m['nyt_word']
            nyt_score = m['nyt_score'] or 0
            diff = m['point_diff']
            print(f'    {game_id} T{turn}: {word}({score})'
                  f' -> {nyt_word}({nyt_score}) [{diff:+d}]')

    # Opponent's biggest misses
    opp_misses = []
    for s in all_stats:
        for m in s['opp_move_details']:
            if m['point_diff'] > 0:
                opp_misses.append((s['game_id'], m))
    opp_misses.sort(key=lambda x: x[1]['point_diff'], reverse=True)

    if opp_misses:
        print(f'\n  Opponent biggest misses:')
        for game_id, m in opp_misses[:10]:
            turn = m['turn']
            word = m['word']
            score = m['score'] or 0
            nyt_word = m['nyt_word']
            nyt_score = m['nyt_score'] or 0
            diff = m['point_diff']
            print(f'    {game_id} T{turn}: {word}({score})'
                  f' -> {nyt_word}({nyt_score}) [{diff:+d}]')

    # Per-game summary table
    print(f'\n  Per-game breakdown:')
    print(f'  {"Game":<16s} {"Result":>6s} {"Spread":>7s} {"Strat":>5s}'
          f' {"Best%":>6s} {"Top2%":>6s} {"PtsLost":>8s}')
    print(f'  {"-"*16} {"-"*6} {"-"*7} {"-"*5} {"-"*6} {"-"*6} {"-"*8}')
    for s in all_stats:
        gid = s['game_id'][:16]
        res = s['result'].upper()[:3]
        spr = f'{s["spread"]:+d}'
        strat = str(s['strategy_you'] or '?')
        bp = f'{s["best_pct"]:.0f}%'
        t2 = f'{s["top2_pct"]:.0f}%'
        pl = str(s['your_pts_lost'])
        print(f'  {gid:<16s} {res:>6s} {spr:>7s} {strat:>5s}'
              f' {bp:>6s} {t2:>6s} {pl:>8s}')


def main():
    args = sys.argv[1:]
    show_detail = '--detail' in args
    if '--detail' in args:
        args.remove('--detail')

    game_filter = args[0] if args else None

    records = load_archive()
    all_stats = []

    for record in records:
        if game_filter and record.get('game_id') != game_filter:
            continue
        stats = analyze_game(record, show_detail)
        if stats:
            all_stats.append(stats)

    if not all_stats:
        if game_filter:
            print(f'No NYT data found for game {game_filter}')
        else:
            print('No games with NYT data found in archive.')
        return

    for stats in all_stats:
        print_game_summary(stats)
        if show_detail:
            print_move_detail(stats)

    if len(all_stats) > 1:
        print_aggregate(all_stats)
    elif len(all_stats) == 1 and not show_detail:
        print('\n  (Use --detail to see move-by-move breakdown)')


if __name__ == '__main__':
    main()
