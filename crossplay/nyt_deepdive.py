"""Deep dive: Run full MC analysis on all positions across archived NYT games.

Reconstructs board positions from archive data and runs the complete
analyze() pipeline (1-ply + risk + MC 2-ply) to compare engine
recommendations vs NYT vs actual play.

Since MC evaluation is already parallelized internally (multi-worker),
positions run sequentially to avoid CPU contention.

Usage:
    python -m crossplay.nyt_deepdive [--game GAME_ID] [--save] [--quiet]

    Without --game: runs all games with NYT data
    --game ID: run only the specified game
    --save: write JSON results to games/analysis/
    --quiet: suppress per-position analyze() output (show summary only)
"""

import json
import sys
import io
import os
import time
from datetime import datetime


def load_archive():
    """Load all archive records keyed by game_id."""
    archive_path = os.path.join(os.path.dirname(__file__), 'games', 'archive.jsonl')
    records = {}
    with open(archive_path) as f:
        for line in f:
            rec = json.loads(line)
            records[rec['game_id']] = rec
    return records


def reconstruct_position(record, turn_num):
    """Reconstruct game state just before turn_num (1-indexed).

    Returns (GameState, move_dict) where move_dict is the move that was played.
    """
    from .game_manager import GameState

    moves = record['moves']
    move = moves[turn_num - 1]
    board_moves = moves[:turn_num - 1]

    if turn_num >= 2:
        prev = moves[turn_num - 2]
        your_score = prev['cumulative'][0]
        opp_score = prev['cumulative'][1]
    else:
        your_score = 0
        opp_score = 0

    rack = move.get('rack')
    all_blanks = record.get('blank_positions', [])

    state = GameState(
        name="Replay %s T%d" % (record['game_id'], turn_num),
        board_moves=board_moves,
        blank_positions=all_blanks,
        your_score=your_score,
        opp_score=opp_score,
        your_rack=rack or "",
        is_your_turn=(move.get('player') == 'me'),
        opponent_name=record.get('opponent', ''),
        created_at=record.get('created_at', ''),
        updated_at='',
    )
    return state, move


def moves_match(word_a, row_a, col_a, dir_a, word_b, row_b, col_b, dir_b):
    """Check if two moves are the same play."""
    return (word_a == word_b and row_a == row_b
            and col_a == col_b and dir_a == dir_b)


def analyze_position(record, turn_num, game_obj, quiet=False):
    """Run full analyze() on a single position and capture structured results.

    Args:
        record: Archive record dict
        turn_num: 1-indexed turn number
        game_obj: Reusable Game object (board will be rebuilt)
        quiet: If True, suppress analyze() console output

    Returns dict with analysis results.
    """
    move = record['moves'][turn_num - 1]
    played_word = move.get('word', '?')
    played_score = move.get('score', 0)
    played_row = move.get('row', 0)
    played_col = move.get('col', 0)
    played_dir = move.get('dir', 'H')
    is_exchange = move.get('is_exchange', False)
    rack = move.get('rack', '')
    bag = move.get('bag', 0)
    nyt = move.get('nyt', {})
    nyt_word = nyt.get('word', '?') if nyt else None
    nyt_score = nyt.get('score', 0) if nyt else None
    nyt_rating = move.get('nyt_rating')
    nyt_strategy = move.get('nyt_strategy')

    if not rack:
        return {
            'turn': turn_num,
            'skipped': True,
            'reason': 'no rack data',
            'played': played_word,
            'played_score': played_score,
        }

    # Reconstruct position
    state, _ = reconstruct_position(record, turn_num)

    # Rebuild game object with new state
    from .game_manager import GameState, Game
    from .board import Board

    game_obj.state = state
    game_obj.board = Board()
    for m in state.board_moves:
        if isinstance(m, dict):
            if m.get('is_exchange'):
                continue
            word = m['word']
            row, col = m['row'], m['col']
            horiz = m.get('dir', 'H') == 'H' if 'dir' in m else m.get('horizontal', True)
        else:
            word, row, col, horiz = m[0], m[1], m[2], m[3]
        game_obj.board.place_word(word, row, col, horiz)
    game_obj.bag = game_obj._reconstruct_bag()
    game_obj.blocked_cache.initialize(game_obj.board, game_obj.dictionary)
    game_obj._threats_cache = None
    game_obj._cached_baseline_risk = 0.0
    game_obj._cached_baseline_threats = []
    game_obj.last_analysis = None

    bag_size = len(game_obj.bag)

    # Run full analysis
    t0 = time.time()
    if quiet:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

    try:
        game_obj.analyze(rack=rack, top_n=10)
    except Exception as e:
        if quiet:
            sys.stdout = old_stdout
        return {
            'turn': turn_num,
            'skipped': True,
            'reason': 'analyze error: %s' % str(e),
            'played': played_word,
            'played_score': played_score,
        }

    if quiet:
        sys.stdout = old_stdout

    elapsed = time.time() - t0

    # Extract engine recommendation from last_analysis
    engine_top = None
    engine_top2 = None
    engine_top3 = None
    mc_ran = False
    eval_type = 'unknown'
    engine_recommended_exchange = False

    if game_obj.last_analysis:
        # Prefer mc_top (includes exchanges) over top3 (filters exchanges)
        mc_top = game_obj.last_analysis.get('mc_top') or []
        top3 = game_obj.last_analysis.get('top3') or []

        # Check if engine recommended exchange (mc_top[0] is exchange)
        if mc_top and mc_top[0].get('is_exchange', False):
            engine_recommended_exchange = True

        # For engine recommendation, use first non-exchange from mc_top,
        # falling back to top3 (which already filters exchanges)
        non_exch = [m for m in mc_top if not m.get('is_exchange', False)]
        best_list = non_exch if non_exch else top3

        if len(best_list) >= 1:
            engine_top = best_list[0]
        if len(best_list) >= 2:
            engine_top2 = best_list[1]
        if len(best_list) >= 3:
            engine_top3 = best_list[2]

        if engine_top or top3 or mc_top:
            mc_ran = True

        # Determine eval type
        if bag_size == 0:
            eval_type = 'endgame_exact'
        elif 1 <= bag_size <= 8:
            eval_type = 'near_endgame'
        else:
            eval_type = 'mc_2ply'

    # Check agreements
    eng_word = engine_top['word'] if engine_top else None
    eng_row = engine_top['row'] if engine_top else None
    eng_col = engine_top['col'] if engine_top else None
    eng_dir = engine_top.get('dir') if engine_top else None
    eng_score = engine_top['score'] if engine_top else None
    eng_equity = engine_top.get('equity', 0) if engine_top else None
    eng_risk_eq = engine_top.get('risk_eq', 0) if engine_top else None

    engine_agrees_nyt = False
    engine_agrees_played = False
    nyt_agrees_played = False

    if eng_word and nyt_word:
        engine_agrees_nyt = (eng_word == nyt_word)
    if eng_word and not is_exchange:
        engine_agrees_played = moves_match(
            eng_word, eng_row, eng_col, eng_dir,
            played_word, played_row, played_col, played_dir)
    if nyt_word and not is_exchange:
        nyt_agrees_played = (nyt_word == played_word)

    # Move finder diagnostic: check if NYT word is in full candidate list
    nyt_in_movefinder = None
    nyt_movefinder_rank = None
    nyt_movefinder_score = None
    if nyt_word and nyt_word != '?' and not is_exchange:
        try:
            from .move_finder_c import find_all_moves_c
            all_candidates = find_all_moves_c(
                game_obj.board, game_obj.gaddag, rack)
            nyt_matches = [m for m in all_candidates if m['word'] == nyt_word]
            nyt_in_movefinder = len(nyt_matches) > 0
            if nyt_matches:
                best_nyt = max(nyt_matches, key=lambda m: m['score'])
                nyt_movefinder_score = best_nyt['score']
                # Find rank by raw score
                all_candidates.sort(key=lambda x: -x['score'])
                for i, m in enumerate(all_candidates):
                    if m['word'] == nyt_word:
                        nyt_movefinder_rank = i + 1
                        break
        except Exception:
            pass  # diagnostic is best-effort

    result = {
        'turn': turn_num,
        'skipped': False,
        'rack': rack,
        'bag_size': bag_size,
        'eval_type': eval_type,
        'elapsed_s': round(elapsed, 1),
        # What was played
        'played': played_word,
        'played_score': played_score,
        'played_row': played_row,
        'played_col': played_col,
        'played_dir': played_dir,
        'is_exchange': is_exchange,
        # NYT recommendation
        'nyt_word': nyt_word,
        'nyt_score': nyt_score,
        'nyt_rating': nyt_rating,
        'nyt_strategy': nyt_strategy,
        # Engine recommendation (best non-exchange move)
        'engine_word': eng_word,
        'engine_score': eng_score,
        'engine_equity': round(eng_equity, 1) if eng_equity is not None else None,
        'engine_risk_eq': round(eng_risk_eq, 1) if eng_risk_eq is not None else None,
        'engine_recommended_exchange': engine_recommended_exchange,
        # Engine top 3 (non-exchange)
        'engine_top3': [
            {
                'word': t['word'],
                'score': t['score'],
                'equity': round(t.get('equity', 0), 1),
                'risk_eq': round(t.get('risk_eq', t.get('equity', 0)), 1),
                'row': t['row'],
                'col': t['col'],
                'dir': t.get('dir'),
            }
            for t in [engine_top, engine_top2, engine_top3]
            if t is not None
        ],
        # Move finder diagnostic
        'nyt_in_movefinder': nyt_in_movefinder,
        'nyt_movefinder_rank': nyt_movefinder_rank,
        'nyt_movefinder_score': nyt_movefinder_score,
        # Agreements
        'engine_agrees_nyt': engine_agrees_nyt,
        'engine_agrees_played': engine_agrees_played,
        'nyt_agrees_played': nyt_agrees_played,
        'all_three_agree': engine_agrees_nyt and engine_agrees_played and nyt_agrees_played,
    }

    return result


def run_game(game_id, records, quiet=False):
    """Run full analysis on all your-move positions in a game."""
    from .game_manager import Game

    rec = records[game_id]
    moves = rec['moves']

    # Find all analyzable positions (your moves with rack)
    positions = []
    for i, m in enumerate(moves):
        if m.get('player') == 'me' and m.get('rack'):
            positions.append(i + 1)  # 1-indexed

    if not positions:
        print("  No analyzable positions in %s" % game_id)
        return {'game_id': game_id, 'positions': [], 'skipped': True}

    print("=" * 70)
    print("GAME: %s vs %s -- %s %d-%d (%+d)" % (
        game_id, rec.get('opponent', '?'),
        rec.get('result', '?').upper(),
        rec.get('your_score', 0), rec.get('opp_score', 0),
        rec.get('spread', 0)))
    print("  NYT Strategy: %s/%s" % (
        rec.get('nyt_strategy_you', '?'), rec.get('nyt_strategy_opp', '?')))
    print("  Positions to analyze: %d" % len(positions))
    print("=" * 70)

    # Create ONE Game object, reuse for all positions (avoid re-loading GADDAG)
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    game = Game()
    sys.stdout = old_stdout

    results = []
    for pos_num, turn in enumerate(positions):
        move = moves[turn - 1]
        played = move.get('word', '?')
        score = move.get('score', 0)
        nyt = move.get('nyt', {})
        nyt_word = nyt.get('word', '?') if nyt else '?'
        nyt_rating = move.get('nyt_rating', '')

        print("\n  [%d/%d] T%d: %s(%d) | NYT: %s | Rating: %s | Rack: %s" % (
            pos_num + 1, len(positions), turn, played, score,
            nyt_word, nyt_rating, move.get('rack', '?')))

        t0 = time.time()
        result = analyze_position(rec, turn, game, quiet=quiet)
        elapsed = time.time() - t0

        if result.get('skipped'):
            print("    SKIPPED: %s" % result.get('reason', '?'))
        else:
            eng = result.get('engine_word') or '?'
            eng_eq = result.get('engine_equity') or 0
            eng_score = result.get('engine_score') or 0
            e_n = "Y" if result.get('engine_agrees_nyt') else "N"
            e_p = "Y" if result.get('engine_agrees_played') else "N"
            print("    -> Engine: %s(%d, %.1feq) | E=NYT:%s E=Play:%s | %.1fs" % (
                eng, eng_score, eng_eq, e_n, e_p, elapsed))

        results.append(result)

    # Compute game summary
    analyzed = [r for r in results if not r.get('skipped')]
    if analyzed:
        eng_nyt = sum(1 for r in analyzed if r.get('engine_agrees_nyt'))
        eng_play = sum(1 for r in analyzed if r.get('engine_agrees_played'))
        nyt_play = sum(1 for r in analyzed if r.get('nyt_agrees_played'))
        all3 = sum(1 for r in analyzed if r.get('all_three_agree'))
        n = len(analyzed)

        # Equity over played: engine equity - played score (approximation)
        # More meaningful: how often engine top pick != played
        engine_different = sum(1 for r in analyzed
                               if not r.get('engine_agrees_played') and not r.get('is_exchange'))

        # Avg time
        avg_time = sum(r.get('elapsed_s', 0) for r in analyzed) / n

        summary = {
            'game_id': game_id,
            'opponent': rec.get('opponent'),
            'result': rec.get('result'),
            'your_score': rec.get('your_score'),
            'opp_score': rec.get('opp_score'),
            'spread': rec.get('spread'),
            'nyt_strategy_you': rec.get('nyt_strategy_you'),
            'nyt_strategy_opp': rec.get('nyt_strategy_opp'),
            'positions_analyzed': n,
            'positions_skipped': len(results) - n,
            'engine_agrees_nyt': eng_nyt,
            'engine_agrees_nyt_pct': round(100 * eng_nyt / n, 1),
            'engine_agrees_played': eng_play,
            'engine_agrees_played_pct': round(100 * eng_play / n, 1),
            'nyt_agrees_played': nyt_play,
            'nyt_agrees_played_pct': round(100 * nyt_play / n, 1),
            'all_three_agree': all3,
            'all_three_agree_pct': round(100 * all3 / n, 1),
            'engine_would_change': engine_different,
            'avg_time_per_position': round(avg_time, 1),
            'total_time': round(sum(r.get('elapsed_s', 0) for r in analyzed), 1),
        }

        print()
        print("-" * 70)
        print("  GAME SUMMARY: %s" % game_id)
        print("  Analyzed: %d positions" % n)
        print("  Engine agrees with NYT:  %d/%d (%.1f%%)" % (eng_nyt, n, 100*eng_nyt/n))
        print("  Engine agrees with play: %d/%d (%.1f%%)" % (eng_play, n, 100*eng_play/n))
        print("  NYT agrees with play:    %d/%d (%.1f%%)" % (nyt_play, n, 100*nyt_play/n))
        print("  All three agree:         %d/%d (%.1f%%)" % (all3, n, 100*all3/n))
        print("  Engine would change:     %d moves" % engine_different)
        print("  Avg time/position:       %.1fs" % avg_time)
        print("-" * 70)
    else:
        summary = {'game_id': game_id, 'positions_analyzed': 0, 'skipped': True}

    return {
        'summary': summary,
        'positions': results,
    }


def print_aggregate(game_results):
    """Print aggregate summary across all games."""
    all_positions = []
    for gr in game_results:
        all_positions.extend([r for r in gr.get('positions', []) if not r.get('skipped')])

    if not all_positions:
        print("No positions analyzed.")
        return

    n = len(all_positions)
    eng_nyt = sum(1 for r in all_positions if r.get('engine_agrees_nyt'))
    eng_play = sum(1 for r in all_positions if r.get('engine_agrees_played'))
    nyt_play = sum(1 for r in all_positions if r.get('nyt_agrees_played'))
    all3 = sum(1 for r in all_positions if r.get('all_three_agree'))

    # Disagreement breakdown
    eng_diff_nyt = [r for r in all_positions
                    if not r.get('engine_agrees_nyt') and r.get('nyt_word')]

    print()
    print("#" * 70)
    print("  AGGREGATE DEEP DIVE -- %d games, %d positions" % (
        len(game_results), n))
    print("#" * 70)
    print()
    print("  Engine agrees with NYT:  %d/%d (%.1f%%)" % (eng_nyt, n, 100*eng_nyt/n))
    print("  Engine agrees with play: %d/%d (%.1f%%)" % (eng_play, n, 100*eng_play/n))
    print("  NYT agrees with play:    %d/%d (%.1f%%)" % (nyt_play, n, 100*nyt_play/n))
    print("  All three agree:         %d/%d (%.1f%%)" % (all3, n, 100*all3/n))
    print()

    # Disagreement category breakdown
    if eng_diff_nyt:
        # Categorize each disagreement
        cat_exchange = []     # engine recommended exchange
        cat_not_found = []    # NYT word not in move finder
        cat_lower_eq = []     # NYT word found but lower equity
        cat_unknown = []      # no diagnostic data
        cat_engine_higher = [] # engine scores same/higher than NYT

        for r in eng_diff_nyt:
            if r.get('engine_recommended_exchange'):
                cat_exchange.append(r)
            elif r.get('nyt_in_movefinder') is False:
                cat_not_found.append(r)
            elif r.get('nyt_in_movefinder') is True:
                eng_s = r.get('engine_score') or 0
                nyt_s = r.get('nyt_score') or 0
                if eng_s >= nyt_s:
                    cat_engine_higher.append(r)
                else:
                    cat_lower_eq.append(r)
            elif r.get('eval_type') == 'unknown':
                cat_unknown.append(r)
            else:
                # No diagnostic data but not unknown eval type
                cat_lower_eq.append(r)

        print("  Disagreement categories (%d total):" % len(eng_diff_nyt))
        if cat_exchange:
            print("    Engine recommended EXCHANGE:  %d" % len(cat_exchange))
            for r in cat_exchange:
                print("      %s T%d: NYT=%s(%s) | Engine top non-exch=%s(%s)" % (
                    r.get('game_id', '?'), r['turn'],
                    r.get('nyt_word', '?'), r.get('nyt_score', '?'),
                    r.get('engine_word', '?'), r.get('engine_score', '?')))
        if cat_not_found:
            print("    NYT word NOT in move finder:  %d  ** MOVE FINDER BUG **" % len(cat_not_found))
            for r in cat_not_found:
                print("      %s T%d: NYT=%s(%s) not generated by engine" % (
                    r.get('game_id', '?'), r['turn'],
                    r.get('nyt_word', '?'), r.get('nyt_score', '?')))
        if cat_lower_eq:
            print("    NYT word found, lower equity: %d  (strategic diff)" % len(cat_lower_eq))
            for r in cat_lower_eq:
                rank = r.get('nyt_movefinder_rank', '?')
                print("      %s T%d: Engine=%s(%s) vs NYT=%s(%s, rank #%s by score)" % (
                    r.get('game_id', '?'), r['turn'],
                    r.get('engine_word', '?'), r.get('engine_score', '?'),
                    r.get('nyt_word', '?'), r.get('nyt_score', '?'), rank))
        if cat_engine_higher:
            print("    Engine scores >= NYT:         %d  (engine correct)" % len(cat_engine_higher))
            for r in cat_engine_higher:
                print("      %s T%d: Engine=%s(%s) vs NYT=%s(%s)" % (
                    r.get('game_id', '?'), r['turn'],
                    r.get('engine_word', '?'), r.get('engine_score', '?'),
                    r.get('nyt_word', '?'), r.get('nyt_score', '?')))
        if cat_unknown:
            print("    Unknown (no engine data):     %d" % len(cat_unknown))
            for r in cat_unknown:
                print("      %s T%d: NYT=%s(%s) | eval_type=%s" % (
                    r.get('game_id', '?'), r['turn'],
                    r.get('nyt_word', '?'), r.get('nyt_score', '?'),
                    r.get('eval_type', '?')))
        print()

    # Top disagreements by score gap (for reference)
    if eng_diff_nyt:
        print("  Top disagreements by score gap:")
        sorted_dis = sorted(eng_diff_nyt,
                            key=lambda r: abs((r.get('nyt_score') or 0) - (r.get('engine_score') or 0)),
                            reverse=True)
        for r in sorted_dis[:10]:
            gid = r.get('game_id', '?')
            exch_tag = " [EXCH]" if r.get('engine_recommended_exchange') else ""
            diag_tag = ""
            if r.get('nyt_in_movefinder') is True:
                diag_tag = " [found #%s]" % (r.get('nyt_movefinder_rank', '?'))
            elif r.get('nyt_in_movefinder') is False:
                diag_tag = " [NOT FOUND]"
            print("    %s T%d: Engine=%s(%s) vs NYT=%s(%s) | gap=%d%s%s" % (
                gid, r['turn'],
                r.get('engine_word') or '?', r.get('engine_score') or '?',
                r.get('nyt_word') or '?', r.get('nyt_score') or '?',
                abs((r.get('nyt_score') or 0) - (r.get('engine_score') or 0)),
                exch_tag, diag_tag))

    # Eval type breakdown
    types = {}
    for r in all_positions:
        t = r.get('eval_type', 'unknown')
        types[t] = types.get(t, 0) + 1
    print()
    print("  Evaluation types: %s" % ', '.join('%s=%d' % (k, v) for k, v in sorted(types.items())))

    # Total time
    total_t = sum(r.get('elapsed_s', 0) for r in all_positions)
    print("  Total analysis time: %.1fs (%.1f min)" % (total_t, total_t / 60))


def save_results(game_results, output_dir=None):
    """Save results to JSON."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'games', 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Per-game files
    for gr in game_results:
        gid = gr.get('summary', {}).get('game_id', 'unknown')
        path = os.path.join(output_dir, '%s_deepdive.json' % gid)
        with open(path, 'w') as f:
            json.dump(gr, f, indent=2)
        print("  Saved: %s" % path)

    # Aggregate summary
    summaries = [gr['summary'] for gr in game_results if 'summary' in gr]
    agg_path = os.path.join(output_dir, 'deepdive_summary.json')
    with open(agg_path, 'w') as f:
        json.dump({
            'generated': datetime.now().isoformat(),
            'engine_version': '18.0.0',
            'games': summaries,
        }, f, indent=2)
    print("  Saved: %s" % agg_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Full MC deep dive on NYT games')
    parser.add_argument('--game', type=str, default=None,
                        help='Run only the specified game ID')
    parser.add_argument('--save', action='store_true',
                        help='Save JSON results to games/analysis/')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-position analyze() output')
    args = parser.parse_args()

    records = load_archive()

    # Find games with NYT data
    nyt_games = [gid for gid, rec in records.items() if rec.get('nyt_strategy_you')]

    if args.game:
        if args.game not in records:
            print("Game %s not found in archive" % args.game)
            return
        nyt_games = [args.game]

    if not nyt_games:
        print("No games with NYT data found")
        return

    print("NYT DEEP DIVE ANALYSIS (FULL MC 2-PLY)")
    print("Games: %s" % ', '.join(nyt_games))

    # Count total positions
    total_pos = 0
    for gid in nyt_games:
        rec = records[gid]
        total_pos += sum(1 for m in rec['moves']
                         if m.get('player') == 'me' and m.get('rack'))
    print("Total positions: %d (estimated %d-%d minutes)" % (
        total_pos, total_pos * 20 // 60, total_pos * 40 // 60))
    print()

    t_total = time.time()
    game_results = []
    for gid in nyt_games:
        result = run_game(gid, records, quiet=args.quiet)
        # Tag positions with game_id for aggregate reporting
        for pos in result.get('positions', []):
            pos['game_id'] = gid
        game_results.append(result)

    elapsed = time.time() - t_total
    print_aggregate(game_results)

    print()
    print("=" * 70)
    print("ALL GAMES COMPLETE in %.1fs (%.1f min)" % (elapsed, elapsed / 60))
    print("=" * 70)

    if args.save:
        print()
        save_results(game_results)


if __name__ == '__main__':
    main()
