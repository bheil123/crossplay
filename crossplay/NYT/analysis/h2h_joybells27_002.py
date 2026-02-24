"""Head-to-head analysis: engine vs NYT for joybells27_002."""
import sys
sys.path.insert(0, '.')
from crossplay.game_manager import Game
from crossplay.move_finder_c import find_all_moves_c
from crossplay.leave_eval import evaluate_leave

moves = [
    {'word':'BOHEA','row':4,'col':8,'dir':'V','player':'opp','score':26},
    {'word':'ANTINUKE','row':8,'col':8,'dir':'H','player':'me','score':72,'rack':'NENITUK'},
    {'word':'DENI','row':9,'col':11,'dir':'H','player':'opp','score':24},
    {'word':'WEAVE','row':4,'col':15,'dir':'V','player':'me','score':42,'rack':'AWVXETA'},
    {'word':'HANKIES','row':9,'col':2,'dir':'H','player':'opp','score':65},
    {'word':'FETE','row':8,'col':1,'dir':'H','player':'me','score':32,'rack':'FPEXETA'},
    {'word':'OH','row':7,'col':1,'dir':'H','player':'opp','score':28},
    {'word':'SPITZ','row':7,'col':6,'dir':'V','player':'me','score':39,'rack':'ZPIXSTA'},
    {'word':'MOVE','row':1,'col':14,'dir':'V','player':'opp','score':28},
    {'word':'AXIOM','row':1,'col':10,'dir':'H','player':'me','score':42,'rack':'ORIXLOA'},
    {'word':'RACY','row':2,'col':7,'dir':'H','player':'opp','score':22},
    {'word':'ROYAL','row':1,'col':4,'dir':'H','player':'me','score':40,'rack':'ORYSLCA'},
    {'word':'EQUIDS','row':5,'col':11,'dir':'V','player':'opp','score':39},
    {'word':'DE','row':5,'col':14,'dir':'H','player':'me','score':16,'rack':'MTLSCD?'},
    {'word':'PUB','row':3,'col':5,'dir':'H','player':'opp','score':18},
    {'word':'MUSCAT','row':4,'col':1,'dir':'H','player':'me','score':43,'rack':'MTLSCA?'},
    {'word':'AFAR','row':11,'col':7,'dir':'V','player':'opp','score':20},
    {'word':'TROWELS','row':15,'col':1,'dir':'H','player':'me','score':90,'rack':'WTLORSE'},
    {'word':'ODOR','row':14,'col':2,'dir':'H','player':'opp','score':25},
    {'word':'GEN','row':11,'col':8,'dir':'V','player':'me','score':39,'rack':'GNIIEIE'},
]

# NYT best recommendations for each of our turns
nyt_best = {
    2:  {'word':'ANTINUKE','score':72,'strat':99,'rating':'Best'},
    4:  {'word':'WEAVE','score':42,'strat':99,'rating':'Best'},
    6:  {'word':'FETE','score':32,'strat':99,'rating':'Best'},
    8:  {'word':'SPITZ','score':39,'strat':99,'rating':'Best'},
    10: {'word':'AXIOM','score':42,'strat':99,'rating':'Best'},
    12: {'word':'ROYAL','score':40,'strat':99,'rating':'Best'},
    14: {'word':'CLOTHED','score':16,'strat':99,'rating':'Best'},
    16: {'word':'SATCOM','score':44,'strat':99,'rating':'Best'},
    18: {'word':'TROWELS','score':90,'strat':99,'rating':'Best'},
    20: {'word':'GENE','score':43,'strat':99,'rating':'Best'},
}

game = Game()
blank_positions = []

print('=' * 90)
print('ENGINE vs NYT HEAD-TO-HEAD: joybells27_002 (WIN 455-310)')
print('Engine version: 18.0.0 | Analysis: 1-ply (score + leave equity)')
print('=' * 90)

results = []

for i, m in enumerate(moves):
    horiz = m['dir'] == 'H'
    if i == 4:
        blank_positions.append((9, 5, 'K'))

    if m['player'] == 'me':
        rack = m.get('rack', '')
        turn_num = i + 1

        game.state.your_rack = rack
        game.state.blank_positions = list(blank_positions)

        all_moves = find_all_moves_c(game.board, game.gaddag, rack,
                                     board_blanks=blank_positions)

        for mv in all_moves:
            tiles_used = mv.get('tiles_used', mv.get('used', mv['word']))
            rack_copy = list(rack.upper())
            for t in str(tiles_used).upper():
                if t in rack_copy:
                    rack_copy.remove(t)
                elif '?' in rack_copy:
                    rack_copy.remove('?')
            leave = ''.join(rack_copy)
            mv['leave'] = leave
            mv['leave_val'] = evaluate_leave(leave)
            mv['equity'] = mv['score'] + mv['leave_val']

        all_moves.sort(key=lambda x: -x['equity'])
        top5 = all_moves[:5]

        played_word = m['word']
        played_score = m['score']

        print(f"\nTurn {turn_num} | Rack: {rack} | Played: {played_word} ({played_score} pts)")
        print(f"  Engine 1-ply top 5:")
        for j, mv in enumerate(top5):
            d = mv.get('direction', 'H')
            pos = f"R{mv['row']}C{mv['col']} {d}"
            match = ' <-- PLAYED' if mv['word'].upper() == played_word.upper() else ''
            print(f"    {j+1}. {mv['word']:12} {pos:12} {mv['score']:3}pts  "
                  f"leave={mv['leave']:7} eq={mv['equity']:+.1f}{match}")

        # Find played move rank + equity
        played_rank = None
        played_eq = 0
        for j, mv in enumerate(all_moves):
            if mv['word'].upper() == played_word.upper():
                played_rank = j + 1
                played_eq = mv['equity']
                break

        if played_rank and played_rank > 5:
            print(f"    ... {played_word} ranked #{played_rank} (eq={played_eq:+.1f})")

        top_eq = top5[0]['equity'] if top5 else 0
        top_word = top5[0]['word'] if top5 else '?'
        top_score = top5[0]['score'] if top5 else 0

        eq_loss = played_eq - top_eq if played_rank else -999

        # Check if NYT best is in our move list
        nyt = nyt_best.get(turn_num, {})
        nyt_word = nyt.get('word', '?')
        nyt_score = nyt.get('score', 0)
        nyt_rank = None
        nyt_eq = 0
        for j, mv in enumerate(all_moves):
            if mv['word'].upper() == nyt_word.upper():
                nyt_rank = j + 1
                nyt_eq = mv['equity']
                break

        if abs(eq_loss) <= 0.5:
            verdict = 'MATCH'
        elif eq_loss > -3:
            verdict = 'CLOSE'
        else:
            verdict = f'MISS ({eq_loss:+.1f})'

        # Compare engine vs NYT
        nyt_vs_engine = ''
        if nyt_rank:
            if top_word.upper() == nyt_word.upper():
                nyt_vs_engine = 'AGREE'
            else:
                diff = top_eq - nyt_eq
                if diff > 1:
                    nyt_vs_engine = f'ENGINE +{diff:.1f}'
                elif diff < -1:
                    nyt_vs_engine = f'NYT +{-diff:.1f}'
                else:
                    nyt_vs_engine = 'TIE'
        else:
            nyt_vs_engine = f'NYT:{nyt_word} NOT FOUND'

        results.append({
            'turn': turn_num,
            'played': played_word,
            'played_score': played_score,
            'played_eq': played_eq,
            'played_rank': played_rank,
            'engine_top': top_word,
            'engine_score': top_score,
            'engine_eq': top_eq,
            'nyt_word': nyt_word,
            'nyt_score': nyt_score,
            'nyt_eq': nyt_eq,
            'nyt_rank': nyt_rank,
            'eq_loss': eq_loss,
            'verdict': verdict,
            'nyt_vs_engine': nyt_vs_engine,
        })

        print(f"  Engine top: {top_word} ({top_score}pts, eq={top_eq:+.1f})")
        print(f"  NYT best:   {nyt_word} ({nyt_score}pts, eq={nyt_eq:+.1f}, rank #{nyt_rank})")
        print(f"  You played: {played_word} ({played_score}pts, eq={played_eq:+.1f}, rank #{played_rank})")
        print(f"  -> Engine vs NYT: {nyt_vs_engine}")

    if i == 15:
        blank_positions.append((4, 2, 'U'))
    game.board.place_word(m['word'], m['row'], m['col'], horiz)


print("\n" + "=" * 90)
print("SUMMARY TABLE")
print("=" * 90)
print(f"{'Turn':>4} {'Played':>10} {'Pts':>4} {'PlayEq':>7} | "
      f"{'EngTop':>10} {'Pts':>4} {'EngEq':>7} | "
      f"{'NYTBest':>10} {'Pts':>4} {'NYTEq':>7} | {'EvN':>12} {'Verdict':>10}")
print("-" * 100)

total_loss = 0
eng_matches = 0
eng_beats_nyt = 0
nyt_beats_eng = 0
ties = 0

for r in results:
    print(f"{r['turn']:4} {r['played']:>10} {r['played_score']:4} {r['played_eq']:+7.1f} | "
          f"{r['engine_top']:>10} {r['engine_score']:4} {r['engine_eq']:+7.1f} | "
          f"{r['nyt_word']:>10} {r['nyt_score']:4} {r['nyt_eq']:+7.1f} | "
          f"{r['nyt_vs_engine']:>12} {r['verdict']:>10}")
    if r['eq_loss'] > -900:
        total_loss += r['eq_loss']
    if r['verdict'] in ('MATCH', 'CLOSE'):
        eng_matches += 1
    if 'ENGINE' in r['nyt_vs_engine']:
        eng_beats_nyt += 1
    elif 'NYT' in r['nyt_vs_engine']:
        nyt_beats_eng += 1
    elif r['nyt_vs_engine'] in ('AGREE', 'TIE'):
        ties += 1

print("-" * 100)
print(f"\nYou matched engine top: {eng_matches}/10")
print(f"Total equity loss vs engine: {total_loss:+.1f}")
print(f"Engine vs NYT: Engine wins {eng_beats_nyt}, NYT wins {nyt_beats_eng}, "
      f"Agree/Tie {ties}")
