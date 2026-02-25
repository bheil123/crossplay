# Engine Tournament Project

## Vision
"Bring your own Scrabble engine" tournament. Family event with Bill + 3 kids
in CA. Each participant uses Claude Code to build their engine. Engines compete
head-to-head in automated round-robin matches.

## Participants
- Bill (dad) + 3 kids (all in CA, all Claude-savvy)
- Each person runs their own Claude Code instance locally
- No centralized hosting needed

## Architecture

### Starter Kit (provided to all participants)
Subset of crossplay codebase -- the building blocks:
- `board.py` -- Board representation
- `scoring.py` -- Move scoring with bonuses and crosswords
- `config.py` -- Tile values, distribution, bonus squares
- `dictionary.py` -- Word validation (196K words)
- `move_finder_c.py` / `move_finder_opt.py` -- Move generation
- `gaddag.py` / `gaddag_compact.py` -- GADDAG data structure (auto-builds on first run)
- `tile_tracker.py` -- Bag tracking
- `leave_eval.py` -- Leave evaluation (optional, advanced)

### Engine Interface Spec
Single Python file implementing a class:
```python
class MyEngine(BaseEngine):
    def pick_move(self, board, rack, scores, bag_count, blank_positions):
        """Return (word, row, col, horizontal) or None to exchange."""
        ...

    def notify_opponent_move(self, word, row, col, horizontal, score):
        """Optional: track opponent moves for strategy."""
        pass
```
- Time limit: 30s per move
- No external deps beyond stdlib + numpy
- Can use any provided building blocks (move finder, scoring, etc.)
- Strategy is the differentiator: which move to pick and why

### Example Engines (seed the tournament)
1. `RandomBot` -- picks a random legal move
2. `GreedyBot` -- highest-scoring move (already have fast_bot.py)
3. `LeaveBot` -- greedy + leave evaluation
4. `MCBot` -- full engine (the one to beat)

### Tournament Runner
```
python tournament.py --games 100 --engines engines/*.py
```
- Loads all submitted engines
- Round-robin: every pair plays N games (home/away)
- Elo ratings, win/loss matrix, average spread
- Results stored in JSON/JSONL

### Submission Process
- Shared private GitHub repo: `crossplay-tournament/`
- Each player pushes: `engines/{name}.py`
- Tournament runner validates engine passes test rig before accepting
- Dad runs tournament on his machine, posts results

## What to Build
1. Engine interface spec + BaseEngine class
2. Tournament runner script (referee, scoring, Elo)
3. Starter kit packaging (subset of codebase)
4. 2-3 example engines
5. Test rig for engine validation
6. Results display (simple HTML or terminal table)

## Status
- Concept phase -- discussed 2026-02-25
- Not started yet
