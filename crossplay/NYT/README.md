# NYT Crossplay Analysis

Screenshots and analysis of NYT Crossplay games for improving the engine.

## Directory Structure

```
NYT/
  games/                    <- Game screenshots organized by date/session
  |   20260220_session1/    <- Screenshots from Feb 20 session (40 images)
  |   YYYYMMDD_sessionN/    <- Future sessions
  analysis/                 <- Analysis notes, comparisons, findings
  README.md                 <- This file
```

## Naming Convention

- Game folders: `YYYYMMDD_sessionN/` (date of game + session number)
- Screenshots: Keep original filenames (timestamped by phone)
- For specific opponent games: `YYYYMMDD_vs_OPPONENT/`

## What We're Looking For

1. **NYT move recommendations** - What does NYT suggest vs our engine?
2. **Word acceptance** - Which words does NYT reject that we recommend?
3. **Scoring differences** - NYT uses different tile values and bonuses
4. **Strategic patterns** - When does NYT play differently from our engine?
5. **Engine improvement opportunities** - Where our engine could do better

## Fields Added to Game JSON

Move records now track NYT-specific data:
- `engine` - Our engine's recommendation at the time of the move
- `nyt` - NYT's suggested move (if captured from screenshot)
- `win_pct` - NYT's win probability display (if visible)
