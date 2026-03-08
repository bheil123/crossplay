# Bill3 Training Instructions -- Gen8 SuperLeaves

## STATUS: Ready to start Gen8 training

Gen8 uses research-derived Crossplay leave values as its starting point
(unlike gen3-6 which started from the hand-tuned formula). Training target
is 100M games for deep convergence.

## Your Benchmark Results

CPU: AMD Ryzen 9 7950X3D 16-Core Processor
Cores/Threads: 16C/32T
L3 Cache: 128MB (3D V-Cache)

| Workers | g/s (total) | g/s (game-only) |
|---------|-------------|-----------------|
|       1 |        11.3 |            12.0 |
|       8 |        50.1 |            64.8 |
|      16 |        79.6 |           103.7 |
|      24 |        85.1 |           132.5 |
|      28 |        91.9 |           150.1 |
|      29 |        93.8 |           154.4 |

## Step 1: Pull Latest Code

```
cd crossplay
git pull
git lfs pull
```

## Step 2: Start Gen8 Training

Run from the repo root in a **native Windows terminal** (CMD or PowerShell, NOT Git Bash):

**CMD:**
```
python -m crossplay.superleaves.remote_train ^
  --generation 8 --games 100000000 ^
  --alpha-start 0.1 --alpha-end 0.001 ^
  --workers 28 --push-every 250000 --resume
```

**PowerShell:**
```powershell
python -m crossplay.superleaves.remote_train `
  --generation 8 --games 100000000 `
  --alpha-start 0.1 --alpha-end 0.001 `
  --workers 28 --push-every 250000 --resume
```

**Note:** Uses `--resume` to pick up from the checkpoint that was pushed to
git. The `remote_train` wrapper handles auto-pushing checkpoints to GitHub
every 250K games and accepts remote control signals via git.

## What Gen8 Does Differently

- **Research bootstrap**: Started from research-derived Crossplay leave
  values instead of the hand-tuned formula. These values were calibrated
  specifically for Crossplay's 3-blank, 40-point-sweep rules.
- **100M games**: Much longer training run for deeper convergence.
- **250K push interval**: Checkpoints pushed every ~25 minutes for
  close monitoring (was 1M in gen6).

## Remote Control (V2)

Dad can now remotely control training by pushing `restart_config.json`
to git. The monitor thread polls git every 60 seconds and detects signals
automatically. **You don't need to do anything** -- it's fully automatic.

### Available remote actions:

| Action | What it does |
|--------|-------------|
| `pause` | Stop training, save checkpoint, push final state, exit |
| `resume` | Resume from latest checkpoint (optionally with new params) |
| `restart` | Fresh start with new parameters |
| `validate` | Pause training, run validation test, push results, resume |
| `tournament` | Pause training, run bot tournament, push results, resume |
| `recalibrate` | Pause training, recalibrate MC speed, resume |
| `update_code` | Pull latest code changes, optionally rebuild Cython, resume |
| `status` | Force immediate status push (no training interruption) |

### Example remote configs (dad pushes these via git):

**Pause training:**
```json
{"action": "pause", "message": "Pausing for maintenance"}
```

**Run validation:**
```json
{
  "action": "validate",
  "table": "latest",
  "opponent": "formula",
  "validate_games": 5000,
  "message": "Check gen8 progress vs formula"
}
```

**Run tournament:**
```json
{
  "action": "tournament",
  "bot1": "dadbot_v6",
  "bot2": "my_bot",
  "tourney_games": 20,
  "tier": "fast",
  "message": "Quick bot check"
}
```

After validate/tournament/recalibrate, training **automatically resumes**
from the latest checkpoint. No manual intervention needed.

## Estimated Time

- Total games: 100,000,000
- Estimated sustained rate: ~90 g/s (28 workers)
- ETA: ~309 hours (~12.9 days)
- Checkpoints pushed every 250K games (~46 min intervals)

## Important Notes

1. Workers take 30-90 seconds to load the GADDAG before progress appears
2. GADDAG auto-builds on first run (~25s on your machine) if not cached
3. Training runs fine in background -- you can use the computer normally
4. If you need to stop: Ctrl+C (saves checkpoint before exiting)
5. To resume after manual stop: run the same command again (--resume)
6. Git pushes happen automatically every 250K games
7. If a git push fails, it retries next milestone -- no training interruption
8. The monitor thread polls git every 60s for remote control signals

## If Something Goes Wrong

- **Training crashes**: Just run the command again with `--resume`.
  It will find the latest checkpoint and continue.
- **Git push fails**: Check your internet connection. The push will
  retry at the next 250K milestone automatically.
- **"No checkpoint found"**: The gen8 checkpoint hasn't been pushed
  to git yet. Tell dad to push one.
- **Orphaned processes**: If you see extra Python processes after a
  crash, the trainer cleans them up automatically on restart.

## Questions?

Text dad or leave a note in a git commit message.
