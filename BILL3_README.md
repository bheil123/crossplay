# Bill3 Training Instructions -- Gen6 SuperLeaves

## STATUS: Gen5 training is being stopped remotely

Gen5 had a training issue (bad seed values from gen4, alpha too low).
The trainer will auto-stop at the next checkpoint via .recalibrate_request.

**When you see the trainer exit, follow the Gen6 instructions below.**

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

## Step 2: Start Gen6 Training (from formula -- clean start)

Run from the repo root in a **native Windows terminal** (CMD or PowerShell, NOT Git Bash):

**CMD:**
```
python -m crossplay.superleaves.remote_train ^
  --generation 6 --games 36000000 ^
  --alpha-start 0.1 --alpha-end 0.001 ^
  --workers 28 --push-every 500000
```

**PowerShell:**
```powershell
python -m crossplay.superleaves.remote_train `
  --generation 6 --games 36000000 `
  --alpha-start 0.1 --alpha-end 0.001 `
  --workers 28 --push-every 500000
```

**Note: NO --init-from and NO --resume.** This starts fresh from the hand-tuned
formula, which is what gen3 (our best generation) did successfully.

## What Changed from Gen5

- **No init-from**: Gen5 was seeded from gen4 which had drifted tile values
  (S inflated to +23, E to +18, Q positive when it should be negative).
  Gen6 starts clean from the formula baseline.
- **Higher alpha**: 0.1 -> 0.001 (proven schedule from gen3) instead of
  gen5's too-conservative 0.03 -> 0.0003.
- **remote_train.py now supports remote restart**: If dad pushes a
  restart_config.json to git, the trainer will auto-stop and restart
  with new parameters. No need to manually intervene.

## Estimated Time

- Total games: 36,000,000
- Estimated sustained rate: ~160 g/s
- ETA: ~62 hours (~2.5 days)

## Important Notes

1. Workers take 30-90 seconds to load the GADDAG before progress appears
2. GADDAG auto-builds on first run (~25s) if not cached
3. Training runs fine in background -- you can use the computer normally
4. If you need to stop: Ctrl+C (saves checkpoint before exiting)
5. To resume after stopping: add `--resume` to the same command
6. Git pushes happen automatically every 500K games
7. If a git push fails, it retries next milestone -- no training interruption

## Remote Restart Feature (NEW)

Dad can now remotely retask the trainer by pushing a `restart_config.json`
to git. When the monitor thread does its next `git pull` (at each push
milestone), it detects the config and restarts the trainer automatically.
You don't need to do anything -- it's fully automatic.

## Questions?

Text dad or leave a note in a git commit message.
