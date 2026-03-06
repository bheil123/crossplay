# Bill3 Training Instructions -- Gen5 SuperLeaves

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

Best throughput: 93.8 g/s (total) at 29 workers
Sustained game-only rate: ~154 g/s (init overhead amortized over long runs)

For comparison: Dad's i7-8700 gets ~26.5 g/s -- your machine is 3.5x faster!

## Step 1: Pull Latest Code

```
cd crossplay
git pull
git lfs pull
```

This gets the gen5 checkpoint at 1.49M games (21MB file via Git LFS).

## Step 2: Start Training

Run from the repo root in a **native Windows terminal** (CMD or PowerShell, NOT Git Bash):

**CMD:**
```
python -m crossplay.superleaves.trainer --resume --generation 5 ^
  --games 36000000 ^
  --init-from crossplay\superleaves\gen5_1490000.pkl ^
  --alpha-start 0.03 --alpha-end 0.0003 ^
  --workers 28
```

**PowerShell:**
```powershell
python -m crossplay.superleaves.trainer --resume --generation 5 `
  --games 36000000 `
  --init-from crossplay\superleaves\gen5_1490000.pkl `
  --alpha-start 0.03 --alpha-end 0.0003 `
  --workers 28
```

## What This Does

- Resumes gen5 training from the 1.49M game checkpoint
- Trains to 36M total games using TD(0) temporal difference learning
- Alpha (learning rate) decays exponentially from 0.03 to 0.0003
- Checkpoints saved every 10,000 games
- Uses 28 workers (reserves 4 threads for OS)

## Estimated Time

- Remaining games: 34,510,000
- Estimated sustained rate: ~120-140 g/s
- ETA: ~70-80 hours (~3 days)

## Important Notes

1. Workers take 30-90 seconds to load the GADDAG before progress appears -- don't kill it
2. GADDAG auto-builds on first run (~25 seconds on your machine) if not cached
3. Training runs fine in background -- you can use the computer normally
4. If you need to stop: Ctrl+C (saves checkpoint before exiting)
5. To resume after stopping: just run the same command again

## Pushing Results

Push checkpoints periodically so dad can monitor progress:

```
cd crossplay
git add -f crossplay\superleaves\gen5_*.pkl
git add crossplay\superleaves\status.json
git commit -m "Gen5 training progress from Bill3 7950X3D"
git push
```

Or just push the final result when training completes.

## Questions?

Text dad or leave a note in a git commit message.
