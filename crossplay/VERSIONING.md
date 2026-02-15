# Versioning Rules -- Crossplay Engine

## Version Format

    MAJOR.MINOR.PATCH    (e.g., 14.0.0)

Stored in `__init__.py` as `__version__` and tagged in git as `vMAJOR.MINOR.PATCH`.

## When to Bump

### MAJOR (14.x.x -> 15.0.0)

Bump the major version when:
- Breaking changes to saved game format (old saves won't load)
- Breaking changes to the Python package API (imports change)
- Fundamental architecture change (e.g., replacing GADDAG with DAWG)
- Renaming the `crossplay` package directory

Major bumps reset minor and patch to 0.

### MINOR (14.0.x -> 14.1.0)

Bump the minor version when:
- New feature that changes engine behavior or output
  - New evaluation component (e.g., positional adjustments in MC)
  - New game mode or command
  - New AI capability (e.g., exchange evaluation, endgame solver)
- Performance optimization that changes move recommendations
  - MC throughput improvements that affect N/K parameters
  - New heuristics that change candidate ranking
- Significant bug fix that changes move recommendations
- Dictionary changes (adding/removing valid words)

Minor bumps reset patch to 0.

### PATCH (14.0.0 -> 14.0.1)

Bump the patch version when:
- Bug fix that doesn't change move recommendations
- Display/formatting fixes
- Documentation updates
- Code cleanup / refactoring with no behavior change
- Game state updates (new saved games, score corrections)
- Build system or CI changes

## Where Version Lives

| Location | Format | Example |
|----------|--------|---------|
| `__init__.py` | `__version__ = "X.Y.Z"` | `__version__ = "14.0.0"` |
| Git tag | `vX.Y.Z` | `v14.0.0` |
| File docstrings | `CROSSPLAY V{MAJOR}` | `CROSSPLAY V14` |
| README.md title | `Crossplay V{MAJOR}.{MINOR}` | `Crossplay V14.0` |

File docstrings use only the major version (`CROSSPLAY V14`) and do not
need updating on minor/patch bumps. Only update them when the major
version changes.

## How to Release

1. Update `__version__` in `__init__.py`
2. Update README.md title if minor version changed
3. Commit: `git commit -m "Bump version to X.Y.Z"`
4. Tag: `git tag vX.Y.Z`
5. Push: `git push origin main --tags`

## Version History

| Version | Date | Summary |
|---------|------|---------|
| 14.0.0 | 2026-02 | Flattened repo structure (crossplay_v9 -> crossplay), major version bump |
| 13.2.0 | 2025-06 | Unified versioning, MC positional adjustments, blank correction |
| 13.1.0 | 2025-05 | GitHub ready: .gitignore, CLAUDE.md, saved game registry |
| 12.1.0 | -- | C word/rack/blank buffers, BoardContext MC fast path |
| 9.0.0 | -- | Bingo-aware leave evaluation |
| 8.0.0 | -- | 3-ply exhaustive endgame |
| 7.0.0 | -- | Original GADDAG engine, MC 2-ply, parallel eval |

