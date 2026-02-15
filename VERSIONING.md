# Versioning Rules -- Crossplay Engine

## Version Format

    MAJOR.MINOR.PATCH    (e.g., 13.2.0)

Stored in `__init__.py` as `__version__` and tagged in git as `vMAJOR.MINOR.PATCH`.

## When to Bump

### MAJOR (13.x.x -> 14.0.0)

Bump the major version when:
- Breaking changes to saved game format (old saves won't load)
- Breaking changes to the Python package API (imports change)
- Fundamental architecture change (e.g., replacing GADDAG with DAWG)
- Renaming the `crossplay_v9` package directory

Major bumps reset minor and patch to 0.

### MINOR (13.2.x -> 13.3.0)

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

### PATCH (13.2.0 -> 13.2.1)

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
| `__init__.py` | `__version__ = "X.Y.Z"` | `__version__ = "13.2.0"` |
| Git tag | `vX.Y.Z` | `v13.2.0` |
| File docstrings | `CROSSPLAY V{MAJOR}` | `CROSSPLAY V13` |
| README.md title | `Crossplay V{MAJOR}.{MINOR}` | `Crossplay V13.2` |

File docstrings use only the major version (`CROSSPLAY V13`) and do not
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
| 13.2.0 | 2025-06 | Unified versioning, MC positional adjustments, blank correction |
| 13.1.0 | 2025-05 | GitHub ready: .gitignore, CLAUDE.md, saved game registry |
| 12.1.0 | -- | C word/rack/blank buffers, BoardContext MC fast path |
| 9.0.0 | -- | Bingo-aware leave evaluation |
| 8.0.0 | -- | 3-ply exhaustive endgame |
| 7.0.0 | -- | Original GADDAG engine, MC 2-ply, parallel eval |

## Package Name Note

The Python package directory is `crossplay_v9` for historical reasons
(the codebase was first imported at that version). All internal imports
use `from crossplay_v9.module import ...`. Renaming the directory would
require updating ~90 import statements across 20+ files and would
constitute a MAJOR version bump.
