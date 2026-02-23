# Crossplay -- New Computer Setup

How to get the Crossplay engine running on a new Windows machine.

## Prerequisites

- **Python 3.12** (CPython, not Windows Store version)
  - Install from https://www.python.org/downloads/
  - During install: check "Add Python to PATH"
  - Verify: `python --version` should show `Python 3.12.x`
- **Git** with Git LFS
  - Install Git: https://git-scm.com/download/win
  - Install Git LFS: `git lfs install` (run once per machine)
- **Claude Code** (CLI)
  - Install via npm: `npm install -g @anthropic-ai/claude-code`

## 1. Clone the repo

```powershell
cd C:\Users\billh
git clone https://github.com/bheil123/crossplay.git
cd crossplay
```

Git LFS automatically pulls large files (deployed_leaves.pkl, gen1_350000.pkl)
during clone. Verify:

```powershell
# Should show actual file sizes (~21MB each), not tiny LFS pointers
dir crossplay\superleaves\deployed_leaves.pkl
dir crossplay\superleaves\gen1_350000.pkl
```

If they show ~130 bytes instead of ~21MB, run:
```powershell
git lfs pull
```

## 2. Directory structure

The repo lives at `C:\Users\billh\crossplay\` with this layout:

```
C:\Users\billh\crossplay\           <- repo root
  crossplay\                        <- Python package (working directory)
    game_manager.py                 <- main engine
    superleaves\deployed_leaves.pkl <- gen3 trained leave table (LFS)
    games\                          <- persistent game library (git-tracked)
    ...
  run.py                            <- entry point
  CLAUDE.md                         <- engine documentation
  SETUP.md                          <- this file
```

## 3. First run (builds GADDAG cache)

```powershell
cd C:\Users\billh\crossplay
python run.py
```

First run will:
1. Build the GADDAG data structure (~48 seconds, cached as `gaddag_compact.bin`)
2. Run MC calibration (~3 seconds, cached as `.mc_calibration.json`)
3. Load game library from `crossplay/games/`

Subsequent runs start in ~2 seconds.

## 4. Suppress Claude Code permission prompts

Claude Code asks for permission on every tool use by default. To bypass this,
create settings files in these locations:

### Global settings (all projects on this machine)

Create `C:\Users\billh\.claude\settings.json`:

```json
{
  "permissions": {
    "allow": [
      "Bash(*)",
      "WebSearch(*)",
      "WebFetch(*)",
      "Read(*)",
      "Write(*)",
      "Edit(*)",
      "Glob(*)",
      "Grep(*)"
    ],
    "defaultMode": "bypassPermissions"
  }
}
```

### Project settings (already in repo, but gitignored)

The repo has `.claude/settings.local.json` at both the repo root and inside
the `crossplay/` package directory. These are gitignored so each machine
needs its own copy. Create both:

**Repo root:** `C:\Users\billh\crossplay\.claude\settings.local.json`
**Package dir:** `C:\Users\billh\crossplay\crossplay\.claude\settings.local.json`

Same content as the global settings above.

Quick setup via PowerShell:
```powershell
# Global
mkdir -Force "$env:USERPROFILE\.claude" | Out-Null
@'
{
  "permissions": {
    "allow": ["Bash(*)", "WebSearch(*)", "WebFetch(*)", "Read(*)", "Write(*)", "Edit(*)", "Glob(*)", "Grep(*)"],
    "defaultMode": "bypassPermissions"
  }
}
'@ | Set-Content "$env:USERPROFILE\.claude\settings.json"

# Project (repo root)
mkdir -Force "C:\Users\billh\crossplay\.claude" | Out-Null
Copy-Item "$env:USERPROFILE\.claude\settings.json" "C:\Users\billh\crossplay\.claude\settings.local.json"

# Project (package dir)
mkdir -Force "C:\Users\billh\crossplay\crossplay\.claude" | Out-Null
Copy-Item "$env:USERPROFILE\.claude\settings.json" "C:\Users\billh\crossplay\crossplay\.claude\settings.local.json"
```

## 5. Multi-computer game sync

Games are git-tracked. To switch between machines:

**Leaving a computer:**
```powershell
cd C:\Users\billh\crossplay
git add crossplay/games/
git commit -m "Save game state"
git push
```

**Starting on another computer:**
```powershell
cd C:\Users\billh\crossplay
git pull
```

Then in the game engine (or tell Claude Code): use the `reload` command to
refresh in-memory game state from the updated JSON files on disk.

## 6. Cython extension (optional rebuild)

Pre-built extensions are included for both platforms:
- Windows: `gaddag_accel.cp312-win_amd64.pyd`
- Linux: `gaddag_accel.cpython-312-x86_64-linux-gnu.so`

If you need to rebuild (e.g., after modifying `gaddag_accel.pyx`):

```powershell
pip install cython
cd C:\Users\billh\crossplay\crossplay
python setup_accel.py build_ext --inplace
```

## 7. Python dependencies

The engine uses only Python standard library -- no pip packages required
for normal operation. Cython is only needed if rebuilding the C extension.

## Troubleshooting

**"python" opens Windows Store:** Use the full path instead:
```
C:\Users\billh\AppData\Local\Programs\Python\Python312\python.exe run.py
```

**LFS files are tiny pointer files:** Run `git lfs pull` to download actual data.

**GADDAG build fails:** Ensure `crossplay_dict.pkl` exists (196K words, ~2.3MB).
This is committed to the repo and should be present after clone.

**Permission prompts still appearing:** Restart the Claude Code session after
creating settings files. Check that `defaultMode` is `"bypassPermissions"`.

**Stale game state after git pull:** Use the `reload` command in the engine
to refresh in-memory state from disk.
