# Getting Crossplay v13.1 onto GitHub & Claude Code

A step-by-step guide. You'll need about 20 minutes.

---

## Step 1: Install Git (if you don't have it)

**Windows:**
Download from https://git-scm.com/download/win and run the installer.
Accept all defaults. This gives you "Git Bash" — a terminal you'll use below.

**Mac:**
Open Terminal and type `git --version`. If it's not installed, it will
prompt you to install Xcode Command Line Tools. Say yes.

**Check it works:**
```
git --version
```
You should see something like `git version 2.43.0`.

---

## Step 2: Configure Git with your identity

Open a terminal (Git Bash on Windows, Terminal on Mac) and run:

```
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"
```

Use the same email as your GitHub account.

---

## Step 3: Create a new repository on GitHub

1. Go to https://github.com/new
2. Repository name: `crossplay` (or whatever you like)
3. Description: `Crossplay Scrabble engine with Monte Carlo AI`
4. Choose **Private** (your game data is in there)
5. Do NOT check "Add a README" or ".gitignore" (we already have both)
6. Click **Create repository**
7. You'll see a page with setup instructions — leave this tab open

---

## Step 4: Unzip and initialize the repo locally

Unzip `Crossplay_v13-1__GitHub_Ready.zip` somewhere on your computer.
Then in your terminal:

```bash
# Navigate to wherever you unzipped it
cd /path/to/crossplay_v9

# Initialize git
git init

# Add all files
git add .

# Make your first commit
git commit -m "Crossplay v13.1 - initial commit"

# Connect to your GitHub repo (copy the URL from Step 3)
# It will look like one of these:
git remote add origin https://github.com/YOUR_USERNAME/crossplay.git
# or
git remote add origin git@github.com:YOUR_USERNAME/crossplay.git

# Push it up
git branch -M main
git push -u origin main
```

If prompted for credentials, GitHub now requires a **Personal Access Token**
instead of your password:
1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name like "crossplay"
4. Check the "repo" scope
5. Click Generate, copy the token
6. Use this token as your password when git asks

After `git push` succeeds, refresh your GitHub page — you should see all
your files there.

---

## Step 5: Install Claude Code

Claude Code is Anthropic's terminal AI tool. It reads your codebase and
can edit files, run commands, and help you develop.

**Recommended: Native installer (no Node.js needed)**

**Mac:**
```bash
curl -fsSL https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/installer.sh | sh
```

**Windows (in PowerShell as Administrator):**
```powershell
winget install Anthropic.ClaudeCode
```
Then use it from Git Bash or WSL.

**Linux:**
```bash
curl -fsSL https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/installer.sh | sh
```

**Verify:**
```bash
claude --version
```

**Authentication:** Claude Code uses your existing Claude Pro/Max
subscription. First time you run `claude`, it will open a browser for
a one-time OAuth login. Same account as claude.ai.

---

## Step 6: Use Claude Code with Crossplay

```bash
cd /path/to/crossplay_v9
claude
```

Claude Code will read the CLAUDE.md file automatically to understand
the project. You can then do things like:

```
> run game_manager.py and show me slot 4
> benchmark MC throughput on this machine
> rebuild the Cython extension for my platform
> what moves are available for rack TCDQHRR?
```

### Useful Claude Code commands

| Command | What it does |
|---------|-------------|
| `/help` | Show all commands |
| `/doctor` | Check your setup |
| `/compact` | Summarize conversation to save context |

### First run on your i7

The first run will:
1. Build the GADDAG (~48s, cached after that)
2. Run MC calibration (~3s, tells you your sims/sec)
3. The calibration number is what matters — my cloud env gets ~1,400
   dense sims/sec on 4 workers. Your i7 should do better.

If the pre-built `.so` doesn't work on your system (it's for CPython 3.12
x86_64 Linux), rebuild it:

```bash
pip install cython
python3 setup_accel.py build_ext --inplace
```

---

## Step 7: Save your work back to GitHub

After making changes with Claude Code:

```bash
git add .
git commit -m "describe what changed"
git push
```

---

## Step 8: Pull code back into Claude Chat later

When you want to bring the code back into a Claude chat session:

**Option A — Upload the zip:**
1. Download your repo as a zip from GitHub:
   `https://github.com/YOUR_USERNAME/crossplay/archive/refs/heads/main.zip`
2. Upload it to Claude chat

**Option B — Just tell Claude the repo URL:**
Say "clone my repo at github.com/YOUR_USERNAME/crossplay and load it"
(only works if the repo is public, or you provide a token)

**Option C — Zip locally and upload:**
```bash
cd /path/to
zip -r crossplay_latest.zip crossplay_v9/ -x "*__pycache__*" "*.bin" "*.json"
```
Then upload to Claude chat.

---

## Quick reference: Git commands you'll actually use

| What you want to do | Command |
|---------------------|---------|
| See what changed | `git status` |
| See the diff | `git diff` |
| Save your changes | `git add . && git commit -m "message"` |
| Push to GitHub | `git push` |
| Pull latest from GitHub | `git pull` |
| See commit history | `git log --oneline` |
| Undo uncommitted changes | `git checkout -- filename` |

---

## Troubleshooting

**"Permission denied" on git push:**
You need a Personal Access Token (see Step 4) or set up SSH keys.

**Claude Code says "command not found":**
Close and reopen your terminal, or add to PATH manually.

**`.so` file doesn't work on my machine:**
The pre-built extension is for Linux x86_64 CPython 3.12. Rebuild with:
`pip install cython && python3 setup_accel.py build_ext --inplace`

**GADDAG build is slow:**
Normal — 48s first time. After that it loads from cache in <0.1s.
