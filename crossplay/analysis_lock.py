"""
Graceful lock mechanism for coordinating game analysis and training.
Prevents both from running simultaneously while handling crashes gracefully.
"""
import json
import os
import time
from pathlib import Path

LOCK_FILE = Path('crossplay/.analysis_lock')
LOCK_TIMEOUT = 300  # 5 minutes - stale lock detection


def acquire_lock(purpose='game_analysis'):
    """Request pause from training. Self-healing with timeout."""
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(json.dumps({
        'acquired_at': time.time(),
        'purpose': purpose,
        'pid': os.getpid(),
    }))


def release_lock():
    """Resume training."""
    LOCK_FILE.unlink(missing_ok=True)


def is_lock_stale():
    """Check if lock is older than timeout (crashed process)."""
    if not LOCK_FILE.exists():
        return False
    try:
        data = json.loads(LOCK_FILE.read_text())
        age = time.time() - data['acquired_at']

        # Check age
        if age > LOCK_TIMEOUT:
            return True

        # On Unix: check if process still alive
        pid = data.get('pid')
        if pid and os.name == 'posix':
            try:
                os.kill(pid, 0)  # Signal 0 = check existence
                return False  # Process alive
            except ProcessLookupError:
                return True  # Process dead

        # On Windows: just use age timeout
        return False
    except Exception:
        return True  # Corrupted lock, treat as stale


def wait_for_analysis(max_wait=300, logger=None):
    """
    Trainer: wait for analysis to complete or timeout.

    Args:
        max_wait: Maximum seconds to wait (default 5 min)
        logger: Optional logger for debug output
    """
    start = time.time()
    while LOCK_FILE.exists():
        if is_lock_stale():
            if logger:
                logger.warning(f"Stale lock detected (age > {LOCK_TIMEOUT}s), clearing")
            LOCK_FILE.unlink(missing_ok=True)
            break
        if time.time() - start > max_wait:
            if logger:
                logger.warning(f"Lock timeout after {max_wait}s, resuming anyway")
            break
        time.sleep(0.1)
