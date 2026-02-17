"""
CROSSPLAY V16 - NYT Curated Word Filter

NYT Crossplay uses a curated version of the NASPA Word List 2023 that
removes trademarks, obscenities, and common slurs. Our engine uses the
full NASPA dictionary, so it may recommend words that Crossplay rejects.

This module loads the curated word list and provides a fast lookup to
flag suspect recommendations with a warning.
"""

import os
from typing import Set, Optional

# Module-level cache
_curated_words: Optional[Set[str]] = None


def load_curated_words() -> Set[str]:
    """Load the NYT curated word list from nyt_curated_words.txt."""
    global _curated_words
    if _curated_words is not None:
        return _curated_words

    curated = set()
    path = os.path.join(os.path.dirname(__file__), 'nyt_curated_words.txt')
    if not os.path.exists(path):
        _curated_words = curated
        return curated

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            curated.add(line.upper())

    _curated_words = curated
    return curated


def is_nyt_curated(word: str) -> bool:
    """Check if a word is likely removed from NYT Crossplay's dictionary.

    Returns True if the word is on the curated list (i.e., probably
    invalid in Crossplay even though it's valid in NASPA/TWL).
    """
    return word.upper() in load_curated_words()


def nyt_warning(word: str) -> str:
    """Return a short warning string if the word is NYT-curated, else empty.

    Use this to append a flag to engine output lines.
    Example: "FUCK  8A H  32  -12.3  [NYT?]"
    """
    if is_nyt_curated(word):
        return " [NYT?]"
    return ""


def nyt_warning_tag(word: str) -> str:
    """Return just the tag 'NYT?' if curated, else empty string.

    For use in structured output (e.g., move dicts).
    """
    if is_nyt_curated(word):
        return "NYT?"
    return ""
