"""
CROSSPLAY V13 - Pattern Index Module

Pre-computed index for fast word pattern lookups.
Used by risk analyzer to quickly find words matching board constraints.

Index types:
- by_length: All words grouped by length
- by_letter_position: Words with specific letter at specific position
  Key: (letter, position, word_length) -> set of words

Usage:
    from pattern_index import get_pattern_index, find_matching_words
    
    index = get_pattern_index()
    
    # Find 5-letter words with A at position 1 and E at position 3 (0-indexed)
    matches = find_matching_words(index, length=5, constraints={1: 'A', 3: 'E'})
"""

import os
import pickle
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

# Path to pre-built index
INDEX_PATH = os.path.join(os.path.dirname(__file__), 'pattern_index.pkl')
DICT_PATH = os.path.join(os.path.dirname(__file__), 'crossplay_dict.pkl')

# Cached index
_PATTERN_INDEX = None


def build_pattern_index(words: Set[str]) -> Dict:
    """
    Build pattern index from word set.
    
    Args:
        words: Set of valid words
        
    Returns:
        Dict with 'by_length' and 'by_letter_position' indexes
    """
    by_length = defaultdict(list)
    by_letter_position = defaultdict(set)
    
    for word in words:
        word = word.upper()
        length = len(word)
        
        if length < 2 or length > 15:
            continue
        
        by_length[length].append(word)
        
        # Index by letter@position
        for i, letter in enumerate(word):
            by_letter_position[(letter, i, length)].add(word)
    
    return {
        'by_length': dict(by_length),
        'by_letter_position': {k: v for k, v in by_letter_position.items()}
    }


def save_pattern_index(index: Dict, path: str = INDEX_PATH) -> None:
    """Save pattern index to pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(index, f)
    print(f"Pattern index saved to {path}")


def load_pattern_index(path: str = INDEX_PATH) -> Dict:
    """Load pattern index from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_pattern_index() -> Dict:
    """
    Get pattern index, loading from file or building if needed.
    
    Returns:
        Pattern index dict
    """
    global _PATTERN_INDEX
    
    if _PATTERN_INDEX is not None:
        return _PATTERN_INDEX
    
    # Try to load from file
    if os.path.exists(INDEX_PATH):
        _PATTERN_INDEX = load_pattern_index(INDEX_PATH)
        return _PATTERN_INDEX
    
    # Build from dictionary
    print("Building pattern index (first time only)...")
    with open(DICT_PATH, 'rb') as f:
        words = pickle.load(f)
    
    _PATTERN_INDEX = build_pattern_index(words)
    
    # Save for next time
    save_pattern_index(_PATTERN_INDEX)
    
    return _PATTERN_INDEX


def find_matching_words(
    index: Dict,
    length: int,
    constraints: Dict[int, str],
    rack: str = None
) -> Set[str]:
    """
    Find words matching length and letter constraints.
    
    Args:
        index: Pattern index from get_pattern_index()
        length: Word length to find
        constraints: Dict of {position: letter} (0-indexed)
        rack: Optional rack to filter by playable tiles
        
    Returns:
        Set of matching words
        
    Example:
        # Find 5-letter words with A at position 1 and E at position 3
        find_matching_words(index, 5, {1: 'A', 3: 'E'})
    """
    if not constraints:
        # No constraints, return all words of this length
        return set(index['by_length'].get(length, []))
    
    # Start with words matching first constraint
    result = None
    by_lp = index['by_letter_position']
    
    for pos, letter in constraints.items():
        key = (letter.upper(), pos, length)
        matches = by_lp.get(key, set())
        
        if result is None:
            result = set(matches)
        else:
            result &= matches
        
        # Early termination if no matches
        if not result:
            return set()
    
    # Filter by rack if provided
    if rack and result:
        rack_upper = rack.upper()
        blank_count = rack_upper.count('?')
        rack_letters = [c for c in rack_upper if c != '?']
        
        filtered = set()
        for word in result:
            # Check if word can be played with rack (excluding constrained letters)
            needed = list(word)
            for pos, letter in constraints.items():
                needed[pos] = None  # Already on board
            
            needed = [c for c in needed if c is not None]
            
            # Check if rack covers needed letters
            available = rack_letters.copy()
            blanks_needed = 0
            
            can_play = True
            for letter in needed:
                if letter in available:
                    available.remove(letter)
                elif blanks_needed < blank_count:
                    blanks_needed += 1
                else:
                    can_play = False
                    break
            
            if can_play:
                filtered.add(word)
        
        return filtered
    
    return result or set()


def find_words_at_anchor(
    index: Dict,
    board_letters: Dict[int, str],
    word_length: int,
    rack: str = None
) -> Set[str]:
    """
    Find words that fit at an anchor position given existing board letters.
    
    Args:
        index: Pattern index
        board_letters: Dict of {position: letter} for letters already on board
        word_length: Length of word to find
        rack: Optional rack to filter by
        
    Returns:
        Set of valid words
    """
    return find_matching_words(index, word_length, board_letters, rack)


def get_words_by_length(index: Dict, length: int) -> List[str]:
    """Get all words of a specific length."""
    return index['by_length'].get(length, [])


def rebuild_index() -> Dict:
    """Force rebuild of pattern index from dictionary."""
    global _PATTERN_INDEX
    
    print("Rebuilding pattern index...")
    with open(DICT_PATH, 'rb') as f:
        words = pickle.load(f)
    
    _PATTERN_INDEX = build_pattern_index(words)
    save_pattern_index(_PATTERN_INDEX)
    
    return _PATTERN_INDEX


# =============================================================================
# CLI for building/testing
# =============================================================================

if __name__ == "__main__":
    import time
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'build':
        # Force rebuild
        rebuild_index()
    else:
        # Load and test
        print("Loading pattern index...")
        start = time.time()
        index = get_pattern_index()
        load_time = time.time() - start
        print(f"Load time: {load_time*1000:.1f}ms")
        
        # Stats
        total_words = sum(len(words) for words in index['by_length'].values())
        print(f"Total words indexed: {total_words}")
        print(f"Length entries: {len(index['by_length'])}")
        print(f"Letter-position entries: {len(index['by_letter_position'])}")
        
        # Test queries
        print("\nTest queries:")
        
        # Query 1: _A_E_ (5 letters)
        start = time.time()
        result = find_matching_words(index, 5, {1: 'A', 3: 'E'})
        query_time = time.time() - start
        print(f"  '_A_E_' (5 letters): {len(result)} matches in {query_time*1000:.2f}ms")
        
        # Query 2: ___ING (6 letters)
        start = time.time()
        result = find_matching_words(index, 6, {3: 'I', 4: 'N', 5: 'G'})
        query_time = time.time() - start
        print(f"  '___ING' (6 letters): {len(result)} matches in {query_time*1000:.2f}ms")
        
        # Query 3: With rack filter
        start = time.time()
        result = find_matching_words(index, 5, {1: 'A', 3: 'E'}, rack='RSTLNE')
        query_time = time.time() - start
        print(f"  '_A_E_' with rack RSTLNE: {len(result)} matches in {query_time*1000:.2f}ms")
        print(f"    Sample: {list(result)[:5]}")
