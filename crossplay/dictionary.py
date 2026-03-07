"""
CROSSPLAY V15 - Dictionary Module
Word validation and lookup with optional enhanced features.
"""

from typing import Set, List, Optional, Dict
import pickle
import os
from .config import VALID_TWO_LETTER


class Dictionary:
    """
    Word dictionary for Crossplay.
    
    Wraps a set of valid words with helper methods.
    Supports enhanced mode with pre-computed hooks and scores.
    """
    
    def __init__(self, words: Optional[Set[str]] = None, use_pattern_index: bool = True):
        """
        Initialize dictionary.
        
        Args:
            words: Set of valid words (uppercase). If None, loads default.
            use_pattern_index: Whether to load pattern index for fast lookups.
        """
        if words is not None:
            self._words = {w.upper() for w in words}
        else:
            self._words = set()
        
        # Enhanced features (populated if available)
        self._front_hooks: Dict[str, Set[str]] = {}
        self._back_hooks: Dict[str, Set[str]] = {}
        self._base_scores: Dict[str, int] = {}
        self._enhanced = False
        
        self._by_length: Dict[int, List[str]] = {}
        self._pattern_cache: Dict[str, List[str]] = {}
        self._pattern_index = None
        self._build_index()
        
        if use_pattern_index:
            self._load_pattern_index()
    
    def _build_index(self):
        """Index words by length for faster pattern matching."""
        self._by_length = {}
        for word in self._words:
            length = len(word)
            if length not in self._by_length:
                self._by_length[length] = []
            self._by_length[length].append(word)
    
    def _load_pattern_index(self):
        """Load pattern index for fast lookups."""
        try:
            from .pattern_index import get_pattern_index
            self._pattern_index = get_pattern_index()
        except Exception:
            self._pattern_index = None
    
    @classmethod
    def load(cls, filepath: str, use_pattern_index: bool = True) -> 'Dictionary':
        """Load dictionary from pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Check if it's enhanced format (dict) or simple format (set)
        if isinstance(data, dict):
            d = cls(data.get('words', set()), use_pattern_index=use_pattern_index)
            d._front_hooks = data.get('front_hooks', {})
            d._back_hooks = data.get('back_hooks', {})
            d._base_scores = data.get('base_scores', {})
            d._enhanced = bool(d._front_hooks or d._back_hooks)
        else:
            d = cls(data, use_pattern_index=use_pattern_index)
        
        return d
    
    def save(self, filepath: str) -> None:
        """Save dictionary to pickle file."""
        if self._enhanced:
            data = {
                'words': self._words,
                'front_hooks': self._front_hooks,
                'back_hooks': self._back_hooks,
                'base_scores': self._base_scores
            }
        else:
            data = self._words
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def is_valid(self, word: str) -> bool:
        """Check if a word is valid. O(1)"""
        word = word.upper()
        if len(word) == 2:
            return word in VALID_TWO_LETTER
        return word in self._words
    
    # Enhanced features
    
    def get_front_hooks(self, word: str) -> Set[str]:
        """Get letters that can prefix this word. O(1) if enhanced."""
        word = word.upper()
        if self._enhanced:
            return self._front_hooks.get(word, set())
        # Fallback: check all letters
        hooks = set()
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if self.is_valid(letter + word):
                hooks.add(letter)
        return hooks
    
    def get_back_hooks(self, word: str) -> Set[str]:
        """Get letters that can suffix this word. O(1) if enhanced."""
        word = word.upper()
        if self._enhanced:
            return self._back_hooks.get(word, set())
        # Fallback: check all letters
        hooks = set()
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if self.is_valid(word + letter):
                hooks.add(letter)
        return hooks
    
    def can_extend_front(self, word: str) -> bool:
        """Check if word can be extended at front. O(1) if enhanced."""
        return bool(self.get_front_hooks(word))
    
    def can_extend_back(self, word: str) -> bool:
        """Check if word can be extended at back. O(1) if enhanced."""
        return bool(self.get_back_hooks(word))
    
    def can_hook_front(self, word: str, letter: str) -> bool:
        """Check if letter+word is valid. O(1) if enhanced."""
        if self._enhanced:
            return letter.upper() in self._front_hooks.get(word.upper(), set())
        return self.is_valid(letter + word)
    
    def can_hook_back(self, word: str, letter: str) -> bool:
        """Check if word+letter is valid. O(1) if enhanced."""
        if self._enhanced:
            return letter.upper() in self._back_hooks.get(word.upper(), set())
        return self.is_valid(word + letter)
    
    def get_base_score(self, word: str) -> int:
        """Get sum of tile values (no multipliers). O(1) if enhanced."""
        word = word.upper()
        if self._enhanced and word in self._base_scores:
            return self._base_scores[word]
        # Fallback: calculate
        from .config import TILE_VALUES
        return sum(TILE_VALUES.get(c, 0) for c in word)
    
    @property
    def is_enhanced(self) -> bool:
        """Check if enhanced features are available."""
        return self._enhanced
    
    def add_word(self, word: str) -> None:
        """Add a word to the dictionary."""
        self._words.add(word.upper())
    
    def remove_word(self, word: str) -> None:
        """Remove a word from the dictionary."""
        self._words.discard(word.upper())
    
    def find_words(self, pattern: str) -> List[str]:
        """
        Find words matching a pattern.
        
        Args:
            pattern: Pattern with '?' as wildcard (e.g., 'QU??T')
        
        Returns:
            List of matching words
        """
        # Try pattern index first
        if self._pattern_index is not None:
            try:
                from .pattern_index import find_matching_words
                constraints = {}
                for i, c in enumerate(pattern.upper()):
                    if c != '?':
                        constraints[i] = c
                return find_matching_words(self._pattern_index, len(pattern), constraints)
            except Exception:
                pass
        
        # Fallback to linear scan
        pattern = pattern.upper()
        length = len(pattern)
        
        if length in self._by_length:
            candidates = self._by_length[length]
        else:
            candidates = [w for w in self._words if len(w) == length]
        
        matches = []
        for word in candidates:
            match = True
            for p, w in zip(pattern, word):
                if p != '?' and p != w:
                    match = False
                    break
            if match:
                matches.append(word)
        
        return sorted(matches)
    
    def find_anagrams(self, letters: str) -> List[str]:
        """Find all words that can be made from given letters."""
        from collections import Counter
        
        letters = letters.upper()
        available = Counter(letters)
        blanks = available.get('?', 0)
        available_no_blanks = Counter(l for l in letters if l != '?')
        
        matches = []
        
        for word in self._words:
            word_counter = Counter(word)
            blanks_needed = 0
            can_make = True
            
            for letter, count in word_counter.items():
                have = available_no_blanks.get(letter, 0)
                if have < count:
                    blanks_needed += count - have
                    if blanks_needed > blanks:
                        can_make = False
                        break
            
            if can_make:
                matches.append(word)
        
        return sorted(matches, key=lambda w: (-len(w), w))
    
    def __contains__(self, word: str) -> bool:
        return self.is_valid(word)
    
    def __len__(self) -> int:
        return len(self._words)
    
    def __iter__(self):
        return iter(self._words)


# Global dictionary instance (lazy loaded)
_global_dict: Optional[Dictionary] = None


def get_dictionary() -> Dictionary:
    """Get the global dictionary instance."""
    global _global_dict
    if _global_dict is None:
        # Try enhanced first
        enhanced_path = os.path.join(os.path.dirname(__file__), 'crossplay_dict_enhanced.pkl')
        if os.path.exists(enhanced_path):
            _global_dict = Dictionary.load(enhanced_path)
        else:
            # Fall back to basic
            default_path = os.path.join(os.path.dirname(__file__), 'crossplay_dict.pkl')
            if os.path.exists(default_path):
                _global_dict = Dictionary.load(default_path)
            else:
                raise FileNotFoundError("Dictionary not found")
    return _global_dict


def is_valid_word(word: str) -> bool:
    """Check if a word is valid using global dictionary."""
    return get_dictionary().is_valid(word)


def modify_dictionary(add=None, remove=None, rebuild_gaddag=True):
    """Add/remove words from the dictionary and optionally rebuild GADDAG.

    This is the single entry point for dictionary changes. It:
      1. Loads crossplay_dict.pkl
      2. Adds/removes specified words
      3. Saves updated crossplay_dict.pkl
      4. Invalidates and rebuilds the GADDAG (~25-48s)
      5. Resets module-level singleton caches

    Args:
        add: List of words to add (case-insensitive).
        remove: List of words to remove (case-insensitive).
        rebuild_gaddag: If True (default), rebuild GADDAG after changes.

    Returns:
        dict with 'added', 'removed', 'already_existed', 'not_found' lists.
    """
    import time
    global _global_dict

    add = [w.upper() for w in (add or [])]
    remove = [w.upper() for w in (remove or [])]

    # Load current dictionary
    base_dir = os.path.dirname(__file__)
    dict_path = os.path.join(base_dir, 'crossplay_dict.pkl')
    dictionary = Dictionary.load(dict_path, use_pattern_index=False)

    result = {
        'added': [],
        'removed': [],
        'already_existed': [],
        'not_found': [],
    }

    # Apply additions
    for word in add:
        if word in dictionary._words:
            result['already_existed'].append(word)
        else:
            dictionary.add_word(word)
            result['added'].append(word)

    # Apply removals
    for word in remove:
        if word in dictionary._words:
            dictionary.remove_word(word)
            result['removed'].append(word)
        else:
            result['not_found'].append(word)

    # Save if anything changed
    if result['added'] or result['removed']:
        dictionary.save(dict_path)
        print(f"  Dictionary saved: {len(dictionary)} words")

        # Reset dictionary singleton
        _global_dict = None

        if rebuild_gaddag:
            from . import gaddag as gaddag_mod
            from .game_manager import invalidate_resources

            print("  Rebuilding GADDAG (this takes 25-48s)...")
            t0 = time.perf_counter()
            gaddag_mod.rebuild()
            elapsed = time.perf_counter() - t0
            print(f"  GADDAG rebuilt in {elapsed:.1f}s")

            # Reset game_manager cached resources
            invalidate_resources()

        # Summary
        parts = []
        if result['added']:
            parts.append(f"Added {len(result['added'])}: {', '.join(result['added'])}")
        if result['removed']:
            parts.append(f"Removed {len(result['removed'])}: {', '.join(result['removed'])}")
        if result['already_existed']:
            parts.append(f"Already existed: {', '.join(result['already_existed'])}")
        if result['not_found']:
            parts.append(f"Not found: {', '.join(result['not_found'])}")
        print(f"  {'. '.join(parts)}.")
    else:
        print("  No changes to dictionary.")

    return result


if __name__ == "__main__":
    d = get_dictionary()
    print(f"Dictionary loaded with {len(d)} words")
    print(f"Enhanced: {d.is_enhanced}")
    
    if d.is_enhanced:
        print(f"\nOXYGEN back hooks: {d.get_back_hooks('OXYGEN')}")
        print(f"OXYGEN front hooks: {d.get_front_hooks('OXYGEN')}")
        print(f"OXYGEN base score: {d.get_base_score('OXYGEN')}")
