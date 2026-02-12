"""
CROSSPLAY V7 - GADDAG Data Structure
A GADDAG for fast Scrabble move generation.

Based on Steven A. Gordon's 1994 paper "A Faster Scrabble Move Generation Algorithm"

For word CARE, GADDAG stores:
  C+ARE   (start at C, go right to spell ARE)
  AC+RE   (start at A, C is to left, go right to spell RE)  
  RAC+E   (start at R, AC is to left, go right to spell E)
  ERAC+   (start at E, RAC is to left, nothing to right)

The '+' delimiter marks the transition point.
"""

import pickle
from typing import Dict, Set, Optional, Tuple


DELIMITER = '+'


class GADDAGNode:
    """A node in the GADDAG trie."""
    __slots__ = ['children', 'is_terminal']
    
    def __init__(self):
        self.children: Dict[str, 'GADDAGNode'] = {}
        self.is_terminal: bool = False


class GADDAG:
    """GADDAG data structure for efficient move generation."""
    
    def __init__(self):
        self.root = GADDAGNode()
        self._word_count = 0
    
    def add_word(self, word: str) -> None:
        """Add a word to the GADDAG."""
        word = word.upper()
        if len(word) < 2:
            return
        
        self._word_count += 1
        
        # Add all entry points for this word
        # For word of length N, we add N paths
        for i in range(len(word)):
            # Path: reversed_prefix + DELIMITER + suffix
            # reversed_prefix = word[0:i+1] reversed
            # suffix = word[i+1:]
            
            prefix_reversed = word[:i+1][::-1]
            suffix = word[i+1:]
            
            path = prefix_reversed + DELIMITER + suffix
            self._add_path(path)
    
    def _add_path(self, path: str) -> None:
        """Add a path to the trie."""
        node = self.root
        for char in path:
            if char not in node.children:
                node.children[char] = GADDAGNode()
            node = node.children[char]
        node.is_terminal = True
    
    def build_from_words(self, words: Set[str]) -> None:
        """Build GADDAG from a set of words."""
        # Add all dictionary words
        for word in words:
            if len(word) >= 2:
                self.add_word(word)
        
        # Also add VALID_TWO_LETTER words (Scrabble standard 2-letter words)
        from .config import VALID_TWO_LETTER
        for word in VALID_TWO_LETTER:
            self.add_word(word)
    
    def get_node(self, path: str) -> Optional[GADDAGNode]:
        """Get node at end of path, or None if doesn't exist."""
        node = self.root
        for char in path.upper():
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def has_path(self, path: str) -> bool:
        """Check if path exists (may or may not be terminal)."""
        return self.get_node(path) is not None
    
    def is_complete_word(self, path: str) -> bool:
        """Check if path exists AND is terminal (complete word)."""
        node = self.get_node(path)
        return node is not None and node.is_terminal
    
    def get_children(self, path: str) -> Set[str]:
        """Get valid next letters after path."""
        node = self.get_node(path)
        if node is None:
            return set()
        return set(node.children.keys())
    
    def is_word(self, word: str) -> bool:
        """Check if word is valid (convenience method)."""
        from .config import VALID_TWO_LETTER
        word = word.upper()
        # For 2-letter words, use VALID_TWO_LETTER (Scrabble standard)
        if len(word) == 2:
            return word in VALID_TWO_LETTER
        # For longer words, check GADDAG
        path = word[::-1] + DELIMITER
        return self.is_complete_word(path)
    
    def save(self, filepath: str) -> None:
        """Save GADDAG to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump((self.root, self._word_count), f)
    
    @classmethod
    def load(cls, filepath: str) -> 'GADDAG':
        """Load GADDAG from pickle file."""
        gaddag = cls()
        with open(filepath, 'rb') as f:
            gaddag.root, gaddag._word_count = pickle.load(f)
        return gaddag
    
    def __len__(self) -> int:
        return self._word_count
    
    def stats(self) -> Dict:
        """Get statistics about the GADDAG."""
        def count_nodes(node: GADDAGNode) -> Tuple[int, int]:
            total = 1
            terminals = 1 if node.is_terminal else 0
            for child in node.children.values():
                t, term = count_nodes(child)
                total += t
                terminals += term
            return total, terminals
        
        total_nodes, terminal_nodes = count_nodes(self.root)
        return {
            'words': self._word_count,
            'total_nodes': total_nodes,
            'terminal_nodes': terminal_nodes,
        }


# Global instance
_gaddag: Optional[GADDAG] = None


def get_gaddag():
    """Get or build the global GADDAG instance.
    
    Load priority:
      1. Compact binary (gaddag_compact.bin) — 0.05s load
      2. Pickle format (gaddag.pkl) — ~7s load
      3. Build from dictionary + cache as compact binary — ~48s first time only
    """
    global _gaddag
    
    if _gaddag is not None:
        return _gaddag
    
    import os, time
    base_dir = os.path.dirname(__file__)
    compact_path = os.path.join(base_dir, 'gaddag_compact.bin')
    gaddag_path = os.path.join(base_dir, 'gaddag.pkl')
    
    # Prefer compact binary format (0.05s load)
    if os.path.exists(compact_path):
        from .gaddag_compact import CompactGADDAG
        _gaddag = CompactGADDAG.load(compact_path)
        return _gaddag
    
    # Fall back to pickle format (~7s load)
    if os.path.exists(gaddag_path):
        _gaddag = GADDAG.load(gaddag_path)
        return _gaddag
    
    # Build from dictionary and cache as compact binary
    dict_path = os.path.join(base_dir, 'crossplay_dict.pkl')
    if not os.path.exists(dict_path):
        raise FileNotFoundError(
            f"No GADDAG or dictionary found in {base_dir}. "
            f"Need gaddag_compact.bin, gaddag.pkl, or crossplay_dict.pkl")
    
    print(f"GADDAG not found -- building from dictionary (one-time, ~48s)...")
    t0 = time.perf_counter()
    
    with open(dict_path, 'rb') as f:
        words = pickle.load(f)
    
    trie = GADDAG()
    trie.build_from_words(words)
    t_trie = time.perf_counter() - t0
    print(f"  Trie built: {trie.stats()} ({t_trie:.0f}s)")
    
    # Convert to compact format and save for future fast loading
    from .gaddag_compact import CompactGADDAG
    t1 = time.perf_counter()
    _gaddag = CompactGADDAG.build_from_gaddag(trie)
    t_compact = time.perf_counter() - t1
    
    try:
        _gaddag.save(compact_path)
        total = time.perf_counter() - t0
        print(f"  Saved {compact_path} ({total:.0f}s total)")
        print(f"  Next startup will load in <0.1s")
    except (OSError, PermissionError) as e:
        print(f"  Warning: could not cache GADDAG ({e})")
        print(f"  Will rebuild on next startup")
    
    return _gaddag


if __name__ == "__main__":
    import time
    import os
    
    base_dir = os.path.dirname(__file__)
    dict_path = os.path.join(base_dir, 'crossplay_dict.pkl')
    gaddag_path = os.path.join(base_dir, 'gaddag.pkl')
    
    # Remove old GADDAG to rebuild
    if os.path.exists(gaddag_path):
        os.remove(gaddag_path)
    
    print("Loading dictionary...")
    with open(dict_path, 'rb') as f:
        words = pickle.load(f)
    print(f"Dictionary has {len(words)} words")
    
    print("\nBuilding GADDAG...")
    start = time.time()
    gaddag = GADDAG()
    gaddag.build_from_words(words)
    build_time = time.time() - start
    
    print(f"Build time: {build_time:.2f}s")
    print(f"Stats: {gaddag.stats()}")
    
    print("\nSaving GADDAG...")
    gaddag.save(gaddag_path)
    print(f"Saved to {gaddag_path}")
    
    # Test word validation
    print("\nWord validation tests:")
    test_words = ['CARE', 'RETINAS', 'NASTIER', 'QI', 'ZA', 'XU', 'ZXQJ']
    for word in test_words:
        valid = gaddag.is_word(word)
        print(f"  {word}: {'OK valid' if valid else 'X invalid'}")
    
    # Test GADDAG paths for CARE
    print("\nPath tests for 'CARE':")
    for path in ['C+ARE', 'AC+RE', 'RAC+E', 'ERAC+']:
        exists = gaddag.is_complete_word(path)
        print(f"  {path}: {'OK' if exists else 'X'}")
