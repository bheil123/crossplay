"""
CROSSPLAY V14 - Compact GADDAG Data Structure

Packs the entire GADDAG trie into a flat bytearray for:
- ~27 MB on disk (vs 106 MB pickle)
- ~28 MB in RAM (vs 2.4 GB)  
- Sub-second load (vs 7.5s)
- Same API as original GADDAG (drop-in replacement)

Binary format per node at offset `pos`:
  byte 0:     flags -> bit 7 = is_terminal, bits 0-4 = num_children (0-27)
  bytes 1..:  N child entries, each 5 bytes:
              byte 0: char_index (0-26: A=0..Z=25, +=26)
              bytes 1-4: uint32 child offset (little-endian)

Children are sorted by char_index for fast lookup.
"""

import struct
import os
from typing import Optional, Set


DELIMITER = '+'

# Char mapping: A-Z=0-25, +=26
_CHAR_TO_IDX = {chr(65 + i): i for i in range(26)}
_CHAR_TO_IDX['+'] = 26
_IDX_TO_CHAR = {v: k for k, v in _CHAR_TO_IDX.items()}


class CompactChildren:
    """Proxy object that mimics dict-like access into the packed bytearray."""
    __slots__ = ['_data', '_pos', '_count']

    def __init__(self, data: bytearray, pos: int, count: int):
        self._data = data
        self._pos = pos
        self._count = count

    def __contains__(self, char: str) -> bool:
        idx = _CHAR_TO_IDX.get(char)
        if idx is None:
            return False
        data = self._data
        pos = self._pos
        for i in range(self._count):
            ci = data[pos + i * 5]
            if ci == idx:
                return True
            if ci > idx:
                return False
        return False

    def __getitem__(self, char: str) -> 'CompactNode':
        idx = _CHAR_TO_IDX.get(char)
        if idx is None:
            raise KeyError(char)
        data = self._data
        pos = self._pos
        for i in range(self._count):
            off = pos + i * 5
            ci = data[off]
            if ci == idx:
                child_offset = int.from_bytes(data[off + 1:off + 5], 'little')
                return CompactNode(data, child_offset)
            if ci > idx:
                break
        raise KeyError(char)

    def __iter__(self):
        data = self._data
        pos = self._pos
        for i in range(self._count):
            yield _IDX_TO_CHAR[data[pos + i * 5]]

    def __len__(self):
        return self._count

    def keys(self):
        return list(self)

    def values(self):
        data = self._data
        pos = self._pos
        result = []
        for i in range(self._count):
            off = pos + i * 5
            child_offset = int.from_bytes(data[off + 1:off + 5], 'little')
            result.append(CompactNode(data, child_offset))
        return result

    def items(self):
        data = self._data
        pos = self._pos
        result = []
        for i in range(self._count):
            off = pos + i * 5
            ci = data[off]
            child_offset = int.from_bytes(data[off + 1:off + 5], 'little')
            result.append((_IDX_TO_CHAR[ci], CompactNode(data, child_offset)))
        return result

    def get(self, char: str, default=None):
        try:
            return self[char]
        except KeyError:
            return default


class CompactNode:
    """Lightweight proxy into the packed bytearray. Mimics GADDAGNode API."""
    __slots__ = ['_data', '_offset', '_children']

    def __init__(self, data: bytearray, offset: int):
        self._data = data
        self._offset = offset
        self._children = None

    @property
    def is_terminal(self) -> bool:
        return bool(self._data[self._offset] & 0x80)

    @property
    def children(self) -> CompactChildren:
        if self._children is None:
            flags = self._data[self._offset]
            count = flags & 0x1F
            self._children = CompactChildren(self._data, self._offset + 1, count)
        return self._children


class CompactGADDAG:
    """Drop-in replacement for GADDAG using packed bytearray storage.
    
    Supports two access modes:
    1. Object API (via .root, CompactNode, CompactChildren) - compatible with existing code
    2. Offset API (via _has_child, _get_child, etc.) - zero allocation, for fast move gen
    """

    def __init__(self):
        self._data: Optional[bytearray] = None
        self._word_count: int = 0
        self._delim_idx = 26  # '+' char index

    @property
    def root(self) -> CompactNode:
        return CompactNode(self._data, 0)

    # ===== Fast offset-based API (no object creation) =====
    
    def _is_terminal(self, offset: int) -> bool:
        """Check if node at offset is terminal. No objects created."""
        return bool(self._data[offset] & 0x80)
    
    def _child_count(self, offset: int) -> int:
        """Get number of children at node offset."""
        return self._data[offset] & 0x1F
    
    def _get_child(self, offset: int, char_idx: int) -> int:
        """Get child offset for char_idx at node. Returns -1 if not found. No objects."""
        data = self._data
        count = data[offset] & 0x1F
        pos = offset + 1
        for i in range(count):
            off = pos + i * 5
            ci = data[off]
            if ci == char_idx:
                return int.from_bytes(data[off + 1:off + 5], 'little')
            if ci > char_idx:
                return -1
        return -1
    
    def _get_child_char(self, offset: int, char: str) -> int:
        """Get child offset for char. Returns -1 if not found."""
        idx = _CHAR_TO_IDX.get(char)
        if idx is None:
            return -1
        return self._get_child(offset, idx)
    
    def _iter_children(self, offset: int):
        """Yield (char, child_offset) for all children. Minimal allocation."""
        data = self._data
        count = data[offset] & 0x1F
        pos = offset + 1
        for i in range(count):
            off = pos + i * 5
            ci = data[off]
            child_off = int.from_bytes(data[off + 1:off + 5], 'little')
            yield (_IDX_TO_CHAR[ci], child_off)
    
    def _iter_children_letters(self, offset: int):
        """Yield only char labels of children. For blank iteration."""
        data = self._data
        count = data[offset] & 0x1F
        pos = offset + 1
        for i in range(count):
            yield _IDX_TO_CHAR[data[pos + i * 5]]

    # ===== Standard API =====

    def get_node(self, path: str) -> Optional[CompactNode]:
        node = self.root
        for char in path:
            ch = node.children
            if char not in ch:
                return None
            node = ch[char]
        return node

    def has_path(self, path: str) -> bool:
        return self.get_node(path) is not None

    def is_complete_word(self, path: str) -> bool:
        node = self.get_node(path)
        return node is not None and node.is_terminal

    def get_children(self, path: str) -> Set[str]:
        node = self.get_node(path)
        if node is None:
            return set()
        return set(node.children)

    def is_word(self, word: str) -> bool:
        from .config import VALID_TWO_LETTER
        word = word.upper()
        if len(word) == 2:
            return word in VALID_TWO_LETTER
        path = word[::-1] + DELIMITER
        return self.is_complete_word(path)

    def __len__(self) -> int:
        return self._word_count

    # === Fast offset-based API (no object creation) ===
    # Used by move_finder_gaddag.py fast path

    def _get_child(self, offset: int, char_idx: int) -> int:
        """Return child offset or -1 if not found. Zero object creation."""
        data = self._data
        count = data[offset] & 0x1F
        pos = offset + 1
        for i in range(count):
            off = pos + i * 5
            ci = data[off]
            if ci == char_idx:
                return int.from_bytes(data[off + 1:off + 5], 'little')
            if ci > char_idx:
                return -1
        return -1

    def _is_terminal(self, offset: int) -> bool:
        """Check terminal flag. Zero object creation."""
        return bool(self._data[offset] & 0x80)

    def _iter_children(self, offset: int):
        """Yield (letter, child_offset) pairs. Minimal object creation."""
        data = self._data
        count = data[offset] & 0x1F
        pos = offset + 1
        for i in range(count):
            off = pos + i * 5
            ci = data[off]
            child_off = int.from_bytes(data[off + 1:off + 5], 'little')
            yield _IDX_TO_CHAR[ci], child_off

    def save(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            f.write(b'CGDG')
            f.write(struct.pack('<I', self._word_count))
            f.write(struct.pack('<I', len(self._data)))
            f.write(self._data)

    @classmethod
    def load(cls, filepath: str) -> 'CompactGADDAG':
        gaddag = cls()
        with open(filepath, 'rb') as f:
            magic = f.read(4)
            if magic != b'CGDG':
                raise ValueError(f"Invalid compact GADDAG file (magic={magic})")
            gaddag._word_count = struct.unpack('<I', f.read(4))[0]
            data_len = struct.unpack('<I', f.read(4))[0]
            gaddag._data = bytearray(f.read(data_len))
        return gaddag

    @classmethod
    def build_from_gaddag(cls, old_gaddag) -> 'CompactGADDAG':
        """Build compact GADDAG from existing GADDAG with GADDAGNode tree."""
        compact = cls()
        compact._word_count = old_gaddag._word_count

        from collections import deque

        node_offsets = {}
        queue = deque()
        queue.append(old_gaddag.root)
        offset = 0
        order = []
        seen = set()
        seen.add(id(old_gaddag.root))
        order.append(old_gaddag.root)

        while queue:
            node = queue.popleft()
            node_size = 1 + len(node.children) * 5
            node_offsets[id(node)] = offset
            offset += node_size
            for char in sorted(node.children.keys(), key=lambda c: _CHAR_TO_IDX[c]):
                child = node.children[char]
                if id(child) not in seen:
                    seen.add(id(child))
                    queue.append(child)
                    order.append(child)

        total_size = offset
        print(f"  Compact GADDAG: {len(order):,} nodes, {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")

        data = bytearray(total_size)
        for node in order:
            pos = node_offsets[id(node)]
            num_children = len(node.children)
            flags = num_children & 0x1F
            if node.is_terminal:
                flags |= 0x80
            data[pos] = flags
            child_items = sorted(node.children.items(), key=lambda x: _CHAR_TO_IDX[x[0]])
            for i, (char, child) in enumerate(child_items):
                entry_pos = pos + 1 + i * 5
                data[entry_pos] = _CHAR_TO_IDX[char]
                struct.pack_into('<I', data, entry_pos + 1, node_offsets[id(child)])

        compact._data = data
        return compact


_compact_gaddag: Optional[CompactGADDAG] = None


def get_compact_gaddag() -> CompactGADDAG:
    global _compact_gaddag
    if _compact_gaddag is not None:
        return _compact_gaddag

    base_dir = os.path.dirname(__file__)
    compact_path = os.path.join(base_dir, 'gaddag_compact.bin')

    if os.path.exists(compact_path):
        _compact_gaddag = CompactGADDAG.load(compact_path)
        return _compact_gaddag

    print("Building compact GADDAG from pickle (one-time operation)...")
    from .gaddag import GADDAG as OrigGADDAG
    old = OrigGADDAG.load(os.path.join(base_dir, 'gaddag.pkl'))
    _compact_gaddag = CompactGADDAG.build_from_gaddag(old)
    _compact_gaddag.save(compact_path)
    print(f"  Saved to {compact_path}")
    return _compact_gaddag
