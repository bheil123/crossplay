"""
CROSSPLAY V15 - SuperLeaves Table

Dict mapping sorted tuple -> float leave value.
Supports bootstrapping from formula, EMA updates, and persistence.
"""

import os
import pickle
import tempfile
from collections import defaultdict
from itertools import combinations_with_replacement


class LeaveTable:
    """Lookup table for empirical leave values.

    Keys are sorted tuples of uppercase letters, blanks as '?'.
    Values are floats representing equity contribution.
    """

    def __init__(self):
        self._table = {}
        self._stv_cache = None  # cached single_tile_values

    def __len__(self):
        return len(self._table)

    def __contains__(self, key):
        return key in self._table

    def get(self, leave_key, default=0.0):
        """Look up leave value by sorted tuple key."""
        return self._table.get(leave_key, default)

    def set(self, leave_key, value):
        """Set leave value for a key."""
        self._table[leave_key] = value

    def keys(self):
        return self._table.keys()

    def items(self):
        return self._table.items()

    def bootstrap_from_formula(self):
        """Initialize all valid leaves from the hand-tuned formula.

        Enumerates every valid leave of size 1-6 respecting
        TILE_DISTRIBUTION counts, then evaluates with the formula.
        """
        from ..leave_eval import _formula_evaluate
        from ..config import TILE_DISTRIBUTION

        # Build tile alphabet with max counts
        tiles = []
        max_counts = {}
        for letter, count in sorted(TILE_DISTRIBUTION.items()):
            tiles.append(letter)
            max_counts[letter] = count

        total = 0
        for size in range(1, 7):
            for combo in combinations_with_replacement(tiles, size):
                # Validate: each tile count <= distribution count
                valid = True
                seen = {}
                for t in combo:
                    seen[t] = seen.get(t, 0) + 1
                    if seen[t] > max_counts[t]:
                        valid = False
                        break
                if not valid:
                    continue

                leave_key = tuple(combo)  # already sorted by c_w_r
                leave_str = ''.join(combo)
                self._table[leave_key] = _formula_evaluate(leave_str)
                total += 1

        return total

    def batch_update_ema(self, observations, alpha):
        """Apply EMA updates from a batch of observations.

        Args:
            observations: list of (leave_key, equity_signal, weight)
            alpha: EMA learning rate
        """
        # Aggregate observations per key: weighted sum / total weight
        agg = defaultdict(lambda: [0.0, 0.0])  # key -> [weighted_sum, total_weight]
        for leave_key, signal, weight in observations:
            agg[leave_key][0] += signal * weight
            agg[leave_key][1] += weight

        # Apply EMA: new = (1-alpha) * old + alpha * observed_mean
        stv_dirty = False
        for leave_key, (wsum, wtotal) in agg.items():
            observed_mean = wsum / wtotal
            old = self._table.get(leave_key, 0.0)
            self._table[leave_key] = (1 - alpha) * old + alpha * observed_mean
            if len(leave_key) == 1:
                stv_dirty = True
        if stv_dirty:
            self._stv_cache = None

    def single_tile_values(self):
        """Return dict of single-tile leave values for validation (cached)."""
        if self._stv_cache is not None:
            return self._stv_cache
        result = {}
        for key, val in self._table.items():
            if len(key) == 1:
                result[key[0]] = val
        self._stv_cache = result
        return result

    def save(self, path):
        """Save table to disk atomically."""
        dir_name = os.path.dirname(path) or '.'
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as f:
                pickle.dump(self._table, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path):
        """Load table from disk."""
        table = cls()
        with open(path, 'rb') as f:
            table._table = pickle.load(f)
        return table
