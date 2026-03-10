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

    Optionally tracks per-key observation counts and per-phase
    running sums for game-phase-dependent analysis (added in gen8).

    Phase encoding: 0 = early (bag 60+), 1 = mid (bag 30-59), 2 = late (bag 1-29)
    """

    # Phase boundaries (bag tile counts)
    PHASE_EARLY = 60   # bag >= 60
    PHASE_MID = 30     # 30 <= bag < 60
    # late = bag < 30

    def __init__(self):
        self._table = {}
        self._counts = defaultdict(int)  # per-key observation counts
        self._counts_start_games = 0  # game count when counting began
        self._stv_cache = None  # cached single_tile_values
        # Phase-dependent tracking: keyed by (leave_key, phase_int)
        self._phase_counts = defaultdict(int)    # observation count per (key, phase)
        self._phase_sums = defaultdict(float)    # running signal sum per (key, phase)
        # Online linear regression accumulators per key:
        # n, sum_x (bag_size), sum_y (signal), sum_xy, sum_xx
        # Allows computing slope (how leave value changes with bag size)
        # and intercept at any time with no additional storage per observation.
        self._reg_n = defaultdict(int)
        self._reg_sum_x = defaultdict(float)
        self._reg_sum_y = defaultdict(float)
        self._reg_sum_xy = defaultdict(float)
        self._reg_sum_xx = defaultdict(float)

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

    def bootstrap_from_quackle(self):
        """Initialize all valid leaves from Quackle-derived per-tile values.

        Same enumeration as bootstrap_from_formula, but uses Quackle's
        per-tile values (adapted for Crossplay) instead of our hand-tuned
        formula. This provides an alternative starting point for training.
        """
        from ..leave_eval import QUACKLE_TILE_VALUES
        from ..config import TILE_DISTRIBUTION

        tiles = []
        max_counts = {}
        for letter, count in sorted(TILE_DISTRIBUTION.items()):
            tiles.append(letter)
            max_counts[letter] = count

        total = 0
        for size in range(1, 7):
            for combo in combinations_with_replacement(tiles, size):
                valid = True
                seen = {}
                for t in combo:
                    seen[t] = seen.get(t, 0) + 1
                    if seen[t] > max_counts[t]:
                        valid = False
                        break
                if not valid:
                    continue

                leave_key = tuple(combo)
                value = sum(QUACKLE_TILE_VALUES.get(t, -1.0) for t in combo)
                self._table[leave_key] = value
                total += 1

        return total

    def bootstrap_from_research(self):
        """Initialize all valid leaves using research-derived Crossplay values.

        Uses _research_evaluate() which applies MAGPIE/Quackle 2025 per-tile
        values adapted for Crossplay differences, with interaction terms
        (V:C balance, duplicate penalties, Q-without-U, bingo stems).
        """
        from ..leave_eval import _research_evaluate
        from ..config import TILE_DISTRIBUTION

        tiles = []
        max_counts = {}
        for letter, count in sorted(TILE_DISTRIBUTION.items()):
            tiles.append(letter)
            max_counts[letter] = count

        total = 0
        for size in range(1, 7):
            for combo in combinations_with_replacement(tiles, size):
                valid = True
                seen = {}
                for t in combo:
                    seen[t] = seen.get(t, 0) + 1
                    if seen[t] > max_counts[t]:
                        valid = False
                        break
                if not valid:
                    continue

                leave_key = tuple(combo)
                leave_str = ''.join(combo)
                self._table[leave_key] = _research_evaluate(leave_str)
                total += 1

        return total

    @staticmethod
    def _bag_to_phase(bag_size):
        """Convert bag size to phase int: 0=early, 1=mid, 2=late."""
        if bag_size >= LeaveTable.PHASE_EARLY:
            return 0
        elif bag_size >= LeaveTable.PHASE_MID:
            return 1
        else:
            return 2

    def batch_update_ema(self, observations, alpha):
        """Apply EMA updates from a batch of observations.

        Args:
            observations: list of (leave_key, equity_signal, weight[, bag_size])
                          bag_size is optional for backward compatibility
            alpha: EMA learning rate
        """
        # Aggregate observations per key: weighted sum / total weight
        agg = defaultdict(lambda: [0.0, 0.0])  # key -> [weighted_sum, total_weight]
        # Phase-specific aggregation: (key, phase) -> [sum, count]
        phase_agg = defaultdict(lambda: [0.0, 0])

        # Regression accumulation: (key) -> [n, sum_x, sum_y, sum_xy, sum_xx]
        reg_agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])

        for obs in observations:
            if len(obs) >= 4:
                leave_key, signal, weight, bag_size = obs[0], obs[1], obs[2], obs[3]
                phase = self._bag_to_phase(bag_size)
                phase_agg[(leave_key, phase)][0] += signal
                phase_agg[(leave_key, phase)][1] += 1
                # Regression accumulators (unweighted -- raw signal vs bag_size)
                r = reg_agg[leave_key]
                r[0] += 1
                r[1] += bag_size
                r[2] += signal
                r[3] += bag_size * signal
                r[4] += bag_size * bag_size
            else:
                leave_key, signal, weight = obs[0], obs[1], obs[2]
            agg[leave_key][0] += signal * weight
            agg[leave_key][1] += weight

        # Apply EMA: new = (1-alpha) * old + alpha * observed_mean
        stv_dirty = False
        for leave_key, (wsum, wtotal) in agg.items():
            observed_mean = wsum / wtotal
            old = self._table.get(leave_key, 0.0)
            self._table[leave_key] = (1 - alpha) * old + alpha * observed_mean
            self._counts[leave_key] += 1  # count unique observations per batch
            if len(leave_key) == 1:
                stv_dirty = True
        if stv_dirty:
            self._stv_cache = None

        # Accumulate phase data (running sums for offline analysis)
        for (leave_key, phase), (sig_sum, count) in phase_agg.items():
            pk = (leave_key, phase)
            self._phase_sums[pk] += sig_sum
            self._phase_counts[pk] += count

        # Accumulate regression data
        for leave_key, (n, sx, sy, sxy, sxx) in reg_agg.items():
            self._reg_n[leave_key] += n
            self._reg_sum_x[leave_key] += sx
            self._reg_sum_y[leave_key] += sy
            self._reg_sum_xy[leave_key] += sxy
            self._reg_sum_xx[leave_key] += sxx

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

    def observation_counts(self):
        """Return the per-key observation counts dict."""
        return dict(self._counts)

    def count_stats(self):
        """Return summary statistics about observation counts.

        Returns dict with: total_observed (keys with >0 counts),
        total_unobserved, total_observations, min/max/mean counts,
        counts_start_games, and distribution by leave size.
        """
        total_keys = len(self._table)
        observed = {k: v for k, v in self._counts.items() if v > 0}
        n_observed = len(observed)
        counts_list = list(observed.values()) if observed else [0]

        # Per-size breakdown
        from collections import Counter
        size_total = Counter(len(k) for k in self._table)
        size_observed = Counter()
        size_obs_counts = defaultdict(list)
        for k, c in observed.items():
            size_observed[len(k)] += 1
            size_obs_counts[len(k)].append(c)

        by_size = {}
        for sz in sorted(size_total):
            obs_list = size_obs_counts.get(sz, [0])
            by_size[sz] = {
                'total': size_total[sz],
                'observed': size_observed.get(sz, 0),
                'pct': 100 * size_observed.get(sz, 0) / max(size_total[sz], 1),
                'mean_count': sum(obs_list) / max(len(obs_list), 1),
            }

        return {
            'total_keys': total_keys,
            'total_observed': n_observed,
            'total_unobserved': total_keys - n_observed,
            'pct_observed': 100 * n_observed / max(total_keys, 1),
            'total_observations': sum(counts_list),
            'min_count': min(counts_list),
            'max_count': max(counts_list),
            'mean_count': sum(counts_list) / max(len(counts_list), 1),
            'counts_start_games': self._counts_start_games,
            'by_size': by_size,
        }

    def regression(self, leave_key):
        """Return linear regression (slope, intercept, n) for a leave key.

        slope: how much the leave value changes per tile in the bag
               (positive = better with more tiles in bag, i.e. better early)
        intercept: estimated value when bag is empty
        n: number of observations

        Returns None if insufficient data (< 3 observations).
        """
        n = self._reg_n.get(leave_key, 0)
        if n < 3:
            return None
        sx = self._reg_sum_x[leave_key]
        sy = self._reg_sum_y[leave_key]
        sxy = self._reg_sum_xy[leave_key]
        sxx = self._reg_sum_xx[leave_key]
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-10:
            return None
        slope = (n * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / n
        return slope, intercept, n

    def regression_summary(self, min_obs=50):
        """Return regression analysis for keys with enough data.

        Returns list of dicts sorted by |slope| (biggest phase effects first).
        """
        results = []
        for leave_key in self._reg_n:
            reg = self.regression(leave_key)
            if reg is None:
                continue
            slope, intercept, n = reg
            if n < min_obs:
                continue
            # Estimated values at early (bag=70) and late (bag=10)
            val_early = intercept + slope * 70
            val_late = intercept + slope * 10
            results.append({
                'key': leave_key,
                'slope': slope,
                'intercept': intercept,
                'n': n,
                'val_early': val_early,
                'val_late': val_late,
                'spread': val_early - val_late,
            })
        results.sort(key=lambda x: -abs(x['slope']))
        return results

    def phase_values(self, leave_key):
        """Return per-phase mean values for a leave key.

        Returns dict: {0: mean_early, 1: mean_mid, 2: mean_late}
        Only includes phases with observations.
        """
        result = {}
        for phase in (0, 1, 2):
            pk = (leave_key, phase)
            count = self._phase_counts.get(pk, 0)
            if count > 0:
                result[phase] = self._phase_sums[pk] / count
        return result

    def phase_summary(self, min_obs=20):
        """Return phase-dependent analysis for keys with enough data.

        Args:
            min_obs: minimum observations per phase to include

        Returns list of dicts with key, early/mid/late means, spread, counts.
        Sorted by spread (biggest phase differences first).
        """
        phase_names = {0: 'early', 1: 'mid', 2: 'late'}
        results = []

        # Find keys with min_obs in all 3 phases
        seen_keys = set()
        for (leave_key, phase) in self._phase_counts:
            seen_keys.add(leave_key)

        for leave_key in seen_keys:
            counts = {}
            means = {}
            for phase in (0, 1, 2):
                pk = (leave_key, phase)
                c = self._phase_counts.get(pk, 0)
                if c < min_obs:
                    break
                counts[phase] = c
                means[phase] = self._phase_sums[pk] / c
            else:
                # All 3 phases have enough data
                spread = max(means.values()) - min(means.values())
                results.append({
                    'key': leave_key,
                    'early': means[0],
                    'mid': means[1],
                    'late': means[2],
                    'spread': spread,
                    'counts': counts,
                })

        results.sort(key=lambda x: -x['spread'])
        return results

    def save(self, path):
        """Save table to disk atomically.

        Format v4: phase buckets + linear regression accumulators.
        Backward compatible -- load() handles v1 (raw), v2, v3, and v4.
        """
        dir_name = os.path.dirname(path) or '.'
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
        try:
            # Phase dicts: (tuple, int) keys -> list of (key, phase, value)
            phase_counts_list = [
                (k, p, c) for (k, p), c in self._phase_counts.items() if c > 0
            ]
            phase_sums_list = [
                (k, p, s) for (k, p), s in self._phase_sums.items() if s != 0.0
            ]
            # Regression: per-key -> list of (key, n, sx, sy, sxy, sxx)
            reg_list = [
                (k, self._reg_n[k], self._reg_sum_x[k], self._reg_sum_y[k],
                 self._reg_sum_xy[k], self._reg_sum_xx[k])
                for k in self._reg_n if self._reg_n[k] > 0
            ]
            data = {
                '_format': 'v4',
                'table': self._table,
                'counts': dict(self._counts),
                'counts_start_games': self._counts_start_games,
                'phase_counts': phase_counts_list,
                'phase_sums': phase_sums_list,
                'regression': reg_list,
            }
            with os.fdopen(fd, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path):
        """Load table from disk.

        Handles old format (raw dict), v2 (counts), v3 (phase), v4 (regression).
        """
        table = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)

        fmt = data.get('_format', '') if isinstance(data, dict) else ''

        if fmt in ('v3', 'v4'):
            table._table = data['table']
            table._counts = defaultdict(int, data.get('counts', {}))
            table._counts_start_games = data.get('counts_start_games', 0)
            # Restore phase data from lists
            table._phase_counts = defaultdict(int)
            for k, p, c in data.get('phase_counts', []):
                table._phase_counts[(k, p)] = c
            table._phase_sums = defaultdict(float)
            for k, p, s in data.get('phase_sums', []):
                table._phase_sums[(k, p)] = s
            # Restore regression data (v4+)
            for k, n, sx, sy, sxy, sxx in data.get('regression', []):
                table._reg_n[k] = n
                table._reg_sum_x[k] = sx
                table._reg_sum_y[k] = sy
                table._reg_sum_xy[k] = sxy
                table._reg_sum_xx[k] = sxx
        elif fmt == 'v2':
            table._table = data['table']
            table._counts = defaultdict(int, data.get('counts', {}))
            table._counts_start_games = data.get('counts_start_games', 0)
        else:
            # Old format: raw dict of leave values
            table._table = data
            table._counts = defaultdict(int)
            table._counts_start_games = 0

        return table
