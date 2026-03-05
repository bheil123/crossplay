"""
CROSSPLAY V21.3 - CMA-ES Per-Tile Leave Value Optimizer

Trains 27 optimal per-tile leave values from scratch using CMA-ES
(Covariance Matrix Adaptation Evolution Strategy). Each candidate
is evaluated by playing N games vs a fixed opponent, measuring
average spread.

Uses separable CMA-ES (diagonal covariance) for simplicity.
Plugs into existing fast_bot.py via TileValueAdapter.

Usage:
    # Smoke test (~5 min)
    python -m crossplay.superleaves.tile_trainer --games-per-eval 30 --generations 5 --population 8

    # Phase 1: broad search (~3-4 hours)
    python -m crossplay.superleaves.tile_trainer --games-per-eval 150 --generations 60

    # Phase 2: fine-tune from checkpoint
    python -m crossplay.superleaves.tile_trainer --resume tile_cma_gen60.pkl --games-per-eval 300 --generations 40

    # Custom starting point
    python -m crossplay.superleaves.tile_trainer --sigma 3.0 --population 24
"""

import os
import sys
import json
import math
import time
import random
import pickle
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..board import Board
from ..config import TILE_DISTRIBUTION, RACK_SIZE
from .fast_bot import select_best_move, compute_leave


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TILES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['?']
N_PARAMS = len(TILES)  # 27

# Quackle-derived starting point (Crossplay-calibrated)
QUACKLE_INIT = {
    '?': 20.0, 'S': 7.0, 'Z': 5.12, 'X': 3.31, 'R': 1.10,
    'C': 0.85, 'H': 0.60, 'M': 0.58, 'D': 0.45, 'E': 0.35,
    'N': 0.22, 'L': 0.20, 'T': -0.10, 'P': -0.46, 'K': -0.20,
    'Y': -0.63, 'A': -0.63, 'J': -0.80, 'B': -1.50, 'I': -2.07,
    'F': -2.21, 'O': -2.50, 'G': -1.80, 'W': -4.50, 'U': -4.00,
    'V': -6.50, 'Q': -6.79,
}


def _default_workers():
    """cpu_count - 3 (min 1), matching trainer/validator."""
    try:
        cpus = os.cpu_count() or 4
    except Exception:
        cpus = 4
    return max(1, cpus - 3)


# ---------------------------------------------------------------------------
# TileValueAdapter -- plugs per-tile values into fast_bot.select_best_move()
# ---------------------------------------------------------------------------

class TileValueAdapter:
    """Wraps per-tile values behind LeaveTable.get() interface."""

    def __init__(self, tile_values_dict):
        self._values = tile_values_dict

    def get(self, leave_key, default=0.0):
        if not leave_key:
            return 0.0
        return sum(self._values.get(t, -1.0) for t in leave_key)


# ---------------------------------------------------------------------------
# Separable CMA-ES (diagonal covariance)
# ---------------------------------------------------------------------------

class SepCMAES:
    """Separable CMA-ES optimizer for N_PARAMS continuous parameters.

    Uses diagonal covariance matrix (no cross-parameter correlations).
    Standard CMA-ES update rules adapted for separable case.
    """

    def __init__(self, mean_dict, sigma, population_size=20):
        self.n = N_PARAMS
        self.lam = population_size  # lambda (population size)
        self.mu = self.lam // 2     # number of parents

        # Mean vector (ordered by TILES)
        self.mean = [mean_dict.get(t, 0.0) for t in TILES]

        # Step size
        self.sigma = sigma

        # Diagonal variances (start uniform)
        self.diag = [1.0] * self.n

        # Evolution path for step-size adaptation (CSA)
        self.ps = [0.0] * self.n

        # Evolution path for covariance adaptation
        self.pc = [0.0] * self.n

        # Recombination weights (log-linear)
        raw_w = [math.log(self.mu + 0.5) - math.log(i + 1) for i in range(self.mu)]
        w_sum = sum(raw_w)
        self.weights = [w / w_sum for w in raw_w]

        # Effective mu
        self.mu_eff = 1.0 / sum(w * w for w in self.weights)

        # Adaptation parameters
        self.cs = (self.mu_eff + 2) / (self.n + self.mu_eff + 5)
        self.ds = 1.0 + 2.0 * max(0, math.sqrt((self.mu_eff - 1) / (self.n + 1)) - 1) + self.cs
        self.cc = (4 + self.mu_eff / self.n) / (self.n + 4 + 2 * self.mu_eff / self.n)
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mu_eff - 2 + 1 / self.mu_eff) /
                       ((self.n + 2) ** 2 + self.mu_eff))

        # Expected norm of N(0,I) for step-size control
        self.chi_n = math.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))

        # Generation counter
        self.generation = 0

        # Track best ever
        self.best_fitness = float('-inf')
        self.best_values = dict(zip(TILES, self.mean))

        # Current population (set by ask())
        self._population = []
        self._z_samples = []

    def ask(self):
        """Sample lambda candidates from current distribution.

        Returns list of dicts (tile -> value).
        """
        self._population = []
        self._z_samples = []

        for _ in range(self.lam):
            z = [random.gauss(0, 1) for _ in range(self.n)]
            x = [self.mean[i] + self.sigma * math.sqrt(self.diag[i]) * z[i]
                 for i in range(self.n)]
            self._z_samples.append(z)
            self._population.append(dict(zip(TILES, x)))

        return self._population

    def tell(self, fitnesses):
        """Update distribution from fitness results.

        Args:
            fitnesses: list of floats (one per candidate, higher = better)
        """
        assert len(fitnesses) == self.lam

        # Sort by fitness (descending -- we want to maximize)
        indices = sorted(range(self.lam), key=lambda i: fitnesses[i], reverse=True)

        # Weighted mean of top mu candidates
        old_mean = self.mean[:]
        self.mean = [0.0] * self.n
        for rank in range(self.mu):
            idx = indices[rank]
            w = self.weights[rank]
            for i in range(self.n):
                self.mean[i] += w * self._population[idx][TILES[i]]

        # Mean shift (normalized)
        dm = [(self.mean[i] - old_mean[i]) / self.sigma for i in range(self.n)]

        # Update evolution path for sigma (CSA)
        c_s = self.cs
        for i in range(self.n):
            self.ps[i] = (1 - c_s) * self.ps[i] + math.sqrt(c_s * (2 - c_s) * self.mu_eff) * dm[i] / math.sqrt(self.diag[i])

        # ps norm for step-size update
        ps_norm = math.sqrt(sum(p * p for p in self.ps))

        # Update sigma
        self.sigma *= math.exp((c_s / self.ds) * (ps_norm / self.chi_n - 1))

        # Clamp sigma to reasonable range
        self.sigma = max(0.01, min(self.sigma, 20.0))

        # Update evolution path for covariance
        # h_sig: stall indicator
        threshold = (1.4 + 2.0 / (self.n + 1)) * self.chi_n * math.sqrt(1 - (1 - c_s) ** (2 * (self.generation + 1)))
        h_sig = 1.0 if ps_norm < threshold else 0.0

        c_c = self.cc
        for i in range(self.n):
            self.pc[i] = (1 - c_c) * self.pc[i] + h_sig * math.sqrt(c_c * (2 - c_c) * self.mu_eff) * dm[i]

        # Update diagonal covariance
        for i in range(self.n):
            # Rank-1 update
            rank1 = self.c1 * self.pc[i] ** 2

            # Rank-mu update
            rank_mu = 0.0
            for rank in range(self.mu):
                idx = indices[rank]
                z_i = self._z_samples[idx][i]
                rank_mu += self.weights[rank] * z_i ** 2

            self.diag[i] = ((1 - self.c1 - self.cmu) * self.diag[i]
                            + rank1
                            + self.cmu * rank_mu)

            # Clamp to prevent degenerate variances
            self.diag[i] = max(0.001, min(self.diag[i], 100.0))

        # Update best
        best_idx = indices[0]
        if fitnesses[best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[best_idx]
            self.best_values = dict(self._population[best_idx])

        self.generation += 1

    def current_mean_dict(self):
        """Return current mean as tile -> value dict."""
        return dict(zip(TILES, self.mean))

    def save(self, path):
        """Save CMA-ES state to pickle."""
        state = {
            'mean': self.mean,
            'sigma': self.sigma,
            'diag': self.diag,
            'ps': self.ps,
            'pc': self.pc,
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'best_values': self.best_values,
            'lam': self.lam,
        }
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(state, f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        """Load CMA-ES state from checkpoint."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.n = N_PARAMS
        obj.lam = state['lam']
        obj.mu = obj.lam // 2
        obj.mean = state['mean']
        obj.sigma = state['sigma']
        obj.diag = state['diag']
        obj.ps = state['ps']
        obj.pc = state['pc']
        obj.generation = state['generation']
        obj.best_fitness = state['best_fitness']
        obj.best_values = state['best_values']
        obj._population = []
        obj._z_samples = []

        # Recompute constants from lam/mu
        raw_w = [math.log(obj.mu + 0.5) - math.log(i + 1) for i in range(obj.mu)]
        w_sum = sum(raw_w)
        obj.weights = [w / w_sum for w in raw_w]
        obj.mu_eff = 1.0 / sum(w * w for w in obj.weights)
        obj.cs = (obj.mu_eff + 2) / (obj.n + obj.mu_eff + 5)
        obj.ds = 1.0 + 2.0 * max(0, math.sqrt((obj.mu_eff - 1) / (obj.n + 1)) - 1) + obj.cs
        obj.cc = (4 + obj.mu_eff / obj.n) / (obj.n + 4 + 2 * obj.mu_eff / obj.n)
        obj.c1 = 2 / ((obj.n + 1.3) ** 2 + obj.mu_eff)
        obj.cmu = min(1 - obj.c1,
                      2 * (obj.mu_eff - 2 + 1 / obj.mu_eff) /
                      ((obj.n + 2) ** 2 + obj.mu_eff))
        obj.chi_n = math.sqrt(obj.n) * (1 - 1 / (4 * obj.n) + 1 / (21 * obj.n ** 2))
        return obj


# ---------------------------------------------------------------------------
# Game playing (adapted from validate.py)
# ---------------------------------------------------------------------------

def _play_game(gaddag, finder_cls, table_a, table_b):
    """Play one game: table_a is P1, table_b is P2.

    Returns (score_a, score_b).
    """
    board = Board()
    bag = []
    for letter, count in TILE_DISTRIBUTION.items():
        bag.extend([letter] * count)
    random.shuffle(bag)

    rack1 = [bag.pop() for _ in range(RACK_SIZE)]
    rack2 = [bag.pop() for _ in range(RACK_SIZE)]
    score1, score2 = 0, 0

    turn = 1
    consecutive_passes = 0
    max_turns = 100
    final_turns = None

    try:
        finder = finder_cls(board, gaddag, skip_postvalidation=True)
    except TypeError:
        finder = finder_cls(board, gaddag)

    while consecutive_passes < 4 and turn <= max_turns:
        if final_turns is not None and final_turns <= 0:
            break

        is_p1 = (turn % 2 == 1)
        current_rack = rack1 if is_p1 else rack2
        current_table = table_a if is_p1 else table_b
        bag_size = len(bag)

        rack_str = ''.join(current_rack)
        moves = finder.find_all_moves(rack_str)

        if not moves:
            consecutive_passes += 1
            if final_turns is not None:
                final_turns -= 1
            turn += 1
            continue

        best_move, best_leave, _ = select_best_move(
            board, moves, current_rack, current_table, bag_size
        )

        if best_move is None:
            consecutive_passes += 1
            if final_turns is not None:
                final_turns -= 1
            turn += 1
            continue

        board.place_word(best_move['word'], best_move['row'], best_move['col'],
                         best_move['direction'] == 'H')

        if is_p1:
            score1 += best_move['score']
        else:
            score2 += best_move['score']

        current_rack.clear()
        current_rack.extend(best_leave)

        draw_count = min(RACK_SIZE - len(current_rack), len(bag))
        for _ in range(draw_count):
            current_rack.append(bag.pop())

        if len(bag) == 0 and final_turns is None:
            final_turns = 2
        if final_turns is not None:
            final_turns -= 1

        consecutive_passes = 0
        turn += 1

    return score1, score2


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

_worker_gaddag = None
_worker_finder_cls = None
_worker_opp_adapter = None


def _init_worker(opp_values_json):
    """Load GADDAG + move finder + fixed opponent adapter. Once per worker."""
    global _worker_gaddag, _worker_finder_cls, _worker_opp_adapter
    pid = os.getpid()

    try:
        t0 = time.time()
        print(f"  [Worker {pid}] Loading GADDAG...", flush=True)
        from ..gaddag import get_gaddag
        _worker_gaddag = get_gaddag()
        print(f"  [Worker {pid}] GADDAG loaded ({time.time()-t0:.1f}s)", flush=True)

        # Set up move finder
        from ..move_finder_c import is_available as c_available
        if c_available():
            from ..move_finder_c import CMoveFinder
            _worker_finder_cls = CMoveFinder
            print(f"  [Worker {pid}] Move finder: C-accelerated", flush=True)
        else:
            from ..move_finder_gaddag import GADDAGMoveFinder
            _worker_finder_cls = GADDAGMoveFinder
            print(f"  [Worker {pid}] Move finder: Python", flush=True)

        # Fixed opponent
        opp_values = json.loads(opp_values_json)
        _worker_opp_adapter = TileValueAdapter(opp_values)
        print(f"  [Worker {pid}] Ready! ({time.time()-t0:.1f}s)", flush=True)

    except Exception as e:
        print(f"  [Worker {pid}] INIT FAILED: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


def _eval_batch(candidate_json, game_nums):
    """Play games with candidate values vs fixed opponent.

    Args:
        candidate_json: JSON string of tile -> value dict
        game_nums: list of game numbers (used for alternating first player)

    Returns:
        (total_spread, n_games)
    """
    candidate_values = json.loads(candidate_json)
    candidate_adapter = TileValueAdapter(candidate_values)

    total_spread = 0
    n_games = 0

    for game_num in game_nums:
        if game_num % 2 == 0:
            # Candidate is P1
            s1, s2 = _play_game(_worker_gaddag, _worker_finder_cls,
                                candidate_adapter, _worker_opp_adapter)
            total_spread += s1 - s2
        else:
            # Candidate is P2
            s1, s2 = _play_game(_worker_gaddag, _worker_finder_cls,
                                _worker_opp_adapter, candidate_adapter)
            total_spread += s2 - s1
        n_games += 1

    return total_spread, n_games


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(generations, games_per_eval, population_size, sigma, workers,
          resume_path=None):
    """Run CMA-ES optimization.

    Args:
        generations: number of CMA-ES generations
        games_per_eval: games per candidate evaluation
        population_size: CMA-ES lambda
        sigma: initial step size
        workers: number of parallel workers
        resume_path: optional path to resume from checkpoint
    """
    superleaves_dir = os.path.dirname(os.path.abspath(__file__))

    # Initialize CMA-ES
    if resume_path:
        print(f"Resuming from {resume_path}")
        cma = SepCMAES.load(resume_path)
        start_gen = cma.generation
        print(f"  Generation: {start_gen}, sigma: {cma.sigma:.4f}")
        print(f"  Best fitness: {cma.best_fitness:+.2f}")
    else:
        print("Starting from Quackle values")
        cma = SepCMAES(QUACKLE_INIT, sigma, population_size)
        start_gen = 0

    end_gen = start_gen + generations

    # Opponent: Quackle values (fixed throughout training)
    opp_json = json.dumps(QUACKLE_INIT)

    print(f"\nCMA-ES Tile Value Optimizer")
    print(f"  Parameters: {N_PARAMS}")
    print(f"  Population: {cma.lam} (mu={cma.mu})")
    print(f"  Sigma: {cma.sigma:.3f}")
    print(f"  Games/eval: {games_per_eval}")
    print(f"  Generations: {start_gen} -> {end_gen}")
    print(f"  Workers: {workers}")
    print(f"  Games/gen: {cma.lam * games_per_eval:,}")
    est_time = cma.lam * games_per_eval * generations / 12.0  # ~12 g/s estimate
    print(f"  Est. time: {est_time/3600:.1f} hours ({est_time:.0f}s)")
    print(f"  Opponent: Quackle values (fixed)")
    print("=" * 60)

    # Start worker pool
    print(f"\nStarting {workers} workers (30-90s init)...", flush=True)
    t_start = time.time()

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(opp_json,)
    ) as pool:

        for gen in range(start_gen, end_gen):
            t_gen = time.time()

            # Sample population
            population = cma.ask()
            fitnesses = [0.0] * cma.lam

            total_games = 0

            # Evaluate each candidate
            for cand_idx, candidate in enumerate(population):
                cand_json = json.dumps(candidate)

                # Distribute games across workers
                game_nums = list(range(games_per_eval))
                batch_size = max(1, len(game_nums) // workers)
                batches = []
                for i in range(0, len(game_nums), batch_size):
                    batches.append(game_nums[i:i + batch_size])

                # Fan out to workers
                futures = [pool.submit(_eval_batch, cand_json, batch)
                           for batch in batches]

                # Collect results
                cand_spread = 0
                cand_games = 0
                for fut in as_completed(futures):
                    try:
                        spread, n = fut.result(timeout=300)
                        cand_spread += spread
                        cand_games += n
                    except Exception as e:
                        print(f"  [ERROR] Batch failed: {e}", flush=True)

                fitnesses[cand_idx] = cand_spread / max(1, cand_games)
                total_games += cand_games

            # Update CMA-ES
            cma.tell(fitnesses)

            # Progress report
            elapsed_gen = time.time() - t_gen
            elapsed_total = time.time() - t_start
            gps = total_games / elapsed_gen if elapsed_gen > 0 else 0
            mean_dict = cma.current_mean_dict()

            # Fitness stats
            fit_sorted = sorted(fitnesses, reverse=True)
            fit_best = fit_sorted[0]
            fit_median = fit_sorted[len(fit_sorted) // 2]
            fit_worst = fit_sorted[-1]

            remaining_gens = end_gen - gen - 1
            eta = remaining_gens * elapsed_gen if elapsed_gen > 0 else 0

            print(
                f"  Gen {gen+1:3d}/{end_gen} | "
                f"best={fit_best:+6.1f} med={fit_median:+6.1f} worst={fit_worst:+6.1f} | "
                f"sigma={cma.sigma:.3f} | "
                f"{gps:.1f} g/s | "
                f"{elapsed_gen:.0f}s | "
                f"ETA {eta/60:.0f}m",
                flush=True
            )

            # Print top 5 / bottom 5 tile values every 5 generations
            if (gen + 1) % 5 == 0 or gen == start_gen:
                sorted_tiles = sorted(mean_dict.items(), key=lambda x: -x[1])
                top5 = ' '.join(f"{t}={v:+.1f}" for t, v in sorted_tiles[:5])
                bot5 = ' '.join(f"{t}={v:+.1f}" for t, v in sorted_tiles[-5:])
                print(f"         Top: {top5}", flush=True)
                print(f"         Bot: {bot5}", flush=True)
                print(f"         All-time best: {cma.best_fitness:+.2f}", flush=True)

            # Checkpoint every 10 generations
            if (gen + 1) % 10 == 0:
                ckpt_path = os.path.join(superleaves_dir,
                                         f"tile_cma_gen{gen+1}.pkl")
                cma.save(ckpt_path)
                print(f"  -> Checkpoint: {ckpt_path}", flush=True)

        # Final save
        final_path = os.path.join(superleaves_dir,
                                  f"tile_cma_gen{end_gen}.pkl")
        cma.save(final_path)

        # Print final results
        print(f"\n{'=' * 60}")
        print(f"TRAINING COMPLETE ({end_gen} generations, {time.time()-t_start:.0f}s)")
        print(f"  Best fitness (avg spread): {cma.best_fitness:+.2f}")
        print(f"  Final sigma: {cma.sigma:.4f}")
        print(f"\n  Best per-tile values:")
        sorted_best = sorted(cma.best_values.items(), key=lambda x: -x[1])
        for tile, val in sorted_best:
            q = QUACKLE_INIT.get(tile, 0.0)
            delta = val - q
            print(f"    {tile}: {val:+7.2f}  (quackle: {q:+6.2f}, delta: {delta:+6.2f})")
        print(f"\n  Saved to: {final_path}")

        # Also save best values as simple JSON for easy use
        json_path = os.path.join(superleaves_dir, "tile_values_best.json")
        with open(json_path, 'w') as f:
            json.dump(cma.best_values, f, indent=2, sort_keys=True)
        print(f"  Values JSON: {json_path}")


class FormulaAdapter:
    """Wraps the hand-tuned formula leave evaluator behind LeaveTable.get()."""

    def get(self, leave_key, default=0.0):
        if not leave_key:
            return 0.0
        from ..leave_eval import _formula_evaluate
        # leave_key is a sorted tuple of chars, formula expects a string
        leave_str = ''.join(leave_key) if isinstance(leave_key, tuple) else leave_key
        return _formula_evaluate(leave_str)


def validate(games, workers, values_path=None):
    """Play CMA-ES best values vs Quackle and Formula opponents.

    Args:
        games: number of games per matchup
        workers: number of parallel workers
        values_path: path to tile_values_best.json (auto-detected if None)
    """
    superleaves_dir = os.path.dirname(os.path.abspath(__file__))

    # Load CMA-ES best values
    if values_path is None:
        values_path = os.path.join(superleaves_dir, 'tile_values_best.json')
    if not os.path.exists(values_path):
        print(f"ERROR: {values_path} not found. Run training first.")
        return

    with open(values_path) as f:
        cmaes_values = json.load(f)

    cmaes_json = json.dumps(cmaes_values)
    quackle_json = json.dumps(QUACKLE_INIT)

    # Also load mean values if available
    mean_path = os.path.join(superleaves_dir, 'tile_values_mean.json')
    has_mean = os.path.exists(mean_path)
    if has_mean:
        with open(mean_path) as f:
            mean_values = json.load(f)
        mean_json = json.dumps(mean_values)

    matchups = [
        ('CMA-ES best vs Quackle', cmaes_json, quackle_json),
        ('CMA-ES best vs Formula', cmaes_json, '__formula__'),
    ]
    if has_mean:
        matchups.extend([
            ('CMA-ES mean vs Quackle', mean_json, quackle_json),
            ('CMA-ES mean vs Formula', mean_json, '__formula__'),
        ])

    for name, p1_json, p2_json in matchups:
        print(f"\n{'='*60}")
        print(f"  {name}  ({games} games, {workers} workers)")
        print(f"{'='*60}")

        # For formula opponent, we use a special init that creates FormulaAdapter
        if p2_json == '__formula__':
            opp_json = '__formula__'
        else:
            opp_json = p2_json

        # Start workers
        print(f"Starting {workers} workers...", flush=True)
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker_validate,
            initargs=(opp_json,),
        ) as pool:
            game_nums = list(range(games))
            batch_size = max(1, len(game_nums) // workers)
            batches = []
            for i in range(0, len(game_nums), batch_size):
                batches.append(game_nums[i:i + batch_size])

            t0 = time.time()
            total_spread = 0
            total_games = 0
            wins = 0
            losses = 0
            ties = 0

            futures = {
                pool.submit(_eval_batch, p1_json, batch): len(batch)
                for batch in batches
            }

            for future in as_completed(futures):
                spread, n = future.result()
                total_spread += spread
                total_games += n
                elapsed = time.time() - t0
                gps = total_games / elapsed if elapsed > 0 else 0
                avg = total_spread / total_games if total_games > 0 else 0
                print(f"  {total_games}/{games} games | "
                      f"spread: {total_spread:+d} | "
                      f"avg: {avg:+.1f} | "
                      f"{gps:.1f} g/s",
                      flush=True)

        avg_spread = total_spread / total_games if total_games > 0 else 0
        elapsed = time.time() - t0
        print(f"\n  Result: {avg_spread:+.1f} avg spread over {total_games} games "
              f"({elapsed:.0f}s)")
        print(f"  {'P1 WINS' if avg_spread > 0 else 'P2 WINS' if avg_spread < 0 else 'DRAW'}")


def _init_worker_validate(opp_json):
    """Worker init for validation -- supports formula opponent."""
    global _worker_gaddag, _worker_finder_cls, _worker_opp_adapter
    pid = os.getpid()

    try:
        t0 = time.time()
        from ..gaddag import get_gaddag
        _worker_gaddag = get_gaddag()

        from ..move_finder_c import is_available as c_available
        if c_available():
            from ..move_finder_c import CMoveFinder
            _worker_finder_cls = CMoveFinder
        else:
            from ..move_finder_gaddag import GADDAGMoveFinder
            _worker_finder_cls = GADDAGMoveFinder

        if opp_json == '__formula__':
            _worker_opp_adapter = FormulaAdapter()
        else:
            opp_values = json.loads(opp_json)
            _worker_opp_adapter = TileValueAdapter(opp_values)

        print(f"  [Worker {pid}] Ready ({time.time()-t0:.1f}s)", flush=True)

    except Exception as e:
        print(f"  [Worker {pid}] INIT FAILED: {e}", flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(
        description='CMA-ES per-tile leave value optimizer'
    )
    sub = parser.add_subparsers(dest='command')

    # Train subcommand (default)
    train_p = sub.add_parser('train', help='Run CMA-ES training')
    train_p.add_argument('--games-per-eval', type=int, default=150)
    train_p.add_argument('--generations', type=int, default=60)
    train_p.add_argument('--population', type=int, default=20)
    train_p.add_argument('--sigma', type=float, default=2.0)
    train_p.add_argument('--workers', type=int, default=None)
    train_p.add_argument('--resume', type=str, default=None)

    # Validate subcommand
    val_p = sub.add_parser('validate', help='Validate CMA-ES values')
    val_p.add_argument('--games', type=int, default=500,
                       help='Games per matchup (default: 500)')
    val_p.add_argument('--workers', type=int, default=None)
    val_p.add_argument('--values', type=str, default=None,
                       help='Path to tile_values_best.json')

    args = parser.parse_args()
    workers = getattr(args, 'workers', None) or _default_workers()

    if args.command == 'validate':
        validate(games=args.games, workers=workers,
                 values_path=args.values)
    else:
        # Default: train (backwards compat with no subcommand)
        # Re-parse for train args
        if args.command is None:
            # No subcommand -- treat all args as train args
            parser2 = argparse.ArgumentParser()
            parser2.add_argument('--games-per-eval', type=int, default=150)
            parser2.add_argument('--generations', type=int, default=60)
            parser2.add_argument('--population', type=int, default=20)
            parser2.add_argument('--sigma', type=float, default=2.0)
            parser2.add_argument('--workers', type=int, default=None)
            parser2.add_argument('--resume', type=str, default=None)
            args = parser2.parse_args()
            workers = args.workers if args.workers else _default_workers()

        resume_path = args.resume
        if resume_path and not os.path.isabs(resume_path):
            candidate = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), resume_path)
            if os.path.exists(candidate):
                resume_path = candidate

        train(
            generations=args.generations,
            games_per_eval=args.games_per_eval,
            population_size=args.population,
            sigma=args.sigma,
            workers=workers,
            resume_path=resume_path,
        )


if __name__ == '__main__':
    main()
