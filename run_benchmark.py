"""
Crossplay Benchmark Script
Run before/after hardware changes to compare performance.
Saves results to .benchmark_baseline.json
"""
import time
import json
import os
import sys

# Run from parent dir for proper imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
os.chdir(PARENT_DIR)

from crossplay_v9.mc_calibrate import get_env_info, calibrate


def run_benchmark():
    info = get_env_info()
    print("=" * 60)
    print("CROSSPLAY BENCHMARK")
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print(f"CPU:      {info['cpu_model']}")
    print(f"Cores:    {info['total_cores']} total, {info['mc_workers']} MC workers")
    print(f"Platform: {info['platform']}")
    print(f"Python:   {info['python']} ({info['arch']})")
    print()

    # 1. Fresh calibration (multi-worker)
    print("--- MC Calibration (force fresh) ---")
    cal = calibrate(force=True, quiet=True)
    engine = cal.get("engine", "?")
    nw = cal["n_workers"]
    print(f"Engine:          {engine}")
    print(f"Sparse  (1w):    {cal['sps_sparse_1w']:.1f} sims/s")
    print(f"Dense   (1w):    {cal['sps_dense_1w']:.1f} sims/s")
    print(f"VDense  (1w):    {cal['sps_vdense_1w']:.1f} sims/s")
    print(f"Sparse  ({nw}w):   {cal['sps_sparse_nw']:.1f} sims/s")
    print(f"Dense   ({nw}w):   {cal['sps_dense_nw']:.1f} sims/s")
    print(f"VDense  ({nw}w):   {cal['sps_vdense_nw']:.1f} sims/s")
    print()

    # 2. Single-worker per-game benchmarks
    from crossplay_v9.game_manager import _create_saved_game_5, _create_saved_game_6
    from crossplay_v9.move_finder_opt import find_best_score_opt
    from crossplay_v9.gaddag import get_gaddag
    from crossplay_v9.dictionary import get_dictionary
    from crossplay_v9.config import (
        TILE_VALUES, BONUS_SQUARES, VALID_TWO_LETTER, BINGO_BONUS, RACK_SIZE,
    )

    gaddag = get_gaddag()
    gdata_bytes = bytes(gaddag._data)
    dictionary = get_dictionary()
    word_set = dictionary._words

    # Scoring data for Cython
    tv_list = [0] * 26
    for c, v in TILE_VALUES.items():
        if c != "?":
            tv_list[ord(c) - 65] = v
    bonus_data = [[(1, 1)] * 15 for _ in range(15)]
    for r in range(15):
        for c in range(15):
            bt = BONUS_SQUARES.get((r + 1, c + 1), "")
            if bt == "2L":
                bonus_data[r][c] = (2, 1)
            elif bt == "3L":
                bonus_data[r][c] = (3, 1)
            elif bt == "2W":
                bonus_data[r][c] = (1, 2)
            elif bt == "3W":
                bonus_data[r][c] = (1, 3)

    # Try importing Cython
    try:
        import gaddag_accel as _accel
        has_cython = hasattr(_accel, "prepare_board_context")
    except ImportError:
        has_cython = False

    game_results = {}

    for label, game_fn in [
        ("Game 5 (endgame)", _create_saved_game_5),
        ("Game 6 (early)", _create_saved_game_6),
    ]:
        game = game_fn()
        grid = game.board._grid
        bb_set = {(r - 1, c - 1) for r, c, _ in game.state.blank_positions}
        rack = game.state.your_rack

        # Python path
        cc = {}
        t0 = time.perf_counter()
        n_py = 0
        while time.perf_counter() - t0 < 3.0:
            find_best_score_opt(grid, gaddag._data, rack, bb_set, cross_cache=cc)
            n_py += 1
        py_elapsed = time.perf_counter() - t0
        py_sps = n_py / py_elapsed

        # Cython path
        cy_sps = 0.0
        speedup = 0.0
        if has_cython:
            ctx = _accel.prepare_board_context(
                grid, gdata_bytes, bb_set, word_set,
                VALID_TWO_LETTER, tv_list, bonus_data,
                BINGO_BONUS, RACK_SIZE,
            )
            t0 = time.perf_counter()
            n_cy = 0
            while time.perf_counter() - t0 < 3.0:
                _accel.find_best_score_c(ctx, rack)
                n_cy += 1
            cy_elapsed = time.perf_counter() - t0
            cy_sps = n_cy / cy_elapsed
            speedup = cy_sps / py_sps if py_sps > 0 else 0

        print(f"--- {label} (rack {rack}) ---")
        print(f"  Python:  {py_sps:.1f} sims/s  ({1000 / py_sps:.1f} ms/sim)")
        if has_cython:
            print(f"  Cython:  {cy_sps:.1f} sims/s  ({1000 / cy_sps:.1f} ms/sim)")
            print(f"  Speedup: {speedup:.1f}x")
        print()

        game_results[label] = {
            "rack": rack,
            "py_sps": round(py_sps, 1),
            "cy_sps": round(cy_sps, 1),
            "speedup": round(speedup, 1),
        }

    # 3. Summary
    nw_sparse = cal["sps_sparse_nw"]
    nw_dense = cal["sps_dense_nw"]
    nw_vdense = cal["sps_vdense_nw"]
    print("--- N=25 K=2000 estimated wall time ---")
    print(f"  Sparse:  {25 * 2000 / nw_sparse:.0f}s")
    print(f"  Dense:   {25 * 2000 / nw_dense:.0f}s")
    print(f"  VDense:  {25 * 2000 / nw_vdense:.0f}s")
    print()

    # Save
    benchmark = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": info,
        "calibration": {
            "engine": engine,
            "sps_sparse_1w": cal["sps_sparse_1w"],
            "sps_dense_1w": cal["sps_dense_1w"],
            "sps_vdense_1w": cal["sps_vdense_1w"],
            "sps_sparse_nw": cal["sps_sparse_nw"],
            "sps_dense_nw": cal["sps_dense_nw"],
            "sps_vdense_nw": cal["sps_vdense_nw"],
            "n_workers": nw,
        },
        "game_benchmarks": game_results,
    }

    save_path = os.path.join(SCRIPT_DIR, ".benchmark_baseline.json")
    # If baseline exists, load it and show comparison
    if os.path.exists(save_path):
        with open(save_path) as f:
            old = json.load(f)
        print("=" * 60)
        print(f"COMPARISON vs baseline from {old['timestamp']}")
        print("=" * 60)
        old_cal = old["calibration"]
        print(f"{'Metric':<22} {'Before':>10} {'Now':>10} {'Change':>10}")
        print("-" * 54)
        for key, label in [
            ("sps_sparse_1w", "Sparse (1w)"),
            ("sps_dense_1w", "Dense (1w)"),
            ("sps_vdense_1w", "VDense (1w)"),
            ("sps_sparse_nw", f"Sparse ({nw}w)"),
            ("sps_dense_nw", f"Dense ({nw}w)"),
            ("sps_vdense_nw", f"VDense ({nw}w)"),
        ]:
            old_v = old_cal.get(key, 0)
            new_v = cal[key]
            if old_v > 0:
                pct = (new_v - old_v) / old_v * 100
                print(f"{label:<22} {old_v:>10.1f} {new_v:>10.1f} {pct:>+9.1f}%")
            else:
                print(f"{label:<22} {'?':>10} {new_v:>10.1f}")

        for gname in game_results:
            if gname in old.get("game_benchmarks", {}):
                old_g = old["game_benchmarks"][gname]
                new_g = game_results[gname]
                print()
                print(f"  {gname}:")
                for path in ["py_sps", "cy_sps"]:
                    plabel = "Python" if path == "py_sps" else "Cython"
                    ov = old_g.get(path, 0)
                    nv = new_g.get(path, 0)
                    if ov > 0 and nv > 0:
                        pct = (nv - ov) / ov * 100
                        print(f"    {plabel:<10} {ov:>8.1f} -> {nv:>8.1f}  ({pct:>+.1f}%)")
        print()

    with open(save_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"Saved to: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
