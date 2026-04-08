# experiments/metric_loader.py
"""
Load and organize metric replay results for analysis.

Provides clean data structures for exploratory analysis and plotting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

from metric_replay import load_replay_results


def load_and_organize_results(
    replay_dir: str | Path,
    horizon_idx: int = 0,
    disamb_bins: List[int] | None = None,
    verbose: bool = True,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Load replay results and organize into clean data structures.

    Args:
        replay_dir: Path to replay directory containing .npz files
        horizon_idx: Which horizon to extract (0 = first horizon)
        disamb_bins: Bin edges for DD time stratification (e.g., [0, 40, 80, 120, 200])
        verbose: Print loading info

    Returns:
        (results_raw, data_by_traj_seed, aggregated_by_traj, stratified)

        results_raw:
            Dict[engine_name] -> raw loaded .npz data

        data_by_traj_seed:
            Dict[engine_name][traj_idx][seed_idx] -> {
                'pll': [T], 'mse': [T], 'ba': [T],
                'cov50': [T], 'cov90': [T], 'ess': [T],
                'p_pos': [T], 'p_neg': [T],
                'dd_time': int, 'delayed': bool
            }

        aggregated_by_traj:
            Dict[engine_name][traj_idx] -> {
                'pll': float, 'mse': float, 'ba': float,
                'cov50': float, 'cov90': float, 'ess': float,
                'dd_time': int, 'delayed': bool
            }
            (Metrics are time-averaged then seed-averaged)

        stratified:
            Dict[engine_name][bin_label] -> List[{
                'pll': float, 'mse': float, 'ba': float,
                'cov50': float, 'cov90': float, 'ess': float,
                'traj_idx': int, 'dd_time': int
            }]
    """
    replay_dir = Path(replay_dir)

    if disamb_bins is None:
        disamb_bins = [0, 20, 40, 60, 80, 100, 120]

    bin_labels = [f"DD_{disamb_bins[i]}_{disamb_bins[i+1]}"
                  for i in range(len(disamb_bins)-1)]

    # ========================================================================
    # Load raw results
    # ========================================================================

    results_raw = {}
    for npz_path in sorted(replay_dir.glob("*.npz")):
        result = load_replay_results(npz_path)
        engine_name = result["engine_name"]
        results_raw[engine_name] = result

        if verbose:
            N = result['pll'].shape[0]
            T = result['pll'].shape[1]
            H_count = result['pll'].shape[2]
            n_delayed = result['delayed_flag'].sum()
            print(f"{engine_name:20s}: N={N:5d}, T={T:3d}, H={H_count}, "
                  f"delayed={n_delayed}/{N}")

    engines = list(results_raw.keys())

    if not engines:
        raise ValueError(f"No .npz files found in {replay_dir}")

    if verbose:
        sample = results_raw[engines[0]]
        horizons = sample['horizons']
        T = sample['T']
        print(f"\nLoaded {len(engines)} engines: {engines}")
        print(f"Horizons: {horizons}")
        print(f"Selected horizon_idx={horizon_idx} → H={horizons[horizon_idx]}")

    # ========================================================================
    # Organize by (engine, traj, seed)
    # ========================================================================

    data_by_traj_seed = {}

    for engine_name in engines:
        result = results_raw[engine_name]
        data_by_traj_seed[engine_name] = {}

        N = len(result['traj_index'])

        for i in range(N):
            traj_idx = int(result['traj_index'][i])
            seed_idx = int(result['seed_index'][i])

            if traj_idx not in data_by_traj_seed[engine_name]:
                data_by_traj_seed[engine_name][traj_idx] = {}

            # Store full timeseries for all metrics at selected horizon
            data_by_traj_seed[engine_name][traj_idx][seed_idx] = {
                'pll': result['pll'][i, :, horizon_idx],
                'mse': result['mse'][i, :, horizon_idx],
                'ba': result['branch_acc'][i, :, horizon_idx],
                'cov50': result['coverage_50'][i, :, horizon_idx],
                'cov90': result['coverage_90'][i, :, horizon_idx],
                'ess': result['ess'][i, :, horizon_idx],
                'p_pos': result['p_positive'][i, :, horizon_idx],
                'p_neg': result['p_negative'][i, :, horizon_idx],
                'dd_time': int(result['disamb_time'][i]),
                'delayed': bool(result['delayed_flag'][i]),
            }

    # Get unique indices
    all_traj_indices = set()
    all_seed_indices = set()
    for engine_data in data_by_traj_seed.values():
        all_traj_indices.update(engine_data.keys())
        for traj_data in engine_data.values():
            all_seed_indices.update(traj_data.keys())

    all_traj_indices = sorted(list(all_traj_indices))
    all_seed_indices = sorted(list(all_seed_indices))

    if verbose:
        print(f"\nUnique trajectories: {len(all_traj_indices)}")
        print(f"Inference seeds: {len(all_seed_indices)} → {all_seed_indices}")

    # ========================================================================
    # Aggregate across seeds (per trajectory)
    # ========================================================================

    aggregated_by_traj = {}

    for engine_name in engines:
        aggregated_by_traj[engine_name] = {}

        for traj_idx in all_traj_indices:
            if traj_idx not in data_by_traj_seed[engine_name]:
                continue

            traj_data = data_by_traj_seed[engine_name][traj_idx]

            # Collect time-averaged metrics across seeds
            pll_vals = []
            mse_vals = []
            ba_vals = []
            cov50_vals = []
            cov90_vals = []
            ess_vals = []

            for seed_idx in all_seed_indices:
                if seed_idx not in traj_data:
                    continue

                pll_vals.append(np.nanmean(traj_data[seed_idx]['pll']))
                mse_vals.append(np.nanmean(traj_data[seed_idx]['mse']))
                ba_vals.append(np.nanmean(traj_data[seed_idx]['ba']))
                cov50_vals.append(np.nanmean(traj_data[seed_idx]['cov50']))
                cov90_vals.append(np.nanmean(traj_data[seed_idx]['cov90']))
                ess_vals.append(np.nanmean(traj_data[seed_idx]['ess']))

            if len(pll_vals) > 0:
                aggregated_by_traj[engine_name][traj_idx] = {
                    'pll': np.mean(pll_vals),
                    'mse': np.mean(mse_vals),
                    'ba': np.mean(ba_vals),
                    'cov50': np.mean(cov50_vals),
                    'cov90': np.mean(cov90_vals),
                    'ess': np.mean(ess_vals),
                    'dd_time': traj_data[list(traj_data.keys())[0]]['dd_time'],
                    'delayed': traj_data[list(traj_data.keys())[0]]['delayed'],
                }

    if verbose:
        print(f"Aggregated {len(aggregated_by_traj[engines[0]])} trajectories per engine")

    # ========================================================================
    # Stratify by DD time bins
    # ========================================================================

    stratified = {}

    for engine_name in engines:
        stratified[engine_name] = {label: [] for label in bin_labels}
        stratified[engine_name]['DD_never'] = []

        for traj_idx, data in aggregated_by_traj[engine_name].items():
            dd_time = data['dd_time']

            if dd_time == -1:
                bin_label = 'DD_never'
            else:
                assigned = False
                for i in range(len(disamb_bins) - 1):
                    if disamb_bins[i] <= dd_time < disamb_bins[i+1]:
                        bin_label = bin_labels[i]
                        assigned = True
                        break
                if not assigned:
                    bin_label = bin_labels[-1]

            stratified[engine_name][bin_label].append({
                'pll': data['pll'],
                'mse': data['mse'],
                'ba': data['ba'],
                'cov50': data['cov50'],
                'cov90': data['cov90'],
                'ess': data['ess'],
                'traj_idx': traj_idx,
                'dd_time': dd_time,
            })

    if verbose:
        print(f"\nStratification by DD time:")
        for engine_name in engines:
            print(f"\n{engine_name}:")
            for bin_label in bin_labels + ['DD_never']:
                n = len(stratified[engine_name][bin_label])
                print(f"  {bin_label:15s}: {n:4d} trajectories")

    return results_raw, data_by_traj_seed, aggregated_by_traj, stratified


if __name__ == "__main__":
    """Test/demo of metric_loader functionality."""
    from pathlib import Path

    # ADJUST THIS to your actual experiment path
    project_root = Path(__file__).parent.parent  # Go up to project root
    results_dir = project_root / "results" / "test_run_3"
    replay_dir = results_dir / "replay"

    if not replay_dir.exists():
        print(f"Error: {replay_dir} does not exist")
        print("Please adjust the path in the __main__ block")
        exit(1)

    print(f"Loading from: {replay_dir}\n")

    # Load data
    results_raw, data_by_traj_seed, aggregated_by_traj, stratified = load_and_organize_results(
        replay_dir=replay_dir,
        horizon_idx=0,
        disamb_bins=[0, 40, 80, 120, 200],
        verbose=True,
    )

    engines = list(results_raw.keys())

    # ========================================================================
    # Inspect data structures
    # ========================================================================

    print(f"\n{'='*80}")
    print("DATA STRUCTURE INSPECTION")
    print(f"{'='*80}\n")

    # 1. Raw results
    print("1. results_raw structure:")
    print(f"   Keys: {list(results_raw.keys())}")
    print(f"   Example (first engine):")
    first_engine = engines[0]
    print(f"     {first_engine}:")
    for key in ['pll', 'mse', 'branch_acc', 'traj_index', 'seed_index']:
        if key in results_raw[first_engine]:
            print(f"       {key:15s}: shape {results_raw[first_engine][key].shape}")

    # 2. Organized by (traj, seed)
    print(f"\n2. data_by_traj_seed structure:")
    print(f"   Engines: {len(data_by_traj_seed)}")
    first_traj = list(data_by_traj_seed[first_engine].keys())[0]
    first_seed = list(data_by_traj_seed[first_engine][first_traj].keys())[0]
    print(f"   Example: {first_engine}[traj={first_traj}][seed={first_seed}]:")
    sample_data = data_by_traj_seed[first_engine][first_traj][first_seed]
    for key, val in sample_data.items():
        if isinstance(val, np.ndarray):
            print(f"       {key:10s}: shape {val.shape}")
        else:
            print(f"       {key:10s}: {val}")

    # 3. Aggregated by trajectory
    print(f"\n3. aggregated_by_traj structure:")
    print(f"   Engines: {len(aggregated_by_traj)}")
    print(f"   Trajectories per engine: {len(aggregated_by_traj[first_engine])}")
    print(f"   Example: {first_engine}[traj={first_traj}]:")
    sample_agg = aggregated_by_traj[first_engine][first_traj]
    for key, val in sample_agg.items():
        print(f"       {key:10s}: {val}")

    # 4. Stratified by DD bins
    print(f"\n4. stratified structure:")
    print(f"   Engines: {len(stratified)}")
    bins = list(stratified[first_engine].keys())
    print(f"   Bins: {bins}")
    first_bin = [b for b in bins if b != 'DD_never'][0]
    print(f"   Example: {first_engine}[{first_bin}] has {len(stratified[first_engine][first_bin])} trajectories")
    if len(stratified[first_engine][first_bin]) > 0:
        sample_strat = stratified[first_engine][first_bin][0]
        print(f"   First trajectory in that bin:")
        for key, val in sample_strat.items():
            print(f"       {key:10s}: {val}")

    # ========================================================================
    # Quick stats
    # ========================================================================

    print(f"\n{'='*80}")
    print("QUICK PERFORMANCE SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Engine':<20} {'PLL':<20} {'MSE':<20} {'BA':<20}")
    print("-" * 80)

    for engine_name in engines:
        all_pll = [d['pll'] for d in aggregated_by_traj[engine_name].values()]
        all_mse = [d['mse'] for d in aggregated_by_traj[engine_name].values()]
        all_ba = [d['ba'] for d in aggregated_by_traj[engine_name].values()]

        pll_str = f"{np.mean(all_pll):.4f}±{np.std(all_pll):.4f}"
        mse_str = f"{np.mean(all_mse):.6f}±{np.std(all_mse):.6f}"
        ba_str = f"{np.mean(all_ba):.4f}±{np.std(all_ba):.4f}"

        print(f"{engine_name:<20} {pll_str:<20} {mse_str:<20} {ba_str:<20}")

    print(f"\n{'='*80}")
    print("SUCCESS - metric_loader.py is working correctly")
    print(f"{'='*80}\n")