# experiments/metric_loader.py
"""
Load and organize metric replay results for analysis.

Provides clean data structures for exploratory analysis and plotting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from metric_replay import load_replay_results


class MetricLoader:
    """
    Load and organize metric replay results.

    Usage:
        loader = MetricLoader(replay_dir="results/exp/replay")
        loader.load()
        loader.organize(horizon_idx=0)

        # Access data
        loader.results_raw          # Raw .npz data
        loader.data_by_traj_seed    # Organized by (engine, traj, seed)
    """

    def __init__(
        self,
        replay_dir: str | Path,
        verbose: bool = True,
    ):
        self.replay_dir = Path(replay_dir)
        self.verbose = verbose

        # Data containers (populated by methods)
        self.results_raw: Optional[Dict[str, Any]] = None
        self.data_by_traj_seed: Optional[Dict] = None

        # Metadata
        self.engines: Optional[List[str]] = None
        self.horizons: Optional[np.ndarray] = None
        self.T: Optional[int] = None
        self.selected_horizon_idx: Optional[int] = None
        self.all_traj_indices: Optional[List[int]] = None
        self.all_seed_indices: Optional[List[int]] = None

    def load(self) -> 'MetricLoader':
        """Load all .npz files from replay directory."""
        self.results_raw = {}

        for npz_path in sorted(self.replay_dir.glob("*.npz")):
            result = load_replay_results(npz_path)
            engine_name = result["engine_name"]
            self.results_raw[engine_name] = result

            if self.verbose:
                N = result['pll'].shape[0]
                T = result['pll'].shape[1]
                H_count = result['pll'].shape[2]
                n_delayed = result['delayed_flag'].sum()
                print(f"{engine_name:20s}: N={N:5d}, T={T:3d}, H={H_count}, "
                      f"delayed={n_delayed}/{N}")

        if not self.results_raw:
            raise ValueError(f"No .npz files found in {self.replay_dir}")

        # Extract metadata
        self.engines = list(self.results_raw.keys())
        sample = self.results_raw[self.engines[0]]
        self.horizons = sample['horizons']
        self.T = int(sample['T'])

        if self.verbose:
            print(f"\nLoaded {len(self.engines)} engines: {self.engines}")
            print(f"Horizons: {self.horizons}")
            print(f"T (trajectory length): {self.T}")

        return self

    def organize(self, horizon_idx: int = 0) -> 'MetricLoader':
        """
        Organize raw results by (engine, traj, seed) with timeseries extracted.

        Args:
            horizon_idx: Which horizon to extract (0 = first horizon)

        Returns:
            self.data_by_traj_seed[engine_name][traj_idx][seed_idx] -> {
                # Forecast metrics (at selected horizon)
                'pll': [T],
                'obs_mse': [T],
                'ba': [T],
                'cov50': [T],
                'cov90': [T],

                # Filtering metrics (no horizon dimension)
                'ba_filt': [T],
                'p_pos_filt': [T],
                'p_neg_filt': [T],
                'bimodality': [T],
                'brier': [T],
                'latent_mse': [T],
                'latent_bias': [T],
                'latent_variance': [T],
                'ess': [T],

                # DD metrics
                'dd_time_predicted': int,
                'dd_time_truth': int,
                'dd_error': int,
                'delayed': bool
            }
        """
        if self.results_raw is None:
            raise RuntimeError("Must call load() first")

        self.selected_horizon_idx = horizon_idx

        if self.verbose:
            print(f"\nOrganizing by (engine, traj, seed) at horizon_idx={horizon_idx} "
                  f"(H={self.horizons[horizon_idx]})")

        self.data_by_traj_seed = {}

        for engine_name in self.engines:
            result = self.results_raw[engine_name]
            self.data_by_traj_seed[engine_name] = {}

            N = len(result['traj_index'])

            for i in range(N):
                traj_idx = int(result['traj_index'][i])
                seed_idx = int(result['seed_index'][i])

                if traj_idx not in self.data_by_traj_seed[engine_name]:
                    self.data_by_traj_seed[engine_name][traj_idx] = {}

                # Store full timeseries for all metrics
                self.data_by_traj_seed[engine_name][traj_idx][seed_idx] = {
                    # Forecast metrics (with horizon dimension)
                    'pll': result['pll'][i, :, horizon_idx],
                    'obs_mse': result['mse_obs'][i, :, horizon_idx],
                    'ba': result['ba'][i, :, horizon_idx],
                    'cov50': result['coverage_50'][i, :, horizon_idx],
                    'cov90': result['coverage_90'][i, :, horizon_idx],

                    # Filtering metrics (no horizon dimension)
                    'ba_filt': result['ba_filt'][i, :],
                    'p_pos_filt': result['p_positive_filt'][i, :],
                    'p_neg_filt': result['p_negative_filt'][i, :],
                    'bimodality': result['bimodality'][i, :],
                    'brier': result['brier'][i, :],
                    'latent_mse': result['latent_mse'][i, :],
                    'latent_bias': result['latent_bias'][i, :],
                    'latent_variance': result['latent_variance'][i, :],
                    'ess': result['ess'][i, :],

                    # DD metrics
                    'dd_time_predicted': int(result['dd_time_predicted'][i]),
                    'dd_time_truth': int(result['dd_time_truth'][i]),
                    'dd_error': int(result['dd_error'][i]),
                    'delayed': bool(result['delayed_flag'][i]),
                }

        # Get unique indices
        all_traj_indices = set()
        all_seed_indices = set()
        for engine_data in self.data_by_traj_seed.values():
            all_traj_indices.update(engine_data.keys())
            for traj_data in engine_data.values():
                all_seed_indices.update(traj_data.keys())

        self.all_traj_indices = sorted(list(all_traj_indices))
        self.all_seed_indices = sorted(list(all_seed_indices))

        if self.verbose:
            print(f"Unique trajectories: {len(self.all_traj_indices)}")
            print(f"Inference seeds: {len(self.all_seed_indices)} → {self.all_seed_indices}")

        return self


if __name__ == "__main__":
    """Test/demo of MetricLoader."""
    from pathlib import Path

    # ADJUST THIS to your actual experiment path
    project_root = Path(__file__).parent.parent
    replay_dir = project_root / "results" / "test_run_3" / "replay"

    if not replay_dir.exists():
        print(f"Error: {replay_dir} does not exist")
        print("Please adjust the path in the __main__ block")
        exit(1)

    print(f"Loading from: {replay_dir}\n")

    # Load and organize data
    loader = MetricLoader(replay_dir=replay_dir, verbose=True)
    loader.load().organize(horizon_idx=0)

    # Inspect structure
    print(f"\n{'='*80}")
    print("DATA STRUCTURE")
    print(f"{'='*80}\n")

    first_engine = loader.engines[0]
    first_traj = loader.all_traj_indices[0]
    first_seed = loader.all_seed_indices[0]

    print(f"Example: {first_engine}[traj={first_traj}][seed={first_seed}]:")
    sample_data = loader.data_by_traj_seed[first_engine][first_traj][first_seed]
    for key, val in sample_data.items():
        if isinstance(val, np.ndarray):
            print(f"  {key:20s}: shape {val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key:20s}: {val}")

    print(f"\n{'='*80}")
    print("SUCCESS - MetricLoader is working correctly")
    print(f"{'='*80}\n")