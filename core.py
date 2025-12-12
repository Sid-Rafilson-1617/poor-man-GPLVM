import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform



"""
Data loading and preprocessing
"""

def _load_cluster_labels(kilosort_dir: str) -> pd.DataFrame:
    """
    Load a Phy/Kilosort cluster label file that includes:
      cluster_id, KSLabel (or 'group')
    Prefers 'cluster_group.tsv' or 'cluster_KSLabel.tsv' in your folder; falls back to 'cluster_group.tsv'.
    """
    candidates = ["cluster_group.tsv", "cluster_KSLabel.tsv", "cluster_group.tsv"]
    path = None
    for fn in candidates:
        p = os.path.join(kilosort_dir, fn)
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError(
            f"Could not find cluster label file in {kilosort_dir}. Tried: {', '.join(candidates)}"
        )

    df = pd.read_csv(path, sep="\t")
    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    id_col  = cols.get("cluster_id") or cols.get("id")
    lbl_col = cols.get("kslabel") or cols.get("group")

    if id_col is None or lbl_col is None:
        raise ValueError(f"Expected columns like cluster_id and KSLabel/group in {path}; found {list(df.columns)}")

    out = df[[id_col, lbl_col]].rename(columns={id_col: "cluster_id", lbl_col: "label"})
    out["cluster_id"] = pd.to_numeric(out["cluster_id"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["cluster_id"]).astype({"cluster_id": int})
    out["label"] = out["label"].astype(str).str.lower()
    return out

def compute_spike_rates(kilosort_dir: str, window_size: float = 1.0, step_size: float = 0.5, use_units: str = 'all', sigma: float = 2.5, zscore: bool = True, adj = None):
    
    """
    Compute smoothed spike rates using a sliding window approach from Kilosort output data.
    
    This function processes spike times and cluster assignments from Kilosort/Phy2, separates units by 
    brain region based on channel mapping, calculates firing rates within sliding time windows, and 
    applies Gaussian smoothing. Optionally, z-scoring can be applied to normalize firing rates.
    
    Parameters
    ----------
    kilosort_dir : str
        Path to the directory containing Kilosort output files.
    window_size : float, optional
        Size of the sliding window in seconds, default is 1.0.
    step_size : float, optional
        Step size for sliding window advancement in seconds, default is 0.5.
    use_units : str, optional
        Filter for unit types to include:
        - 'all': Include all units
        - 'good': Include only good units
        - 'mua': Include only multi-unit activity
        - 'good/mua': Include both good units and multi-unit activity
        - 'noise': Include only noise units
        Default is 'all'.
    sigma : float, optional
        Standard deviation for Gaussian smoothing kernel, default is 2.5.
    zscore : bool, optional
        Whether to z-score the firing rates, default is True.
    
    Returns
    -------
    spike_rate_matrix : ndarray
        Matrix of spike rates (shape: num_units × num_windows).
    time_bins : ndarray
        Array of starting times for each window (s).
    
    Notes
    -----
    - Firing rates are computed in Hz (spikes per second)
    
    Raises
    ------
    FileNotFoundError
        If any required Kilosort output files are missing.
    """    

    # Load spike times and cluster assignments
    spike_times_path = os.path.join(kilosort_dir, f"spike_times{adj}.npy")
    spike_clusters_path = os.path.join(kilosort_dir, "spike_clusters.npy")  # Cluster assignments from Phy2 manual curation


    # extracting the sampling rate from params.py
    params_path = os.path.join(kilosort_dir, "params.py")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.py not found in {kilosort_dir}")
    with open(params_path, 'r') as f:
        params_content = f.read()
    sampling_rate_line = [line for line in params_content.splitlines() if 'sample_rate' in line]
    if not sampling_rate_line:
        raise ValueError("sample_rate not found in params.py")
    sampling_rate = float(sampling_rate_line[0].split('=')[1].strip())  


    # Ensure all required files exist
    if not all(os.path.exists(p) for p in [spike_times_path, spike_clusters_path]):
        raise FileNotFoundError("Missing required Kilosort output files.")

    # Loading spikes
    if adj == '_sec_adj':
        spike_times = np.load(spike_times_path)  # Already in seconds
    else:
        spike_times = np.load(spike_times_path) / sampling_rate  # Convert to seconds
    spike_clusters = np.load(spike_clusters_path)

    # load cluster labels
    info = _load_cluster_labels(kilosort_dir)



   # Which labels count
    use_units = use_units.lower()
    if use_units == "all":
        keep_ids = info["cluster_id"].values
    elif use_units == "good":
        keep_ids = info.loc[info["label"].eq("good"), "cluster_id"].values
    elif use_units == "mua":
        keep_ids = info.loc[info["label"].eq("mua"), "cluster_id"].values
    elif use_units in ("good/mua", "good+mua", "goodmua"):
        keep_ids = info.loc[info["label"].isin(["good", "mua"]), "cluster_id"].values
    elif use_units == "noise":
        keep_ids = info.loc[info["label"].eq("noise"), "cluster_id"].values
    else:
        raise ValueError(f"Unknown use_units='{use_units}'")
    

    # Filter to spikes belonging to kept units only
    keep_mask = np.isin(spike_clusters, keep_ids)
    spike_times = spike_times[keep_mask]
    spike_clusters = spike_clusters[keep_mask] 


    # Return early if no spikes
    if spike_times.size == 0:
        # no spikes in kept units
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.float64), np.array([], dtype=int)


    # Get total duration of the recording
    recording_duration = float(spike_times.max())
    if recording_duration < window_size:
        # No full window fits; return empty with units list
        units = np.unique(spike_clusters)
        return np.zeros((len(units), 0), dtype=np.float64), np.zeros((0,), dtype=np.float64), units


    num_windows = 1 + int(np.floor((recording_duration - window_size) / step_size))
    time_bins = np.arange(num_windows, dtype=np.float64) * step_size  # start times


    # Bin index per spike (window start index)
    # For each spike at time t, it contributes to the window whose start satisfies:
    # t ∈ [start, start+window_size)  => start = floor((t - 0)/step_size) but must ensure t >= start and t < start+window_size.
    # We’ll first assign spikes to the *window start index* by floor division,
    # then later drop contributions where spike time >= start+window_size (rare at boundaries).
    start_idx = np.floor(spike_times / step_size).astype(np.int64)
    valid = (start_idx >= 0) & (start_idx < num_windows)
    start_idx = start_idx[valid]
    spike_times_v = spike_times[valid]
    spike_clusters_v = spike_clusters[valid]

    # Guard against spikes that land in a start bin whose window would end before the spike (float edge cases)
    win_end = (start_idx * step_size) + window_size
    valid2 = spike_times_v < win_end
    start_idx = start_idx[valid2]
    spike_clusters_v = spike_clusters_v[valid2]

    # Map kept units to row indices
    units = np.unique(spike_clusters_v)          # actual cluster IDs present post-filter
    unit_to_row = {u: i for i, u in enumerate(units)}
    rows = np.fromiter((unit_to_row[u] for u in spike_clusters_v), dtype=np.int64, count=spike_clusters_v.size)

    # Accumulate counts per (unit,row, window,col)
    # Build a sparse-like COO accumulation with np.add.at into a dense array
    spike_counts = np.zeros((units.size, num_windows), dtype=np.float64)
    np.add.at(spike_counts, (rows, start_idx), 1.0)

    # Convert counts to rates (Hz)
    spike_rate_matrix = spike_counts / float(window_size)

    # Optional smoothing (σ in bins)
    if sigma and sigma > 0:
        for r in range(spike_rate_matrix.shape[0]):
            spike_rate_matrix[r, :] = gaussian_filter1d(spike_rate_matrix[r, :], sigma=sigma, mode='nearest')

    # Optional z-score per unit
    if zscore:
        mean = spike_rate_matrix.mean(axis=1, keepdims=True)
        std = spike_rate_matrix.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        spike_rate_matrix = (spike_rate_matrix - mean) / std

    return spike_rate_matrix, time_bins, units



def compute_spike_counts(
    kilosort_dir: str,
    window_size: float = 1.0,
    step_size: float = 0.5,
    use_units: str = 'all',
    sigma: float = 2.5,
    zscore: bool = True,
    adj = None,
):
    """
    Compute spike counts using a sliding window approach from Kilosort output data.
    
    This function processes spike times and cluster assignments from Kilosort/Phy2, filters
    units by label, and computes spike counts within overlapping sliding windows. Optionally,
    Gaussian smoothing and z-scoring can be applied across time for each unit.

    Parameters
    ----------
    kilosort_dir : str
        Path to the directory containing Kilosort output files.
    window_size : float, optional
        Size of the sliding window in seconds, default is 1.0.
    step_size : float, optional
        Step size for sliding window advancement in seconds, default is 0.5.
    use_units : str, optional
        Filter for unit types to include:
        - 'all': Include all units
        - 'good': Include only good units
        - 'mua': Include only multi-unit activity
        - 'good/mua': Include both good units and multi-unit activity
        - 'noise': Include only noise units
        Default is 'all'.
    sigma : float, optional
        Standard deviation (in window steps) for Gaussian smoothing kernel. If 0 or None,
        no smoothing is applied. Default is 2.5.
    zscore : bool, optional
        Whether to z-score the spike counts over time for each unit, default is True.
    adj : str or None, optional
        Suffix for spike_times file (e.g., '_sec_adj'), consistent with your existing code.

    Returns
    -------
    spike_count_matrix : ndarray
        Matrix of spike counts (shape: num_units × num_windows).
    time_bins : ndarray
        Array of starting times for each window (seconds).
    units : ndarray
        Array of unit (cluster) IDs corresponding to rows of `spike_count_matrix`.

    Notes
    -----
    - Counts are raw spike counts per window before optional smoothing/z-scoring.
      To get rates later, you can divide by `window_size`.
    """

    # Paths to Kilosort outputs
    spike_times_path = os.path.join(kilosort_dir, f"spike_times{adj}.npy")
    spike_clusters_path = os.path.join(kilosort_dir, "spike_clusters.npy")

    # Extract sampling rate from params.py
    params_path = os.path.join(kilosort_dir, "params.py")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"params.py not found in {kilosort_dir}")
    with open(params_path, 'r') as f:
        params_content = f.read()
    sampling_rate_line = [line for line in params_content.splitlines()
                          if 'sample_rate' in line]
    if not sampling_rate_line:
        raise ValueError("sample_rate not found in params.py")
    sampling_rate = float(sampling_rate_line[0].split('=')[1].strip())

    # Ensure required files exist
    if not all(os.path.exists(p) for p in [spike_times_path, spike_clusters_path]):
        raise FileNotFoundError("Missing required Kilosort output files.")

    # Load spikes
    if adj == '_sec_adj':
        spike_times = np.load(spike_times_path)  # already in seconds
    else:
        spike_times = np.load(spike_times_path) / sampling_rate  # convert to seconds
    spike_clusters = np.load(spike_clusters_path)

    # Load cluster labels (your helper)
    info = _load_cluster_labels(kilosort_dir)

    # Decide which units to keep based on labels
    use_units = use_units.lower()
    if use_units == "all":
        keep_ids = info["cluster_id"].values
    elif use_units == "good":
        keep_ids = info.loc[info["label"].eq("good"), "cluster_id"].values
    elif use_units == "mua":
        keep_ids = info.loc[info["label"].eq("mua"), "cluster_id"].values
    elif use_units in ("good/mua", "good+mua", "goodmua"):
        keep_ids = info.loc[info["label"].isin(["good", "mua"]), "cluster_id"].values
    elif use_units == "noise":
        keep_ids = info.loc[info["label"].eq("noise"), "cluster_id"].values
    else:
        raise ValueError(f"Unknown use_units='{use_units}'")

    # Filter spikes to kept units
    keep_mask = np.isin(spike_clusters, keep_ids)
    spike_times = spike_times[keep_mask]
    spike_clusters = spike_clusters[keep_mask]

    # Return early if no spikes
    if spike_times.size == 0:
        return (
            np.zeros((0, 0), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.array([], dtype=int),
        )

    # Total duration of recording
    recording_duration = float(spike_times.max())
    if recording_duration < window_size:
        # No full window fits; return empty with units list
        units = np.unique(spike_clusters)
        return (
            np.zeros((len(units), 0), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            units,
        )

    # Number of windows and their start times
    num_windows = 1 + int(np.floor((recording_duration - window_size) / step_size))
    time_bins = np.arange(num_windows, dtype=np.float64) * step_size  # window starts

    # Assign each spike to a window start index
    start_idx = np.floor(spike_times / step_size).astype(np.int64)
    valid = (start_idx >= 0) & (start_idx < num_windows)
    start_idx = start_idx[valid]
    spike_times_v = spike_times[valid]
    spike_clusters_v = spike_clusters[valid]

    # Guard against spikes that land in a start bin whose window would end before the spike
    win_end = (start_idx * step_size) + window_size
    valid2 = spike_times_v < win_end
    start_idx = start_idx[valid2]
    spike_clusters_v = spike_clusters_v[valid2]

    # Map units to row indices
    units = np.unique(spike_clusters_v)  # actual units present post-filter
    unit_to_row = {u: i for i, u in enumerate(units)}
    rows = np.fromiter(
        (unit_to_row[u] for u in spike_clusters_v),
        dtype=np.int64,
        count=spike_clusters_v.size,
    )

    # Accumulate counts: (unit, window) -> spike count
    spike_count_matrix = np.zeros((units.size, num_windows), dtype=np.float64)
    np.add.at(spike_count_matrix, (rows, start_idx), 1.0)

    # Optional smoothing (on counts)
    if sigma and sigma > 0:
        for r in range(spike_count_matrix.shape[0]):
            spike_count_matrix[r, :] = gaussian_filter1d(
                spike_count_matrix[r, :],
                sigma=sigma,
                mode='nearest',
            )

    # Optional z-score per unit
    if zscore:
        mean = spike_count_matrix.mean(axis=1, keepdims=True)
        std = spike_count_matrix.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        spike_count_matrix = (spike_count_matrix - mean) / std

    return spike_count_matrix, time_bins, units



def align_brain_and_behavior(events: pd.DataFrame, spike_rates: np.ndarray, units: np.ndarray, time_bins: np.ndarray, window_size: float = 0.1, speed_threshold: float = 4.0, interp_method='linear', order=None):
    
    """
    Align neural spike rate data with behavioral tracking data using time windows.
    
    This function matches neural activity from spike rates with behavioral metrics (position, velocity, speed)
    by finding the closest behavioral event to the middle of each time bin. It creates a unified dataframe
    containing both neural and behavioral data, removes outliers based on speed threshold, and interpolates
    missing values.
    
    Parameters
    ----------
    events : pd.DataFrame
        Behavioral tracking data containing columns:
        - 'timestamp_ms': Timestamps in milliseconds
        - 'position_x', 'position_y': Position coordinates
        - 'velocity_x', 'velocity_y': Velocity components
        - 'speed': Overall movement speed
        - 'reward_state': Reward state indicator
    
    spike_rates : np.ndarray
        Matrix of spike rates with shape (n_units, n_time_bins).
    
    units : np.ndarray
        Array of unit IDs corresponding to rows in spike_rates.
    
    time_bins : np.ndarray
        Array of starting times for each time bin in seconds.
    
    window_size : float, optional
        Size of each time window in seconds, default is 0.1.
    
    speed_threshold : float, optional
        Threshold for removing speed outliers, expressed as a multiplier of the
        standard deviation. Default is 4.0 (values > 4 × std are treated as
        outliers).
    interp_method : str, optional
        Interpolation method for filling missing values. Passed to
        ``DataFrame.interpolate``. Default is ``'linear'``.
    order : int, optional
        Polynomial order to use when ``interp_method`` is ``'polynomial'``.
    
    Returns
    -------
    pd.DataFrame
        Combined dataframe with aligned neural and behavioral data containing:
        - Unit columns: Spike rates for each neural unit
        - 'x', 'y': Position coordinates
        - 'v_x', 'v_y': Velocity components
        - 'speed': Movement speed
        - 'reward_state': Reward indicator
        - 'time': Time bin start times
        
    Notes
    -----
    - For each time bin, the behavioral event closest to the middle of the bin is selected
    - Speed outliers are removed using a threshold based on standard deviation
    - Missing values are interpolated using linear interpolation
    - Rows with missing behavioral data (typically at beginning/end of recording) are removed
    """

    # Initialize arrays for holding aligned data
    mean_positions_x = np.full(len(time_bins), np.nan)
    mean_positions_y = np.full(len(time_bins), np.nan)
    mean_velocities_x = np.full(len(time_bins), np.nan)
    mean_velocities_y = np.full(len(time_bins), np.nan)
    mean_speeds = np.full(len(time_bins), np.nan)
    mean_rewards = np.full(len(time_bins), np.nan)

    # getting event times in seconds
    event_times = events['timestamp_ms'].values / 1000

    # Calculate mean behavior in each time bin
    for i, t_start in enumerate(time_bins):
        t_end = t_start + window_size
        middle = t_start + window_size / 2

        if np.any(event_times < middle):
            nearest_event_index = np.argmin(np.abs(event_times - middle))
            mean_positions_x[i] = events['position_x'].iloc[nearest_event_index]
            mean_positions_y[i] = events['position_y'].iloc[nearest_event_index]
            mean_velocities_x[i] = events['velocity_x'].iloc[nearest_event_index]
            mean_velocities_y[i] = events['velocity_y'].iloc[nearest_event_index]
            mean_speeds[i] = events['speed'].iloc[nearest_event_index]
            mean_rewards[i] = events['reward_state'].iloc[nearest_event_index]
        else:
            mean_positions_x[i] = np.nan
            mean_positions_y[i] = np.nan
            mean_velocities_x[i] = np.nan
            mean_velocities_y[i] = np.nan
            mean_speeds[i] = np.nan
            mean_rewards[i] = np.nan


    # converting the spike rate matrix to a DataFrame
    data = pd.DataFrame(spike_rates.T, columns=[f"Unit {i}" for i in units])

    # adding the tracking data to the DataFrame
    conversion = 5.1
    data['x'] = mean_positions_x / conversion # convert to cm
    data['y'] = mean_positions_y / conversion # convert to cm
    data['v_x'] = mean_velocities_x / conversion # convert to cm/s
    data['v_y'] = mean_velocities_y / conversion # convert to cm/s
    data['speed'] = mean_speeds / conversion # convert to cm/s
    data['time'] = time_bins  # in seconds
    data['reward_state'] = mean_rewards

    # Remove speed outliers based on a standard deviation threshold
    speed_std = np.nanstd(data['speed'])
    if speed_std == 0:
        speed_std = 1  # avoid zero division / blanket removal
    data.loc[data['speed'] > speed_threshold * speed_std, ['x', 'y', 'v_x', 'v_y', 'speed']] = np.nan

    # interpolating the tracking data to fill in NaN values
    data.interpolate(method=interp_method, inplace=True, order = order)

    # Finding the trial number and getting the click time
    trial_ids = np.zeros(data.shape[0])
    click_event = np.zeros(data.shape[0])
    for i in range(1, len(data)):
        trial_ids[i] = trial_ids[i-1]
        if data['reward_state'].iloc[i-1] and not data['reward_state'].iloc[i]:
            trial_ids[i] += 1
            click_event[i] = 1
    data = data.assign(trial_id = trial_ids, click = click_event)

    return data

def get_presence_ratio(
    est_counts_per_bin: np.ndarray,
    time_bins: np.ndarray,
    n_coarse_bins: int
) -> np.ndarray:
    """
    Compute presence ratio per unit using coarse time bins.
    A unit is 'present' in a coarse bin if its estimated spike count sum in that bin > 0.
    Vectorized via a (n_time x n_coarse_bins) binning matrix.
    """
    if time_bins.ndim != 1:
        raise ValueError("time_bins must be 1D (monotonic increasing).")
    if est_counts_per_bin.shape[1] != time_bins.size:
        raise ValueError("est_counts_per_bin columns must match len(time_bins).")

    edges = np.linspace(time_bins[0], time_bins[-1], n_coarse_bins + 1)
    # Map each fine time bin into a coarse bin index [0, n_coarse_bins-1]
    bin_idx = np.digitize(time_bins, edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, n_coarse_bins - 1)

    # Build a (n_time, n_coarse_bins) one-hot binning matrix (uint8 to save memory)
    n_time = time_bins.size
    B = np.zeros((n_time, n_coarse_bins), dtype=np.uint8)
    B[np.arange(n_time), bin_idx] = 1

    # Sum counts within each coarse bin for every unit: (n_units x n_time) @ (n_time x n_bins)
    coarse_sums = est_counts_per_bin @ B  # shape: (n_units, n_coarse_bins)

    # Presence if sum>0 in a coarse bin; ratio across bins
    presence = (coarse_sums > 0).mean(axis=1)
    return presence

def load_behavior(behavior_file: str, tracking_file: str = None) -> pd.DataFrame:

    """Load and preprocess behavioral tracking data.

    Parameters
    ----------
    behavior_file : str
        Directory containing ``events.csv`` with task variables and timestamps.
    tracking_file : str, optional
        Path to an optional SLEAP ``*.analysis.h5`` tracking file. When provided,
        nose coordinates from this file replace those in ``events.csv``.

    Returns
    -------
    pandas.DataFrame
        Dataframe with columns:
        - ``position_x`` and ``position_y``: Zero-centered spatial coordinates
        - ``velocity_x`` and ``velocity_y``: Velocity components
        - ``reward_state``: Task variable indicating reward delivery
        - ``speed``: Movement speed in pixels per sample
        - ``timestamp_ms``: Timestamps in milliseconds

    Notes
    -----
    - Position coordinates are normalized by subtracting the mean to center around
      zero
    - Velocity is calculated using first-order differences (current - previous
      position)
    - The first velocity value uses the first position value as the "previous"
      position
    - Speed is calculated as the Euclidean distance between consecutive positions
    """

    # Load the behavior data
    events = pd.read_csv(os.path.join(behavior_file, 'events.csv'))

    if tracking_file:
        # Load the SLEAP tracking data from the HDF5 file
        f = h5py.File(tracking_file, 'r')
        nose = f['tracks'][:].T[:, 0, :]
        nose = nose[:np.shape(events)[0], :]
        mean_x, mean_y = np.nanmean(nose[:, 0]), np.nanmean(nose[:, 1])
        events['position_x'] = nose[:, 0] - mean_x
        events['position_y'] = nose[:, 1] - mean_y
        
    else:
        # zero-mean normalize the x and y coordinates
        mean_x, mean_y = np.nanmean(events['centroid_x']), np.nanmean(events['centroid_y'])
        events['position_x'] = events['centroid_x'] - mean_x
        events['position_y'] = events['centroid_y'] - mean_y

    # Estimating velocity and speed
    events['velocity_x'] = np.diff(events['position_x'], prepend=events['position_x'].iloc[0])
    events['velocity_y'] = np.diff(events['position_y'], prepend=events['position_y'].iloc[0])
    events['speed'] = np.sqrt(events['velocity_x']**2 + events['velocity_y']**2)



    # keeping only the columns we need
    events = events[['position_x', 'position_y', 'velocity_x', 'velocity_y', 'reward_state', 'speed', 'timestamp_ms']]
    return events

# ---- Helper to collect units for a given (region_name, cell_type_key) across probes ----
def collect_matrix_for(spike_rate_matrices, region_name, regions, cell_type_key, cell_types, probes, cell_type_groups):

    labels = set(cell_type_groups[cell_type_key])
    collected = []
    for p in probes:
        mat, _, _ = spike_rate_matrices[p]
        ct  = cell_types[p]
        rgn = regions[p]
        # Boolean mask: matching region + cell-type (after previous filtering)
        mask = (rgn == region_name) & np.isin(ct, list(labels))
        if np.any(mask):
            collected.append(mat[mask, :])
    if len(collected) == 0:
        return None
    return np.vstack(collected)


def collect_matrix_for_bilat(spike_rate_matrices, region_name, regions, cell_type_key, cell_types, probes, cell_type_groups, hemi, hemisphere_dict):

    labels = set(cell_type_groups[cell_type_key])
    collected = []
    for p in probes:
        mat, _, _ = spike_rate_matrices[p]
        ct  = cell_types[p]
        rgn = regions[p]
        hemisphere = hemisphere_dict[p]
        # Boolean mask: matching region + cell-type (after previous filtering)
        mask = (rgn == region_name) & np.isin(ct, list(labels)) & (hemisphere == hemi)
        if np.any(mask):
            collected.append(mat[mask, :])
    if len(collected) == 0:
        return None
    return np.vstack(collected)


# ---- Optional within-panel unit ordering: sort units by peak time or mean rate
def sort_units(matrix, mode='corr'):  # 'mean', 'peak', or 'corr'
    if matrix is None or matrix.shape[0] == 0:
        return matrix

    if mode == 'mean':
        order = np.argsort(matrix.mean(axis=1))
        return matrix[order]

    if mode == 'peak':
        order = np.argsort(np.argmax(matrix, axis=1))
        return matrix[order]

    if mode == 'corr':
        # If 1 or fewer units, nothing to sort
        if matrix.shape[0] <= 1:
            return matrix

        X = matrix.astype(float, copy=True)

        # Z-score each unit across time to normalize scale/offset
        X -= X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        nonzero = (std.squeeze() > 0)

        # Keep track of zero-variance (flat) units; put them at the end
        valid_idx   = np.where(nonzero)[0]
        invalid_idx = np.where(~nonzero)[0]

        if valid_idx.size <= 1:
            # Nothing meaningful to cluster; just append flat units
            order = np.r_[valid_idx, invalid_idx]
            return matrix[order]

        X[valid_idx] /= std[valid_idx]

        # Pairwise correlation among valid rows (units)
        C = np.corrcoef(X[valid_idx])
        # Replace NaNs (can happen if a row was all zeros after z-scoring)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)

        # Distance = 1 - correlation (symmetric, zero diagonal)
        D = 1.0 - C
        D = np.clip(D, 0.0, 2.0)  # numerical safety

        # Condensed distance for linkage
        dvec = squareform(D, checks=False)

        # Hierarchical clustering + optimal leaf ordering
        Z = linkage(dvec, method='average')  # 'average' works well with 1 - corr
        Z_opt = optimal_leaf_ordering(Z, dvec)
        leaf_order = leaves_list(Z_opt)

        ordered_valid = valid_idx[leaf_order]
        order = np.r_[ordered_valid, invalid_idx]  # flat units at bottom
        return matrix[order]

    # Fallback: no sorting
    return matrix

"""
Models and training utilities
"""

def cv_split(data, k, k_CV=10, n_blocks=10):
    '''
    Perform cross-validation split of the data, following the Hardcastle et 
    al paper.
    
    Parameters
    --
    data : An array of data.
    
    k : Which CV subset to hold out as testing data (integer from 0 to k_CV-1).
    
    k_CV : Number of CV splits (integer).
        
    n_blocks : Number of blocks for initially partitioning the data. The testing
        data will consist of a fraction 1/k_CV of the data from each of these
        blocks.
        
    Returns
    --
    data_train, data_test, switch_indices : 
        - Data arrays after performing the train/test split
        - Indices in the train and test data where new blocks begin
    '''

    block_size = len(data)//n_blocks
    mask_test = [False for _ in data]
    
    # Keep track of which indices in the original data are the start of test blocks
    test_block_starts = []
    
    for block in range(n_blocks):
        i_start = int((block + k/k_CV)*block_size)
        i_stop = int(i_start + block_size//k_CV)
        mask_test[i_start:i_stop] = [True for _ in range(block_size//k_CV)]
        test_block_starts.append(i_start)
        
    mask_train = [not a for a in mask_test]
    data_test = data[mask_test]
    data_train = data[mask_train]

    train_switch_indices = [0]
    test_switch_indices = [0]
    train_count = 0
    test_count = 0
    for i in range(len(data)-1):
        if mask_train[i]:
            train_count += 1
        if mask_test[i]:
            test_count += 1
        if not mask_train[i] and mask_train[i + 1]:
            train_switch_indices.append(train_count)
        if not mask_test[i] and mask_test[i + 1]:
            test_switch_indices.append(test_count)

    train_switch_indices = np.unique(train_switch_indices)
    test_switch_indices = np.unique(test_switch_indices)

    
    return data_train, data_test, train_switch_indices, test_switch_indices

class DecoderDataset:
    """
    parameters:
    -----------
    X: (N, T) continuous features, e.g., spike rates
    Y: (T,) discrete targets in [0, K-1], e.g., position bin indices
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        if X.ndim != 2:
            raise ValueError("X must have shape (N, T)")
        if Y.ndim != 1:
            raise ValueError("Y must have shape (T,)")
        if X.shape[1] != Y.shape[0]:
            raise ValueError("X.shape[1] must equal Y.shape[0]")
        self.X = X
        self.Y = Y

    def split(self, k: int, k_CV: int = 10, n_blocks: int = 10):
        """
        Splits the dataset into training and testing sets for k-fold cross-validation.

        Returns:
        (X_train, Y_train), (X_test, Y_test), train_switch_ind, test_switch_ind
        """
        X_train, X_test, train_switch_ind, test_switch_ind = cv_split(self.X.T, k, k_CV, n_blocks)
        Y_train, Y_test, _, _ = cv_split(self.Y, k, k_CV, n_blocks)
        return (X_train.T, Y_train.T), (X_test.T, Y_test.T), train_switch_ind, test_switch_ind


class GaussianBayesDecoder:
    """
    A Gaussian Naive Bayes decoder for discrete states based on continuous observations.
    """

    def __init__(self, n_bins: int, var_floor: float = 1e-4, uniform_prior: bool = False):
        self.n_bins = n_bins
        self.var_floor = var_floor
        self.uniform_prior = uniform_prior

        self.mu_ = None
        self.var_ = None
        self.log_prior_ = None

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Fit the Gaussian Naive Bayes model to the training data.
        X: (N, T) features
        Y: (T,) labels
        """
        if X.ndim != 2 or Y.ndim != 1 or X.shape[1] != Y.shape[0]:
            raise ValueError("X must be (N, T) and Y must be (T,) with matching timepoints.")

        N, T = X.shape
        K = self.n_bins

        self.mu_ = np.zeros((N, K))
        self.var_ = np.zeros((N, K))
        self.log_prior_ = np.zeros(K)

        # Compute mean and variance per bin
        for k in range(K):
            idx = (Y == k)
            if np.any(idx):
                X_k = X[:, idx]
                self.mu_[:, k] = X_k.mean(axis=1)
                v = X_k.var(axis=1)
                self.var_[:, k] = np.maximum(v, self.var_floor)
            else:
                self.var_[:, k] = self.var_floor

        # Compute priors
        if self.uniform_prior:
            self.log_prior_[:] = -np.log(K)
        else:
            counts = np.bincount(Y, minlength=K)
            probs = (counts + 1) / (counts.sum() + K)  # Laplace smoothing
            self.log_prior_ = np.log(probs)

        return self

    def predict_log_probabilities(self, X: np.ndarray):
        """
        Predict log-probabilities log p(y=k | x)
        Returns: (K, T)
        """
        if self.mu_ is None or self.var_ is None or self.log_prior_ is None:
            raise RuntimeError("Model must be fitted before calling predict_log_probabilities().")

        N, T = X.shape
        K = self.n_bins
        log_probs = np.zeros((K, T))
        two_pi = 2 * np.pi

        for k in range(K):
            mu_k = self.mu_[:, [k]]
            var_k = self.var_[:, [k]]
            const_term = -0.5 * np.sum(np.log(two_pi * var_k))
            quad_term = -0.5 * np.sum(((X - mu_k) ** 2) / var_k, axis=0)
            log_probs[k, :] = const_term + quad_term + self.log_prior_[k]

        # numerical stability
        m = log_probs.max(axis=0, keepdims=True)
        return log_probs - m

    def predict(self, X: np.ndarray):
        """Return MAP class indices (argmax over log-probabilities)."""
        return np.argmax(self.predict_log_probabilities(X), axis=0)
