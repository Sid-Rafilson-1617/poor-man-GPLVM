import numpy as np
import jax.numpy as jnp
import jax.random as jr
import poor_man_gplvm as pmg
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.io import savemat
import os
import glob
import mat73

from core import *

# ----------------------------- Config ---------------------------------

# Data paths and session info
DATA_DIR = r"z:\Homes\voerom01\HPC_transverse\HPC_transverse_M01\HPC_TR_M01_20250615"
FIG_DIR = r"z:\Homes\voerom01\HPC_transverse\HPC_transverse_M01\HPC_TR_M01_20250615\gplvmFigures"
SESSION_NAME = "HPC_TR_M01_20250615"
BASE_NAME = "HPC_TR_M01_20250615"
PROBES = [0, 1]


# Spike rate computation parameters
WINDOW_SIZE = 0.01         # s
STEP_SIZE   = 0.01       # s
USE_UNITS   = "good"       # {'all', 'good', 'mua'}


# Filtering thresholds
MIN_TOTAL_SPIKES   = 500
MIN_MEAN_RATE_HZ   = 0.01
MIN_PRESENCE_RATIO = 0.50
N_TIME_BINS        = 100    # coarse bins for presence ratio


# defining the region dictionary. Keys are probe numbers, values are dictionaries which are further mapping shanks to regions
#region_dict = {0: {0: 'CA1', 1: 'RSP', 2: 'CA3', 3: 'CA1', 4: 'RSP', 5: 'CA3', 6: 'CA1', 7: 'RSP'}, 1: {0: 'CA3', 1: 'CA1', 2: 'CA2', 3: 'A1', 4: 'S1'}, 2: {0: 'CA3', 1: 'Th', 2: 'V1', 3: 'Th', 4: 'CA1', 5: 'V1', 6: 'CA1', 7: 'V1'}} UDS
region_dict = {
    0: {0: 'CA1', 1: 'CA1', 2: 'CA1', 3: 'CA1'},
    1: {0: 'CA1', 1: 'CA1', 2: 'CA1', 3: 'CA1'},
    2: {0: 'CA1', 1: 'CA1', 2: 'CA1', 3: 'Th'}, 
    3: {0: 'CA1', 1: 'CA1', 2: 'CA1', 3: 'CA1'}
} #This is going to need to be replaced by manual unit labels....


# defining the hemisphere dictionary. Keys are probe numbers, values are 'L' or 'R' for left/right hemisphere
hemisphere_dict = {0: 'L', 1: 'L', 2: 'R', 3: 'R'}

# Create figure directory if it doesn't exist
os.makedirs(FIG_DIR, exist_ok=True)





# ---------------------------Config------------------------------------
use_mat73 = False  # Set to True if .mat files are v7.3 format


# ------------------------ Load spike rate matrices ------------------------
spike_count_matrices = {}
cell_types = {}
regions = {}
for probe in PROBES:

    if probe == 0:
        use_mat73 = False
    else:
        use_mat73 = False

        
    # Get the raw spike counts and the zscored and smoothed spike rates
    #ks_dir = os.path.join(DATA_DIR, f'Kilosort_imec{probe}_ks4')
    pattern = os.path.join(DATA_DIR, f"Kilosort*imec{probe}*")
    matches = glob.glob(pattern)
    ks_dir = matches[0]  # assuming there's only one match
    print(f"Loading spike counts from Kilosort directory: {ks_dir}")
    #spike_count_matrix, time_bins, units = compute_spike_counts(str(ks_dir), WINDOW_SIZE, STEP_SIZE, USE_UNITS, 0.0, False, adj = '_sec_adj')
    spike_count_matrix, time_bins, units = compute_spike_counts(str(ks_dir), WINDOW_SIZE, STEP_SIZE, USE_UNITS, 0.0, False, adj = '')


    # computing the filtering metrics
    total_spikes   = spike_count_matrix.sum(axis=1)         
    mean_rates   = spike_count_matrix.mean(axis=1) / WINDOW_SIZE
    presence_ratio = get_presence_ratio(spike_count_matrix, time_bins, N_TIME_BINS)

    mask = (
        (total_spikes >= MIN_TOTAL_SPIKES) &
        (mean_rates >= MIN_MEAN_RATE_HZ) &
        (presence_ratio >= MIN_PRESENCE_RATIO)
    )


    # Apply mask to spike rate matrix and unit list
    spike_count_matrix = spike_count_matrix[mask, :]
    units = [u for u, keep in zip(units, mask) if keep]
    print(f"Probe {probe}: Keeping {mask.sum()} / {mask.sum() + (~mask).sum()} units after filtering.\nspike_count_matrix shape: {spike_count_matrix.shape}\n")

    # remove units based on presence ratio
    spike_count_matrices[probe] = (spike_count_matrix, time_bins, units)


# Load the cell metrics
    cell_metrics_file = os.path.join(DATA_DIR, f'{BASE_NAME}_imec{probe}.cell_metrics.cellinfo.mat')
    print(f"Loading cell metrics from: {cell_metrics_file}")
    if use_mat73:
        cell_metrics_data = mat73.loadmat(cell_metrics_file)
        cm = cell_metrics_data['cell_metrics']
        cell_type = np.array(cm['putativeCellType'])
        shank_ids = np.array(cm['shankID'])
        CluIDs = np.array(cm['cluID'])
    else:
        cm = loadmat(cell_metrics_file)['cell_metrics']
        cell_type = cm['putativeCellType'][0, 0]
        shank_ids = cm['shankID'][0, 0]
        CluIDs = cm['cluID'][0, 0]

    # Ensure 1D arrays
    CluIDs = np.asarray(CluIDs).reshape(-1)
    shank_ids = np.asarray(shank_ids).reshape(-1)

    # flatten cell_type and turn into strings
    cell_type = np.array([
        ct[0] if isinstance(ct, (np.ndarray, list)) else ct
        for ct in np.asarray(cell_type).reshape(-1)
    ])

    # ------------------ ALIGN USING INTERSECTION ------------------
    units = np.asarray(units)  # ensure ndarray for indexing
    print(f"cell_metrics has {len(CluIDs)} entries; spike-count units: {len(units)}")

    # map cluID -> index into cell_metrics arrays
    id_to_idx = {int(cid): i for i, cid in enumerate(CluIDs)}

    keep_rows = []
    aligned_cell_type = []
    aligned_shank_ids = []
    aligned_units = []

    for row, uid in enumerate(units):
        idx = id_to_idx.get(int(uid), None)
        if idx is None:
            # this unit has no cell_metrics entry; drop it
            print(f"Warning: Unit ID {uid} not found in cell_metrics; dropping from analysis.")
            continue
        aligned_units.append(uid)
        aligned_cell_type.append(cell_type[idx])
        aligned_shank_ids.append(shank_ids[idx])
        keep_rows.append(row)

    keep_rows = np.asarray(keep_rows, dtype=int)

    # apply to spike_count_matrix and units
    spike_count_matrix = spike_count_matrix[keep_rows, :]
    units = np.asarray(aligned_units)
    cell_type = np.asarray(aligned_cell_type)
    shank_ids = np.asarray(aligned_shank_ids)

    print(f"After alignment: {len(units)} units remain with cell_metrics info.")

    # store per-probe results
    spike_count_matrices[probe] = (spike_count_matrix, time_bins, units)
    cell_types[probe] = cell_type
    regions[probe] = np.array([region_dict[probe][sid - 1] for sid in shank_ids])  # shank IDs are 1-based
    # --------------------------------------------------------------


    # ------------------------ Plotting spike raster for each hemisphere and the binarized positions------------------------
REGIONS = ['CA1']
UNIT_TYPES = ['Pyramidal Cell']

# preparing the lists to collect spike rates
selected_spike_counts_L = []
selected_spike_counts_R = []

# making a fake timebase ref since no position info is available
no_position = True
if no_position:
    start_time = 0.0
    # getting the end time from the shortest probe
    end_time = None
    for probe in PROBES:
        _, tb_probe, _ = spike_count_matrices[probe]
        if end_time is None:
            end_time = np.max(tb_probe)
        elif np.max(tb_probe) < end_time:
            end_time = np.max(tb_probe)
    tb_ref = np.arange(start_time, end_time, STEP_SIZE)

    # setting the mask to all timebins
    mask = np.ones_like(tb_ref, dtype=bool)

    # making constant position bins
    pos_bins = np.zeros(np.sum(mask), dtype=int)

# looping over probes to collect spike rates
for probe in PROBES:

    spike_count_matrix, probe_times, units = spike_count_matrices[probe]  # (n_units, n_timebins)
    hemisphere = hemisphere_dict[probe]


    # cropping the probe times to match the reference timebins
    print(f'Probe {probe} timebins length {len(probe_times)}: reference timebins length {len(tb_ref)}')
    start_time, end_time = np.min(tb_ref), np.max(tb_ref)
    valid_time_probe = (probe_times >= start_time) & (probe_times <= end_time)

    # check if all time values are now equal
    probe_times = probe_times[valid_time_probe]
    spike_count_matrix = spike_count_matrix[:, valid_time_probe]
        
    if np.allclose(probe_times, tb_ref, rtol=0, atol=1e-9):
        print(f'Probe {probe} timebins successfully matched to reference timebins after cropping.')
        print(f'sum of differences: {np.sum(probe_times - tb_ref)}\n')
    else:
        raise ValueError(f'Probe {probe} timebins do not match reference timebins after cropping. Lengths: {len(probe_times)} vs {len(tb_ref)}')


    # filtering out timebins outside the TTL range and speed threshold
    spike_counts = spike_count_matrix[:, mask]



    # getting cell types and regions for the current probe
    ct_probe = np.asarray(cell_types[probe])                    
    rgn_arr = regions[probe]                                   

    # select units from the selected regions
    unit_sel = np.where(
        (np.isin(ct_probe, UNIT_TYPES)) &
        (np.isin(rgn_arr, REGIONS))
    )[0]
    if unit_sel.size == 0:
        continue

    if hemisphere == 'L':
        selected_spike_counts_L.append(spike_counts[unit_sel, :])
    elif hemisphere == 'R':
        selected_spike_counts_R.append(spike_counts[unit_sel, :])
    else:
        raise ValueError(f'Unknown hemisphere {hemisphere} for probe {probe}')

# concatenating the selected spike rates across probes
if (len(selected_spike_counts_L) == 0 and len(selected_spike_counts_R) == 0):
    raise RuntimeError("No units found for the chosen regions after filtering.")

spike_counts_concat_L = np.vstack(selected_spike_counts_L) if len(selected_spike_counts_L) > 0 else np.empty((0, 0))
spike_counts_concat_R = np.vstack(selected_spike_counts_R) if len(selected_spike_counts_R) > 0 else np.empty((0, 0))

# sort the units based on correlation
sorted_spike_counts_concat_L = sort_units(spike_counts_concat_L, mode='corr')
sorted_spike_counts_concat_R = sort_units(spike_counts_concat_R, mode='corr')

# getting the time vector in seconds
times = tb_ref[mask] - tb_ref[0]

print(f'sorted_counts_concat_L shape: {sorted_spike_counts_concat_L.shape}')
print(f'sorted_counts_concat_R shape: {sorted_spike_counts_concat_R.shape}')
print(f'pos_bins shape: {pos_bins.shape}')
#print(f'pos_ref shape: {pos_ref.shape}')
print(f'times shape: {times.shape}')




# define the models for the left and right hemispheres
N_SPATIAL_BINS = 100
model=pmg.PoissonGPLVMJump1D(sorted_spike_counts_concat_L.shape[0], n_latent_bin=N_SPATIAL_BINS + 1, movement_variance=1, tuning_lengthscale=1)
em_res_l = model.fit_em(sorted_spike_counts_concat_L.T, key=jr.PRNGKey(3), n_iter=30, log_posterior_init=None, ma_neuron=None, ma_latent=None, n_time_per_chunk=10000)

#tuning_curves_r = model_r.tuning
tuning_curves_l = model.tuning
for neuron in range(10):
    plt.figure()
    #plt.plot(tuning_curves_r[:, neuron], label='Right Hemisphere')
    plt.plot(tuning_curves_l[:, neuron])
    plt.title(f'Tuning Curve for Unit {neuron}')
    plt.xlabel('latent Bin')
    plt.ylabel('Tuning Value')
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'tuning_curve_unit_{neuron}.png'))


decode_res = model.decode_latent(sorted_spike_counts_concat_L.T)

latent_posteriors_l = decode_res['posterior_latent_marg']
jump_prob_l = decode_res['posterior_dynamics_marg'][:,1]
continuous_prob_l = decode_res['posterior_dynamics_marg'][:,0]

print(f'latent_posteriors_l shape: {latent_posteriors_l.shape}')
print(f'jump_prob_l shape: {jump_prob_l.shape}')
print(f'continuous_prob_l shape: {continuous_prob_l.shape}')

# making a time vector in seconds
time_array = np.arange(latent_posteriors_l.shape[0]) * STEP_SIZE
print(f'time_array shape: {time_array.shape}')
print(time_array)


# save the posterior results
np.savez_compressed(
    os.path.join(DATA_DIR, 'gplvm', f'{SESSION_NAME}_gplvm_posteriors_100bins.npz'),
    latent_posteriors=latent_posteriors_l,
    jump_prob=jump_prob_l,
    continuous_prob=continuous_prob_l,
    time_array=time_array

)


fn = os.path.join(DATA_DIR, 'gplvm', f'{SESSION_NAME}_gplvm_posteriors_100bins.npz')

# load npz
data = np.load(fn)

# convert to .mat
savemat(os.path.join(DATA_DIR, 'gplvm', f'{SESSION_NAME}_gplvm_posteriors_100bins.mat'), {
    'latent_posteriors': data['latent_posteriors'],
    'jump_prob': data['jump_prob'],
    'continuous_prob': data['continuous_prob'],
    'time_seconds': data['time_array']
})

