import numpy as np
import jax.numpy as jnp
import jax.random as jr
import poor_man_gplvm as pmg
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.interpolate import interp1d
import os
import glob
import mat73

from core import *

# ----------------------------- Config ---------------------------------

# Data paths and session info
DATA_DIR = r"Z:\Homes\voerom01\Bilat_HPC\Bilat_R02\Bilat_R02_20251106"
FIG_DIR = os.path.join(DATA_DIR, "figures")
ANALYSIS_DIR = os.path.join(DATA_DIR, "gplvm_results")
SESSION_NAME = os.path.basename(DATA_DIR)
BASE_NAME = SESSION_NAME
PROBES = [1, 3]  # List of probe numbers to include

REGIONS = ['CA1']
UNIT_TYPES = ['Pyramidal Cell', "Wide Interneuron", "Narrow Interneuron"]

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
region_dict = {
    0: {0: 'CA1', 1: 'CA1', 2: 'Th', 3: 'CA1', 4: 'Th', 5: 'CA1'},
    1: {0: 'CA1', 1: 'CA1', 2: 'CA1', 3: 'CA1', 4: 'CA1', 5: 'CA1'},
    2: {0: 'CA1', 1: 'CA1', 2: 'CA1', 3: 'Ctx', 4: 'Th', 5: 'CA1'}, 
    3: {0: 'CA1', 1: 'CA1', 2: 'CA1', 3: 'CA1', 4: 'CA1', 5: 'CA1'}
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

        
    # Get the raw spike counts and the zscored and smoothed spike rates
    pattern = os.path.join(DATA_DIR, f"Kilosort*imec{probe}*")
    matches = glob.glob(pattern)
    ks_dir = matches[0]  # assuming there's only one match
    print(f"Loading spike counts from Kilosort directory: {ks_dir}")
    spike_count_matrix, time_bins, units = compute_spike_counts(str(ks_dir), WINDOW_SIZE, STEP_SIZE, USE_UNITS, 0.0, False, adj = '_sec_adj')



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

    
use_position = 'x'  # 'x', 'y', or 'z'

# Loading the manually extracted TTL times
manual_ttl_file = os.path.join(DATA_DIR, 'manual_ttl_extraction.csv')
manual_ttls_df = pd.read_csv(manual_ttl_file)
ttls = manual_ttls_df['ttl_times_sec'].values

# Loadding in the tracking data
tracking_file = os.path.join(DATA_DIR, "Bilat_R02_20251106_linear_maze_000.csv")
tracking_data = np.genfromtxt(tracking_file, delimiter=",", skip_header=7)
frames = tracking_data[:,0]
print(f'Number of frames: {len(frames)}')
print(f'first row of tracking data: {tracking_data[0,:]}')


# Sometimes there are a few more frames after the last TTL or more TTLs than frames. We will trim the data to the shortest length.
if len(frames) < len(ttls):
    print(f'Trimming TTLs from {len(ttls)} to {len(frames)}')
    ttls = ttls[:len(frames)]
    tracking_data = tracking_data[:len(frames),:]
elif len(frames) > len(ttls):
    print(f'Trimming frames from {len(frames)} to {len(ttls)}')
    frames = frames[:len(ttls)]
    tracking_data = tracking_data[:len(ttls),:]

# printing the start and end times
print(f'Tracking start time: {ttls[0]:.3f} s, end time: {ttls[-1]:.3f} s, duration: {ttls[-1]-ttls[0]:.3f} s')


# fill missing values or nan with linear interpolation
for i in range(tracking_data.shape[1]):
    col = tracking_data[:,i]
    nans = np.isnan(col)
    if np.any(nans):
        not_nans = ~nans
        interp_func_col = interp1d(frames[not_nans], col[not_nans], bounds_error=False, fill_value="extrapolate")
        col[nans] = interp_func_col(frames[nans])
        tracking_data[:,i] = col

# extract x, y, z positions and rotations
x_rotations = tracking_data[:,2]
y_rotations = tracking_data[:,3]
z_rotations = tracking_data[:,4]
w_rotations = tracking_data[:,5]
x_positions = tracking_data[:,6]
y_positions = tracking_data[:,7]
z_positions = tracking_data[:,8]

# setting the pos_raw to the selected position axis
if use_position == 'x':
    pos_raw = x_positions
elif use_position == 'y':
    pos_raw = y_positions
elif use_position == 'z':
    pos_raw = z_positions

N_SPATIAL_BINS = 25
POS_MIN = -0.6
POS_MAX = 0.8
SPEED_THRESHOLD = 0.05  # m/s

from scipy.ndimage import gaussian_filter1d

# compute the spatial bin edges
bin_edges = np.linspace(POS_MIN, POS_MAX, N_SPATIAL_BINS + 1)

# Creating interpolation function for position
pos_interp_func = interp1d(ttls, pos_raw, kind = 'linear', bounds_error=False, fill_value=np.nan)

# getting reference time from probe with shortest timebins
tb_ref = None
for probe in PROBES:
    _, tb_probe, _ = spike_count_matrices[probe]
    if tb_ref is None:
        tb_ref = tb_probe
    elif len(tb_probe) < len(tb_ref):
        tb_ref = tb_probe

# Build canonical masks on the *reference* tb_ref
pos_ref = pos_interp_func(tb_ref)
valid_ttl  = (tb_ref >= np.min(ttls)) & (tb_ref <= np.max(ttls))
print(f'number of valid TTL timebins: {valid_ttl.sum()} / {len(tb_ref)}')
finite_pos = np.isfinite(pos_ref)
print(f'number of finite position timebins: {finite_pos.sum()} / {len(tb_ref)}')
# filtering when slower than median speed
# smooth the position signal before computing speed
pos_ref_smooth = gaussian_filter1d(pos_ref, sigma=10, mode='nearest')
speed = np.abs(np.gradient(pos_ref_smooth) / np.gradient(tb_ref))
# smooth the speed signal
speed = gaussian_filter1d(speed, sigma=50, mode='nearest')
speed_mask = (speed > SPEED_THRESHOLD)
print(f'number of timebins with speed > {SPEED_THRESHOLD}: {speed_mask.sum()} / {len(speed)}')

mask = valid_ttl & finite_pos & speed_mask

# digitize positions into spatial bins
pos_bins = np.digitize(pos_ref, bin_edges) - 1  # bins are 0-indexed

# ------------------------ Plotting spike raster for each hemisphere and the binarized positions------------------------


tb_ref = None
for probe in PROBES:
    _, tb_probe, _ = spike_count_matrices[probe]
    if tb_ref is None:
        tb_ref = tb_probe
    elif len(tb_probe) < len(tb_ref):
        tb_ref = tb_probe

# preparing the lists to collect spike rates
selected_spike_counts_L = []
selected_spike_counts_R = []


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
    #spike_counts = spike_count_matrix[:, mask]

    # setting spike counts outside mask to 0
    spike_counts = spike_count_matrix.copy()
    spike_counts[:, ~mask] = 0



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

# get the mean firing rate across all the units for each time bin
mean_fr_l = np.mean(sorted_spike_counts_concat_L, axis=0) / WINDOW_SIZE  # in Hz
mean_fr_r = np.mean(sorted_spike_counts_concat_R, axis=0) / WINDOW_SIZE  # in Hz

# getting the time vector in seconds
times = tb_ref - tb_ref[0]


# removing all the data before the first TTL and after the last TTL
ttl_mask = (tb_ref >= np.min(ttls)) & (tb_ref <= (np.max(ttls)))
sorted_spike_counts_concat_L = sorted_spike_counts_concat_L[:, ttl_mask]
sorted_spike_counts_concat_R = sorted_spike_counts_concat_R[:, ttl_mask]
mean_fr_l = mean_fr_l[ttl_mask]
mean_fr_r = mean_fr_r[ttl_mask]
pos_bins = pos_bins[ttl_mask]
pos_ref = pos_ref[ttl_mask]
times = times[ttl_mask]
speed = speed[ttl_mask]

print(f'sorted_counts_concat_L shape: {sorted_spike_counts_concat_L.shape}')
print(f'sorted_counts_concat_R shape: {sorted_spike_counts_concat_R.shape}')
print(f'Mean FR L shape: {mean_fr_l.shape}')
print(f'Mean FR R shape: {mean_fr_r.shape}')
print(f'pos_bins shape: {pos_bins.shape}')
print(f'pos_ref shape: {pos_ref.shape}')
print(f'times shape: {times.shape}')
print(f'speed shape: {speed.shape}')

# define the models for the left and right hemispheres
N_SPATIAL_BINS = 100
MOVEMENT_VARIANCE = 0.5
TUNING_LENGTHSCALE = 10
model_l=pmg.PoissonGPLVMJump1D(sorted_spike_counts_concat_L.shape[0], n_latent_bin=N_SPATIAL_BINS, movement_variance=MOVEMENT_VARIANCE, tuning_lengthscale=TUNING_LENGTHSCALE)
model_r=pmg.PoissonGPLVMJump1D(sorted_spike_counts_concat_R.shape[0], n_latent_bin=N_SPATIAL_BINS, movement_variance=MOVEMENT_VARIANCE, tuning_lengthscale=TUNING_LENGTHSCALE)


# randomize the position bins for training
random_positions = False
if random_positions:
    pos_bins = np.random.randint(0, N_SPATIAL_BINS, size=sorted_spike_counts_concat_L.shape[1])

print(pos_bins.shape)
print(sorted_spike_counts_concat_L.shape)
print(sorted_spike_counts_concat_R.shape)

# getting the log posterior init
log_posterior_init = np.zeros((sorted_spike_counts_concat_L.shape[1], N_SPATIAL_BINS))

print(log_posterior_init.shape)
for t in range(sorted_spike_counts_concat_L.shape[1]):
    log_posterior_init[t, pos_bins[t]] += 1

# adding a small value to avoid log(0)
log_posterior_init += 1e-6

#normalizing so sums to 1
log_posterior_init /= log_posterior_init.sum(axis=1, keepdims=True)

# converting to log space
log_posterior_init = np.log(log_posterior_init)

print(f'log_posterior_init shape: {log_posterior_init.shape}')



# getting the time bins where there are at least one spike in both hemispheres
valid_time_bins = np.where(
    (np.sum(sorted_spike_counts_concat_L, axis=0) > 0) &
    (np.sum(sorted_spike_counts_concat_R, axis=0) > 0)
)[0]

# fitting the models
em_res_r = model_r.fit_em(sorted_spike_counts_concat_R[:, valid_time_bins].T, key=jr.PRNGKey(3), n_iter=15, log_posterior_init=None, ma_neuron=None, ma_latent=None, n_time_per_chunk=10000)
em_res_l = model_l.fit_em(sorted_spike_counts_concat_L[:, valid_time_bins].T, key=jr.PRNGKey(3), n_iter=1, log_posterior_init=None, ma_neuron=None, ma_latent=None, n_time_per_chunk=10000)

#decode_res_r = model_r.decode_latent(sorted_spike_counts_concat_R.T)
decode_res_l = model_l.decode_latent(sorted_spike_counts_concat_L.T)
decode_res_r = model_r.decode_latent(sorted_spike_counts_concat_R.T)

latent_posteriors_r = decode_res_r['posterior_latent_marg']
jump_prob_r = decode_res_r['posterior_dynamics_marg'][:,1]
continuous_prob_r = decode_res_r['posterior_dynamics_marg'][:,0]

print(f'latent_posteriors_r shape: {latent_posteriors_r.shape}')
print(f'jump_prob_r shape: {jump_prob_r.shape}')
print(f'continuous_prob_r shape: {continuous_prob_r.shape}')


latent_posteriors_l = decode_res_l['posterior_latent_marg']
jump_prob_l = decode_res_l['posterior_dynamics_marg'][:,1]
continuous_prob_l = decode_res_l['posterior_dynamics_marg'][:,0]

print(f'latent_posteriors_l shape: {latent_posteriors_l.shape}')
print(f'jump_prob_l shape: {jump_prob_l.shape}')
print(f'continuous_prob_l shape: {continuous_prob_l.shape}')

tuning_curves_r = model_r.tuning
tuning_curves_l = model_l.tuning

# save the posterior results
os.makedirs(ANALYSIS_DIR, exist_ok=True)
date = pd.Timestamp.now().strftime('%Y%m%d')
time = pd.Timestamp.now().strftime('%H%M%S')
path_name = os.path.join(ANALYSIS_DIR, f'{SESSION_NAME}_gplvm_posteriors_{N_SPATIAL_BINS}bins_{date}_{time}.npz')
np.savez_compressed(
    path_name,
    latent_posteriors_r=latent_posteriors_r,
    jump_prob_r=jump_prob_r,
    continuous_prob_r=continuous_prob_r,
    latent_posteriors_l=latent_posteriors_l,
    jump_prob_l=jump_prob_l,
    continuous_prob_l=continuous_prob_l,
    time_array=times,
    pos_bins=pos_bins,
    pos_ref=pos_ref,
    mean_fr_l=mean_fr_l,
    mean_fr_r=mean_fr_r,
    WINDOW_SIZE=WINDOW_SIZE,
    STEP_SIZE=STEP_SIZE,
    USE_UNITS=USE_UNITS,
    UNIT_TYPES=UNIT_TYPES,
    N_SPATIAL_BINS=N_SPATIAL_BINS,
    POS_MIN=POS_MIN,
    POS_MAX=POS_MAX,
    SPEED_THRESHOLD=SPEED_THRESHOLD,
    MOVEMENT_VARIANCE=MOVEMENT_VARIANCE,
    TUNING_LENGTHSCALE=TUNING_LENGTHSCALE,
    tuning_functions_l=tuning_curves_l,
    tuning_functions_r=tuning_curves_r
)




from scipy.io import savemat



# load npz
data = np.load(path_name)
mat_path_name = path_name.replace('.npz', '.mat')

# convert to .mat
savemat(mat_path_name, 
        {
    'latent_posteriors_r': data['latent_posteriors_r'],
    'jump_prob_r': data['jump_prob_r'],
    'continuous_prob_r': data['continuous_prob_r'],
    'latent_posteriors_l': data['latent_posteriors_l'],
    'jump_prob_l': data['jump_prob_l'],
    'continuous_prob_l': data['continuous_prob_l'],
    'time_seconds': data['time_array'],
    'pos_bins': data['pos_bins'],
    'pos_ref': data['pos_ref'],
    'mean_fr_l': data['mean_fr_l'],
    'mean_fr_r': data['mean_fr_r'],
    'WINDOW_SIZE': data['WINDOW_SIZE'].item(),
    'STEP_SIZE': data['STEP_SIZE'].item(),
    'USE_UNITS': data['USE_UNITS'].item(),
    'UNIT_TYPES': data['UNIT_TYPES'].tolist(),
    'N_SPATIAL_BINS': data['N_SPATIAL_BINS'].item(),
    'POS_MIN': data['POS_MIN'].item(),
    'POS_MAX': data['POS_MAX'].item(),
    'SPEED_THRESHOLD': data['SPEED_THRESHOLD'].item(),
    'MOVEMENT_VARIANCE': data['MOVEMENT_VARIANCE'].item(),
    'TUNING_LENGTHSCALE': data['TUNING_LENGTHSCALE'].item(),
    'tuning_functions_l': data['tuning_functions_l'],
    'tuning_functions_r': data['tuning_functions_r']   
})