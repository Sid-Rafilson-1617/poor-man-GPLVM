import numpy as np
import jax.numpy as jnp
import jax.random as jr
import poor_man_gplvm as pmg
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.io import savemat
import seaborn as sns
import os

from core import *


# ---------------------------USER CONFIGURATION-------------------------
data_file = r'D:\sid\moser_vollan_2025\navigation\lt\28229_3.mat'
fig_dir = r'D:\sid\moser_vollan_2025\figures'
analysis_results_dir = r'D:\sid\moser_vollan_2025\analysis_results'

# Spike rate computation parameters
WINDOW_SIZE = 0.01         # s
STEP_SIZE   = 0.01        # s
USE_UNITS   = "good"       # {'all', 'good', 'mua'}3


# Filtering thresholds
MIN_TOTAL_SPIKES   = 500
MIN_MEAN_RATE_HZ   = .05
MIN_PRESENCE_RATIO = 0.5
N_TIME_BINS        = 100    # coarse bins for presence ratio
LOCATIONS = 'mec' # {'mec', 'hc', 'both'} # which regions to use


N_SPATIAL_BINS = 25 # number of spatial bins for decoding
POS_MIN = -0.8 # in meters
POS_MAX = 0.8
POSITION_AXIS = 'y'  # {'x', 'y', 'z'}

N_LATENT_BINS = 25
MOVEMENT_VARIANCE = 1.0
TUNING_LENGTHSCALE = 10.0
N_ITER = 10

N_FIGS = 50 # number of partitions of the data to plot the figures for


# -------------------------------END USER CONFIGURATION----------------------------

print('Preprocessing data...')

# Extract session name from data file path
session_name = os.path.splitext(os.path.basename(data_file))[0]
print(f'Session name: {session_name}')

# Extract spike count matrix and position data
spike_count_matrix, time_bins, units, x_win, y_win, z_win = preprocess_moser_data(data_file, window_size=WINDOW_SIZE, step_size=STEP_SIZE, use_units=USE_UNITS, locations=LOCATIONS)
spike_count_matrix = sort_units(spike_count_matrix, mode='corr')
print(f'spike count matrix shape: {spike_count_matrix.shape}')
print(f'time bins shape: {time_bins.shape}')
print(f'units shape: {units.shape}')
print(f'x_win shape: {x_win.shape}')
print(f'y_win shape: {y_win.shape}')
print(f'z_win shape: {z_win.shape}')


# Compute position and speed
if POSITION_AXIS == 'x':
    pos_raw = x_win
elif POSITION_AXIS == 'y':
    pos_raw = y_win
elif POSITION_AXIS == 'z':
    pos_raw = z_win
else:
    raise ValueError(f'Invalid POSITION_AXIS: {POSITION_AXIS}')

speed = np.abs(np.gradient(pos_raw) / np.gradient(time_bins))  # speed in m/s



# Plotting the tracking data
tracking_figure_dir = os.path.join(fig_dir, session_name,'tracking_data')
os.makedirs(tracking_figure_dir, exist_ok=True)
fig, axs = plt.subplots(3, 1, figsize=(15,8))
sns.lineplot(ax=axs[0], x=time_bins, y=x_win, label='X Position')
sns.lineplot(ax=axs[0], x=time_bins, y=y_win, label='Y Position')
sns.lineplot(ax=axs[0], x=time_bins, y=z_win, label='Z Position')
axs[0].set_ylabel('Position')
axs[0].set_title('Tracking Data - Positions')
sns.histplot(ax=axs[1], x=pos_raw, bins=100)
axs[1].set_xlabel('Position (meters)')
axs[1].set_ylabel('Count')
axs[1].set_title(f'Position Histogram')
sns.histplot(ax=axs[2], x=speed[speed < np.percentile(speed, 99)], bins=100)
axs[2].set_xlabel('Speed (m/s)')
axs[2].set_ylabel('Count')
axs[2].set_title('Speed Histogram')
axs[2].axvline(x = np.median(speed), color='red', linestyle='--', label=f'Median Speed: {np.median(speed):.4f} m/s')
axs[2].legend()
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(tracking_figure_dir, f'{session_name}_tracking_data.png'))



# Binning the position data
pos_bins = np.linspace(POS_MIN, POS_MAX, N_SPATIAL_BINS)
pos_bin_centers = (pos_bins[:-1] + pos_bins[1:]) / 2
pos_idx = np.digitize(pos_raw, pos_bins) - 1
pos_idx = np.clip(pos_idx, 0, N_SPATIAL_BINS - 1)

# Computing average firing rate across all units
fr = np.sum(spike_count_matrix, axis=0) / len(units) / WINDOW_SIZE  # in Hz



# Plot spike counts and position over time in segments
counts_fig_dir = os.path.join(fig_dir, session_name, 'spike_counts_and_position')
os.makedirs(counts_fig_dir, exist_ok=True)
for i in range(N_FIGS):
    start_idx = int(i * len(time_bins) / N_FIGS)
    end_idx = int((i + 1) * len(time_bins) / N_FIGS)
    spike_count_matrix_segment = spike_count_matrix[:, start_idx:end_idx]
    time_bins_segment = time_bins[start_idx:end_idx]
    pos_idx_segment = pos_idx[start_idx:end_idx]
    fr_segment = fr[start_idx:end_idx]

    fig, axs = plt.subplots(2, 1, figsize=(15,8), sharex = True)
    dt = np.mean(np.diff(time_bins))  # step_size
    axs[0].imshow(spike_count_matrix_segment, cmap='Greys', aspect='auto',extent=[time_bins_segment[0], time_bins_segment[-1] + dt, 0, spike_count_matrix.shape[0]], origin='lower', vmin=0, vmax=1)
    axs[0].twinx().plot(time_bins_segment, fr_segment, color='k', label='Firing Rate (Hz)', alpha = 0.2)
    axs[1].plot(time_bins_segment, pos_idx_segment, color='b', label='Position Bin Index', alpha = 0.7)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Position Bin Index')
    axs[1].set_title('Position Bins Over Time')
    axs[0].set_ylabel('Unit Index')
    axs[0].set_title('Spike Counts Over Time')
    plt.tight_layout()
    sns.despine()
    plt.savefig(os.path.join(counts_fig_dir, f'{session_name}_figure_{i}_spike_counts_and_position.png'))


# Define poisson Bayesian decoder model
model = PoissonBayesDecoder(n_bins=N_SPATIAL_BINS, rate_floor=1e-4, uniform_prior=True)

# fitting the model
model.fit(spike_count_matrix, pos_idx)

# poisson bayessian decoding
log_probs = model.predict_log_probabilities(spike_count_matrix)
poisson_posterior = np.exp(log_probs - logsumexp(log_probs, axis=0, keepdims=True))

poisson_fig_dir = os.path.join(fig_dir, session_name, 'poisson_bayesian_decoder')
os.makedirs(poisson_fig_dir, exist_ok=True)
for i in range(N_FIGS):
    start_idx = int(i * len(time_bins) / N_FIGS)
    end_idx = int((i + 1) * len(time_bins) / N_FIGS)
    poisson_posterior_segment = poisson_posterior[:, start_idx:end_idx]
    time_bins_segment = time_bins[start_idx:end_idx]
    pos_idx_segment = pos_idx[start_idx:end_idx]
    fr_segment = fr[start_idx:end_idx]

    fig, axs = plt.subplots(1, 1, figsize=(15,8))
    dt = np.mean(np.diff(time_bins))  # step_size
    axs.imshow(poisson_posterior_segment, cmap='viridis', aspect='auto', extent=[time_bins_segment[0], time_bins_segment[-1] + dt, 0, N_SPATIAL_BINS], origin='lower')
    axs.plot(time_bins_segment, pos_idx_segment, color='k', label='True Position', alpha=0.5, linewidth=3)
    axs.plot(time_bins_segment, fr_segment, color='r', label='Mean Firing Rate', alpha=0.5, linewidth=2)
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Position Bin Index')
    axs.set_title('Posterior Probability Over Time')
    cbar = plt.colorbar(axs.imshow(poisson_posterior_segment, cmap='viridis', aspect='auto', extent=[time_bins_segment[0], time_bins_segment[-1] + dt, 0, N_SPATIAL_BINS], origin='lower'))
    cbar.set_label('Posterior Probability')
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(poisson_fig_dir, f'{session_name}_figure_{i}_poisson_bayesian_decoder.png'))






# Fitting GPLVM model
n_neuron = spike_count_matrix.shape[0]
latent_model =pmg.PoissonGPLVMJump1D(n_neuron, n_latent_bin=N_LATENT_BINS, movement_variance=MOVEMENT_VARIANCE, tuning_lengthscale=TUNING_LENGTHSCALE)
em_res = latent_model.fit_em(spike_count_matrix.T, key=jr.PRNGKey(3), n_iter=N_ITER, log_posterior_init=None, ma_neuron=None, ma_latent=None, n_time_per_chunk=10000)

# GPLVM decoding
GPLVM_plots_dir = os.path.join(fig_dir, session_name, 'GPLVM_decoder')
os.makedirs(GPLVM_plots_dir, exist_ok=True)
plt.figure()
plt.plot(em_res['log_marginal_l'], linewidth=3)
plt.xlabel('EM Iteration')
plt.ylabel('Log Marginal Likelihood')
plt.title('EM Algorithm Convergence')
plt.tight_layout()
plt.savefig(os.path.join(GPLVM_plots_dir, f'{session_name}_EM_convergence.png'))

# plotting tuning curves
tuning_curve_dir = os.path.join(GPLVM_plots_dir, 'tuning_curves')
os.makedirs(tuning_curve_dir, exist_ok=True)

for neuron in range(n_neuron):
    plt.figure()
    plt.plot(latent_model.latent_bin_centers, latent_model.tuning_curves[neuron], linewidth=2)
    plt.xlabel('Latent Variable Bin Centers')
    plt.ylabel('Firing Rate (Hz)')
    plt.title(f'Neuron {neuron} Tuning Curve')
    plt.tight_layout()
    plt.savefig(os.path.join(tuning_curve_dir, f'{session_name}_neuron_{neuron}_tuning_curve.png'))
    plt.close()


# decoding latent variable
decode_res = latent_model.decode_latent(spike_count_matrix.T)
latent_posterior = decode_res['posterior_latent_marg']


# plotting latent posterior over time
latent_posterior_dir = os.path.join(GPLVM_plots_dir, 'latent_posterior')
for i in range(N_FIGS):
    plt.figure(figsize=(15,8))
    plt.imshow(latent_posterior.T, cmap='viridis', aspect='auto', extent=[time_bins[0], time_bins[-1] + dt, 0, N_LATENT_BINS], origin='upper')
    #overlay true position
    plt.plot(time_bins, pos_idx, color='k', label='True Position', alpha=0.5, linewidth=3)
    plt.plot(time_bins, fr, color='r', label='Mean Firing Rate', alpha=0.5, linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Latent Bin Index')
    plt.title('Posterior Probability Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(latent_posterior_dir, f'{session_name}_figure_{i}_latent_posterior.png'))
    plt.close()

# iterations, regions, etc.

# saving all the results to NPZ and MAT files
analysis_results_session_dir = os.path.join(analysis_results_dir, session_name)
os.makedirs(analysis_results_session_dir, exist_ok=True)
np.savez_compressed(os.path.join(analysis_results_session_dir, f'{session_name}_analysis_results.npz'),
                    spike_count_matrix=spike_count_matrix,
                    time_bins=time_bins,
                    units=units,
                    pos_raw=pos_raw,
                    pos_idx=pos_idx,
                    fr=fr,
                    poisson_posterior=poisson_posterior,
                    latent_posterior=latent_posterior,
                    tuning_curves=latent_model.tuning_curves,
                    latent_bin_centers=latent_model.latent_bin_centers,
                    em_log_marginal_l=em_res['log_marginal_l']
                   )

# saving to mat file
savemat(os.path.join(analysis_results_session_dir, f'{session_name}_analysis_results.mat'),
        {
            'spike_count_matrix': spike_count_matrix,
            'time_bins': time_bins,
            'units': units,
            'pos_raw': pos_raw,
            'pos_idx': pos_idx,
            'fr': fr,
            'poisson_posterior': poisson_posterior,
            'latent_posterior': latent_posterior,
            'tuning_curves': latent_model.tuning_curves,
            'latent_bin_centers': latent_model.latent_bin_centers,
            'em_log_marginal_l': em_res['log_marginal_l']
        })