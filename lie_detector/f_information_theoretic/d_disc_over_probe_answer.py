import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import json
from util.util import YamlConfig

# Load config
config_path = sys.argv[1]
args = YamlConfig(config_path)

# Extract key parameters from config
system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name
initial_answers_args_name = args.initial_answers_args_name
probe_responses_args_name = args.probe_responses_args_name
probe_analysis_args_name = args.probe_analysis_args_name
probe_file_name = args.probe_file_name
num_layers = args.num_layers
prompt_index = args.prompt_idx
limit_to_lying = args.limit_to_lying

# Set up paths
save_base = os.path.join('lie_detector_results/f_information_theoretic', args.args_name)

# Load the saved numpy arrays. 
# Projections of shape [questions, probe_questions, layers, tokens] onto vector shared amongst all cases
all_lie_projections = np.load(os.path.join(save_base, 'all_lie_projections_over_probe_question.npy'))
all_truth_projections = np.load(os.path.join(save_base, 'all_truth_projections_over_probe_question.npy'))

# Load the dedicated projection arrays [questions, probe_questions, layers, tokens] onto 
# Projections of shape [questions, probe_questions, layers, tokens] onto vector dedicated to that token within that probe question
all_lie_dedicated_projections = np.load(os.path.join(save_base, 'all_lie_dedicated_projections_over_probe_question.npy'))
all_truth_dedicated_projections = np.load(os.path.join(save_base, 'all_truth_dedicated_projections_over_probe_question.npy'))

# Load related data files
initial_answers_path = os.path.join('lie_detector_results/a_initial_answers', initial_answers_args_name, 'initial_answers.csv')
response_data = pd.read_csv(initial_answers_path)

initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')
probe_questions = pd.read_csv(f'data/probe_questions/{probe_file_name}.csv')['probe']

# Print basic info about loaded data
print(f"Loaded projections with shape:")
print(f"  Generic lie projections: {all_lie_projections.shape}")
print(f"  Generic truth projections: {all_truth_projections.shape}")
print(f"  Dedicated lie projections: {all_lie_dedicated_projections.shape}")
print(f"  Dedicated truth projections: {all_truth_dedicated_projections.shape}")
print(f"  Shape meaning: [questions, probe_questions, layers, tokens]")
print(f"  Number of layers: {num_layers}")
print(f"  Number of probe questions: {len(probe_questions)}")

# Filter to same trainable questions as original script
if limit_to_lying:
    knows_answer = response_data[(response_data['knows_answer']) & (response_data['told_truth'])]
    trainable_answers = knows_answer[(knows_answer['prompt_idx'] == prompt_index) & (knows_answer['told_lie'])]
else:
    trainable_answers = response_data
trainable_questions_idxs = trainable_answers['question_idx']

print(f"Number of trainable questions: {len(trainable_questions_idxs)}")

# Load reference SNRs and take absolute value to make them unsigned
discriminability_data_path = os.path.join('lie_detector_results/c_probe_discrimination', probe_analysis_args_name, 'discriminability_results.json')
with open(discriminability_data_path, 'r') as f:
    discriminability_data = json.load(f)
reference_snrs = np.abs(np.array([res["effect_size"] for res in discriminability_data['probe_results']]))
print(f"Reference SNRs shape (unsigned): {reference_snrs.shape}")

# Calculate unsigned SNRs for generic projections
print("Calculating unsigned SNRs for generic projections...")
diff_generic = all_lie_projections - all_truth_projections  # shape: [questions, probe_questions, layers, tokens]
mean_diff_generic = np.nanmean(diff_generic, axis=0)  # shape: [probe_questions, layers, tokens]
std_diff_generic = np.nanstd(diff_generic, axis=0)    # shape: [probe_questions, layers, tokens]
std_diff_generic[std_diff_generic < 1e-10] = np.nan
snr_generic = np.abs(mean_diff_generic) / std_diff_generic  # Take absolute value for unsigned SNR

# Calculate unsigned SNRs for dedicated projections
print("Calculating unsigned SNRs for dedicated projections...")
diff_dedicated = all_lie_dedicated_projections - all_truth_dedicated_projections  # shape: [questions, probe_questions, layers, tokens]
mean_diff_dedicated = np.nanmean(diff_dedicated, axis=0)  # shape: [probe_questions, layers, tokens]
std_diff_dedicated = np.nanstd(diff_dedicated, axis=0)    # shape: [probe_questions, layers, tokens]
std_diff_dedicated[std_diff_dedicated < 1e-10] = np.nan
snr_dedicated = np.abs(mean_diff_dedicated) / std_diff_dedicated  # Take absolute value for unsigned SNR


print(f"Generic SNR shape: {snr_generic.shape} [probe_questions, layers, tokens]")
print(f"Dedicated SNR shape: {snr_dedicated.shape} [probe_questions, layers, tokens]")

# Calculate correlation coefficients for both generic and dedicated projections
print("Calculating correlation coefficients...")
num_tokens = snr_generic.shape[2]
correlation_coeffs_generic = np.full((num_layers, num_tokens), np.nan)
correlation_coeffs_dedicated = np.full((num_layers, num_tokens), np.nan)

for layer in range(num_layers):
    for timestep in range(num_tokens):
        # Generic correlations
        layer_timestep_snrs_generic = snr_generic[:, layer, timestep]
        valid_mask_generic = ~np.isnan(layer_timestep_snrs_generic)
        
        if np.sum(valid_mask_generic) > 2:  # Need at least 3 points for meaningful correlation
            valid_snrs_generic = layer_timestep_snrs_generic[valid_mask_generic]
            valid_refs_generic = reference_snrs[valid_mask_generic]
            
            if len(valid_snrs_generic) > 1 and np.std(valid_snrs_generic) > 0 and np.std(valid_refs_generic) > 0:
                correlation_coeffs_generic[layer, timestep] = np.corrcoef(valid_snrs_generic, valid_refs_generic)[0, 1]
        
        # Dedicated correlations
        layer_timestep_snrs_dedicated = snr_dedicated[:, layer, timestep]
        valid_mask_dedicated = ~np.isnan(layer_timestep_snrs_dedicated)
        
        if np.sum(valid_mask_dedicated) > 2:  # Need at least 3 points for meaningful correlation
            valid_snrs_dedicated = layer_timestep_snrs_dedicated[valid_mask_dedicated]
            valid_refs_dedicated = reference_snrs[valid_mask_dedicated]
            
            if len(valid_snrs_dedicated) > 1 and np.std(valid_snrs_dedicated) > 0 and np.std(valid_refs_dedicated) > 0:
                correlation_coeffs_dedicated[layer, timestep] = np.corrcoef(valid_snrs_dedicated, valid_refs_dedicated)[0, 1]

print(f"Generic correlation coefficients shape: {correlation_coeffs_generic.shape} [layers, tokens]")
print(f"Dedicated correlation coefficients shape: {correlation_coeffs_dedicated.shape} [layers, tokens]")

# Calculate number of probe questions remaining at each timestep
probe_counts = np.zeros(num_tokens)
for timestep in range(num_tokens):
    probe_counts[timestep] = np.sum(~np.isnan(snr_generic[:, 0, timestep]))  # Use layer 0 as reference

# Find the maximum timestep with actual data
max_timestep = np.where(probe_counts > 0)[0][-1] + 1 if np.any(probe_counts > 0) else num_tokens

# Create three-panel plot
print("Creating plot...")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, 
                                    gridspec_kw={'height_ratios': [3, 3, 1], 'hspace': 0.1})

# Set up colormap for layers
colors = plt.cm.magma(np.linspace(0.1, 0.9, num_layers))

# Top subplot: Generic correlation coefficients
for layer in range(num_layers):
    valid_timesteps = ~np.isnan(correlation_coeffs_generic[layer, :max_timestep])
    if np.any(valid_timesteps):
        timesteps = np.arange(max_timestep)[valid_timesteps]
        coeffs = correlation_coeffs_generic[layer, :max_timestep][valid_timesteps]
        ax1.plot(timesteps, coeffs, color=colors[layer], alpha=0.8, linewidth=1.5)

ax1.set_ylabel('Correlation Coefficient')
ax1.set_title('Generic Projections: Correlation between Unsigned SNRs and Reference SNRs vs Time')
ax1.grid(True, alpha=0.3)

# Middle subplot: Dedicated correlation coefficients
for layer in range(num_layers):
    valid_timesteps = ~np.isnan(correlation_coeffs_dedicated[layer, :max_timestep])
    if np.any(valid_timesteps):
        timesteps = np.arange(max_timestep)[valid_timesteps]
        coeffs = correlation_coeffs_dedicated[layer, :max_timestep][valid_timesteps]
        ax2.plot(timesteps, coeffs, color=colors[layer], alpha=0.8, linewidth=1.5)

ax2.set_ylabel('Correlation Coefficient')
ax2.set_title('Dedicated Projections: Correlation between Unsigned SNRs and Reference SNRs vs Time')
ax2.grid(True, alpha=0.3)

# Add colorbar legend for layers using separate axes to prevent squashing
sm = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=0, vmax=num_layers-1))
sm.set_array([])
# Create separate axes for colorbar
cbar_ax = fig.add_axes([0.92, 0.35, 0.02, 0.4])  # [left, bottom, width, height]
cbar = fig.colorbar(sm, cax=cbar_ax, label='Layer')
cbar.set_ticks(np.arange(0, num_layers, max(1, num_layers//10)))

# Bottom subplot: Number of probe questions remaining
ax3.bar(range(max_timestep), probe_counts[:max_timestep], color='steelblue', alpha=0.7, width=1.0)
ax3.set_xlabel('Timestep')
ax3.set_ylabel('Probe Questions\nRemaining')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, len(probe_questions) * 1.05)
ax3.set_xlim(-0.5, max_timestep - 0.5)

plt.tight_layout()
plt.subplots_adjust(right=0.9)  # Make room for colorbar
plot_path = os.path.join(save_base, 'correlation_coefficients_vs_time.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")
plt.show()

# Create new grid plot: SNR scatter plots
print("Creating SNR scatter grid plot...")

# Sample timesteps and layers: every 5th + last one
timestep_samples = list(range(0, max_timestep, 5))
if (max_timestep - 1) not in timestep_samples:
    timestep_samples.append(max_timestep - 1)

layer_samples = list(range(0, num_layers, 5))
if (num_layers - 1) not in layer_samples:
    layer_samples.append(num_layers - 1)

print(f"Sampled timesteps: {timestep_samples}")
print(f"Sampled layers: {layer_samples}")

# Create grid of subplots
n_rows = len(layer_samples)
n_cols = len(timestep_samples)
fig_grid, axes_grid = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), 
                                   sharex=True, sharey=True)

# Handle case where we might have only 1 row or 1 col
if n_rows == 1:
    axes_grid = axes_grid.reshape(1, -1)
elif n_cols == 1:
    axes_grid = axes_grid.reshape(-1, 1)

# Function to add best-fit line
def add_bestfit_line(ax, x, y, color, alpha=0.5):
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(valid_mask) > 1:
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        if len(x_valid) > 1 and np.std(x_valid) > 0:
            coeffs = np.polyfit(x_valid, y_valid, 1)
            x_line = np.linspace(x_valid.min(), x_valid.max(), 100)
            y_line = np.polyval(coeffs, x_line)
            ax.plot(x_line, y_line, '--', color=color, alpha=alpha, linewidth=1)

# Plot each subplot
for i, layer in enumerate(layer_samples):
    for j, timestep in enumerate(timestep_samples):
        ax = axes_grid[i, j]
        
        # Get SNR values for this layer/timestep
        generic_snrs = snr_generic[:, layer, timestep]
        dedicated_snrs = snr_dedicated[:, layer, timestep]

        # Create scatter plots
        ax.scatter(reference_snrs, generic_snrs, c='green', alpha=0.6, s=20, label='Generic')
        ax.scatter(reference_snrs, dedicated_snrs, c='purple', alpha=0.6, s=20, label='Dedicated')
        
        # Add best-fit lines
        add_bestfit_line(ax, reference_snrs, generic_snrs, 'green')
        add_bestfit_line(ax, reference_snrs, dedicated_snrs, 'purple')
        
        # Set title for each subplot
        ax.set_title(f'Layer {layer}, Timestep {timestep}', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add legend only to bottom-right subplot
        if i == n_rows-1 and j == n_cols-1:
            ax.legend(fontsize=10)

# Set common labels
fig_grid.text(0.5, 0.02, 'Reference SNR (Unsigned)', ha='center', fontsize=12)
fig_grid.text(0.02, 0.5, 'Computed SNR (Unsigned)', va='center', rotation=90, fontsize=12)

plt.tight_layout()
plt.subplots_adjust(bottom=0.08, left=0.08)
grid_plot_path = os.path.join(save_base, 'snr_scatter_grid.png')
plt.savefig(grid_plot_path, dpi=300, bbox_inches='tight')
print(f"Grid plot saved to: {grid_plot_path}")
plt.show()

# Save all results
np.save(os.path.join(save_base, 'correlation_coefficients_generic.npy'), correlation_coeffs_generic)
np.save(os.path.join(save_base, 'correlation_coefficients_dedicated.npy'), correlation_coeffs_dedicated)
np.save(os.path.join(save_base, 'snr_values_generic.npy'), snr_generic)
np.save(os.path.join(save_base, 'snr_values_dedicated.npy'), snr_dedicated)
np.save(os.path.join(save_base, 'reference_snrs_unsigned.npy'), reference_snrs)
print("Analysis complete. Saved correlation coefficients and SNR values for both generic and dedicated projections (all unsigned).")