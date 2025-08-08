import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from scipy.stats import spearmanr
from util.util import YamlConfig

# Load config
config_path = sys.argv[1]
args = YamlConfig(config_path)

# Extract key parameters from config
system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name
initial_answers_args_name = args.initial_answers_args_name
probe_file_name = args.probe_file_name
num_layers = args.num_layers
prompt_index = args.prompt_idx
limit_to_lying = args.limit_to_lying

# Set up paths
save_base = os.path.join('lie_detector_results/g_cutting_probes', args.args_name)

# Load projection data
print("Loading projection data...")
all_lie_projections = np.load(os.path.join(save_base, 'all_lie_projections_over_probe_question.npy'))
all_truth_projections = np.load(os.path.join(save_base, 'all_truth_projections_over_probe_question.npy'))
all_lie_dedicated_projections = np.load(os.path.join(save_base, 'all_lie_dedicated_projections_over_probe_question.npy'))
all_truth_dedicated_projections = np.load(os.path.join(save_base, 'all_truth_dedicated_projections_over_probe_question.npy'))

print(f"Projection shapes: {all_lie_projections.shape} [questions, probe_questions, layers, tokens]")

# Load behavioral data
print("Loading behavioral data...")
truncated_response_path = os.path.join(save_base, 'truncated_probe_response.csv')
truncated_df = pd.read_csv(truncated_response_path)
truncated_df['log_odds'] = np.log(truncated_df['prob_yes'] / truncated_df['prob_no'])

# Load related data for filtering
initial_answers_path = os.path.join('lie_detector_results/a_initial_answers', initial_answers_args_name, 'initial_answers.csv')
response_data = pd.read_csv(initial_answers_path)
probe_questions = pd.read_csv(f'data/probe_questions/{probe_file_name}.csv')['probe']

# Filter to same trainable questions
if limit_to_lying:
    knows_answer = response_data[(response_data['knows_answer']) & (response_data['told_truth'])]
    trainable_answers = knows_answer[(knows_answer['prompt_idx'] == prompt_index) & (knows_answer['told_lie'])]
else:
    trainable_answers = response_data
trainable_questions_idxs = trainable_answers['question_idx'].tolist()

# Subsample questions (keeping the same reduction as original)
trainable_questions_idxs = trainable_questions_idxs[:10]            
print('REDUCING trainable_questions_idxs TO JUST SPORTS QUESTIONS!!')
print(f"Using {len(trainable_questions_idxs)} initial questions")

# Filter behavioral data
truncated_df_filtered = truncated_df[truncated_df['question_idx'].isin(trainable_questions_idxs)]

# Determine dimensions
num_probe_questions = len(probe_questions)
max_tokens = truncated_df_filtered['token_position'].max()
num_trainable_questions = len(trainable_questions_idxs)
print(f"Dimensions: {num_probe_questions} probe questions, {num_layers} layers, {max_tokens} max tokens, {num_trainable_questions} trainable questions")

# Set up structured arrays for all data
print("Setting up structured arrays...")

# Behavioral data: [probe_questions, trainable_questions, tokens, 2(truth/lie)]
behavioral_log_odds = np.full((num_probe_questions, num_trainable_questions, max_tokens, 2), np.nan)

# Projection data: [probe_questions, trainable_questions, layers, tokens, 2(truth/lie)]
generic_projections = np.full((num_probe_questions, num_trainable_questions, num_layers, max_tokens, 2), np.nan)
dedicated_projections = np.full((num_probe_questions, num_trainable_questions, num_layers, max_tokens, 2), np.nan)

# Load behavioral data into structured arrays
print("Loading behavioral data into arrays...")
for i, question_idx in enumerate(trainable_questions_idxs):
    question_data = truncated_df_filtered[truncated_df_filtered['question_idx'] == question_idx]
    
    for _, row in question_data.iterrows():
        probe_idx = int(row['probe_question_idx'])
        token_pos = int(row['token_position'])
        truth = int(row['truth'])
        log_odds = row['log_odds']
        
        if probe_idx < num_probe_questions and token_pos <= max_tokens:
            behavioral_log_odds[probe_idx, i, token_pos-1, truth] = log_odds

# Load projection data into structured arrays
print("Loading projection data into arrays...")
for i, question_idx in enumerate(trainable_questions_idxs):
    for probe_idx in range(num_probe_questions):
        for layer in range(num_layers):
            for token_idx in range(min(max_tokens, all_lie_projections.shape[3])):
                # Truth projections
                truth_generic = all_truth_projections[i, probe_idx, layer, token_idx]
                truth_dedicated = all_truth_dedicated_projections[i, probe_idx, layer, token_idx]
                
                # Lie projections  
                lie_generic = all_lie_projections[i, probe_idx, layer, token_idx]
                lie_dedicated = all_lie_dedicated_projections[i, probe_idx, layer, token_idx]
                
                if not np.isnan(truth_generic):
                    generic_projections[probe_idx, i, layer, token_idx, 1] = truth_generic
                if not np.isnan(truth_dedicated):
                    dedicated_projections[probe_idx, i, layer, token_idx, 1] = truth_dedicated
                if not np.isnan(lie_generic):
                    generic_projections[probe_idx, i, layer, token_idx, 0] = lie_generic
                if not np.isnan(lie_dedicated):
                    dedicated_projections[probe_idx, i, layer, token_idx, 0] = lie_dedicated

# Calculate orientation decisions (simplified - just probe-global)
print("Calculating orientation decisions...")
probe_global_orientation = np.full(num_probe_questions, 1.0)  # 1 = no flip, -1 = flip

for probe_idx in range(num_probe_questions):
    # Find the last token with data for this probe
    valid_tokens = ~np.isnan(behavioral_log_odds[probe_idx, :, :, :]).any(axis=(0,2))
    if np.any(valid_tokens):
        last_token_idx = np.where(valid_tokens)[0][-1]
        
        # Calculate probe-global orientation based on last token
        truth_vals = behavioral_log_odds[probe_idx, :, last_token_idx, 1]
        lie_vals = behavioral_log_odds[probe_idx, :, last_token_idx, 0]
        
        valid_truth = truth_vals[~np.isnan(truth_vals)]
        valid_lie = lie_vals[~np.isnan(lie_vals)]
        
        if len(valid_truth) > 0 and len(valid_lie) > 0:
            mean_truth = np.mean(valid_truth)
            mean_lie = np.mean(valid_lie)
            
            # If truth has higher log-odds, flip so lie has higher
            if mean_truth > mean_lie:
                probe_global_orientation[probe_idx] = -1.0

# Calculate behavioral SNR for color coding
print("Calculating behavioral SNR for color coding...")
behavioral_snrs_for_coloring = np.full(num_probe_questions, np.nan)

for probe_idx in range(num_probe_questions):
    valid_tokens = ~np.isnan(behavioral_log_odds[probe_idx, :, :, :]).any(axis=(0,2))
    if np.any(valid_tokens):
        last_token_idx = np.where(valid_tokens)[0][-1]
        
        truth_vals = behavioral_log_odds[probe_idx, :, last_token_idx, 1] * probe_global_orientation[probe_idx]
        lie_vals = behavioral_log_odds[probe_idx, :, last_token_idx, 0] * probe_global_orientation[probe_idx]
        
        valid_truth = truth_vals[~np.isnan(truth_vals)]
        valid_lie = lie_vals[~np.isnan(lie_vals)]
        
        if len(valid_truth) > 0 and len(valid_lie) > 0:
            mean_diff = np.abs(np.mean(lie_vals) - np.mean(valid_truth))
            pooled_std = np.sqrt((np.var(valid_lie) + np.var(valid_truth)) / 2)
            
            if pooled_std > 1e-10:
                behavioral_snrs_for_coloring[probe_idx] = mean_diff / pooled_std

def calculate_layerwise_correlations(behavioral_data, projection_data, orientation):
    """
    Calculate correlations for each layer using 2D spearmanr.
    
    Args:
        behavioral_data: [probe_questions, trainable_questions, tokens, 2(truth/lie)]
        projection_data: [probe_questions, trainable_questions, layers, tokens, 2(truth/lie)]  
        orientation: [probe_questions] orientation multipliers
    
    Returns:
        mean_correlations: [probe_questions, tokens] mean correlation across layers
        std_correlations: [probe_questions, tokens] std correlation across layers
    """
    num_probes, num_questions, num_layers, num_tokens, _ = projection_data.shape
    
    mean_correlations = np.full((num_probes, num_tokens), np.nan)
    std_correlations = np.full((num_probes, num_tokens), np.nan)
    
    for probe_idx in range(num_probes):
        for token_idx in range(num_tokens):
            # Get behavioral data with orientation applied
            behavioral_truth = behavioral_data[probe_idx, :, token_idx, 1] * orientation[probe_idx]
            behavioral_lie = behavioral_data[probe_idx, :, token_idx, 0] * orientation[probe_idx]
            behavioral_combined = np.concatenate([behavioral_truth, behavioral_lie])
            
            # Get projection data for all layers
            projection_truth = projection_data[probe_idx, :, :, token_idx, 1]  # [questions, layers]
            projection_lie = projection_data[probe_idx, :, :, token_idx, 0]    # [questions, layers]
            projection_combined = np.concatenate([projection_truth, projection_lie], axis=0)  # [2*questions, layers]
            
            # Check for valid data
            valid_behavioral = ~np.isnan(behavioral_combined)
            valid_projection = ~np.isnan(projection_combined).any(axis=1)  # Valid if any layer has data
            valid_both = valid_behavioral & valid_projection
            
            if np.sum(valid_both) > 3:  # Need sufficient data points
                behavioral_values = behavioral_combined[valid_both]
                projection_values = projection_combined[valid_both, :]  # [valid_points, layers]
                
                # Remove layers with no variation
                layer_stds = np.nanstd(projection_values, axis=0)
                valid_layers = layer_stds > 1e-10
                
                if np.sum(valid_layers) > 0 and np.std(behavioral_values) > 0:
                    projection_values = projection_values[:, valid_layers]
                    
                    # Calculate correlations for each layer using 2D spearmanr
                    try:
                        correlations, _ = spearmanr(behavioral_values[:, np.newaxis], projection_values, axis=0)
                        # spearmanr returns correlations between first column and all other columns
                        layer_correlations = correlations[0, 1:]  # Skip self-correlation
                        
                        # Calculate mean and std across layers
                        valid_correlations = layer_correlations[~np.isnan(layer_correlations)]
                        if len(valid_correlations) > 0:
                            mean_correlations[probe_idx, token_idx] = np.mean(valid_correlations)
                            if len(valid_correlations) > 1:
                                std_correlations[probe_idx, token_idx] = np.std(valid_correlations)
                            else:
                                std_correlations[probe_idx, token_idx] = 0.0
                    except Exception as e:
                        print(f"Error calculating correlation for probe {probe_idx}, token {token_idx}: {e}")
                        continue
    
    return mean_correlations, std_correlations

def calculate_last_token_layerwise_correlations(behavioral_data, projection_data, orientation):
    """
    Calculate correlations between last token behavioral and each token's projections.
    """
    num_probes, num_questions, num_layers, num_tokens, _ = projection_data.shape
    
    mean_correlations = np.full((num_probes, num_tokens), np.nan)
    std_correlations = np.full((num_probes, num_tokens), np.nan)
    
    for probe_idx in range(num_probes):
        # Find last token for this probe
        valid_tokens = ~np.isnan(behavioral_data[probe_idx, :, :, :]).any(axis=(0,2))
        if not np.any(valid_tokens):
            continue
        last_token_idx = np.where(valid_tokens)[0][-1]
        
        # Get last token behavioral data with orientation
        last_behavioral_truth = behavioral_data[probe_idx, :, last_token_idx, 1] * orientation[probe_idx]
        last_behavioral_lie = behavioral_data[probe_idx, :, last_token_idx, 0] * orientation[probe_idx]
        last_behavioral_combined = np.concatenate([last_behavioral_truth, last_behavioral_lie])
        valid_last_behavioral = ~np.isnan(last_behavioral_combined)
        
        if np.sum(valid_last_behavioral) > 3:
            last_behavioral_values = last_behavioral_combined[valid_last_behavioral]
            
            for token_idx in range(num_tokens):
                # Get projection data for this token
                projection_truth = projection_data[probe_idx, :, :, token_idx, 1]  # [questions, layers]
                projection_lie = projection_data[probe_idx, :, :, token_idx, 0]    # [questions, layers]
                projection_combined = np.concatenate([projection_truth, projection_lie], axis=0)  # [2*questions, layers]
                
                # Only use points where both behavioral and projection data are valid
                valid_projection = ~np.isnan(projection_combined).any(axis=1)
                valid_both = valid_last_behavioral & valid_projection
                
                if np.sum(valid_both) > 3:
                    behavioral_values = last_behavioral_values[valid_both[:len(last_behavioral_values)]]
                    projection_values = projection_combined[valid_both, :]  # [valid_points, layers]
                    
                    # Remove layers with no variation
                    layer_stds = np.nanstd(projection_values, axis=0)
                    valid_layers = layer_stds > 1e-10
                    
                    if np.sum(valid_layers) > 0 and np.std(behavioral_values) > 0:
                        projection_values = projection_values[:, valid_layers]
                        
                        # Calculate correlations for each layer
                        try:
                            correlations, _ = spearmanr(behavioral_values[:, np.newaxis], projection_values, axis=0)
                            layer_correlations = correlations[0, 1:]  # Skip self-correlation
                            
                            # Calculate mean and std across layers
                            valid_correlations = layer_correlations[~np.isnan(layer_correlations)]
                            if len(valid_correlations) > 0:
                                mean_correlations[probe_idx, token_idx] = np.mean(valid_correlations)
                                if len(valid_correlations) > 1:
                                    std_correlations[probe_idx, token_idx] = np.std(valid_correlations)
                                else:
                                    std_correlations[probe_idx, token_idx] = 0.0
                        except Exception as e:
                            print(f"Error calculating last token correlation for probe {probe_idx}, token {token_idx}: {e}")
                            continue
    
    return mean_correlations, std_correlations

# Calculate all correlation types
print("Calculating same-token correlations...")
generic_same_token_mean, generic_same_token_std = calculate_layerwise_correlations(
    behavioral_log_odds, generic_projections, probe_global_orientation)
dedicated_same_token_mean, dedicated_same_token_std = calculate_layerwise_correlations(
    behavioral_log_odds, dedicated_projections, probe_global_orientation)

print("Calculating last-token behavioral correlations...")
generic_last_token_mean, generic_last_token_std = calculate_last_token_layerwise_correlations(
    behavioral_log_odds, generic_projections, probe_global_orientation)
dedicated_last_token_mean, dedicated_last_token_std = calculate_last_token_layerwise_correlations(
    behavioral_log_odds, dedicated_projections, probe_global_orientation)

# Set up plotting
print("Creating plots...")
fig, axes = plt.subplots(5, 2, figsize=(16, 25))

# Color mapping based on SNR
valid_snrs = behavioral_snrs_for_coloring[~np.isnan(behavioral_snrs_for_coloring)]
if len(valid_snrs) > 0:
    snr_ranks = np.argsort(np.argsort(behavioral_snrs_for_coloring))  # Rank ordering
    max_rank = np.nanmax(snr_ranks)
    
    # Normalize ranks to [0.3, 1] for color intensity
    color_intensities = 0.3 + 0.7 * (snr_ranks / max_rank)
    color_intensities[np.isnan(behavioral_snrs_for_coloring)] = 0.5  # Default for NaN
else:
    color_intensities = np.full(num_probe_questions, 0.5)

# Plot correlation data with error bars (top 4 subplots)
correlation_plots = [
    (generic_same_token_mean, generic_same_token_std, 'Generic - Same Token (Mean±Std across layers)', axes[0, 0]),
    (dedicated_same_token_mean, dedicated_same_token_std, 'Dedicated - Same Token (Mean±Std across layers)', axes[0, 1]),
    (generic_last_token_mean, generic_last_token_std, 'Generic - Last Token Behavioral (Mean±Std across layers)', axes[1, 0]),
    (dedicated_last_token_mean, dedicated_last_token_std, 'Dedicated - Last Token Behavioral (Mean±Std across layers)', axes[1, 1])
]

for mean_data, std_data, title, ax in correlation_plots:
    for probe_idx in range(num_probe_questions):
        probe_means = mean_data[probe_idx, :]
        probe_stds = std_data[probe_idx, :]
        valid_tokens = ~np.isnan(probe_means)
        
        if np.any(valid_tokens):
            token_positions = np.arange(1, max_tokens + 1)[valid_tokens]
            valid_means = probe_means[valid_tokens]
            valid_stds = probe_stds[valid_tokens]
            valid_stds[np.isnan(valid_stds)] = 0  # Replace NaN stds with 0
            
            color = plt.cm.viridis(color_intensities[probe_idx])
            ax.errorbar(token_positions, valid_means, yerr=valid_stds, 
                       alpha=0.7, linewidth=2, color=color, capsize=3)
    
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Spearman Correlation')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Plot projection data (averaged across layers for visualization)
projection_plots = [
    ('generic', 'Generic Projections (Mean across layers)', axes[2, 0]),
    ('dedicated', 'Dedicated Projections (Mean across layers)', axes[2, 1])
]

for proj_type, title, ax in projection_plots:
    for probe_idx in range(num_probe_questions):
        # Find valid token range for this probe
        valid_tokens = ~np.isnan(behavioral_log_odds[probe_idx, :, :, :]).any(axis=(0,2))
        if not np.any(valid_tokens):
            continue
        max_valid_token = np.where(valid_tokens)[0][-1] + 1  # Convert to 1-indexed
        
        # Plot truth and lie lines  
        for truth_val, color_base in [(1, 'blue'), (0, 'red')]:
            token_positions = []
            projections = []
            
            for token_idx in range(max_valid_token):
                if proj_type == 'generic':
                    proj_vals = generic_projections[probe_idx, :, :, token_idx, truth_val]
                else:  # dedicated
                    proj_vals = dedicated_projections[probe_idx, :, :, token_idx, truth_val]
                
                # Average across questions and layers
                proj_val = np.nanmean(proj_vals)
                
                if not np.isnan(proj_val):
                    token_positions.append(token_idx + 1)  # Convert back to 1-indexed
                    projections.append(proj_val)
            
            if len(token_positions) > 0:
                # Color based on SNR intensity
                if color_base == 'blue':
                    color = (0, 0, color_intensities[probe_idx])
                else:  # red
                    color = (color_intensities[probe_idx], 0, 0)
                
                ax.plot(token_positions, projections, color=color, alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Projection Value')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

# Plot mean-centered projection data (new row)
projection_centered_plots = [
    ('generic', 'Generic Projections - Mean-Centered (Mean across layers)', axes[3, 0]),
    ('dedicated', 'Dedicated Projections - Mean-Centered (Mean across layers)', axes[3, 1])
]

for proj_type, title, ax in projection_centered_plots:
    for probe_idx in range(num_probe_questions):
        # Find valid token range for this probe
        valid_tokens = ~np.isnan(behavioral_log_odds[probe_idx, :, :, :]).any(axis=(0,2))
        if not np.any(valid_tokens):
            continue
        max_valid_token = np.where(valid_tokens)[0][-1] + 1
        
        # First pass: collect centered values for each token position
        centered_data = {}
        for token_idx in range(max_valid_token):
            if proj_type == 'generic':
                truth_vals = generic_projections[probe_idx, :, :, token_idx, 1]
                lie_vals = generic_projections[probe_idx, :, :, token_idx, 0]
            else:  # dedicated
                truth_vals = dedicated_projections[probe_idx, :, :, token_idx, 1]
                lie_vals = dedicated_projections[probe_idx, :, :, token_idx, 0]
            
            # Average across questions and layers
            truth_mean = np.nanmean(truth_vals)
            lie_mean = np.nanmean(lie_vals)
            
            if not (np.isnan(truth_mean) or np.isnan(lie_mean)):
                overall_mean = (truth_mean + lie_mean) / 2
                centered_data[token_idx] = {
                    'truth': truth_mean - overall_mean,
                    'lie': lie_mean - overall_mean
                }
        
        # Second pass: plot the centered values
        for truth_val, color_base in [(1, 'blue'), (0, 'red')]:
            token_positions = []
            centered_values = []
            
            for token_idx in sorted(centered_data.keys()):
                token_positions.append(token_idx + 1)
                if truth_val == 1:
                    centered_values.append(centered_data[token_idx]['truth'])
                else:
                    centered_values.append(centered_data[token_idx]['lie'])
            
            if len(token_positions) > 0:
                # Color based on SNR intensity
                if color_base == 'blue':
                    color = (0, 0, color_intensities[probe_idx])
                else:  # red
                    color = (color_intensities[probe_idx], 0, 0)
                
                ax.plot(token_positions, centered_values, color=color, alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Mean-Centered Projection Value')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Plot oriented log-odds (bottom row)
# Left subplot: Raw oriented log-odds
ax = axes[4, 0]
for probe_idx in range(num_probe_questions):
    # Find valid token range for this probe
    valid_tokens = ~np.isnan(behavioral_log_odds[probe_idx, :, :, :]).any(axis=(0,2))
    if not np.any(valid_tokens):
        continue
    max_valid_token = np.where(valid_tokens)[0][-1] + 1
    
    # Plot truth and lie lines with probe-global orientation
    for truth_val, color_base in [(1, 'blue'), (0, 'red')]:
        token_positions = []
        log_odds_values = []
        
        for token_idx in range(max_valid_token):
            # Get behavioral data with probe-global orientation
            behavioral_vals = behavioral_log_odds[probe_idx, :, token_idx, truth_val] * probe_global_orientation[probe_idx]
            log_odds_val = np.nanmean(behavioral_vals)
            
            if not np.isnan(log_odds_val):
                token_positions.append(token_idx + 1)
                log_odds_values.append(log_odds_val)
        
        if len(token_positions) > 0:
            # Color based on SNR intensity
            if color_base == 'blue':
                color = (0, 0, color_intensities[probe_idx])
            else:  # red
                color = (color_intensities[probe_idx], 0, 0)
            
            ax.plot(token_positions, log_odds_values, color=color, alpha=0.7, linewidth=2)

ax.set_xlabel('Token Position')
ax.set_ylabel('Oriented Log-Odds')
ax.set_title('Oriented Log-Odds (Higher = More Lie-like)')
ax.grid(True, alpha=0.3)

# Right subplot: Mean-centered oriented log-odds
ax = axes[4, 1]
for probe_idx in range(num_probe_questions):
    # Find valid token range for this probe
    valid_tokens = ~np.isnan(behavioral_log_odds[probe_idx, :, :, :]).any(axis=(0,2))
    if not np.any(valid_tokens):
        continue
    max_valid_token = np.where(valid_tokens)[0][-1] + 1
    
    # First pass: collect centered values for each token position
    centered_data = {}
    for token_idx in range(max_valid_token):
        truth_vals = behavioral_log_odds[probe_idx, :, token_idx, 1] * probe_global_orientation[probe_idx]
        lie_vals = behavioral_log_odds[probe_idx, :, token_idx, 0] * probe_global_orientation[probe_idx]
        
        truth_mean = np.nanmean(truth_vals)
        lie_mean = np.nanmean(lie_vals)
        
        if not (np.isnan(truth_mean) or np.isnan(lie_mean)):
            overall_mean = (truth_mean + lie_mean) / 2
            centered_data[token_idx] = {
                'truth': truth_mean - overall_mean,
                'lie': lie_mean - overall_mean
            }
    
    # Second pass: plot the centered values
    for truth_val, color_base in [(1, 'blue'), (0, 'red')]:
        token_positions = []
        centered_values = []
        
        for token_idx in sorted(centered_data.keys()):
            token_positions.append(token_idx + 1)
            if truth_val == 1:
                centered_values.append(centered_data[token_idx]['truth'])
            else:
                centered_values.append(centered_data[token_idx]['lie'])
        
        if len(token_positions) > 0:
            # Color based on SNR intensity
            if color_base == 'blue':
                color = (0, 0, color_intensities[probe_idx])
            else:  # red
                color = (color_intensities[probe_idx], 0, 0)
            
            ax.plot(token_positions, centered_values, color=color, alpha=0.7, linewidth=2)

ax.set_xlabel('Token Position')
ax.set_ylabel('Mean-Centered Log-Odds')
ax.set_title('Mean-Centered Log-Odds (Relative Discrimination)')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()

# Save plot
plot_path = os.path.join(save_base, 'layerwise_lie_detection_analysis.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")
plt.show()

print("\nAnalysis complete!")
print(f"Behavioral SNRs for coloring: min={np.nanmin(behavioral_snrs_for_coloring):.3f}, max={np.nanmax(behavioral_snrs_for_coloring):.3f}")
print(f"Number of probe questions with valid SNR: {np.sum(~np.isnan(behavioral_snrs_for_coloring))}")
print(f"Mean correlations across all valid probe-token combinations:")
for name, data in [("Generic Same Token", generic_same_token_mean), 
                   ("Dedicated Same Token", dedicated_same_token_mean),
                   ("Generic Last Token", generic_last_token_mean),
                   ("Dedicated Last Token", dedicated_last_token_mean)]:
    valid_corrs = data[~np.isnan(data)]
    if len(valid_corrs) > 0:
        print(f"  {name}: {np.mean(valid_corrs):.3f} ± {np.std(valid_corrs):.3f}")