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
    question_idx_in_original = trainable_questions_idxs.index(question_idx) if question_idx in trainable_questions_idxs else None
    if question_idx_in_original is not None:
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

# Calculate orientation decisions
print("Calculating orientation decisions...")

# Token-specific orientations for top row (same token correlations)
token_specific_orientation = np.full((num_probe_questions, max_tokens), 1.0)  # 1 = no flip, -1 = flip

# Probe-global orientations for other rows (based on final token)
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
    
    # Calculate token-specific orientations for each token
    for token_idx in range(max_tokens):
        truth_vals = behavioral_log_odds[probe_idx, :, token_idx, 1]
        lie_vals = behavioral_log_odds[probe_idx, :, token_idx, 0]
        
        valid_truth = truth_vals[~np.isnan(truth_vals)]
        valid_lie = lie_vals[~np.isnan(lie_vals)]
        
        if len(valid_truth) > 0 and len(valid_lie) > 0:
            mean_truth = np.mean(valid_truth)
            mean_lie = np.mean(valid_lie)
            
            # If truth has higher log-odds, flip so lie has higher
            if mean_truth > mean_lie:
                token_specific_orientation[probe_idx, token_idx] = -1.0

# Calculate behavioral SNR for color coding (based on last token, with probe-global orientation)
print("Calculating behavioral SNR for color coding...")
behavioral_snrs_for_coloring = np.full(num_probe_questions, np.nan)

for probe_idx in range(num_probe_questions):
    # Find the last token with data
    valid_tokens = ~np.isnan(behavioral_log_odds[probe_idx, :, :, :]).any(axis=(0,2))
    if np.any(valid_tokens):
        last_token_idx = np.where(valid_tokens)[0][-1]
        
        # Get oriented values
        truth_vals = behavioral_log_odds[probe_idx, :, last_token_idx, 1] * probe_global_orientation[probe_idx]
        lie_vals = behavioral_log_odds[probe_idx, :, last_token_idx, 0] * probe_global_orientation[probe_idx]
        
        valid_truth = truth_vals[~np.isnan(truth_vals)]
        valid_lie = lie_vals[~np.isnan(lie_vals)]
        
        if len(valid_truth) > 0 and len(valid_lie) > 0:
            mean_diff = np.abs(np.mean(lie_vals) - np.mean(valid_truth))
            pooled_std = np.sqrt((np.var(valid_lie) + np.var(valid_truth)) / 2)
            
            if pooled_std > 1e-10:
                behavioral_snrs_for_coloring[probe_idx] = mean_diff / pooled_std

# Calculate correlations for top row (same token, token-specific orientation)
print("Calculating same-token correlations...")
correlation_data_same_token = {
    'generic': np.full((num_probe_questions, max_tokens), np.nan),
    'dedicated': np.full((num_probe_questions, max_tokens), np.nan)
}

for probe_idx in range(num_probe_questions):
    for token_idx in range(max_tokens):
        # Get behavioral data with token-specific orientation
        behavioral_truth = behavioral_log_odds[probe_idx, :, token_idx, 1] * token_specific_orientation[probe_idx, token_idx]
        behavioral_lie = behavioral_log_odds[probe_idx, :, token_idx, 0] * token_specific_orientation[probe_idx, token_idx]
        
        # Combine truth and lie into single arrays
        behavioral_combined = np.concatenate([behavioral_truth, behavioral_lie])
        valid_behavioral = ~np.isnan(behavioral_combined)
        
        if np.sum(valid_behavioral) > 3:  # Need sufficient data points
            behavioral_values = behavioral_combined[valid_behavioral]
            
            # Get corresponding projection data (averaged across layers)
            generic_truth = np.nanmean(generic_projections[probe_idx, :, :, token_idx, 1], axis=1)
            generic_lie = np.nanmean(generic_projections[probe_idx, :, :, token_idx, 0], axis=1)
            generic_combined = np.concatenate([generic_truth, generic_lie])
            
            dedicated_truth = np.nanmean(dedicated_projections[probe_idx, :, :, token_idx, 1], axis=1)
            dedicated_lie = np.nanmean(dedicated_projections[probe_idx, :, :, token_idx, 0], axis=1)
            dedicated_combined = np.concatenate([dedicated_truth, dedicated_lie])
            
            generic_values = generic_combined[valid_behavioral]
            dedicated_values = dedicated_combined[valid_behavioral]
            
            # Calculate correlations
            if len(behavioral_values) > 2 and np.std(behavioral_values) > 0:
                if np.std(generic_values) > 0:
                    corr, _ = spearmanr(behavioral_values, generic_values)
                    if not np.isnan(corr):
                        correlation_data_same_token['generic'][probe_idx, token_idx] = corr
                
                if np.std(dedicated_values) > 0:
                    corr, _ = spearmanr(behavioral_values, dedicated_values)
                    if not np.isnan(corr):
                        correlation_data_same_token['dedicated'][probe_idx, token_idx] = corr

# Calculate correlations for last token behavioral (probe-global orientation)
print("Calculating last-token behavioral correlations...")
correlation_data_last_token = {
    'generic': np.full((num_probe_questions, max_tokens), np.nan),
    'dedicated': np.full((num_probe_questions, max_tokens), np.nan)
}

for probe_idx in range(num_probe_questions):
    # Find last token for this probe
    valid_tokens = ~np.isnan(behavioral_log_odds[probe_idx, :, :, :]).any(axis=(0,2))
    if not np.any(valid_tokens):
        continue
    last_token_idx = np.where(valid_tokens)[0][-1]
    
    # Get last token behavioral data with probe-global orientation
    last_behavioral_truth = behavioral_log_odds[probe_idx, :, last_token_idx, 1] * probe_global_orientation[probe_idx]
    last_behavioral_lie = behavioral_log_odds[probe_idx, :, last_token_idx, 0] * probe_global_orientation[probe_idx]
    last_behavioral_combined = np.concatenate([last_behavioral_truth, last_behavioral_lie])
    valid_last_behavioral = ~np.isnan(last_behavioral_combined)
    
    if np.sum(valid_last_behavioral) > 3:
        last_behavioral_values = last_behavioral_combined[valid_last_behavioral]
        
        for token_idx in range(max_tokens):
            # Get projection data for this token
            generic_truth = np.nanmean(generic_projections[probe_idx, :, :, token_idx, 1], axis=1)
            generic_lie = np.nanmean(generic_projections[probe_idx, :, :, token_idx, 0], axis=1)
            generic_combined = np.concatenate([generic_truth, generic_lie])
            
            dedicated_truth = np.nanmean(dedicated_projections[probe_idx, :, :, token_idx, 1], axis=1)
            dedicated_lie = np.nanmean(dedicated_projections[probe_idx, :, :, token_idx, 0], axis=1)
            dedicated_combined = np.concatenate([dedicated_truth, dedicated_lie])
            
            # Only use points where both behavioral and projection data are valid
            valid_both = valid_last_behavioral & ~np.isnan(generic_combined) & ~np.isnan(dedicated_combined)
            
            if np.sum(valid_both) > 2:
                behavioral_values = last_behavioral_values[valid_both[:len(last_behavioral_values)]]
                import pdb; pdb.set_trace()
                generic_values = generic_combined[valid_both]
                dedicated_values = dedicated_combined[valid_both]
                
                # Calculate correlations
                if len(behavioral_values) > 2 and np.std(behavioral_values) > 0:
                    if np.std(generic_values) > 0:
                        corr, _ = spearmanr(behavioral_values, generic_values)
                        if not np.isnan(corr):
                            correlation_data_last_token['generic'][probe_idx, token_idx] = corr
                    
                    if np.std(dedicated_values) > 0:
                        corr, _ = spearmanr(behavioral_values, dedicated_values)
                        if not np.isnan(corr):
                            correlation_data_last_token['dedicated'][probe_idx, token_idx] = corr

# Set up plotting
print("Creating plots...")
fig, axes = plt.subplots(4, 2, figsize=(16, 20))

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

# Plot correlation data (top 4 subplots)
correlation_plots = [
    (correlation_data_same_token['generic'], 'Generic - Same Token', axes[0, 0]),
    (correlation_data_same_token['dedicated'], 'Dedicated - Same Token', axes[0, 1]),
    (correlation_data_last_token['generic'], 'Generic - Last Token Behavioral', axes[1, 0]),
    (correlation_data_last_token['dedicated'], 'Dedicated - Last Token Behavioral', axes[1, 1])
]

for corr_data, title, ax in correlation_plots:
    for probe_idx in range(num_probe_questions):
        probe_corrs = corr_data[probe_idx, :]
        valid_tokens = ~np.isnan(probe_corrs)
        
        if np.any(valid_tokens):
            token_positions = np.arange(1, max_tokens + 1)[valid_tokens]
            valid_corrs = probe_corrs[valid_tokens]
            
            ax.plot(token_positions, valid_corrs, alpha=0.7, linewidth=2, 
                   color=plt.cm.viridis(color_intensities[probe_idx]))
    
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Spearman Correlation')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Plot projection data (second row)
projection_plots = [
    ('generic', 'Generic Projections', axes[2, 0]),
    ('dedicated', 'Dedicated Projections', axes[2, 1])
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

# Plot oriented log-odds (bottom row)
# Left subplot: Raw oriented log-odds (probe-global orientation)
ax = axes[3, 0]
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

# Right subplot: Mean-centered oriented log-odds (probe-global orientation)
ax = axes[3, 1]
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
plot_path = os.path.join(save_base, 'enhanced_lie_detection_analysis.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")
plt.show()

print("\nAnalysis complete!")
print(f"Behavioral SNRs for coloring: min={np.nanmin(behavioral_snrs_for_coloring):.3f}, max={np.nanmax(behavioral_snrs_for_coloring):.3f}")
print(f"Number of probe questions with valid SNR: {np.sum(~np.isnan(behavioral_snrs_for_coloring))}")
print(f"Token-specific orientations (first 5 probes, first 10 tokens):")
print(token_specific_orientation[:5, :10])
print(f"Probe-global orientations (first 10 probes): {probe_global_orientation[:10]}")