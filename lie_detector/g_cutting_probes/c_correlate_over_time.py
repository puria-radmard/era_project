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

expected_lens = (~np.isnan(all_lie_dedicated_projections)).sum(-1)[0,:,0]


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

# Subsample questions
trainable_questions_idxs = trainable_questions_idxs[:10]            
print('REDUCING trainable_questions_idxs TO JUST SPORTS QUESTIONS!!')
print(f"Using {len(trainable_questions_idxs)} initial questions")

# Filter behavioral data
truncated_df_filtered = truncated_df[truncated_df['question_idx'].isin(trainable_questions_idxs)]

# Calculate projection SNRs
print("Calculating projection SNRs... !!! DOING MEAN DIFF, NOT SNR !!!")
diff_generic = all_lie_projections - all_truth_projections  # [questions, probe_questions, layers, tokens]
mean_diff_generic = np.nanmean(diff_generic, axis=0)  # [probe_questions, layers, tokens]
std_diff_generic = np.nanstd(diff_generic, axis=0)
std_diff_generic[std_diff_generic < 1e-10] = np.nan
snr_generic = np.abs(mean_diff_generic)#  / std_diff_generic

diff_dedicated = all_lie_dedicated_projections - all_truth_dedicated_projections
mean_diff_dedicated = np.nanmean(diff_dedicated, axis=0)
std_diff_dedicated = np.nanstd(diff_dedicated, axis=0)
std_diff_dedicated[std_diff_dedicated < 1e-10] = np.nan
snr_dedicated = np.abs(mean_diff_dedicated)#  / std_diff_dedicated

# Determine dimensions for structured arrays
num_probe_questions = len(probe_questions)
max_tokens = truncated_df_filtered['token_position'].max()
print(f"Dimensions: {num_probe_questions} probe questions, {num_layers} layers, {max_tokens} max tokens")

# Initialize structured arrays with NaN
behavioral_snrs_structured = np.full((num_probe_questions, max_tokens), np.nan)
projection_snrs_generic_structured = np.full((num_probe_questions, num_layers, max_tokens), np.nan)
projection_snrs_dedicated_structured = np.full((num_probe_questions, num_layers, max_tokens), np.nan)

# Populate structured arrays
print("Populating structured arrays...")
for probe_idx in range(num_probe_questions):
    probe_data = truncated_df_filtered[truncated_df_filtered['probe_question_idx'] == probe_idx]

    for token_pos in probe_data['token_position'].unique():

        token_data = probe_data[probe_data['token_position'] == token_pos]
        
        if len(token_data) > 0:
            # Calculate behavioral SNR for this token position
            lie_logodds = token_data[token_data['truth'] == 0]['log_odds'].values
            truth_logodds = token_data[token_data['truth'] == 1]['log_odds'].values
            
            if len(lie_logodds) > 0 and len(truth_logodds) > 0:
                mean_diff = np.mean(lie_logodds) - np.mean(truth_logodds)
                pooled_std = np.sqrt((np.var(lie_logodds) + np.var(truth_logodds)) / 2)
                
                if pooled_std > 1e-10:
                    behavioral_snr = np.abs(mean_diff)#  / pooled_std
                    token_idx = token_pos - 1  # Convert to 0-indexed

                    # Store behavioral SNR
                    if token_idx < max_tokens:
                        behavioral_snrs_structured[probe_idx, token_idx] = behavioral_snr
                        
                        # Store projection SNRs for all layers
                        if token_idx < snr_generic.shape[2]:
                            for layer in range(num_layers):
                                generic_snr = snr_generic[probe_idx, layer, token_idx]
                                dedicated_snr = snr_dedicated[probe_idx, layer, token_idx]
                                
                                if not np.isnan(generic_snr):
                                    projection_snrs_generic_structured[probe_idx, layer, token_idx] = generic_snr
                                if not np.isnan(dedicated_snr):
                                    projection_snrs_dedicated_structured[probe_idx, layer, token_idx] = dedicated_snr

print("Structured arrays populated!")

# Prepare data for top row (full question vs last token)
print("Preparing data for top row plots...")
behavioral_snrs_full = []
projection_snrs_generic_full = []
projection_snrs_dedicated_full = []
layers_list_full = []

for probe_idx in range(num_probe_questions):
    # Find the last valid token for this probe question
    valid_tokens = ~np.isnan(behavioral_snrs_structured[probe_idx, :])
    if np.any(valid_tokens):
        last_token_idx = np.where(valid_tokens)[0][-1]
        behavioral_snr_full = behavioral_snrs_structured[probe_idx, last_token_idx]
        
        for layer in range(num_layers):
            generic_snr = projection_snrs_generic_structured[probe_idx, layer, last_token_idx]
            dedicated_snr = projection_snrs_dedicated_structured[probe_idx, layer, last_token_idx]
            
            if not (np.isnan(generic_snr) or np.isnan(dedicated_snr)):
                behavioral_snrs_full.append(behavioral_snr_full)
                projection_snrs_generic_full.append(generic_snr)
                projection_snrs_dedicated_full.append(dedicated_snr)
                layers_list_full.append(layer)

# Prepare data for middle row (all matching tokens)
print("Preparing data for middle row plots...")
behavioral_snrs_all = []
projection_snrs_generic_all = []
projection_snrs_dedicated_all = []
layers_list_all = []

for probe_idx in range(num_probe_questions):
    for token_idx in range(max_tokens):
        behavioral_snr = behavioral_snrs_structured[probe_idx, token_idx]
        if not np.isnan(behavioral_snr):
            for layer in range(num_layers):
                generic_snr = projection_snrs_generic_structured[probe_idx, layer, token_idx]
                dedicated_snr = projection_snrs_dedicated_structured[probe_idx, layer, token_idx]
                
                if not (np.isnan(generic_snr) or np.isnan(dedicated_snr)):
                    behavioral_snrs_all.append(behavioral_snr)
                    projection_snrs_generic_all.append(generic_snr)
                    projection_snrs_dedicated_all.append(dedicated_snr)
                    layers_list_all.append(layer)

# Prepare data for bottom row (Spearman correlations across tokens for each probe question and layer)
print("Calculating Spearman correlations for bottom row...")
spearman_correlations_generic = np.full((num_probe_questions, num_layers), np.nan)
spearman_correlations_dedicated = np.full((num_probe_questions, num_layers), np.nan)

for probe_idx in range(num_probe_questions):
    behavioral_tokens = behavioral_snrs_structured[probe_idx, :]
    valid_mask = ~np.isnan(behavioral_tokens)
    
    if np.sum(valid_mask) > 2:  # Need at least 3 points for meaningful correlation
        behavioral_valid = behavioral_tokens[valid_mask]
        
        for layer in range(num_layers):
            # Generic correlations
            projection_tokens_generic = projection_snrs_generic_structured[probe_idx, layer, :]
            projection_valid_generic = projection_tokens_generic[valid_mask]
            
            if not np.any(np.isnan(projection_valid_generic)) and np.std(projection_valid_generic) > 0:
                corr, p_val = spearmanr(behavioral_valid, projection_valid_generic)
                if not np.isnan(corr):
                    spearman_correlations_generic[probe_idx, layer] = corr
            
            # Dedicated correlations
            projection_tokens_dedicated = projection_snrs_dedicated_structured[probe_idx, layer, :]
            projection_valid_dedicated = projection_tokens_dedicated[valid_mask]
            
            if not np.any(np.isnan(projection_valid_dedicated)) and np.std(projection_valid_dedicated) > 0:
                corr, p_val = spearmanr(behavioral_valid, projection_valid_dedicated)
                if not np.isnan(corr):
                    spearman_correlations_dedicated[probe_idx, layer] = corr

# Convert to numpy arrays for plotting
behavioral_snrs_full = np.array(behavioral_snrs_full)
projection_snrs_generic_full = np.array(projection_snrs_generic_full)
projection_snrs_dedicated_full = np.array(projection_snrs_dedicated_full)
layers_array_full = np.array(layers_list_full)

behavioral_snrs_all = np.array(behavioral_snrs_all)
projection_snrs_generic_all = np.array(projection_snrs_generic_all)
projection_snrs_dedicated_all = np.array(projection_snrs_dedicated_all)
layers_array_all = np.array(layers_list_all)

# Create 3x2 plot
print("Creating plots...")
fig, ((ax_top_left, ax_top_right), (ax_mid_left, ax_mid_right), (ax_bot_left, ax_bot_right)) = plt.subplots(3, 2, figsize=(16, 18))

# Set up colormap
layer_colors = plt.cm.magma(np.linspace(0.1, 0.9, num_layers))

# Function to add per-layer regression lines
def add_layer_regressions(ax, x_data, y_data, layers_data):
    for layer in range(num_layers):
        layer_mask = layers_data == layer
        if np.sum(layer_mask) > 1:
            x_layer = x_data[layer_mask]
            y_layer = y_data[layer_mask]
            
            if len(x_layer) > 1 and np.std(x_layer) > 0:
                coeffs = np.polyfit(x_layer, y_layer, 1)
                x_line = np.linspace(x_layer.min(), x_layer.max(), 100)
                y_line = np.polyval(coeffs, x_line)
                ax.plot(x_line, y_line, color=layer_colors[layer], alpha=0.8, linewidth=2)

# Top row: Full question behavioral vs last token projections
scatter1 = ax_top_left.scatter(behavioral_snrs_full, projection_snrs_generic_full, 
                              c=layers_array_full, cmap='magma', alpha=0.7, s=30)
add_layer_regressions(ax_top_left, behavioral_snrs_full, projection_snrs_generic_full, layers_array_full)
ax_top_left.set_xlabel('Behavioral SNR (Full Question)')
ax_top_left.set_ylabel('Generic Projection SNR (Last Token)')
ax_top_left.set_title('Generic: Full Behavioral vs Last Token Projection')
ax_top_left.grid(True, alpha=0.3)

scatter2 = ax_top_right.scatter(behavioral_snrs_full, projection_snrs_dedicated_full, 
                               c=layers_array_full, cmap='magma', alpha=0.7, s=30)
add_layer_regressions(ax_top_right, behavioral_snrs_full, projection_snrs_dedicated_full, layers_array_full)
ax_top_right.set_xlabel('Behavioral SNR (Full Question)')
ax_top_right.set_ylabel('Dedicated Projection SNR (Last Token)')
ax_top_right.set_title('Dedicated: Full Behavioral vs Last Token Projection')
ax_top_right.grid(True, alpha=0.3)

# Middle row: All token positions (matching behavioral and projection)
scatter3 = ax_mid_left.scatter(behavioral_snrs_all, projection_snrs_generic_all, 
                              c=layers_array_all, cmap='magma', alpha=0.7, s=30)
add_layer_regressions(ax_mid_left, behavioral_snrs_all, projection_snrs_generic_all, layers_array_all)
ax_mid_left.set_xlabel('Behavioral SNR (All Tokens)')
ax_mid_left.set_ylabel('Generic Projection SNR (Matching Tokens)')
ax_mid_left.set_title('Generic: All Token Positions (Matched)')
ax_mid_left.grid(True, alpha=0.3)

scatter4 = ax_mid_right.scatter(behavioral_snrs_all, projection_snrs_dedicated_all, 
                               c=layers_array_all, cmap='magma', alpha=0.7, s=30)
add_layer_regressions(ax_mid_right, behavioral_snrs_all, projection_snrs_dedicated_all, layers_array_all)
ax_mid_right.set_xlabel('Behavioral SNR (All Tokens)')
ax_mid_right.set_ylabel('Dedicated Projection SNR (Matching Tokens)')
ax_mid_right.set_title('Dedicated: All Token Positions (Matched)')
ax_mid_right.grid(True, alpha=0.3)

# Bottom row: Spearman correlations across tokens (one line per probe question)
for probe_idx in range(num_probe_questions):
    # Generic correlations
    generic_corrs = spearman_correlations_generic[probe_idx, :]
    valid_layers = ~np.isnan(generic_corrs)
    if np.any(valid_layers):
        ax_bot_left.plot(np.arange(num_layers)[valid_layers], generic_corrs[valid_layers], 
                        alpha=0.7, linewidth=2, label=f'Probe {probe_idx}' if probe_idx < 5 else '')
    
    # Dedicated correlations
    dedicated_corrs = spearman_correlations_dedicated[probe_idx, :]
    valid_layers = ~np.isnan(dedicated_corrs)
    if np.any(valid_layers):
        ax_bot_right.plot(np.arange(num_layers)[valid_layers], dedicated_corrs[valid_layers], 
                         alpha=0.7, linewidth=2, label=f'Probe {probe_idx}' if probe_idx < 5 else '')

ax_bot_left.set_xlabel('Layer')
ax_bot_left.set_ylabel('Spearman Correlation (across tokens)')
ax_bot_left.set_title('Generic: Per-Probe Correlations by Layer')
ax_bot_left.grid(True, alpha=0.3)
ax_bot_left.legend()

ax_bot_right.set_xlabel('Layer')
ax_bot_right.set_ylabel('Spearman Correlation (across tokens)')
ax_bot_right.set_title('Dedicated: Per-Probe Correlations by Layer')
ax_bot_right.grid(True, alpha=0.3)
ax_bot_right.legend()

# Add shared colorbar for top and middle rows
cbar = fig.colorbar(scatter1, ax=[ax_top_left, ax_top_right, ax_mid_left, ax_mid_right], 
                   label='Layer', shrink=0.6)
cbar.set_ticks(np.arange(0, num_layers, max(1, num_layers//10)))

plt.tight_layout()

# Save plot
plot_path = os.path.join(save_base, 'three_row_structured_analysis.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")
plt.show()

# Print summary statistics
print("\nSummary:")
print(f"Full question data points: {len(behavioral_snrs_full)}")
print(f"All token data points: {len(behavioral_snrs_all)}")
print(f"Probe questions with correlations: {np.sum(~np.isnan(spearman_correlations_generic).any(axis=1))}")
print(f"Structured arrays shape:")
print(f"  Behavioral: {behavioral_snrs_structured.shape}")
print(f"  Generic projections: {projection_snrs_generic_structured.shape}")
print(f"  Dedicated projections: {projection_snrs_dedicated_structured.shape}")