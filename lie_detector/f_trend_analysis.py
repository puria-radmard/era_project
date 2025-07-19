#!/usr/bin/env python3

import json
import numpy as np
import sys
import os
from scipy import stats
from util.util import YamlConfig

# Load config
config_path = sys.argv[1]
args = YamlConfig(config_path)

# Set up paths
save_path = args.save_path
questions_data_name = args.questions_data_name
prompt_index = args.prompt_idx
sorting_key = args.icl_sorting_key

probe_file_name = args.probe_file_name
probe_response_type = args.probe_response_type

# Define paths
context_results_path = f"{save_path}/in_context_learning/{probe_file_name}/{probe_response_type}/{questions_data_name}/prompt{prompt_index}/{sorting_key}/context_effect_results.json"
data_path = f"{save_path}/activation_discovery/contextual/{probe_file_name}/{probe_response_type}/{questions_data_name}/prompt{prompt_index}"
prompted_activations_projections_path = os.path.join(data_path, 'prompted_projection_along_average_lie_vector.npy')
contextual_activations_projections_path = os.path.join(data_path, 'contextual_projection_along_average_lie_vector.npy')

# Load all data
print("Loading data...")
with open(context_results_path, 'r') as f:
    context_results = json.load(f)

prompted_activations_projections = np.load(prompted_activations_projections_path, allow_pickle=True).item()
contextual_activations_projections = np.load(contextual_activations_projections_path, allow_pickle=True)

# Get context lengths from the behavioral data
context_lengths = args.context_lengths
context_lengths_activations = args.context_lengths_activations

n_layers = len(prompted_activations_projections)  # Should be 32


pairs_to_compare = [
    ('top_lie_shuffled_together', 'top_truth_shuffled_together'),
    ('top_lie_shuffled_together', 'top_questions_random_answers'),
    ('top_questions_random_answers', 'top_truth_shuffled_together'),
]

more_pairs_to_compare = pairs_to_compare + [

#     ('top_lie_shuffled_together', 'top_questions_random_answers'),
#     ('top_questions_random_answers', 'top_truth_shuffled_together'),

#     ('top_lie_shuffled_together', 'top_lie_questions_shuffled'),
#     ('top_lie_questions_shuffled', 'top_truth_shuffled_together'),

#     ('top_lie_shuffled_together', 'top_truth_questions_shuffled'),
#     ('top_truth_questions_shuffled', 'top_truth_shuffled_together'),
# ]
]



prob_diff_mean_diff = np.zeros([len(more_pairs_to_compare), len(context_lengths)])
prob_diff_ttest_stats = np.zeros([len(more_pairs_to_compare), len(context_lengths)])
prob_diff_p_values = np.zeros([len(more_pairs_to_compare), len(context_lengths)])

num_control_directions = 16

activations_mean_diffs = np.zeros([len(pairs_to_compare), len(context_lengths_activations), ])
activations_ttest_stats = np.zeros([len(pairs_to_compare), len(context_lengths_activations), ])
activations_p_values = np.zeros([len(pairs_to_compare), len(context_lengths_activations), ])
control_activations_mean_diffs = np.zeros([len(pairs_to_compare), len(context_lengths_activations), num_control_directions])
control_activations_ttest_stats = np.zeros([len(pairs_to_compare), len(context_lengths_activations), num_control_directions])




print(f"Found {len(context_lengths)} context lengths: {context_lengths}")
print(f"Found {n_layers} layers")
print(f"Available context types in behavioral data: {list(context_results.keys())}")

# Iterate over context lengths
for i_N, N in enumerate(context_lengths):
    print(f"\n{'='*60}")
    print(f"Processing context length N = {N}")
    print(f"{'='*60}")
    
    # Extract behavioral data for this N
    behavioral_data = {}
    for context_type, results_list in context_results.items():
        for result in results_list:
            if result['context_length'] == N:
                behavioral_data[context_type] = result
                break
    
    print(f"Found behavioral data for {len(behavioral_data)} context types at N={N}")
    print(f"Behavioural context types: {list(behavioral_data.keys())}")

    # ==========================================
    # BEHAVIOURAL ANALYSIS SPACE - ALL DATA ACCESSIBLE HERE
    # ==========================================

    # [behavioural questions]
    if N > 0:
        for i_pair, (pair_context_1, pair_context_2) in enumerate(more_pairs_to_compare):

            question_truth_probs = np.array(behavioral_data[pair_context_1]['question_truth_probs'])
            question_lie_probs = np.array(behavioral_data[pair_context_2]['question_truth_probs'])
            try:
                ttest_stat_beh, p_value_beh = stats.ttest_rel(question_lie_probs, question_truth_probs)
                question_prob_diffs = question_lie_probs - question_truth_probs
                prob_mean_diff = np.mean(question_prob_diffs)#  / np.std(question_prob_diffs)

                prob_diff_mean_diff[i_pair, i_N] = prob_mean_diff
                prob_diff_ttest_stats[i_pair, i_N] = ttest_stat_beh
                prob_diff_p_values[i_pair, i_N] = p_value_beh
            except ValueError:
                prob_diff_ttest_stats[:, i_N] = 0.0
                prob_diff_p_values[:, i_N] = np.nan

    else:
        prob_diff_ttest_stats[:, i_N] = 0.0
        prob_diff_p_values[:, i_N] = np.nan
    



for i_N, N in enumerate(context_lengths_activations):
    
    # Iterate over layers
    for i_layer, layer in enumerate(range(n_layers)):
        print(f"\nProcessing layer {layer}")
        
        # Extract projection data from prompted (no context) condition
        prompted_data = prompted_activations_projections[layer]
        prompted_lie_projs = prompted_data['lie_projs']  # [test_questions]
        prompted_truth_projs = prompted_data['truth_projs']  # [test_questions]
        
        # Extract contextual projection data for this N and layer
        contextual_activations_data = {}
        control_contextual_activations_data = {}
        for entry in contextual_activations_projections:
            if entry['context_length'] == N:
                context_type = entry['context_type']
                test_context_projections = entry['projections'][:, layer]  # [test_questions] for this layer
                control_test_context_projections = entry['control_projections'][:, :, layer]  # [num controls, test_questions] for this layer
                contextual_activations_data[context_type] = test_context_projections
                control_contextual_activations_data[context_type] = control_test_context_projections
        
        print(f"Found contextual data for {len(contextual_activations_data)} context types at N={N}, layer={layer}")
        print(f"Contextual context types: {list(contextual_activations_data.keys())}")
        
        # ==========================================
        # ACTIVATIONS ANALYSIS SPACE - ALL DATA ACCESSIBLE HERE
        # ==========================================
        
        # Available variables:
        # - N: current context length (int)
        # - layer: current layer (int, 0-31)
        # - behavioral_data: dict with context types as keys, behavioral results as values
        #   * Keys: context types (e.g., 'top_lie_shuffled_together', 'top_truth_shuffled_together', etc.)
        #   * Values: dict with fields like 'mean_truth_prob', 'question_truth_probs', etc.
        # - lie_projs: projection values for lie condition (no context) [test_questions]
        # - truth_projs: projection values for truth condition (no context) [test_questions]
        # - activations_data: dict with context types as keys, projection arrays as values
        #   * Keys: context types (e.g., 'top_lie_shuffled_together', 'top_truth_shuffled_together', etc.)
        #   * Values: projection arrays [test_questions] for this specific layer
        
        # Example access patterns:
        # - behavioral_data['top_lie_shuffled_together']['mean_truth_prob']
        # - behavioral_data['top_truth_shuffled_together']['question_truth_probs']
        # - activations_data['top_lie_shuffled_together']  # projections for lie context
        # - activations_data['top_truth_shuffled_together']  # projections for truth context
        # - lie_projs  # no-context lie projections
        # - truth_projs  # no-context truth projections


        for i_pair, (pair_context_1, pair_context_2) in enumerate(pairs_to_compare):

            contextural_projection_1 = contextual_activations_data[pair_context_1]
            contextual_projection_2 = contextual_activations_data[pair_context_2]
            control_contextual_projection_1 = control_contextual_activations_data[pair_context_1]
            control_contextual_projection_2 = control_contextual_activations_data[pair_context_2]

            t_stat_proj, p_value_proj = stats.ttest_rel(contextural_projection_1, contextual_projection_2)                     # XXX replace with a better test
            question_contextual_diffs = contextural_projection_1 - contextual_projection_2
            contextual_mean_diff = np.mean(question_contextual_diffs)# / np.std(question_contextual_diffs)

            t_stat_control_proj, p_value_control_proj = stats.ttest_rel(control_contextual_projection_1, control_contextual_projection_2)                     # XXX replace with a better test
            question_control_contextual_diffs = control_contextual_projection_1 - control_contextual_projection_2
            control_contextual_mean_diff = np.mean(question_control_contextual_diffs, 0)# / np.std(question_control_contextual_diffs, 0)

            activations_mean_diffs[i_pair, i_N] = contextual_mean_diff
            activations_ttest_stats[i_pair, i_N] = t_stat_proj
            activations_p_values[i_pair, i_N] = p_value_proj

            control_activations_mean_diffs[i_pair, i_N] = control_contextual_mean_diff
            control_activations_ttest_stats[i_pair, i_N] = t_stat_control_proj


import matplotlib.pyplot as plt
import numpy as np
import os

# Get output directory
output_dir = os.path.dirname(context_results_path)

# ==============================================
# BEHAVIORAL VISUALIZATION
# ==============================================

fig_beh, ax_beh = plt.subplots(1, 2, figsize=(15, 10))

# Colors for different pairs
colors = plt.cm.tab10(np.linspace(0, 1, len(more_pairs_to_compare)))

for i_pair, (pair_context_1, pair_context_2) in enumerate(more_pairs_to_compare):
    # Line style: first pair solid, others dotted
    linestyle = '-' if i_pair == 0 else '--'
    
    # Plot ttest statistics
    ax_beh[0].plot(context_lengths, prob_diff_ttest_stats[i_pair, :], 
                color=colors[i_pair], linestyle=linestyle, marker='o', 
                label=f'{pair_context_1} vs {pair_context_2}', linewidth=2, markersize=6)

    # Plot mean diff
    ax_beh[1].plot(context_lengths, prob_diff_mean_diff[i_pair, :], 
                color=colors[i_pair], linestyle=linestyle, marker='o', 
                label=f'{pair_context_1} vs {pair_context_2}', linewidth=2, markersize=6)
    
    # Add significance stars
    for i_N, (N, p_val) in enumerate(zip(context_lengths, prob_diff_p_values[i_pair, :])):
        if p_val < 0.05 / 3 and not np.isnan(p_val):
            y_pos = prob_diff_ttest_stats[i_pair, i_N]
            ax_beh[0].scatter(N, y_pos, marker='*', s=100, color=colors[i_pair], 
                          edgecolors='black', linewidth=0.5, zorder=10)

ax_beh[0].set_xlabel('Context Length (N)')
ax_beh[0].set_ylabel('t-test Statistic')
ax_beh[0].set_title('p(truth)\nt-test statistics across context lengths')
ax_beh[1].legend()
ax_beh[0].grid(True, alpha=0.3)

ax_beh[1].set_xlabel('Context Length (N)')
ax_beh[1].set_ylabel('Mean diff')
ax_beh[1].set_title('p(truth)\nmean diff across context lengths')
# ax_beh[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax_beh[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'context_effect_results.png'), dpi=300, bbox_inches='tight')
plt.close()

# ==============================================
# ACTIVATIONS VISUALIZATION
# ==============================================

# NOTE: Your activations arrays need a layer dimension!
# Current shape: [len(pairs_to_compare), len(context_lengths)]
# Should be: [len(pairs_to_compare), len(context_lengths), n_layers]
# For now, I'll assume you only have data from the last layer processed

import matplotlib.gridspec as gridspec

fig_act = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(8, 8, figure=fig_act, wspace=0.15, hspace=0.3)

# Create subplots with custom spacing - smaller gaps within pairs
axes_act = []
for row in range(8):
    row_axes = []
    for col_pair in range(4):  # 4 pairs of columns
        # Left subplot of pair (t-stat)
        ax_left = fig_act.add_subplot(gs[row, col_pair*2])
        # Right subplot of pair (mean difference) 
        ax_right = fig_act.add_subplot(gs[row, col_pair*2 + 1])
        row_axes.extend([ax_left, ax_right])
    axes_act.append(row_axes)

axes_act = np.array(axes_act)

# Reshape axes for easier indexing: [layer, stat_type] where stat_type 0=t-stat, 1=mean difference
axes_act = axes_act.reshape(n_layers, 2)

colors_act = plt.cm.tab10(np.linspace(0, 1, len(pairs_to_compare)))

for layer in range(n_layers):
    # Left subplot: t-test statistics
    ax_tstat = axes_act[layer, 0]
    # Right subplot: mean difference
    ax_mean_diffs = axes_act[layer, 1]
    
    for i_pair, (pair_context_1, pair_context_2) in enumerate(pairs_to_compare):
        # Line style: first pair solid, others dotted
        linestyle = '-' if i_pair == 0 else '--'
        color = colors_act[i_pair]
        
        # Plot t-statistics (left subplot)
        ax_tstat.plot(context_lengths_activations, activations_ttest_stats[i_pair, :], 
                     color=color, linestyle=linestyle, marker='o', 
                     linewidth=1.5, markersize=4)
        
        # Plot mean difference (right subplot)
        ax_mean_diffs.plot(context_lengths_activations, activations_mean_diffs[i_pair, :], 
                      color=color, linestyle=linestyle, marker='o', 
                      linewidth=1.5, markersize=4)
        
        # Add error bars for control data
        # Control stats shape: [len(pairs_to_compare), len(context_lengths), num_control_directions]
        control_tstat_mean = np.mean(control_activations_ttest_stats[i_pair, :, :], axis=1)
        control_tstat_std = np.std(control_activations_ttest_stats[i_pair, :, :], axis=1)
        control_mean_diffs_mean = np.mean(control_activations_mean_diffs[i_pair, :, :], axis=1)
        control_mean_diffs_std = np.std(control_activations_mean_diffs[i_pair, :, :], axis=1)
        
        # Add control error bars (slightly offset x for visibility)
        x_offset = (i_pair - len(pairs_to_compare)/2) * 0.05
        x_positions = np.array(context_lengths_activations) + x_offset
        
        ax_tstat.errorbar(x_positions, control_tstat_mean, yerr=control_tstat_std,
                         color=color, linestyle=':', alpha=0.6, capsize=2, 
                         capthick=1, linewidth=1)
        ax_mean_diffs.errorbar(x_positions, control_mean_diffs_mean, yerr=control_mean_diffs_std,
                          color=color, linestyle=':', alpha=0.6, capsize=2, 
                          capthick=1, linewidth=1)
        
        # Add significance stars to t-stat plot only
        for i_N, (N, p_val) in enumerate(zip(context_lengths_activations, activations_p_values[i_pair, :])):
            if p_val < 0.05 and not np.isnan(p_val):
                y_pos = activations_ttest_stats[i_pair, i_N]
                ax_tstat.scatter(N, y_pos, marker='*', s=50, color=color, 
                               edgecolors='black', linewidth=0.3, zorder=10)
    
    # Formatting
    ax_tstat.set_title(f'Layer {layer}: T-statistics', fontsize=10)
    ax_mean_diffs.set_title(f'Layer {layer}: Mean difference', fontsize=10)
    
    if layer == n_layers - 1:  # Bottom row
        ax_tstat.set_xlabel('Context Length (N)')
        ax_mean_diffs.set_xlabel('Context Length (N)')
    
    ax_tstat.grid(True, alpha=0.3)
    ax_mean_diffs.grid(True, alpha=0.3)
    
    # Smaller font for tick labels
    ax_tstat.tick_params(labelsize=8)
    ax_mean_diffs.tick_params(labelsize=8)

# Add legend to the figure
legend_elements = []
for i_pair, (pair_context_1, pair_context_2) in enumerate(pairs_to_compare):
    linestyle = '-' if i_pair == 0 else '--'
    legend_elements.append(plt.Line2D([0], [0], color=colors_act[i_pair], 
                                     linestyle=linestyle, linewidth=2,
                                     label=f'{pair_context_1} vs {pair_context_2}'))

fig_act.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
               ncol=len(pairs_to_compare), fontsize=10)

plt.suptitle('Activation Context Effects Across Layers', fontsize=14, y=0.98)
# plt.tight_layout()
plt.subplots_adjust(bottom=0.08, top=0.94, hspace=0.3, wspace=0.15)
plt.savefig(os.path.join(output_dir, 'activation_context_effects.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Figures saved to:")
print(f"  - {os.path.join(output_dir, 'context_effect_results.png')}")
print(f"  - {os.path.join(output_dir, 'activation_context_effects.png')}")

# IMPORTANT NOTE: Your activations data arrays need to be 3D to properly store layer information:
# activations_mean_diffs should be shape [len(pairs_to_compare), len(context_lengths), n_layers]
# activations_ttest_stats should be shape [len(pairs_to_compare), len(context_lengths), n_layers]  
# activations_p_values should be shape [len(pairs_to_compare), len(context_lengths), n_layers]
# control_activations_mean_diffs should be shape [len(pairs_to_compare), len(context_lengths), n_layers, num_control_directions]
# control_activations_ttest_stats should be shape [len(pairs_to_compare), len(context_lengths), n_layers, num_control_directions]

