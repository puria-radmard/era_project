#!/usr/bin/env python3

import numpy as np
import sys
import os
from scipy import stats
from util.util import YamlConfig
from lie_detector.e_activation_analysis.viz import plot_behavioral_context_effects, plot_activation_context_effects, plot_absolute_activation_context_effects
from lie_detector.e_activation_analysis.viz import plot_differential_activation_context_effects

import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Load config
config_path = sys.argv[1]
args = YamlConfig(config_path)

# Set up paths
context_results_path = os.path.join('lie_detector_results/d_in_context_lying', args.args_name, 'context_effect_results_{context_type}.npy')
projection_results_path = os.path.join('lie_detector_results/e_activation_analysis', args.args_name, 'projection_results')

# Define paths
prompted_activations_projections_path = os.path.join(projection_results_path, 'prompted_projection_along_average_lie_vector.npy')
contextual_activations_projections_path = os.path.join(projection_results_path, 'contextual_projection_along_average_lie_vector.npy')

# Load all data
print("Loading data...")
context_results = {}
pattern = context_results_path.replace('{context_type}', '*')
for path in glob.glob(pattern):
    # Extract context_type from filename
    filename = os.path.basename(path)
    prefix = 'context_effect_results_'
    suffix = '.npy'
    if filename.startswith(prefix) and filename.endswith(suffix):
        context_type = filename[len(prefix):-len(suffix)]
        context_results[context_type] = np.load(path, allow_pickle=True).item()

prompted_activations_projections = np.load(prompted_activations_projections_path, allow_pickle=True).item()
contextual_activations_projections = np.load(contextual_activations_projections_path, allow_pickle=True)

# Get context lengths from the behavioral data
context_lengths = args.context_lengths
context_lengths_activations = args.context_lengths_activations

n_layers = len(prompted_activations_projections)  # Should be 32


pairs_to_compare = [
    ('top_lie_shuffled_together', 'top_truth_shuffled_together'),
    # ('top_lie_shuffled_together', 'top_questions_random_answers'),
    # ('top_questions_random_answers', 'top_truth_shuffled_together'),
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
    for context_type, context_type_results in context_results.items():
        iN_in_file = context_type_results['context_length'].tolist().index(N)
        behavioral_data[context_type] = {k: v[iN_in_file] for k, v in context_type_results.items()}
    
    print(f"Found behavioral data for {len(behavioral_data)} context types at N={N}")
    print(f"Behavioural context types: {list(behavioral_data.keys())}")

    # ==========================================
    # BEHAVIOURAL ANALYSIS SPACE - ALL DATA ACCESSIBLE HERE
    # ==========================================

    # [behavioural questions]
    if N > 0:
        for i_pair, (pair_context_1, pair_context_2) in enumerate(more_pairs_to_compare):
            
            try:
                question_truth_probs = np.array(behavioral_data[pair_context_1]['question_truth_probs_across_samples'].mean(-1))
                question_lie_probs = np.array(behavioral_data[pair_context_2]['question_truth_probs_across_samples'].mean(-1))
            except KeyError:
                question_truth_probs = np.array(behavioral_data[pair_context_1]['question_truth_lie_diffs_across_samples'].mean(-1))
                question_lie_probs = np.array(behavioral_data[pair_context_2]['question_truth_lie_diffs_across_samples'].mean(-1))


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


# ==============================================
# VISUALIZATION
# ==============================================

# Plot behavioral context effects
plot_behavioral_context_effects(
    context_lengths=context_lengths,
    prob_diff_ttest_stats=prob_diff_ttest_stats,
    prob_diff_mean_diff=prob_diff_mean_diff,
    prob_diff_p_values=prob_diff_p_values,
    more_pairs_to_compare=more_pairs_to_compare,
    projection_results_path=projection_results_path
)

# Plot activation context effects
plot_activation_context_effects(
    context_lengths_activations=context_lengths_activations,
    activations_ttest_stats=activations_ttest_stats,
    activations_mean_diffs=activations_mean_diffs,
    activations_p_values=activations_p_values,
    control_activations_ttest_stats=control_activations_ttest_stats,
    control_activations_mean_diffs=control_activations_mean_diffs,
    pairs_to_compare=pairs_to_compare,
    n_layers=n_layers,
    projection_results_path=projection_results_path
)

# ==============================================
# SINGLE LAYER ABSOLUTE PROJECTION PLOTS
# ==============================================

print("Creating single layer plots...")

# Define your context type aliases
context_type_aliases = {
    'top_lie_shuffled_together': 'Deceptively misaligned answers in context',
    'top_truth_shuffled_together': 'Aligned answers in context',
    'top_questions_random_answers': 'Random yes/no answers in context',
    # Add other context types as needed
}

# Plot for specific layers (e.g., layers 15, 20, 25, 31)
layers_to_plot = [15, 20, 25, 31]

for layer_idx in layers_to_plot:
    layer_contextual_data = {}
    layer_control_data = {}
    
    # Collect data across all context lengths for this layer
    for context_type in context_type_aliases.keys():
        layer_contextual_data[context_type] = []
        layer_control_data[context_type] = []
        
        for entry in contextual_activations_projections:
            if entry['context_type'] == context_type:
                proj_mean = np.mean(entry['projections'][:, layer_idx])
                control_proj_means = np.mean(entry['control_projections'][:, :, layer_idx], axis=1)
                
                layer_contextual_data[context_type].append(proj_mean)
                layer_control_data[context_type].append(control_proj_means)
    
    plot_absolute_activation_context_effects(
        layer_idx=layer_idx,
        context_lengths_activations=context_lengths_activations,
        contextual_activations_data=layer_contextual_data,
        control_contextual_activations_data=layer_control_data,
        context_type_aliases=context_type_aliases,
        projection_results_path=projection_results_path
    )

# ==============================================
# DIFFERENTIAL SINGLE LAYER PROJECTION PLOTS
# ==============================================

print("Creating differential single layer plots...")

# Plot differential effects for the same layers
for layer_idx in layers_to_plot:
    layer_contextual_data = {}
    layer_control_data = {}
    
    # Initialize data structures
    for context_type in context_type_aliases.keys():
        layer_contextual_data[context_type] = []
        layer_control_data[context_type] = []
    
    # Also include the control context type for the baseline
    control_context_type = 'top_questions_random_answers'
    if control_context_type not in layer_contextual_data:
        layer_contextual_data[control_context_type] = []
        layer_control_data[control_context_type] = []
    
    # Organize data by context length, keeping questions dimension
    for length_idx, context_length in enumerate(context_lengths_activations):
        all_context_types = list(context_type_aliases.keys()) + [control_context_type]
        
        for context_type in set(all_context_types):  # Remove duplicates
            # Find entry for this context type and length
            for entry in contextual_activations_projections:
                if entry['context_type'] == context_type and entry['context_length'] == context_length:
                    # Keep full questions dimension (don't take mean)
                    test_projections = entry['projections'][:, layer_idx]  # [questions]
                    control_projections = entry['control_projections'][:, :, layer_idx]  # [num_control_directions, questions]
                    
                    layer_contextual_data[context_type].append(test_projections)
                    layer_control_data[context_type].append(control_projections)
                    break
    
    # Convert to arrays
    for context_type in layer_contextual_data.keys():
        layer_contextual_data[context_type] = np.array(layer_contextual_data[context_type])  # [num_context_lengths, questions]
        layer_control_data[context_type] = np.array(layer_control_data[context_type])  # [num_context_lengths, num_control_directions, questions]
    
    plot_differential_activation_context_effects(
        layer_idx=layer_idx,
        context_lengths_activations=context_lengths_activations,
        contextual_activations_data=layer_contextual_data,
        control_contextual_activations_data=layer_control_data,
        control_context_type=control_context_type,
        context_type_aliases=context_type_aliases,
        projection_results_path=projection_results_path
    )

print(f"Figures saved to:")
print(f"  - {os.path.join(projection_results_path, 'context_effect_results.png')}")
print(f"  - {os.path.join(projection_results_path, 'activation_context_effects.png')}")
print(f"  - Single layer plots: context_absolute_effect_results_layer_{{layer}}.png")
print(f"  - Differential single layer plots: context_differential_effect_results_layer_{{layer}}.png")

# IMPORTANT NOTE: Your activations data arrays need to be 3D to properly store layer information:
# activations_mean_diffs should be shape [len(pairs_to_compare), len(context_lengths), n_layers]
# activations_ttest_stats should be shape [len(pairs_to_compare), len(context_lengths), n_layers]  
# activations_p_values should be shape [len(pairs_to_compare), len(context_lengths), n_layers]
# control_activations_mean_diffs should be shape [len(pairs_to_compare), len(context_lengths), n_layers, num_control_directions]
# control_activations_ttest_stats should be shape [len(pairs_to_compare), len(context_lengths), n_layers, num_control_directions]
