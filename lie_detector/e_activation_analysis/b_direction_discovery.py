import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

from util.util import YamlConfig
from lie_detector.e_activation_analysis.viz import plot_prompted_projections, reconstruct_projection_arrays
from lie_detector.e_activation_analysis.viz import plot_prompted_control_projections, reconstruct_control_projection_arrays

import sys, os

config_path = sys.argv[1]
args = YamlConfig(config_path)

batch_size = args.batch_size
system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name

probe_file_name = args.probe_file_name
probe_response_type = args.probe_response_type

prompt_index = args.prompt_idx

# Save directory should already be made
prompted_results_path = os.path.join('lie_detector_results/e_activation_analysis', args.args_name, 'prompted')
contextual_results_path = os.path.join('lie_detector_results/e_activation_analysis', args.args_name, 'contextual')

# Create place to save results from this analysis
save_base = os.path.join('lie_detector_results/e_activation_analysis', args.args_name, 'projection_results')
os.makedirs(save_base, exist_ok=True)
args.save(save_base)

# Both [questions, layers, neurons]
all_truth_residual = torch.load(os.path.join(prompted_results_path, 'all_truth_residual_with_question.pt'))
all_lie_residual = torch.load(os.path.join(prompted_results_path, 'all_lie_residual_with_question.pt'))

# For labelling
candidate_layers = list(range(32))

# Convert to numpy for easier manipulation
truth_acts = all_truth_residual.numpy()  # [questions, layers, neurons]
lie_acts = all_lie_residual.numpy()      # [questions, layers, neurons]
n_questions, n_layers, n_neurons = truth_acts.shape

# Calculate lie directions using all data
# [layers, neurons]
lie_directions = (lie_acts - truth_acts).mean(0)
lie_directions_normalized = lie_directions / np.sqrt(np.square(lie_directions).sum(-1, keepdims=True))

# [num_controls, layers, neurons] - random normal vectors, normalized along neurons
num_controls = 16
control_lie_directions_normalized = np.random.randn(num_controls, n_layers, n_neurons)
control_lie_directions_normalized /= np.linalg.norm(control_lie_directions_normalized, axis=-1, keepdims=True)

# Project all activations onto lie directions
truth_projections = np.zeros((n_questions, n_layers))
lie_projections = np.zeros((n_questions, n_layers))

truth_projections_control = np.zeros((n_questions, num_controls, n_layers))
lie_projections_control = np.zeros((n_questions, num_controls, n_layers))

results = {}

for layer in range(n_layers):
    direction = lie_directions_normalized[layer, :]
    control_directions = control_lie_directions_normalized[:, layer, :]
    
    # Project each question's activations onto the direction
    for q_idx in range(n_questions):
        truth_projections[q_idx, layer] = np.dot(truth_acts[q_idx, layer, :], direction)
        lie_projections[q_idx, layer] = np.dot(lie_acts[q_idx, layer, :], direction)

        for c_idx in range(num_controls):
            control_direction = control_directions[c_idx]
            truth_projections_control[q_idx, c_idx, layer] = np.dot(truth_acts[q_idx, layer, :], control_direction)
            lie_projections_control[q_idx, c_idx, layer] = np.dot(lie_acts[q_idx, layer, :], control_direction)

    # Statistical analysis for each layer
    truth_projs = truth_projections[:, layer]
    lie_projs = lie_projections[:, layer]
    
    # Paired t-test (since same questions used for both conditions)
    t_stat, p_value = stats.ttest_rel(lie_projs, truth_projs)
    
    # Effect size (Cohen's d for paired samples)
    diff_scores = lie_projs - truth_projs
    cohens_d = np.mean(diff_scores) / np.std(diff_scores)
    
    results[layer] = {
        'lie_mean': float(np.mean(lie_projs)),
        'truth_mean': float(np.mean(truth_projs)),
        'lie_std': float(np.std(lie_projs)),
        'truth_std': float(np.std(truth_projs)),
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'lie_projs': lie_projs,
        'truth_projs': truth_projs,
        'direction': direction,
        'control_directions': control_directions,
    }
    
    print(f"Layer {layer}: Truth={np.mean(truth_projs):.3f}, Lie={np.mean(lie_projs):.3f}, "
          f"Cohen's d={cohens_d:.3f}, p={p_value:.3f}")

# Save results after all layers are processed
np.save(os.path.join(save_base, 'prompted_projection_along_average_lie_vector.npy'), results)

# Visualization
truth_projections, lie_projections = reconstruct_projection_arrays(results)
plot_prompted_projections(truth_projections, lie_projections, results, 
                          os.path.join(save_base, 'projection_along_average_lie_vector.png'))

truth_projections_control, lie_projections_control = reconstruct_control_projection_arrays(truth_acts, lie_acts, results)
plot_prompted_control_projections(truth_projections_control, lie_projections_control, results,
                                 os.path.join(save_base, 'projection_along_control_vector.png'))

# Find layers with strongest separation
effect_sizes = [results[layer]['cohens_d'] for layer in range(n_layers)]
best_layers = np.argsort(np.abs(effect_sizes))[-5:]  # Top 5 layers by effect size
print(f"\nLayers with strongest lie/truth separation: {best_layers}")
print(f"Effect sizes: {[effect_sizes[i] for i in best_layers]}")

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################

# Context composition types
context_types = [
    'top_lie_shuffled_together',      # 1. Top questions + lie answers, shuffled together
    'top_truth_shuffled_together',    # 2. Top questions + truth answers, shuffled together 
    'top_questions_random_answers',     # 5alt. Top questions + 50/50 random answers
]

context_lengths = args.context_lengths_activations

num_context_lengths = len(context_lengths)

fig, contextual_axes = plt.subplots(num_context_lengths, n_layers, figsize = (5 * n_layers, 5 * num_context_lengths))
fig_control, contextual_axes_control = plt.subplots(num_context_lengths, n_layers, figsize = (5 * n_layers, 5 * num_context_lengths))

all_contextual_data = []

for cli, context_length in tqdm(enumerate(context_lengths), total = len(context_lengths)):

    contextual_axes[cli,0].set_ylabel(f'N = {context_length}')
    contextual_axes_control[cli,0].set_ylabel(f'N = {context_length}')

    for context_type in context_types:

        # [num_questions, n_samples, num_candidate_layers, residual_stream_sizes]
        try:
            context_driven_residuals = torch.load(os.path.join(contextual_results_path, f'all_contextual_residual_without_question_N{context_length}_context{context_type}.pt'), weights_only=False)
        except FileNotFoundError:
            continue

        # [questions, n_samples, num_candidate_layers, residual_stream_sizes]
        # Use all questions instead of just test questions

        # [questions, num_candidate_layers, residual_stream_sizes]
        sampled_averaged_context_driven_residuals = context_driven_residuals.mean(1)

        context_projections = np.zeros((n_questions, n_layers))
        context_projections_control = np.zeros((n_questions, num_controls, n_layers))

        for layer in tqdm(range(n_layers)):

            contextual_axes[0, layer].set_title(f'Layer = {layer + 1}')
            
            direction = lie_directions_normalized[layer, :]
            
            # Project each question's activations onto the direction
            context_projections[:,layer] = (sampled_averaged_context_driven_residuals[:, layer, :] * direction[None]).sum(-1)
            context_projections_control[:, :, layer] = (sampled_averaged_context_driven_residuals[:, layer, None] * control_lie_directions_normalized[:, layer, :][None]).sum(-1)

            contextual_axes[cli, layer].hist(context_projections[:, layer], alpha=0.5, label=context_type, bins=20)
            contextual_axes_control[cli, layer].hist(context_projections_control[:, 0, layer], alpha=0.5, label=context_type, bins=20)

        all_contextual_data.append({
            'context_length': context_length,
            'context_type': context_type,
            'projections': context_projections,
            'control_projections': context_projections_control
        })

    contextual_axes[cli,-1].legend(title = 'Context type')
    contextual_axes_control[cli,-1].legend(title = 'Context type')

    # Save contextual results after all processing is complete
    np.save(os.path.join(save_base, 'contextual_projection_along_average_lie_vector.npy'), all_contextual_data)

    fig.savefig(os.path.join(save_base, 'contextual_projection_along_average_lie_vector.png'))
    fig_control.savefig(os.path.join(save_base, 'contextual_projection_along_control_vector.png'))
