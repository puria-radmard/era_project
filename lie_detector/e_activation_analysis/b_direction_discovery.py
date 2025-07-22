import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm

from util.util import YamlConfig

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

# Train/test split (stratified by question)
n_train = int(0.7 * n_questions)
train_indices = np.random.choice(n_questions, n_train, replace=False)
test_indices = np.array([i for i in range(n_questions) if i not in train_indices])

train_truth = truth_acts[train_indices]  # [train_questions, layers, neurons]
train_lie = lie_acts[train_indices]
test_truth = truth_acts[test_indices]    # [test_questions, layers, neurons]
test_lie = lie_acts[test_indices]

# [layers, neurons]
lie_directions = (train_lie - train_truth).mean(0)
lie_directions_normalized = lie_directions / np.sqrt(np.square(lie_directions).sum(-1, keepdims=True))

# [num_controls, layers, neurons] - random normal vectors, normalized along neurons
num_controls = 16
control_lie_directions_normalized = np.random.randn(num_controls, n_layers, n_neurons)
control_lie_directions_normalized /= np.linalg.norm(control_lie_directions_normalized, axis=-1, keepdims=True)

# Project test set activations onto lie directions
test_truth_projections = np.zeros((len(test_indices), n_layers))
test_lie_projections = np.zeros((len(test_indices), n_layers))

test_truth_projections_control = np.zeros((len(test_indices), num_controls, n_layers))
test_lie_projections_control = np.zeros((len(test_indices), num_controls, n_layers))

results = {}

for layer in range(n_layers):
    direction = lie_directions_normalized[layer, :]
    control_directions = control_lie_directions_normalized[:, layer, :]
    
    # Project each test question's activations onto the direction
    for q_idx in range(len(test_indices)):
        test_truth_projections[q_idx, layer] = np.dot(test_truth[q_idx, layer, :], direction)
        test_lie_projections[q_idx, layer] = np.dot(test_lie[q_idx, layer, :], direction)

        for c_idx in range(num_controls):
            control_direction = control_directions[c_idx]
            test_truth_projections_control[q_idx, c_idx, layer] = np.dot(test_truth[q_idx, layer, :], control_direction)
            test_lie_projections_control[q_idx, c_idx, layer] = np.dot(test_lie[q_idx, layer, :], control_direction)

    # Statistical analysis for each layer
    truth_projs = test_truth_projections[:, layer]
    lie_projs = test_lie_projections[:, layer]
    
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

    np.save(os.path.join(save_base, 'prompted_projection_along_average_lie_vector.npy'), results)


# Visualization
fig, axes = plt.subplots(7, 5, figsize=(9, 12))  # Adjust grid size as needed
axes = axes.flatten()

for layer in range(n_layers):  # Show first 25 layers
    ax = axes[layer]
    
    # Histogram both distributions
    ax.hist(test_truth_projections[:, layer], alpha=0.5, label='Truth', bins=20)
    ax.hist(test_lie_projections[:, layer], alpha=0.5, label='Lie', bins=20)
    
    ax.set_title(f'Layer {layer} (d={results[layer]["cohens_d"]:.2f})')
    ax.legend()
    ax.set_xlabel('Projection onto Lie Direction')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(os.path.join(save_base, 'projection_along_average_lie_vector.png'))



# Visualization
fig, axes = plt.subplots(7, 5, figsize=(9, 12))  # Adjust grid size as needed
axes = axes.flatten()

for layer in range(n_layers):  # Show first 25 layers
    ax = axes[layer]
    
    # Histogram both distributions
    ax.hist(test_truth_projections_control[:, 0, layer], alpha=0.5, label='Truth', bins=20)
    ax.hist(test_lie_projections_control[:, 0, layer], alpha=0.5, label='Lie', bins=20)
    
    ax.set_title(f'Layer {layer} (d={results[layer]["cohens_d"]:.2f})')
    ax.legend()
    ax.set_xlabel('Projection onto example control direction')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(os.path.join(save_base, 'projection_along_control_vector.png'))



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

context_type = 'top_lie_shuffled_together'

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

        # [test_questions, n_samples, num_candidate_layers, residual_stream_sizes]
        test_context_driven_residuals = context_driven_residuals[test_indices]

        # [test_questions, num_candidate_layers, residual_stream_sizes]
        sampled_averaged_test_context_driven_residuals = test_context_driven_residuals.mean(1)

        test_context_projections = np.zeros((len(test_indices), n_layers))
        test_context_projections_control = np.zeros((len(test_indices), num_controls, n_layers))

        for layer in tqdm(range(n_layers)):

            contextual_axes[0, layer].set_title(f'Layer = {layer + 1}')
            
            direction = lie_directions_normalized[layer, :]
            
            # Project each test question's activations onto the direction
            test_context_projections[:,layer] = (sampled_averaged_test_context_driven_residuals[:, layer, :] * direction[None]).sum(-1)
            test_context_projections_control[:, :, layer] = (sampled_averaged_test_context_driven_residuals[:, layer, None] * control_lie_directions_normalized[:, layer, :][None]).sum(-1)

            contextual_axes[cli, layer].hist(test_context_projections[:, layer], alpha=0.5, label=context_type, bins=20)
            contextual_axes_control[cli, layer].hist(test_context_projections_control[:, 0, layer], alpha=0.5, label=context_type, bins=20)

        all_contextual_data.append({
            'context_length': context_length,
            'context_type': context_type,
            'projections': test_context_projections,
            'control_projections': test_context_projections_control
        })
    np.save(os.path.join(save_base, 'contextual_projection_along_average_lie_vector.npy'), all_contextual_data)

    contextual_axes[cli,-1].legend(title = 'Context type')
    contextual_axes_control[cli,-1].legend(title = 'Context type')

    fig.savefig(os.path.join(save_base, 'contextual_projection_along_average_lie_vector.png'))
    fig_control.savefig(os.path.join(save_base, 'contextual_projection_along_control_vector.png'))



