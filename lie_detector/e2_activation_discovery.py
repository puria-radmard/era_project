import torch
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from util.util import YamlConfig

import sys, os

config_path = sys.argv[1]
args = YamlConfig(config_path)

batch_size = args.batch_size
system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name
save_path = args.save_path

prompt_index = args.prompt_idx


save_target = f"{save_path}/activation_discovery/{questions_data_name}/prompt{prompt_index}"
os.makedirs(save_target, exist_ok=True)

# Both [questions, layers, neurons]
all_truth_residual = torch.load(os.path.join(save_target, 'all_truth_residual_with_question.pt'))
all_lie_residual = torch.load(os.path.join(save_target, 'all_lie_residual_with_question.pt'))

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

# Project test set activations onto lie directions
test_truth_projections = np.zeros((len(test_indices), n_layers))
test_lie_projections = np.zeros((len(test_indices), n_layers))

for layer in range(n_layers):
    direction = lie_directions_normalized[layer, :]
    
    # Project each test question's activations onto the direction
    for q_idx in range(len(test_indices)):
        test_truth_projections[q_idx, layer] = np.dot(test_truth[q_idx, layer, :], direction)
        test_lie_projections[q_idx, layer] = np.dot(test_lie[q_idx, layer, :], direction)


# Statistical analysis for each layer
results = {}
for layer in range(n_layers):
    truth_projs = test_truth_projections[:, layer]
    lie_projs = test_lie_projections[:, layer]
    
    # Paired t-test (since same questions used for both conditions)
    t_stat, p_value = stats.ttest_rel(lie_projs, truth_projs)
    
    # Effect size (Cohen's d for paired samples)
    diff_scores = lie_projs - truth_projs
    cohens_d = np.mean(diff_scores) / np.std(diff_scores)
    
    results[layer] = {
        'lie_mean': np.mean(lie_projs),
        'truth_mean': np.mean(truth_projs),
        'lie_std': np.std(lie_projs),
        'truth_std': np.std(truth_projs),
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d
    }
    
    print(f"Layer {layer}: Truth={np.mean(truth_projs):.3f}, Lie={np.mean(lie_projs):.3f}, "
          f"Cohen's d={cohens_d:.3f}, p={p_value:.3f}")

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
plt.savefig(os.path.join(save_target, 'projection_along_average_lie_vector.png'))

# Find layers with strongest separation
effect_sizes = [results[layer]['cohens_d'] for layer in range(n_layers)]
best_layers = np.argsort(np.abs(effect_sizes))[-5:]  # Top 5 layers by effect size
print(f"\nLayers with strongest lie/truth separation: {best_layers}")
print(f"Effect sizes: {[effect_sizes[i] for i in best_layers]}")


