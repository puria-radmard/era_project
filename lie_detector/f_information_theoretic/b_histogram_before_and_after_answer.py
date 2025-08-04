## Plan

# 1. Extract vector and projections
# 2. Get cluster means along vector
# 3. Write ablation utils (layer-specific)

import numpy as np

import sys, os, torch
from util.util import YamlConfig

from matplotlib import pyplot as plt

from lie_detector.f_information_theoretic.z_util import prob_mode


config_path = sys.argv[1]
args = YamlConfig(config_path)

batch_size = args.batch_size
system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name

num_layers = args.num_layers

probe_file_name = args.probe_file_name
probe_response_type = args.probe_response_type

prompt_index = args.prompt_idx


save_base = os.path.join('lie_detector_results/f_information_theoretic', args.args_name, 'testbed')
os.makedirs(save_base, exist_ok=True)
args.save(save_base)


activation_analysis_path = os.path.join('lie_detector_results/e_activation_analysis', args.args_name, 'projection_results')


projection_lie_directions_normalized = np.load(os.path.join(activation_analysis_path, 'prompted_projection_along_average_lie_vector.npy'), allow_pickle=True).item()


lie_directions = np.stack([projection_lie_directions_normalized[idx]['direction'] for idx in range(len(projection_lie_directions_normalized))])
truth_means = np.stack([projection_lie_directions_normalized[idx]['truth_mean'] for idx in range(len(projection_lie_directions_normalized))])
truth_std = np.stack([projection_lie_directions_normalized[idx]['truth_std'] for idx in range(len(projection_lie_directions_normalized))])
lie_means = np.stack([projection_lie_directions_normalized[idx]['lie_mean'] for idx in range(len(projection_lie_directions_normalized))])
lie_std = np.stack([projection_lie_directions_normalized[idx]['lie_std'] for idx in range(len(projection_lie_directions_normalized))])


# [questions, layers]
lie_projs = np.stack([projection_lie_directions_normalized[idx]['lie_projs'] for idx in range(len(projection_lie_directions_normalized))], axis = 1)
truth_projs = np.stack([projection_lie_directions_normalized[idx]['truth_projs'] for idx in range(len(projection_lie_directions_normalized))], axis = 1)

lie_probs = prob_mode(lie_projs, truth_means, truth_std, lie_means, lie_std)
truth_probs = prob_mode(truth_projs, truth_means, truth_std, lie_means, lie_std)

#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

test_data = np.linspace(-0.5, 1.5, 500)[:,None].repeat(num_layers, 1)

# Compute transformation parameters
scale = lie_means - truth_means  # Distance between modes
offset = truth_means               # Shift amount

# Apply rescaling
rescaled_truth_means = (truth_means - offset) / scale  # = 0
rescaled_lie_means = (lie_means - offset) / scale  # = 1  
rescaled_truth_stds = truth_std / np.abs(scale)
rescaled_lie_stds = lie_std / np.abs(scale)


probs = prob_mode(test_data, rescaled_truth_means, rescaled_truth_stds, rescaled_lie_means, rescaled_lie_stds)

# Get colors from magma colormap
n_channels = probs.shape[1]  # Number of channels
colors = plt.cm.magma(np.linspace(0.1, 0.9, n_channels))  # Avoid pure black/white

# Plot each probability curve with magma colors
for i, prob in enumerate(probs.T):
    plt.plot(test_data[:,0], prob, color=colors[i], label=f'Channel {i}')

# Add colorbar to show index mapping
sm = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=0, vmax=n_channels-1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())  # Specify current axes
cbar.set_label('Channel Index')
cbar.set_ticks(np.arange(n_channels))

plt.xlabel('Test Data')
plt.ylabel('Probability')

fig_save_path = os.path.join(save_base, 'test_sigmoids.png')
plt.savefig(fig_save_path)
print(fig_save_path)
plt.close('all')

#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################


all_truth_residual_with_initial_answer = torch.load(os.path.join(save_base, 'prompted/all_truth_residual_with_initial_answer.pt'))
all_lie_residual_with_initial_answer = torch.load(os.path.join(save_base, 'prompted/all_lie_residual_with_initial_answer.pt'))

post_answer_lie_projs = (all_lie_residual_with_initial_answer * lie_directions[None]).sum(-1)   # [questions, layers]
post_answer_truth_projs = (all_truth_residual_with_initial_answer * lie_directions[None]).sum(-1)   # [questions, layers]


fig, axes = plt.subplots(7, 6, figsize=(18, 21))
axes = axes.flatten()
colors = ['red', 'green', 'orange', 'lightgreen']
labels = ['lie post', 'truth post', 'lie pre', 'truth pre']

for i in range(num_layers):
    ax = axes[i]
    ax.hist(post_answer_lie_projs[:, i], alpha=0.5, color=colors[0], label=labels[0])
    ax.hist(post_answer_truth_projs[:, i], alpha=0.5, color=colors[1], label=labels[1])
    ax.hist(lie_projs[:, i], alpha=0.5, color=colors[2], label=labels[2])
    ax.hist(truth_projs[:, i], alpha=0.5, color=colors[3], label=labels[3])
    ax.set_title(f'Layer {i}')

ax.legend(fontsize='small')

plt.tight_layout()
fig_save_path = os.path.join(save_base, 'pre_and_post_answer_proj.png')
print(fig_save_path)
plt.savefig(fig_save_path)
plt.close(fig)



#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################

post_answer_lie_probs = prob_mode(post_answer_lie_projs.numpy(), truth_means, truth_std, lie_means, lie_std)
post_answer_truth_probs = prob_mode(post_answer_truth_projs.numpy(), truth_means, truth_std, lie_means, lie_std)

fig, axes = plt.subplots(7, 6, figsize=(18, 21))
axes = axes.flatten()
colors = ['red', 'green', 'orange', 'lightgreen']
labels = ['lie post', 'truth post', 'lie pre', 'truth pre']

for i in range(num_layers):
    ax = axes[i]
    ax.hist(post_answer_lie_probs[:, i], alpha=0.5, color=colors[0], label=labels[0])
    ax.hist(post_answer_truth_probs[:, i], alpha=0.5, color=colors[1], label=labels[1])
    ax.hist(lie_probs[:, i], alpha=0.5, color=colors[2], label=labels[2])
    ax.hist(truth_probs[:, i], alpha=0.5, color=colors[3], label=labels[3])
    ax.set_title(f'Layer {i}')

ax.legend(fontsize='small')

plt.tight_layout()
fig_save_path = os.path.join(save_base, 'pre_and_post_answer_probs.png')
print(fig_save_path)
plt.savefig(fig_save_path)
plt.close(fig)


#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################


post_truth_means = torch.mean(post_answer_truth_projs, axis=0)
post_truth_std = torch.std(post_answer_truth_projs, axis=0)
post_lie_means = torch.mean(post_answer_lie_projs, axis=0)
post_lie_std = torch.std(post_answer_lie_projs, axis=0)

alt_post_answer_lie_probs = prob_mode(post_answer_lie_projs.numpy(), post_truth_means, post_truth_std, post_lie_means, post_lie_std)
alt_post_answer_truth_probs = prob_mode(post_answer_truth_projs.numpy(), post_truth_means, post_truth_std, post_lie_means, post_lie_std)


fig, axes = plt.subplots(7, 6, figsize=(18, 21))
axes = axes.flatten()
colors = ['red', 'green',]
labels = ['lie post', 'truth post',]

for i in range(num_layers):
    ax = axes[i]
    ax.hist(alt_post_answer_lie_probs[:, i], alpha=0.5, color=colors[0], label=labels[0])
    ax.hist(alt_post_answer_truth_probs[:, i], alpha=0.5, color=colors[1], label=labels[1])
    ax.set_title(f'Layer {i}')

ax.legend(fontsize='small')

plt.tight_layout()
fig_save_path = os.path.join(save_base, 'alt_pre_and_post_answer_probs.png')
print(fig_save_path)
plt.savefig(fig_save_path)
plt.close(fig)


#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################


test_data = np.linspace(-0.5, 1.5, 500)[:,None].repeat(num_layers, 1)

# Compute transformation parameters
scale = post_lie_means - post_truth_means  # Distance between modes
offset = post_truth_means               # Shift amount

# Apply rescaling
rescaled_truth_means = (post_truth_means - offset) / scale  # = 0
rescaled_lie_means = (post_lie_means - offset) / scale  # = 1  
rescaled_truth_stds = post_truth_std / np.abs(scale)
rescaled_lie_stds = post_lie_std / np.abs(scale)


probs = prob_mode(test_data, rescaled_truth_means, rescaled_truth_stds, rescaled_lie_means, rescaled_lie_stds)

# Get colors from magma colormap
n_channels = probs.shape[1]  # Number of channels
colors = plt.cm.magma(np.linspace(0.1, 0.9, n_channels))  # Avoid pure black/white

# Plot each probability curve with magma colors
for i, prob in enumerate(probs.T):
    plt.plot(test_data[:,0], prob, color=colors[i], label=f'Channel {i}')

# Add colorbar to show index mapping
sm = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=0, vmax=n_channels-1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca())  # Specify current axes
cbar.set_label('Channel Index')
cbar.set_ticks(np.arange(n_channels))

plt.xlabel('Test Data')
plt.ylabel('Probability')

fig_save_path = os.path.join(save_base, 'alt_test_sigmoids.png')
plt.savefig(fig_save_path)
print(fig_save_path)
plt.close('all')


#########################################################################################################
#########################################################################################################
#########################################################################################################
#########################################################################################################


post_vector_lie_directions = (all_truth_residual_with_initial_answer - all_lie_residual_with_initial_answer).mean(0)
post_vector_lie_directions_normalized = post_vector_lie_directions / np.sqrt(np.square(post_vector_lie_directions).sum(-1, keepdims=True))

post_vector_post_answer_lie_projs = (all_lie_residual_with_initial_answer * post_vector_lie_directions_normalized[None]).sum(-1)   # [questions, layers]
post_vector_post_answer_truth_projs = (all_truth_residual_with_initial_answer * post_vector_lie_directions_normalized[None]).sum(-1)   # [questions, layers]

fig, axes = plt.subplots(7, 6, figsize=(18, 21))
axes = axes.flatten()
colors = ['red', 'green']
labels = ['lie post', 'truth post']

for i in range(num_layers):
    ax = axes[i]
    ax.hist(post_vector_post_answer_lie_projs[:, i], alpha=0.5, color=colors[0], label=labels[0])
    ax.hist(post_vector_post_answer_truth_projs[:, i], alpha=0.5, color=colors[1], label=labels[1])
    ax.set_title(f'Layer {i}')

ax.legend(fontsize='small')

plt.tight_layout()
fig_save_path = os.path.join(save_base, 'post_vector_pre_and_post_answer_proj.png')
print(fig_save_path)
plt.savefig(fig_save_path)
plt.close(fig)

