import torch
import numpy as np
import sys
import os
from util.util import YamlConfig
from lie_detector.e_activation_analysis.viz import (
    plot_prompted_projections, 
    plot_prompted_control_projections,
    plot_contextual_projections, 
    plot_contextual_control_projections,
    plot_single_layer_histogram,
    reconstruct_projection_arrays,
    reconstruct_control_projection_arrays
)

# Same argument signature as original script
config_path = sys.argv[1]
args = YamlConfig(config_path)

# Load paths from config
prompted_results_path = os.path.join('lie_detector_results/e_activation_analysis', args.args_name, 'prompted')
save_base = os.path.join('lie_detector_results/e_activation_analysis', args.args_name, 'projection_results')

# Load original activation data
print("Loading original activation data...")
all_truth_residual = torch.load(os.path.join(prompted_results_path, 'all_truth_residual_with_question.pt'))
all_lie_residual = torch.load(os.path.join(prompted_results_path, 'all_lie_residual_with_question.pt'))

truth_acts = all_truth_residual.numpy()  # [questions, layers, neurons]
lie_acts = all_lie_residual.numpy()      # [questions, layers, neurons]

# Load saved data
print("Loading saved projection results...")
results = np.load(os.path.join(save_base, 'prompted_projection_along_average_lie_vector.npy'), allow_pickle=True).item()

print("Loading saved contextual data...")
all_contextual_data = np.load(os.path.join(save_base, 'contextual_projection_along_average_lie_vector.npy'), allow_pickle=True)

# Reconstruct projection arrays for prompted data
print("Reconstructing projection arrays...")
truth_projections, lie_projections = reconstruct_projection_arrays(results)

print("Reconstructing control projection arrays...")
truth_projections_control, lie_projections_control = reconstruct_control_projection_arrays(truth_acts, lie_acts, results)

# Create post-hoc visualizations
print("Creating post-hoc prompted projections plot...")
plot_prompted_projections(
    truth_projections, 
    lie_projections, 
    results,
    os.path.join(save_base, 'posthoc_projection_along_average_lie_vector.png')
)

print("Creating post-hoc prompted control projections plot...")
plot_prompted_control_projections(
    truth_projections_control,
    lie_projections_control,
    results,
    os.path.join(save_base, 'posthoc_projection_along_control_vector.png')
)

# Contextual projections
if len(all_contextual_data) > 0:
    print("Creating post-hoc contextual projections plot...")
    context_lengths = args.context_lengths_activations
    
    plot_contextual_projections(
        all_contextual_data,
        context_lengths, 
        os.path.join(save_base, 'posthoc_contextual_projection_along_average_lie_vector.png')
    )
    
    print("Creating post-hoc contextual control projections plot...")
    plot_contextual_control_projections(
        all_contextual_data,
        context_lengths,
        os.path.join(save_base, 'posthoc_contextual_projection_along_control_vector.png')
    )
else:
    print("No contextual data found.")


print(f"Creating single layer histogram for layer {args.histogram_layer}...")
plot_single_layer_histogram(
    truth_projections,
    lie_projections, 
    results,
    args.histogram_layer,
    os.path.join(save_base, f'posthoc_projection_along_average_lie_vector_{args.histogram_layer + 1}.png'),
    truth_label = 'Aligned',
    lie_label = 'Deceptively misaligned',
    vector_label = 'Deceptively misaligned contrast vector'
)

print("Post-hoc visualization complete!")

