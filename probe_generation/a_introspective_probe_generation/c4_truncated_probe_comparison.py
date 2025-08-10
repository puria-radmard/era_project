import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os, json
from util.util import YamlConfig
import statsmodels.api as sm

config_path = sys.argv[1]
args = YamlConfig(config_path)

batch_size = args.batch_size
question_instruction = args.question_instruction
questions_data_name = args.questions_data_name
model_name = args.model_name
prompt_idx = args.prompt_idx
probe_file_name = args.probe_file_name
banned_words = args.banned_words

save_base = os.path.join('probe_generation_results/a_introspective_probe_generation', args.args_name)

# Load discriminability results
with open(os.path.join(save_base, 'c_truncated_discriminability_results.json'), 'r') as f:
    truncated_discriminability_results_list = json.load(f)

with open(os.path.join(save_base, 'c_discriminability_results.json'), 'r') as f:
    original_discriminability_results_list = json.load(f)

# Convert to dictionaries keyed by probe_idx for easy lookup
original_discriminability_results = {
    item['probe_idx']: item for item in original_discriminability_results_list
}

truncated_discriminability_results = {
    item['probe_idx']: item for item in truncated_discriminability_results_list
}

# Load the linking dataset
truncated_probe_questions_df = pd.read_csv(os.path.join(save_base, "c_truncated_probe_completions.csv"))
probe_questions = truncated_probe_questions_df[~truncated_probe_questions_df['generated_sequence'].str.lower().apply(lambda x: any(word in x for word in banned_words))]

# Find probe indices that have both original results and at least some augmented results
original_probe_indices = set(original_discriminability_results.keys())
available_augmented_indices = set(truncated_discriminability_results.keys())

# Get augmentations that actually have discriminability results
valid_augmentations = probe_questions[
    probe_questions.index.isin(available_augmented_indices)
].copy()

# Find original probes that have at least one valid augmentation
probes_with_augmentations = set(valid_augmentations['probe_idx'].unique())
plottable_probes = original_probe_indices.intersection(probes_with_augmentations)
plottable_probes = sorted(list(plottable_probes))

n_probes = len(plottable_probes)

if n_probes == 0:
    print("No probes found with both original and augmented discriminability results!")
    sys.exit(1)

# Calculate grid dimensions
n_cols = min(4, n_probes)  # Max 4 columns
n_rows = (n_probes + n_cols - 1) // n_cols

# Create figure and subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
if n_probes == 1:
    axes = [axes]
elif n_rows == 1:
    axes = [axes]
else:
    axes = axes.flatten()

best_augmented_avg_effect_size = 0

# Get unique generation types for consistent coloring
all_generation_types = valid_augmentations['generation_type'].unique()
colors = plt.cm.Set1(np.linspace(0, 1, len(all_generation_types)))
color_map = dict(zip(all_generation_types, colors))

# Process each original probe
for plot_idx, probe_idx in enumerate(plottable_probes):
    ax = axes[plot_idx]
    
    # Get original probe's average absolute effect size
    original_result = original_discriminability_results[probe_idx]
    original_avg_effect_size = np.abs(np.mean(original_result['effect_sizes_by_append']))
    
    # Draw horizontal reference line for original probe
    ax.axhline(y=original_avg_effect_size, color='black', linestyle='--', 
               alpha=0.7, label='Original probe')
    
    # Get all valid augmented versions of this probe
    probe_augmentations = valid_augmentations[
        valid_augmentations['probe_idx'] == probe_idx
    ]
    
    # Plot each augmented version
    for _, row in probe_augmentations.iterrows():
        # The row index corresponds to probe_idx in truncated_discriminability_results
        row_idx = row.name
        
        # Get the discriminability result for this augmentation
        augmented_result = truncated_discriminability_results[row_idx]
        augmented_avg_effect_size = np.abs(np.mean(augmented_result['effect_sizes_by_append']))

        if augmented_avg_effect_size > best_augmented_avg_effect_size:
            best_augmented_probe = row.generated_sequence
            best_augmented_avg_effect_size = augmented_avg_effect_size
            print(best_augmented_probe)
        
        # Plot point
        jitter = np.random.randn() * 0.05
        ax.scatter(row['prefix_length'] + jitter, augmented_avg_effect_size, 
                  c=[color_map[row['generation_type']]], 
                  alpha=0.7, s=50)
    
    # Add best fit lines for each generation type
    generation_types_in_probe = probe_augmentations['generation_type'].unique()
    for gen_type in generation_types_in_probe:
        type_data = probe_augmentations[probe_augmentations['generation_type'] == gen_type]
        if len(type_data) > 1:  # Need at least 2 points for a line
            x_vals = type_data['prefix_length'].values
            y_vals = []
            for _, type_row in type_data.iterrows():
                type_row_idx = type_row.name
                type_result = truncated_discriminability_results[type_row_idx]
                y_vals.append(np.abs(np.mean(type_result['effect_sizes_by_append'])))
            
            # Fit line
            X2 = sm.add_constant(x_vals)
            est = sm.OLS(y_vals, X2).fit()
            
            # Plot line
            x_range = np.linspace(min(x_vals), max(x_vals), 100)
            ax.plot(x_range, est.params[0] + est.params[1] * x_range, color=color_map[gen_type],  
                   alpha=0.5, linewidth=2, label = f'{gen_type} | $p_C=${est.pvalues[0]:.2f} | $p_M=${est.pvalues[1]:.2f}')
    
    ax.legend()

    # Customize subplot
    ax.set_xlabel('Prefix Length')
    ax.set_ylabel('Absolute Average Effect Size')
    ax.set_title(f'Probe {probe_idx}')
    ax.grid(True, alpha=0.3)

# Hide empty subplots
for i in range(n_probes, len(axes)):
    axes[i].set_visible(False)

# Create legend for generation types
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=color_map[gen_type], 
                             markersize=8, label=gen_type)
                  for gen_type in all_generation_types]
legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', 
                                 label='Original probe'))

# Add legend to the figure
fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

plt.tight_layout()
fig.savefig(os.path.join(save_base, 'c_truncated_probe_comparison.png'))

print(f"Plotted {n_probes} probes with augmentations")
print(f"Original probes available: {len(original_probe_indices)}")
print(f"Augmented results available: {len(available_augmented_indices)}")
print(f"Valid augmentations after filtering: {len(valid_augmentations)}")