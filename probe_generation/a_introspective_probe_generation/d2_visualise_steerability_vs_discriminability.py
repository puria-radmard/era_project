import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import sys
from util.util import YamlConfig

# Setup
config_path = sys.argv[1]
args = YamlConfig(config_path)

# Load paths
save_base = os.path.join('probe_generation_results/a_introspective_probe_generation', args.args_name)
input_path = os.path.join(save_base, 'd_in_context_steering')

# Parameters
persona_types = ['aligned', 'misaligned', 'random']
generation_types = ['lie-truth_contrastive', 'truth-lie_contrastive', 'lie-only', 'truth-only']

print("Loading steering experiment results...")

# Load all steering results
steering_results = {}
for generation_type in generation_types:
    steering_results[generation_type] = {}
    for persona_type in persona_types:
        results_path = os.path.join(input_path, f'steering_results_{generation_type}_{persona_type}.npy')
        if os.path.exists(results_path):
            steering_results[generation_type][persona_type] = np.load(results_path, allow_pickle=True).item()
            print(f"Loaded: {generation_type} - {persona_type}")
        else:
            print(f"Missing: {results_path}")

# Load discriminability results for SNR calculation
print("Loading discriminability results...")
with open(os.path.join(save_base, 'c_truncated_discriminability_results.json'), 'r') as f:
    truncated_discriminability_results_list = json.load(f)

# Convert to dictionary keyed by probe_idx for easy lookup
truncated_discriminability_results = {}
for item in truncated_discriminability_results_list:
    probe_idx = item['probe_idx']
    truncated_discriminability_results[probe_idx] = item

print("Processing results...")

# Storage for final results
results_data = []

# Process each generation type
for generation_type in generation_types:
    if generation_type not in steering_results:
        continue
        
    # Check if we have both aligned and misaligned data
    if 'aligned' not in steering_results[generation_type] or 'misaligned' not in steering_results[generation_type]:
        print(f"Skipping {generation_type}: missing aligned or misaligned data")
        continue
        
    print(f"Processing generation type: {generation_type}")
    
    # Get dimensions
    aligned_data = steering_results[generation_type]['aligned']
    n_probe_samples, n_order_samples = aligned_data['ordered_context_indices'].shape[:2]
    n_context_probes = aligned_data['ordered_context_indices'].shape[2]
    
    # Process each probe sample and order sample
    for probe_sample_idx in range(n_probe_samples):
        for order_idx in range(n_order_samples):
            
            # Get ordered context indices and append indices for this specific order
            ordered_context_indices = aligned_data['ordered_context_indices'][probe_sample_idx, order_idx]
            context_append_indices = aligned_data['context_append_indices'][probe_sample_idx, order_idx]
            
            # Skip if this order wasn't processed yet (contains -1s)
            if np.any(ordered_context_indices == -1) or np.any(context_append_indices == -1):
                continue
            
            # Calculate order-specific SNR using actual append indices
            probe_effect_sizes = []
            for probe_pos in range(n_context_probes):
                probe_idx = ordered_context_indices[probe_pos]
                append_idx = context_append_indices[probe_pos]
                
                if probe_idx in truncated_discriminability_results:
                    effect_sizes_by_append = truncated_discriminability_results[probe_idx]['effect_sizes_by_append']
                    # Use the specific append type for this probe in this order
                    specific_effect_size = effect_sizes_by_append[append_idx]
                    probe_effect_sizes.append(abs(specific_effect_size))
            
            if len(probe_effect_sizes) == 0:
                continue
                
            order_snr = np.mean(probe_effect_sizes)
            
            # Get log probabilities for aligned and misaligned personas
            aligned_truth = steering_results[generation_type]['aligned']['question_truth_log_probs'][probe_sample_idx, order_idx]
            aligned_lie = steering_results[generation_type]['aligned']['question_lie_log_probs'][probe_sample_idx, order_idx]
            misaligned_truth = steering_results[generation_type]['misaligned']['question_truth_log_probs'][probe_sample_idx, order_idx]
            misaligned_lie = steering_results[generation_type]['misaligned']['question_lie_log_probs'][probe_sample_idx, order_idx]
            
            # Calculate truth-lie differences for each persona
            # Shape: [n_questions, n_stochastic_samples]
            aligned_diff = aligned_truth - aligned_lie
            misaligned_diff = misaligned_truth - misaligned_lie
            
            # Calculate effectiveness: aligned_diff - misaligned_diff
            # Shape: [n_questions, n_stochastic_samples]
            effectiveness_per_question = aligned_diff - misaligned_diff
            
            # Average across stochastic samples first, then get per-question effectiveness
            # Shape: [n_questions]
            effectiveness_per_question_mean = np.nanmean(effectiveness_per_question, axis=1)
            
            # Filter out questions with no valid data
            valid_questions = ~np.isnan(effectiveness_per_question_mean)
            if np.sum(valid_questions) == 0:
                continue
            
            valid_effectiveness = effectiveness_per_question_mean[valid_questions]
            
            # Calculate mean and std across questions for this order sample
            effectiveness_mean = np.mean(valid_effectiveness)
            effectiveness_std = np.std(valid_effectiveness)
            n_valid_questions = len(valid_effectiveness)
            
            # Store results
            results_data.append({
                'generation_type': generation_type,
                'probe_sample_idx': probe_sample_idx,
                'order_idx': order_idx,
                'order_snr': order_snr,
                'effectiveness_mean': effectiveness_mean,
                'effectiveness_std': effectiveness_std,
                'n_valid_questions': n_valid_questions,
                'ordered_context_indices': ordered_context_indices.tolist(),
                'context_append_indices': context_append_indices.tolist()
            })

# Convert to DataFrame
results_df = pd.DataFrame(results_data)

if len(results_df) == 0:
    print("No valid results found!")
    sys.exit(1)

print(f"Found {len(results_df)} valid data points")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

colors = plt.cm.Set1(np.linspace(0, 1, len(generation_types)))
color_map = dict(zip(generation_types, colors))

for idx, generation_type in enumerate(generation_types):
    ax = axes[idx]
    
    gen_data = results_df[results_df['generation_type'] == generation_type]
    
    if len(gen_data) == 0:
        ax.set_title(f'{generation_type}\n(No data)')
        ax.set_xlabel('Order-Specific SNR')
        ax.set_ylabel('Steering Effectiveness\n(Aligned - Misaligned)')
        continue
    
    x = gen_data['order_snr']
    y = gen_data['effectiveness_mean']
    yerr = gen_data['effectiveness_std']
    
    ax.errorbar(x, y, yerr=yerr, 
               color=color_map[generation_type], 
               marker='o',
               linestyle='none',
               alpha=0.7,
               capsize=3)
    
    ax.set_xlabel('Order-Specific SNR')
    ax.set_ylabel('Steering Effectiveness\n(Aligned - Misaligned)')
    ax.set_title(f'{generation_type}\n({len(gen_data)} order samples)')
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

# Hide empty subplots
for i in range(len(generation_types), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()

# Save figure
output_fig_path = os.path.join(save_base, 'd_steering_ability_vs_discriminability.png')
plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
print(f"Figure saved: {output_fig_path}")

# Save CSV
output_csv_path = os.path.join(save_base, 'd_steering_ability_vs_discriminability.csv')
results_df.to_csv(output_csv_path, index=False)
print(f"Data saved: {output_csv_path}")

# Print summary statistics
print("\nSummary:")
for generation_type in generation_types:
    gen_data = results_df[results_df['generation_type'] == generation_type]
    if len(gen_data) > 0:
        mean_effectiveness = gen_data['effectiveness_mean'].mean()
        mean_snr = gen_data['order_snr'].mean()
        correlation = gen_data[['order_snr', 'effectiveness_mean']].corr().iloc[0, 1]
        print(f"\n{generation_type}: {len(gen_data)} order samples")
        print(f"  Avg SNR: {mean_snr:.3f}, Avg effectiveness: {mean_effectiveness:.3f}")
        print(f"  SNR-effectiveness correlation: {correlation:.3f}")

# Overall correlation analysis
if len(results_df) > 1:
    overall_correlation = results_df[['order_snr', 'effectiveness_mean']].corr().iloc[0, 1]
    print(f"\nOverall SNR-effectiveness correlation: {overall_correlation:.3f}")