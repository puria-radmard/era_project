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
input_path = os.path.join(save_base, 'e_probe_consistency')

# Parameters
context_lengths = list(range(10))
generation_types = ['lie-truth_contrastive', 'truth-lie_contrastive', 'lie-only', 'truth-only']
persona_types = ['aligned', 'misaligned']

print("Loading probe consistency results...")

# Load discriminability results
with open(os.path.join(save_base, 'c_truncated_discriminability_results.json'), 'r') as f:
    truncated_discriminability_results_list = json.load(f)

# Convert to lookup arrays for vectorized access
max_probe_idx = max(item['probe_idx'] for item in truncated_discriminability_results_list) + 1
effect_sizes_lookup = np.full(max_probe_idx, np.nan)
for item in truncated_discriminability_results_list:
    effect_sizes_lookup[item['probe_idx']] = np.mean(item['effect_sizes_by_append'])

# Load probe data
truncated_probe_questions_df = pd.read_csv(os.path.join(save_base, "c_truncated_probe_completions.csv"))
banned_words = args.banned_words
probe_questions = truncated_probe_questions_df[
    ~truncated_probe_questions_df['generated_sequence'].str.lower().apply(
        lambda x: any(word in x for word in banned_words)
    )
]
original_probe_indices = sorted(probe_questions['probe_idx'].unique())

print(f"Found {len(original_probe_indices)} original probe indices")

# Load all consistency results
consistency_results = {}
for generation_type in generation_types:
    consistency_results[generation_type] = {}
    for persona_type in persona_types:
        results_path = os.path.join(input_path, f'consistency_results_{generation_type}_{persona_type}.npy')
        if os.path.exists(results_path):
            consistency_results[generation_type][persona_type] = np.load(results_path, allow_pickle=True).item()
            print(f"Loaded: {generation_type} - {persona_type}")

print("Processing results with vectorized operations...")

def process_generation_type_vectorized(generation_type):
    """Process a single generation type using vectorized operations"""
    
    if generation_type not in consistency_results:
        return None, None
    
    # Check if we have both personas
    if 'aligned' not in consistency_results[generation_type] or 'misaligned' not in consistency_results[generation_type]:
        print(f"Skipping {generation_type}: missing persona data")
        return None, None
    
    # Get both persona results
    aligned_data = consistency_results[generation_type]['aligned']
    misaligned_data = consistency_results[generation_type]['misaligned']
    
    # Extract arrays - shape: [test_probe_pos, context_length, probe_sample, order_sample]
    test_probe_indices = aligned_data['test_probe_indices']  # Same for both personas
    prob_yes_aligned = aligned_data['test_prob_yes']
    prob_no_aligned = aligned_data['test_prob_no']
    prob_yes_misaligned = misaligned_data['test_prob_yes']
    prob_no_misaligned = misaligned_data['test_prob_no']
    
    # Get effect sizes for all test probes - vectorized lookup
    effect_sizes = effect_sizes_lookup[test_probe_indices]
    
    # Create masks for valid data
    valid_mask = (
        (test_probe_indices != -1) & 
        ~np.isnan(prob_yes_aligned) & ~np.isnan(prob_no_aligned) &
        ~np.isnan(prob_yes_misaligned) & ~np.isnan(prob_no_misaligned) &
        (prob_yes_aligned > 0) & (prob_no_aligned > 0) &
        (prob_yes_misaligned > 0) & (prob_no_misaligned > 0) &
        ~np.isnan(effect_sizes)
    )
    
    # Vectorized log-odds calculation
    log_odds_aligned = np.log(prob_yes_aligned) - np.log(prob_no_aligned)
    log_odds_misaligned = np.log(prob_yes_misaligned) - np.log(prob_no_misaligned)
    
    # Normalize by effect direction - broadcasting works here
    normalized_log_odds_aligned = np.sign(effect_sizes) * log_odds_aligned
    normalized_log_odds_misaligned = np.sign(effect_sizes) * log_odds_misaligned
    
    # Apply valid mask
    normalized_log_odds_aligned = np.where(valid_mask, normalized_log_odds_aligned, np.nan)
    normalized_log_odds_misaligned = np.where(valid_mask, normalized_log_odds_misaligned, np.nan)
    
    # Calculate statistics across order samples (axis=3) and probe samples (axis=2)
    # Shape after: [test_probe_pos, context_length]
    aligned_mean_across_orders = np.nanmean(normalized_log_odds_aligned, axis=3)
    misaligned_mean_across_orders = np.nanmean(normalized_log_odds_misaligned, axis=3)
    
    aligned_mean_across_samples = np.nanmean(aligned_mean_across_orders, axis=2)
    misaligned_mean_across_samples = np.nanmean(misaligned_mean_across_orders, axis=2)
    
    aligned_std_across_samples = np.nanstd(aligned_mean_across_orders, axis=2)
    misaligned_std_across_samples = np.nanstd(misaligned_mean_across_orders, axis=2)
    
    # Calculate differences (aligned - misaligned)
    differences = aligned_mean_across_samples - misaligned_mean_across_samples
    
    return {
        'aligned_mean': aligned_mean_across_samples,  # [test_probe_pos, context_length]
        'aligned_std': aligned_std_across_samples,
        'misaligned_mean': misaligned_mean_across_samples,
        'misaligned_std': misaligned_std_across_samples,
        'differences': differences,
        'aligned_individual': aligned_mean_across_orders,  # [test_probe_pos, context_length, probe_sample]
        'misaligned_individual': misaligned_mean_across_orders,
    }, np.sum(valid_mask, axis=(2,3))  # Count of valid samples per [test_probe_pos, context_length]

# Process all generation types
all_processed = {}
for generation_type in generation_types:
    result, valid_counts = process_generation_type_vectorized(generation_type)
    if result is not None:
        all_processed[generation_type] = result
        print(f"Processed {generation_type}: {np.sum(~np.isnan(result['aligned_mean']))} valid data points")

# Create visualization
fig, axes = plt.subplots(4, 2, figsize=(15, 16))
persona_colors = {'aligned': 'blue', 'misaligned': 'red'}
persona_linestyles = {'aligned': '-', 'misaligned': '--'}

for gen_idx, generation_type in enumerate(generation_types):
    if generation_type not in all_processed:
        axes[gen_idx, 0].set_title(f'{generation_type}\n(No data)')
        axes[gen_idx, 1].set_title(f'{generation_type}\n(No data)')
        continue
    
    data = all_processed[generation_type]
    
    # Plot 1: Individual probe responses
    ax1 = axes[gen_idx, 0]
    
    n_test_probes = data['aligned_mean'].shape[0]
    alpha_per_line = min(0.8, 1.0 / n_test_probes * 3)
    
    # Plot individual probe lines
    for test_probe_pos in range(n_test_probes):
        # Get data for this test probe across context lengths
        aligned_means = data['aligned_mean'][test_probe_pos, :]
        misaligned_means = data['misaligned_mean'][test_probe_pos, :]
        aligned_stds = data['aligned_std'][test_probe_pos, :]
        misaligned_stds = data['misaligned_std'][test_probe_pos, :]
        
        # Only plot where we have valid data
        valid_contexts = ~np.isnan(aligned_means) & ~np.isnan(misaligned_means)
        if not np.any(valid_contexts):
            continue
            
        x_vals = np.array(context_lengths)[valid_contexts]
        
        # Plot with error bars
        if test_probe_pos == 0:  # Only label first probe
            label_aligned, label_misaligned = 'aligned', 'misaligned'
        else:
            label_aligned, label_misaligned = "", ""
            
        ax1.errorbar(x_vals, aligned_means[valid_contexts], 
                    yerr=aligned_stds[valid_contexts],
                    color=persona_colors['aligned'], linestyle=persona_linestyles['aligned'],
                    alpha=alpha_per_line, capsize=2, linewidth=1.5, markersize=4,
                    label=label_aligned)
        
        ax1.errorbar(x_vals, misaligned_means[valid_contexts],
                    yerr=misaligned_stds[valid_contexts], 
                    color=persona_colors['misaligned'], linestyle=persona_linestyles['misaligned'],
                    alpha=alpha_per_line, capsize=2, linewidth=1.5, markersize=4,
                    label=label_misaligned)
    
    # Add average trend lines
    aligned_avg = np.nanmean(data['aligned_mean'], axis=0)
    misaligned_avg = np.nanmean(data['misaligned_mean'], axis=0)
    aligned_avg_std = np.nanstd(data['aligned_mean'], axis=0)
    misaligned_avg_std = np.nanstd(data['misaligned_mean'], axis=0)
    
    valid_avg = ~np.isnan(aligned_avg) & ~np.isnan(misaligned_avg)
    if np.any(valid_avg):
        x_vals = np.array(context_lengths)[valid_avg]
        ax1.plot(x_vals, aligned_avg[valid_avg], color=persona_colors['aligned'],
                linewidth=3, alpha=0.9, label='aligned (avg)')
        ax1.plot(x_vals, misaligned_avg[valid_avg], color=persona_colors['misaligned'],
                linewidth=3, alpha=0.9, label='misaligned (avg)')
    
    ax1.set_xlabel('Context Length (N)')
    ax1.set_ylabel('Effect-Normalized Log-Odds')
    ax1.set_title(f'{generation_type}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 2: Differences (vectorized)
    ax2 = axes[gen_idx, 1]
    
    # Individual probe difference lines
    for test_probe_pos in range(n_test_probes):
        diffs = data['differences'][test_probe_pos, :]
        valid_contexts = ~np.isnan(diffs)
        
        if np.any(valid_contexts):
            x_vals = np.array(context_lengths)[valid_contexts]
            label = 'Individual probes' if test_probe_pos == 0 else ""
            ax2.plot(x_vals, diffs[valid_contexts],
                    color='green', alpha=alpha_per_line, 
                    linewidth=1.5, marker='o', markersize=3, label=label)
    
    # Average difference line
    avg_diff = np.nanmean(data['differences'], axis=0)
    std_diff = np.nanstd(data['differences'], axis=0)
    valid_diff = ~np.isnan(avg_diff)
    
    if np.any(valid_diff):
        x_vals = np.array(context_lengths)[valid_diff]
        ax2.errorbar(x_vals, avg_diff[valid_diff], yerr=std_diff[valid_diff],
                    color='darkgreen', linewidth=3, marker='o', markersize=6,
                    capsize=3, label='Average across probes')
    
    ax2.set_xlabel('Context Length (N)')
    ax2.set_ylabel('Aligned - Misaligned')
    ax2.set_title(f'{generation_type} (Differential Effect)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()

# Save figure
output_fig_path = os.path.join(save_base, 'e_probe_consistency_results.png')
plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
print(f"Figure saved: {output_fig_path}")

# Create summary DataFrame (much more efficient)
summary_data = []
for generation_type, data in all_processed.items():
    for context_length in context_lengths:
        if context_length < data['aligned_mean'].shape[1]:
            aligned_vals = data['aligned_mean'][:, context_length]
            misaligned_vals = data['misaligned_mean'][:, context_length]
            diff_vals = data['differences'][:, context_length]
            
            valid_mask = ~np.isnan(aligned_vals) & ~np.isnan(misaligned_vals)
            if np.any(valid_mask):
                summary_data.append({
                    'generation_type': generation_type,
                    'context_length': context_length,
                    'aligned_mean': np.nanmean(aligned_vals),
                    'aligned_std': np.nanstd(aligned_vals),
                    'misaligned_mean': np.nanmean(misaligned_vals),
                    'misaligned_std': np.nanstd(misaligned_vals),
                    'difference_mean': np.nanmean(diff_vals),
                    'difference_std': np.nanstd(diff_vals),
                    'n_probes': np.sum(valid_mask)
                })

summary_df = pd.DataFrame(summary_data)
output_csv_path = os.path.join(save_base, 'e_probe_consistency_summary.csv')
summary_df.to_csv(output_csv_path, index=False)
print(f"Summary data saved: {output_csv_path}")

# Print summary statistics
print("\nSummary:")
for generation_type in generation_types:
    if generation_type in all_processed:
        gen_summary = summary_df[summary_df['generation_type'] == generation_type]
        if len(gen_summary) > 0:
            print(f"\n{generation_type}:")
            baseline = gen_summary[gen_summary['context_length'] == 0]
            max_context = gen_summary[gen_summary['context_length'] == gen_summary['context_length'].max()]
            
            if len(baseline) > 0 and len(max_context) > 0:
                baseline_diff = baseline['difference_mean'].iloc[0]
                max_diff = max_context['difference_mean'].iloc[0]
                print(f"  Difference (aligned-misaligned): {baseline_diff:.3f} → {max_diff:.3f} (Δ={max_diff-baseline_diff:+.3f})")
                print(f"  Max probes tested: {gen_summary['n_probes'].max()}")

plt.show()