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
context_lengths = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
generation_types = ['lie-truth_contrastive', 'truth-lie_contrastive', 'lie-only', 'truth-only']
persona_types = ['aligned', 'misaligned']

print("Loading probe consistency results...")

# Load discriminability results for effect size lookup
with open(os.path.join(save_base, 'c_truncated_discriminability_results.json'), 'r') as f:
    truncated_discriminability_results_list = json.load(f)

truncated_discriminability_results = {}
for item in truncated_discriminability_results_list:
    probe_idx = item['probe_idx']
    truncated_discriminability_results[probe_idx] = item

# Load probe data to get original probe indices (same as generation script)
truncated_probe_questions_df = pd.read_csv(os.path.join(save_base, "c_truncated_probe_completions.csv"))

# Load banned words from args
banned_words = args.banned_words

# Filter out banned words
probe_questions = truncated_probe_questions_df[~truncated_probe_questions_df['generated_sequence'].str.lower().apply(lambda x: any(word in x for word in banned_words))]

# Get the original probe indices that were augmented (same as generation script)
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
        else:
            print(f"Missing: {results_path}")

print("Processing results...")

# Storage for processed data
processed_data = []

# Process each generation type
for generation_type in generation_types:
    if generation_type not in consistency_results:
        continue
    
    # Check if we have both personas
    if 'aligned' not in consistency_results[generation_type] or 'misaligned' not in consistency_results[generation_type]:
        print(f"Skipping {generation_type}: missing persona data")
        continue
    
    print(f"Processing generation type: {generation_type}")
    
    # Get data dimensions
    aligned_data = consistency_results[generation_type]['aligned']
    n_test_original_probes, max_context_length_plus1, n_probe_samples, n_order_samples = aligned_data['test_prob_yes'].shape
    
    # Process each test original probe
    for test_original_probe_idx_pos in range(n_test_original_probes):
        test_original_probe_idx = original_probe_indices[test_original_probe_idx_pos]
        
        # Process each context length
        for context_length in context_lengths:
            if context_length >= max_context_length_plus1:
                continue
                
            # Process each probe sample
            for probe_sample_idx in range(n_probe_samples):
                
                # Collect normalized log-odds across order samples for both personas
                aligned_normalized_log_odds = []
                misaligned_normalized_log_odds = []
                
                for order_idx in range(n_order_samples):
                    
                    # Get test probe information
                    test_probe_index = consistency_results[generation_type]['aligned']['test_probe_indices'][test_original_probe_idx_pos, context_length, probe_sample_idx, order_idx]
                    
                    # Skip if not processed
                    if test_probe_index == -1:
                        continue
                    
                    # Get test probe effect size
                    if test_probe_index not in truncated_discriminability_results:
                        continue
                    
                    test_result = truncated_discriminability_results[test_probe_index]
                    test_effect_size = np.mean(test_result['effect_sizes_by_append'])
                    
                    # Process both personas
                    for persona_type in persona_types:
                        prob_yes = consistency_results[generation_type][persona_type]['test_prob_yes'][test_original_probe_idx_pos, context_length, probe_sample_idx, order_idx]
                        prob_no = consistency_results[generation_type][persona_type]['test_prob_no'][test_original_probe_idx_pos, context_length, probe_sample_idx, order_idx]
                        
                        # Skip if no valid data
                        if np.isnan(prob_yes) or np.isnan(prob_no) or prob_yes <= 0 or prob_no <= 0:
                            continue
                        
                        # Calculate log-odds and normalize by effect direction
                        # i.e. test_effect_size > 0 --> higher log-odds when aligned and vice versa
                        # So always expect higher normalized_log_odds when aligned
                        log_odds = np.log(prob_yes) - np.log(prob_no)
                        normalized_log_odds = np.sign(test_effect_size) * log_odds
                        
                        if persona_type == 'aligned':
                            aligned_normalized_log_odds.append(normalized_log_odds)
                        else:
                            misaligned_normalized_log_odds.append(normalized_log_odds)
                
                # Calculate statistics across order samples for this probe sample
                if len(aligned_normalized_log_odds) > 0:
                    aligned_mean = np.mean(aligned_normalized_log_odds)
                    aligned_std = np.std(aligned_normalized_log_odds) if len(aligned_normalized_log_odds) > 1 else 0.0
                    
                    processed_data.append({
                        'generation_type': generation_type,
                        'test_original_probe_idx': test_original_probe_idx,
                        'test_original_probe_idx_pos': test_original_probe_idx_pos,
                        'context_length': context_length,
                        'probe_sample_idx': probe_sample_idx,
                        'persona_type': 'aligned',
                        'mean_normalized_log_odds': aligned_mean,
                        'std_normalized_log_odds': aligned_std,
                        'n_order_samples': len(aligned_normalized_log_odds)
                    })
                
                if len(misaligned_normalized_log_odds) > 0:
                    misaligned_mean = np.mean(misaligned_normalized_log_odds)
                    misaligned_std = np.std(misaligned_normalized_log_odds) if len(misaligned_normalized_log_odds) > 1 else 0.0
                    
                    processed_data.append({
                        'generation_type': generation_type,
                        'test_original_probe_idx': test_original_probe_idx,
                        'test_original_probe_idx_pos': test_original_probe_idx_pos,
                        'context_length': context_length,
                        'probe_sample_idx': probe_sample_idx,
                        'persona_type': 'misaligned',
                        'mean_normalized_log_odds': misaligned_mean,
                        'std_normalized_log_odds': misaligned_std,
                        'n_order_samples': len(misaligned_normalized_log_odds)
                    })

# Convert to DataFrame
df = pd.DataFrame(processed_data)

if len(df) == 0:
    print("No valid results found!")
    sys.exit(1)

print(f"Found {len(df)} valid data points across {df['generation_type'].nunique()} generation types and {df['test_original_probe_idx'].nunique()} test probes")

# Create visualization: 4 rows (generation types) × 2 columns
fig, axes = plt.subplots(4, 2, figsize=(15, 16))

persona_colors = {'aligned': 'blue', 'misaligned': 'red'}
persona_linestyles = {'aligned': '-', 'misaligned': '--'}

# Process each generation type (one row per generation type)
for gen_idx, generation_type in enumerate(generation_types):
    gen_data = df[df['generation_type'] == generation_type]
    
    if len(gen_data) == 0:
        # Empty plots
        axes[gen_idx, 0].set_title(f'{generation_type}\n(No data)')
        axes[gen_idx, 1].set_title(f'{generation_type}\n(No data)')
        continue
    
    # Plot 1: Effect-normalized log-odds (separate lines per test probe)
    ax1 = axes[gen_idx, 0]
    
    # Get unique test probes for this generation type
    test_probes = sorted(gen_data['test_original_probe_idx'].unique())
    alpha_per_line = min(0.8, 1.0 / len(test_probes) * 3)
    
    for test_probe_idx in test_probes:
        test_probe_data = gen_data[gen_data['test_original_probe_idx'] == test_probe_idx]
        
        for persona_type in persona_types:
            persona_data = test_probe_data[test_probe_data['persona_type'] == persona_type]
            
            if len(persona_data) > 0:
                # Average across probe samples, get std across probe samples
                context_stats = persona_data.groupby('context_length').agg({
                    'mean_normalized_log_odds': ['mean', 'std']
                }).round(4)
                
                context_stats.columns = ['mean', 'std']
                context_stats = context_stats.dropna()
                
                if len(context_stats) > 0:
                    ax1.errorbar(context_stats.index, 
                                context_stats['mean'],
                                yerr=context_stats['std'],
                                color=persona_colors[persona_type],
                                linestyle=persona_linestyles[persona_type],
                                alpha=alpha_per_line,
                                capsize=2, linewidth=1.5, markersize=4,
                                label=f'{persona_type}' if test_probe_idx == test_probes[0] else "")
    
    # Add average trend lines for this generation type
    for persona_type in persona_types:
        persona_all_data = gen_data[gen_data['persona_type'] == persona_type]
        
        # Calculate mean across all test probes and probe samples
        avg_stats = persona_all_data.groupby('context_length')['mean_normalized_log_odds'].agg(['mean', 'std'])
        avg_stats = avg_stats.dropna()
        
        if len(avg_stats) > 0:
            ax1.plot(avg_stats.index, 
                    avg_stats['mean'],
                    color=persona_colors[persona_type],
                    linewidth=3, alpha=0.9,
                    label=f'{persona_type} (avg)')
    
    ax1.set_xlabel('Context Length (N)')
    ax1.set_ylabel('Effect-Normalized Log-Odds')
    ax1.set_title(f'{generation_type}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 2: Difference (aligned - misaligned) per test probe
    ax2 = axes[gen_idx, 1]
    
    # Calculate differences for each test probe
    difference_data = []
    
    for test_probe_idx in test_probes:
        test_probe_data = gen_data[gen_data['test_original_probe_idx'] == test_probe_idx]
        
        for context_length in context_lengths:
            aligned_data = test_probe_data[(test_probe_data['context_length'] == context_length) & 
                                         (test_probe_data['persona_type'] == 'aligned')]
            misaligned_data = test_probe_data[(test_probe_data['context_length'] == context_length) & 
                                            (test_probe_data['persona_type'] == 'misaligned')]
            
            if len(aligned_data) > 0 and len(misaligned_data) > 0:
                # Average across probe samples
                aligned_mean = aligned_data['mean_normalized_log_odds'].mean()
                misaligned_mean = misaligned_data['mean_normalized_log_odds'].mean()
                difference = aligned_mean - misaligned_mean
                
                difference_data.append({
                    'test_probe_idx': test_probe_idx,
                    'context_length': context_length,
                    'difference': difference
                })
    
    if difference_data:
        diff_df = pd.DataFrame(difference_data)
        
        # Plot individual test probe difference lines
        for test_probe_idx in test_probes:
            probe_diff_data = diff_df[diff_df['test_probe_idx'] == test_probe_idx]
            if len(probe_diff_data) > 0:
                probe_diff_data = probe_diff_data.sort_values('context_length')
                ax2.plot(probe_diff_data['context_length'], 
                        probe_diff_data['difference'],
                        color='green', alpha=alpha_per_line, 
                        linewidth=1.5, marker='o', markersize=3,
                        label='Individual probes' if test_probe_idx == test_probes[0] else "")
        
        # Add average difference line
        avg_differences = diff_df.groupby('context_length')['difference'].agg(['mean', 'std'])
        avg_differences = avg_differences.dropna()
        
        if len(avg_differences) > 0:
            ax2.errorbar(avg_differences.index, 
                        avg_differences['mean'],
                        yerr=avg_differences['std'],
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

# Save processed data
output_csv_path = os.path.join(save_base, 'e_probe_consistency_results.csv')
df.to_csv(output_csv_path, index=False)
print(f"Data saved: {output_csv_path}")

# Print summary statistics
print("\nSummary:")
for generation_type in generation_types:
    gen_data = df[df['generation_type'] == generation_type]
    if len(gen_data) > 0:
        print(f"\n{generation_type}:")
        test_probes = gen_data['test_original_probe_idx'].nunique()
        max_context = gen_data['context_length'].max()
        baseline_data = gen_data[gen_data['context_length'] == 0]
        max_data = gen_data[gen_data['context_length'] == max_context]
        
        print(f"  Test probes: {test_probes}, Max context: {max_context}")
        
        if len(baseline_data) > 0 and len(max_data) > 0:
            for persona_type in persona_types:
                baseline_mean = baseline_data[baseline_data['persona_type'] == persona_type]['mean_normalized_log_odds'].mean()
                max_mean = max_data[max_data['persona_type'] == persona_type]['mean_normalized_log_odds'].mean()
                print(f"    {persona_type}: {baseline_mean:.3f} → {max_mean:.3f} (Δ={max_mean-baseline_mean:+.3f})")

plt.show()