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
input_path = os.path.join(save_base, 'e_original_probe_consistency')

# Parameters
context_lengths = list(range(11))  # Now 0-10
persona_types = ['aligned', 'misaligned']

print("Loading probe information and consistency results...")

# Load probe mappings (contains all the info we need!)
with open(os.path.join(input_path, 'context_probes.json'), 'r') as f:
    context_probes = json.load(f)

with open(os.path.join(input_path, 'test_probes.json'), 'r') as f:
    test_probes = json.load(f)

# Create lookup dictionary: probe_idx -> effect_sign
probe_effect_signs = {}
for probe in context_probes + test_probes:
    probe_effect_signs[probe['probe_idx']] = probe['effect_sign']

print(f"Loaded {len(context_probes)} context probes and {len(test_probes)} test probes")

# Load consistency results
consistency_results = {}
for persona_type in persona_types:
    results_path = os.path.join(input_path, f'consistency_results_{persona_type}.npy')
    if os.path.exists(results_path):
        consistency_results[persona_type] = np.load(results_path, allow_pickle=True).item()
        print(f"Loaded: {persona_type}")

# Check if we have both personas
if 'aligned' not in consistency_results or 'misaligned' not in consistency_results:
    print("Missing aligned or misaligned data")
    sys.exit(1)

print("Processing results with vectorized operations...")

def process_consistency_results_vectorized():
    """Process consistency results using vectorized operations"""
    
    # Get both persona results
    aligned_data = consistency_results['aligned']
    misaligned_data = consistency_results['misaligned']
    
    # Extract arrays - shape: [test_probe_idx, context_length, probe_sample, order_sample]
    test_probe_indices = aligned_data['test_probe_indices']  # Same for both personas
    prob_yes_aligned = aligned_data['test_prob_yes']
    prob_no_aligned = aligned_data['test_prob_no']
    prob_yes_misaligned = misaligned_data['test_prob_yes']
    prob_no_misaligned = misaligned_data['test_prob_no']
    
    # Get effect signs for test probes directly from saved info
    n_test_probes = len(test_probes)
    effect_signs = np.array([probe['effect_sign'] for probe in test_probes])
    
    print(f"Found {len(test_probes)} test probes with known effect signs")
    
    # Create effect signs array with same shape as test_probe_indices for broadcasting
    # Shape: [test_probe_idx, context_length, probe_sample, order_sample]
    effect_signs_broadcasted = np.broadcast_to(
        effect_signs[:, np.newaxis, np.newaxis, np.newaxis],
        test_probe_indices.shape
    )
    
    # Create masks for valid data
    valid_mask = (
        (test_probe_indices != -1) & 
        ~np.isnan(prob_yes_aligned) & ~np.isnan(prob_no_aligned) &
        ~np.isnan(prob_yes_misaligned) & ~np.isnan(prob_no_misaligned) &
        (prob_yes_aligned > 0) & (prob_no_aligned > 0) &
        (prob_yes_misaligned > 0) & (prob_no_misaligned > 0)
    )
    
    # Vectorized log-odds calculation
    log_odds_aligned = np.log(prob_yes_aligned) - np.log(prob_no_aligned)
    log_odds_misaligned = np.log(prob_yes_misaligned) - np.log(prob_no_misaligned)
    
    # Normalize by effect direction - broadcasting works here
    normalized_log_odds_aligned = log_odds_aligned * effect_signs_broadcasted
    normalized_log_odds_misaligned = log_odds_misaligned * effect_signs_broadcasted
    
    # Apply valid mask
    normalized_log_odds_aligned = np.where(valid_mask, normalized_log_odds_aligned, np.nan)
    normalized_log_odds_misaligned = np.where(valid_mask, normalized_log_odds_misaligned, np.nan)
    
    # Calculate statistics across order samples (axis=3) and probe samples (axis=2)
    # Shape after: [test_probe_idx, context_length]
    aligned_mean_across_orders = np.nanmean(normalized_log_odds_aligned, axis=3)
    misaligned_mean_across_orders = np.nanmean(normalized_log_odds_misaligned, axis=3)
    
    aligned_mean_across_samples = np.nanmean(aligned_mean_across_orders, axis=2)
    misaligned_mean_across_samples = np.nanmean(misaligned_mean_across_orders, axis=2)
    
    aligned_std_across_samples = np.nanstd(aligned_mean_across_orders, axis=2)
    misaligned_std_across_samples = np.nanstd(misaligned_mean_across_orders, axis=2)

    # Calculate differences (aligned - misaligned)
    differences = aligned_mean_across_samples - misaligned_mean_across_samples
    
    return {
        'aligned_mean': aligned_mean_across_samples,  # [test_probe_idx, context_length]
        'aligned_std': aligned_std_across_samples,
        'misaligned_mean': misaligned_mean_across_samples,
        'misaligned_std': misaligned_std_across_samples,
        'differences': differences,
        'aligned_individual': aligned_mean_across_orders,  # [test_probe_idx, context_length, probe_sample]
        'misaligned_individual': misaligned_mean_across_orders,
        'effect_signs': effect_signs
    }, np.sum(valid_mask, axis=(2,3))  # Count of valid samples per [test_probe_idx, context_length]

# Process results
processed_data, valid_counts = process_consistency_results_vectorized()
print(f"Processed data: {np.sum(~np.isnan(processed_data['aligned_mean']))} valid data points")

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
persona_colors = {'aligned': 'blue', 'misaligned': 'red'}
persona_linestyles = {'aligned': '-', 'misaligned': '--'}
effect_sign_colors = {1: 'green', -1: 'orange'}  # positive: green, negative: orange

data = processed_data

# Plot 1: Individual probe responses
ax1 = axes[0]

n_test_probes = data['aligned_mean'].shape[0]
alpha_per_line = min(0.8, 1.0 / n_test_probes * 3)

# Plot individual probe lines
for test_probe_idx in range(n_test_probes):
    # Get data for this test probe across context lengths
    aligned_means = data['aligned_mean'][test_probe_idx, :]
    misaligned_means = data['misaligned_mean'][test_probe_idx, :]
    aligned_stds = data['aligned_std'][test_probe_idx, :]
    misaligned_stds = data['misaligned_std'][test_probe_idx, :]
    
    # Only plot where we have valid data
    valid_contexts = ~np.isnan(aligned_means) & ~np.isnan(misaligned_means)
    if not np.any(valid_contexts):
        continue
        
    x_vals = np.array(context_lengths)[valid_contexts]
    
    # Plot with error bars
    if test_probe_idx == 0:  # Only label first probe
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

valid_avg = ~np.isnan(aligned_avg) & ~np.isnan(misaligned_avg)
if np.any(valid_avg):
    x_vals = np.array(context_lengths)[valid_avg]
    ax1.plot(x_vals, aligned_avg[valid_avg], color=persona_colors['aligned'],
            linewidth=3, alpha=0.9, label='aligned (avg)')
    ax1.plot(x_vals, misaligned_avg[valid_avg], color=persona_colors['misaligned'],
            linewidth=3, alpha=0.9, label='misaligned (avg)')

ax1.set_xlabel('Context Length (N)')
ax1.set_ylabel('Effect-Normalized Log-Odds')
ax1.set_title('Probe Consistency: Aligned vs Misaligned')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Plot 2: Differences (colored by effect sign)
ax2 = axes[1]

# Individual probe difference lines - colored by effect sign
probe_annotations = []  # Store info for text annotations

for test_probe_idx in range(n_test_probes):
    diffs = data['differences'][test_probe_idx, :]
    effect_sign = data['effect_signs'][test_probe_idx]
    valid_contexts = ~np.isnan(diffs)
    
    if np.any(valid_contexts):
        x_vals = np.array(context_lengths)[valid_contexts]
        y_vals = diffs[valid_contexts]
        
        # Color by effect sign
        color = effect_sign_colors[effect_sign]
        
        # Label only first probe of each type
        if effect_sign == 1 and test_probe_idx == np.where(data['effect_signs'] == 1)[0][0]:
            label = 'Positive effect probes'
        elif effect_sign == -1 and test_probe_idx == np.where(data['effect_signs'] == -1)[0][0]:
            label = 'Negative effect probes'
        else:
            label = ""
            
        ax2.plot(x_vals, y_vals,
                color=color, alpha=alpha_per_line, 
                linewidth=1.5, marker='o', markersize=3, label=label)
        
        # Store annotation info (rightmost point)
        if len(x_vals) > 0:
            final_x = x_vals[-1]
            final_y = y_vals[-1]
            probe_text = test_probes[test_probe_idx]['probe_text']
            probe_annotations.append({
                'x': final_x,
                'y': final_y,
                'text': probe_text,
                'color': color,
                'effect_sign': effect_sign
            })

# Average difference line (black)
avg_diff = np.nanmean(data['differences'], axis=0)
std_diff = np.nanstd(data['differences'], axis=0)
valid_diff = ~np.isnan(avg_diff)

if np.any(valid_diff):
    x_vals = np.array(context_lengths)[valid_diff]
    ax2.errorbar(x_vals, avg_diff[valid_diff], yerr=std_diff[valid_diff],
                color='black', linewidth=3, marker='o', markersize=6,
                capsize=3, label='Average across probes')

# Add probe text annotations on the right side
if probe_annotations:
    # Sort by y-value to handle potential overlaps
    probe_annotations.sort(key=lambda x: x['y'])
    
    # Get plot boundaries for positioning
    xlim = ax2.get_xlim()
    ylim = ax2.get_ylim()
    
    for i, annotation in enumerate(probe_annotations):
        # Truncate long probe texts
        text = annotation['text']
        if len(text) > 60:
            text = text[:57] + "..."
        
        # Position text just outside the right edge of the plot
        text_x = xlim[1] + (xlim[1] - xlim[0]) * 0.02
        text_y = annotation['y']
        
        # Add colored background box
        bbox_props = dict(
            boxstyle="round,pad=0.3", 
            facecolor=annotation['color'], 
            alpha=0.3,
            edgecolor=annotation['color']
        )
        
        ax2.text(text_x, text_y, text, 
                fontsize=8, 
                verticalalignment='center',
                bbox=bbox_props,
                clip_on=False)

ax2.set_xlabel('Context Length (N)')
ax2.set_ylabel('Aligned - Misaligned')
ax2.set_title('Differential Effect (Colored by Probe Effect Sign)\nGreen: higher = aligned more likely to say yes\nYellow: higher = aligned more likely to say no')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Plot 3: Context composition (proportion of "yes" answers)
ax3 = axes[2]

# Calculate proportion of "yes" answers in context for each persona
aligned_data = consistency_results['aligned']
context_probe_indices = aligned_data['context_probe_indices']  # [test_probe_idx, context_length, probe_sample, order_sample, context_position]

# For each context configuration, calculate proportion of "yes" answers
context_yes_proportions = {'aligned': [], 'misaligned': []}

for context_length_idx, context_length in enumerate(context_lengths):
    if context_length == 0:
        # No context, skip
        context_yes_proportions['aligned'].append([])
        context_yes_proportions['misaligned'].append([])
        continue
    
    aligned_proportions_this_length = []
    misaligned_proportions_this_length = []
    
    # Iterate over test probes and probe samples
    for test_probe_idx in range(context_probe_indices.shape[0]):
        for probe_sample_idx in range(context_probe_indices.shape[2]):
            
            # Get context probe IDs for this configuration (average across order samples since composition should be same)
            context_ids = context_probe_indices[test_probe_idx, context_length_idx, probe_sample_idx, 0, :context_length]
            
            # Filter out invalid entries
            valid_context_ids = context_ids[context_ids != -1]
            
            if len(valid_context_ids) == 0:
                continue
            
            # For each persona, determine how many "yes" answers they would give
            aligned_yes_count = 0
            misaligned_yes_count = 0
            
            for context_probe_id in valid_context_ids:
                if context_probe_id in probe_effect_signs:
                    context_effect_sign = probe_effect_signs[context_probe_id]
                    
                    # Aligned persona: "yes" if positive effect, "no" if negative effect
                    if context_effect_sign > 0:
                        aligned_yes_count += 1
                    
                    # Misaligned persona: "no" if positive effect, "yes" if negative effect  
                    if context_effect_sign < 0:
                        misaligned_yes_count += 1
            
            # Calculate proportions
            if len(valid_context_ids) > 0:
                aligned_proportion = aligned_yes_count / len(valid_context_ids)
                misaligned_proportion = misaligned_yes_count / len(valid_context_ids)
                
                aligned_proportions_this_length.append(aligned_proportion)
                misaligned_proportions_this_length.append(misaligned_proportion)
    
    context_yes_proportions['aligned'].append(aligned_proportions_this_length)
    context_yes_proportions['misaligned'].append(misaligned_proportions_this_length)

# Plot the proportions
for persona_type in ['aligned', 'misaligned']:
    means = []
    stds = []
    
    for context_length_idx, context_length in enumerate(context_lengths):
        proportions = context_yes_proportions[persona_type][context_length_idx]
        if len(proportions) > 0:
            means.append(np.mean(proportions))
            stds.append(np.std(proportions))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    
    valid_mask = ~np.isnan(means)
    if np.any(valid_mask):
        x_vals = np.array(context_lengths)[valid_mask]
        y_vals = np.array(means)[valid_mask]
        yerr_vals = np.array(stds)[valid_mask]
        
        ax3.errorbar(x_vals, y_vals, yerr=yerr_vals,
                    color=persona_colors[persona_type], linewidth=2, marker='o', markersize=6,
                    capsize=3, label=f'{persona_type}')

ax3.set_xlabel('Context Length (N)')
ax3.set_ylabel('Proportion of "Yes" Answers in Context')
ax3.set_title('Context Composition by Persona')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-0.05, 1.05)
ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, label='50% baseline')

plt.tight_layout()

# Save figure
output_fig_path = os.path.join(save_base, 'e_original_probe_consistency_results.png')
plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
print(f"Figure saved: {output_fig_path}")

# Create summary DataFrame
summary_data = []
for context_length in context_lengths:
    if context_length < data['aligned_mean'].shape[1]:
        aligned_vals = data['aligned_mean'][:, context_length]
        misaligned_vals = data['misaligned_mean'][:, context_length]
        diff_vals = data['differences'][:, context_length]
        
        valid_mask = ~np.isnan(aligned_vals) & ~np.isnan(misaligned_vals)
        if np.any(valid_mask):
            # Overall stats
            summary_data.append({
                'context_length': context_length,
                'aligned_mean': np.nanmean(aligned_vals),
                'aligned_std': np.nanstd(aligned_vals),
                'misaligned_mean': np.nanmean(misaligned_vals),
                'misaligned_std': np.nanstd(misaligned_vals),
                'difference_mean': np.nanmean(diff_vals),
                'difference_std': np.nanstd(diff_vals),
                'n_probes': np.sum(valid_mask),
                'effect_sign': 'all'
            })
            
            # Stats by effect sign
            for effect_sign in [1, -1]:
                sign_mask = valid_mask & (data['effect_signs'] == effect_sign)
                if np.any(sign_mask):
                    summary_data.append({
                        'context_length': context_length,
                        'aligned_mean': np.nanmean(aligned_vals[sign_mask]),
                        'aligned_std': np.nanstd(aligned_vals[sign_mask]),
                        'misaligned_mean': np.nanmean(misaligned_vals[sign_mask]),
                        'misaligned_std': np.nanstd(misaligned_vals[sign_mask]),
                        'difference_mean': np.nanmean(diff_vals[sign_mask]),
                        'difference_std': np.nanstd(diff_vals[sign_mask]),
                        'n_probes': np.sum(sign_mask),
                        'effect_sign': 'positive' if effect_sign == 1 else 'negative'
                    })

summary_df = pd.DataFrame(summary_data)
output_csv_path = os.path.join(save_base, 'e_original_probe_consistency_summary.csv')
summary_df.to_csv(output_csv_path, index=False)
print(f"Summary data saved: {output_csv_path}")

# Print summary statistics
print("\nSummary:")
overall_summary = summary_df[summary_df['effect_sign'] == 'all']
if len(overall_summary) > 0:
    baseline = overall_summary[overall_summary['context_length'] == 0]
    max_context = overall_summary[overall_summary['context_length'] == overall_summary['context_length'].max()]
    
    if len(baseline) > 0 and len(max_context) > 0:
        baseline_diff = baseline['difference_mean'].iloc[0]
        max_diff = max_context['difference_mean'].iloc[0]
        print(f"Overall difference (aligned-misaligned): {baseline_diff:.3f} → {max_diff:.3f} (Δ={max_diff-baseline_diff:+.3f})")
        print(f"Max probes tested: {overall_summary['n_probes'].max()}")

# Print by effect sign
for effect_sign in ['positive', 'negative']:
    sign_summary = summary_df[summary_df['effect_sign'] == effect_sign]
    if len(sign_summary) > 0:
        baseline = sign_summary[sign_summary['context_length'] == 0]
        max_context = sign_summary[sign_summary['context_length'] == sign_summary['context_length'].max()]
        
        if len(baseline) > 0 and len(max_context) > 0:
            baseline_diff = baseline['difference_mean'].iloc[0]
            max_diff = max_context['difference_mean'].iloc[0]
            print(f"{effect_sign.capitalize()} probes: {baseline_diff:.3f} → {max_diff:.3f} (Δ={max_diff-baseline_diff:+.3f})")

plt.show()