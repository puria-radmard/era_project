#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
from scipy.stats import ttest_rel
import json
import sys
import os


def load_and_preprocess_data_modified(probe_responses_path: str, probe_questions_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess data with probe categories and log odds calculation."""
    
    # Load probe responses
    data = pd.read_csv(probe_responses_path)
    
    # Calculate log odds
    epsilon = 1e-10
    data['prob_yes'] = np.clip(data['prob_yes'], epsilon, 1 - epsilon)
    data['prob_no'] = np.clip(data['prob_no'], epsilon, 1 - epsilon)
    data['log_odds'] = np.log(data['prob_yes'] / data['prob_no'])
    
    # Load probe questions and create categories
    probe_df = pd.read_csv(probe_questions_path)
    probe_df['probe_question_idx'] = probe_df.index
    
    # Create probe categories: N={context_length}|max={lie if lie_maximisation else truth}
    probe_df['probe_type'] = probe_df.apply(
        lambda row: f"N={row['context_length']}|max={'lie' if row['lie_maximisation'] else 'truth'}", 
        axis=1
    )

    # Merge probe categories into main data
    data = data.merge(
        probe_df[['probe_question_idx', 'probe_type', 'generated_question', 'question_achieved']],
        on='probe_question_idx'
    ).rename(columns={'generated_question': 'probe'})
    
    # data = data[data.question_idx < 11]

    data = data[data['question_achieved'] == True]
    probe_df = probe_df[probe_df['question_achieved'] == True]
    
    return data, probe_df


def compute_probe_discriminability_modified(data: pd.DataFrame) -> Dict:
    """Compute discriminability statistics for each probe question with 4 conditions."""
    
    probe_info = data[['probe_question_idx', 'probe', 'probe_type']].drop_duplicates().sort_values('probe_question_idx')
    
    probe_results = []
    significant_count = 0
    effect_sizes = []
    
    for _, probe_row in probe_info.iterrows():
        probe_idx = probe_row['probe_question_idx']
        probe_data = data[data['probe_question_idx'] == probe_idx]
        
        # Analyze each i_append_string value separately
        effect_sizes_by_append = []
        p_values_by_append = []
        stats_by_append = []

        for i_append in [0.0, 1.0]:
            append_data = probe_data[probe_data['i_append_string'] == i_append]
            
            # Pivot to get paired data
            pivot_data = append_data.pivot_table(
                index='question_idx', 
                columns='truth', 
                values='log_odds'
            ).dropna()
            
            truth_0_values = pivot_data[0].values
            truth_1_values = pivot_data[1].values
            stat, p_value = ttest_rel(truth_1_values, truth_0_values)
            
            # Calculate Cohen's d for paired samples
            differences = truth_1_values - truth_0_values
            cohens_d = np.mean(differences) / np.std(differences, ddof=1)
            
            effect_sizes_by_append.append(cohens_d)
            p_values_by_append.append(p_value)
            stats_by_append.append({
                'n_pairs': len(pivot_data),
                'mean_truth_0': float(np.mean(truth_0_values)),
                'mean_truth_1': float(np.mean(truth_1_values)),
                'mean_difference': float(np.mean(differences)),
                'test_statistic': float(stat),
                'p_value': float(p_value),
                'effect_size': float(cohens_d)
            })
        
        # Check if effect sizes have same sign (robustness check)
        if len(effect_sizes_by_append) == 2:
            signs_match = np.sign(effect_sizes_by_append[0]) == np.sign(effect_sizes_by_append[1])
            average_effect_size = np.mean(effect_sizes_by_append)
        else:
            signs_match = True
            average_effect_size = effect_sizes_by_append[0] if effect_sizes_by_append else 0.0
            
        # Determine significance (if either comparison is significant)
        any_significant = any(p < 0.05 for p in p_values_by_append)
        
        probe_results.append({
            'probe_type': probe_row['probe_type'],
            'effect_sizes_by_append': effect_sizes_by_append,
            'p_values_by_append': p_values_by_append,
            'stats_by_append': stats_by_append,
            'signs_match': bool(signs_match),
            'average_effect_size': float(average_effect_size),
            'abs_average_effect_size': float(abs(average_effect_size)),
            'significant': bool(any_significant),
            'robustness_penalty': 0.0 if signs_match else 1.0  # For ordering
        })
        
        if any_significant:
            significant_count += 1
        effect_sizes.append(abs(average_effect_size))
    
    return {
        'probe_results': probe_results,
        'overall_stats': {
            'total_probes': len(probe_results),
            'significant_probes': significant_count,
            'mean_effect_size': float(np.mean(effect_sizes) if effect_sizes else 0)
        }
    }


def create_probe_boxplot_modified(data: pd.DataFrame) -> plt.Axes:
    """Create boxplot with 4 hues based on truth and i_append_string."""
    
    # Create combined category for hue
    data = data.copy()
    data['truth_append'] = data['truth'].astype(str) + '_' + data['i_append_string'].astype(str)
    
    # Define colors: red family for truth=0, blue family for truth=1
    palette = {
        '0_0.0': '#d62728',    # red
        '0_1.0': '#ff7f7f',    # light red  
        '1_0.0': '#1f77b4',    # blue
        '1_1.0': '#87ceeb'     # light blue
    }
    
    # Create box plot
    ax = sns.boxplot(
        data=data, 
        x='probe_question_idx', 
        y='log_odds', 
        hue='truth_append',
        palette=palette,
        width=0.6
    )
    
    return ax


def add_significance_stars_and_crosses_modified(ax: plt.Axes, data: pd.DataFrame, 
                                               probe_info: pd.DataFrame, 
                                               discriminability_results: Dict):
    """Add significance stars and robustness crosses to the plot."""
    
    for i, probe_result in enumerate(discriminability_results['probe_results']):
        probe_idx = probe_info.iloc[i]['probe_question_idx']
        max_y = data[data['probe_question_idx'] == probe_idx]['log_odds'].max()
        
        # Add significance star if significant
        if probe_result['significant']:
            ax.text(i, max_y + 0.15, '*', 
                   ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        # Add red cross if signs don't match
        if not probe_result['signs_match']:
            ax.text(i, max_y + 0.05, '❌', 
                   ha='center', va='center', fontsize=12)


def create_discriminability_ordered_plot_modified(data: pd.DataFrame, 
                                                 discriminability_results: Dict) -> plt.Axes:
    """Create second subplot ordered by discriminability with robustness penalty."""
    
    probe_info = data[['probe_question_idx', 'probe', 'probe_type']].drop_duplicates().sort_values('probe_question_idx')
    
    # Create ordering key
    probe_discriminability = []
    for i, (_, probe_row) in enumerate(probe_info.iterrows()):
        probe_idx = probe_row['probe_question_idx']
        result = discriminability_results['probe_results'][i]
        
        # Use absolute average effect size
        discriminability = result['abs_average_effect_size']
        probe_discriminability.append((probe_idx, discriminability))
    
    # Sort by discriminability (least to most)
    probe_discriminability.sort(key=lambda x: x[1])
    
    # Create mapping from probe_question_idx to new ordered position
    probe_idx_to_new_pos = {probe_idx: new_pos for new_pos, (probe_idx, _) in enumerate(probe_discriminability)}
    
    # Calculate overall mean for each probe
    probe_means = {}
    for _, row in probe_info.iterrows():
        probe_idx = row['probe_question_idx']
        probe_data = data[data['probe_question_idx'] == probe_idx]
        probe_means[probe_idx] = probe_data['log_odds'].mean()
    
    # Create transformed dataset
    transformed_data = []
    for _, row in data.iterrows():
        probe_idx = row['probe_question_idx']
        new_pos = probe_idx_to_new_pos[probe_idx]
        delta_quantity = row['log_odds'] - probe_means[probe_idx]
        
        # Create combined category
        truth_append = str(row['truth']) + '_' + str(row['i_append_string'])
        
        transformed_data.append({
            'ordered_probe_idx': new_pos,
            'delta_log_odds': delta_quantity,
            'truth_append': truth_append,
            'question_idx': row['question_idx']
        })
    
    transformed_df = pd.DataFrame(transformed_data)
    
    # Create the plot with same color palette
    palette = {
        '0_0.0': '#d62728',    # red
        '0_1.0': '#ff7f7f',    # light red  
        '1_0.0': '#1f77b4',    # blue
        '1_1.0': '#87ceeb'     # light blue
    }
    
    ax = sns.boxplot(
        data=transformed_df, 
        x='ordered_probe_idx', 
        y='delta_log_odds', 
        hue='truth_append',
        palette=palette,
        width=0.6
    )
    
    sns.stripplot(
        data=transformed_df, 
        x='ordered_probe_idx', 
        y='delta_log_odds', 
        hue='truth_append',
        palette=palette,
        dodge=True, 
        size=3, 
        alpha=0.6, 
        marker='x', 
        ax=ax
    )
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    return ax


def add_category_braces_modified(ax: plt.Axes, probe_info: pd.DataFrame):
    """Add category braces below the x-axis to group probe types."""
    
    # Group consecutive probe indices by probe_type
    groups = []
    current_type = None
    current_start = None
    
    for i, (_, row) in enumerate(probe_info.iterrows()):
        if row['probe_type'] != current_type:
            if current_type is not None:
                groups.append({
                    'type': current_type,
                    'start': current_start,
                    'end': i - 1
                })
            current_type = row['probe_type']
            current_start = i
    
    # Add the last group
    if current_type is not None:
        groups.append({
            'type': current_type,
            'start': current_start,
            'end': len(probe_info) - 1
        })
    
    # Draw braces and labels
    y_min = ax.get_ylim()[0]
    brace_height = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08
    
    for group in groups:
        start_x = group['start'] - 0.4
        end_x = group['end'] + 0.4
        center_x = (start_x + end_x) / 2
        
        # Draw horizontal line
        ax.plot([start_x, end_x], [y_min - brace_height, y_min - brace_height], 
               'k-', linewidth=1)
        
        # Draw vertical lines at ends
        ax.plot([start_x, start_x], [y_min - brace_height * 0.5, y_min - brace_height], 
               'k-', linewidth=1)
        ax.plot([end_x, end_x], [y_min - brace_height * 0.5, y_min - brace_height], 
               'k-', linewidth=1)
        
        # Add category label
        ax.text(center_x, y_min - brace_height * 2, group['type'], 
               ha='center', va='top', fontsize=10, fontweight='bold')


def plot_probe_type_analysis_modified(data: pd.DataFrame, filepath: str, enable_category_braces: bool = False) -> Dict:
    """Main plotting function with modified analysis for 4-condition design."""
    
    probe_info = data[['probe_question_idx', 'probe', 'probe_type']].drop_duplicates().sort_values('probe_question_idx')
    
    # Compute discriminability statistics
    discriminability_results = compute_probe_discriminability_modified(data)
    
    if filepath:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, len(probe_info)), 15))
        
        # First subplot: Original probe order
        plt.sca(ax1)
        ax1 = create_probe_boxplot_modified(data)
        add_significance_stars_and_crosses_modified(ax1, data, probe_info, discriminability_results)
        
        if enable_category_braces:
            add_category_braces_modified(ax1, probe_info)
        
        ax1.set_title('Log Odds Distribution by Probe Question\n(* = significant, ❌ = non-robust effect)', fontsize=20)
        ax1.set_xlabel('Probe Question Index', fontsize=18)
        ax1.set_ylabel('Log Odds', fontsize=18)
        ax1.tick_params(axis='x', rotation=45)
        
        # Custom legend
        legend_labels = ['Truth=0, Append=0', 'Truth=0, Append=1', 'Truth=1, Append=0', 'Truth=1, Append=1']
        handles, _ = ax1.get_legend_handles_labels()
        ax1.legend(handles[:4], legend_labels, fontsize=14)
        
        # Second subplot: Ordered by discriminability
        plt.sca(ax2)
        ax2 = create_discriminability_ordered_plot_modified(data, discriminability_results)
        
        ax2.set_xlabel('Probe Index (ordered by discriminability)', fontsize=18)
        ax2.set_ylabel('Δ Log Odds (centered at probe mean)', fontsize=18)
        
        # Handle legend for second plot
        handles, _ = ax2.get_legend_handles_labels()
        ax2.legend(handles[:4], legend_labels, fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    # Print results
    print("\nDiscriminability Analysis Results:")
    print("Probe | Append=0 P-val | Append=1 P-val | Avg Effect | Robust | Significant")
    print("-" * 80)
    
    for i, result in enumerate(discriminability_results['probe_results']):
        p0 = result['p_values_by_append'][0] if len(result['p_values_by_append']) > 0 else 1.0
        p1 = result['p_values_by_append'][1] if len(result['p_values_by_append']) > 1 else 1.0
        robust = "✓" if result['signs_match'] else "❌"
        sig = "*" if result['significant'] else ""
        
        print(f"{i:5d} | {p0:13.4f} | {p1:13.4f} | {result['average_effect_size']:10.3f} | {robust:6s} | {sig:11s}")
    
    return discriminability_results


def plot_effect_size_vs_context_length(data: pd.DataFrame, probe_df: pd.DataFrame, 
                                        discriminability_results: Dict, filepath: str):
    """Create scatterplot showing mean effect size vs context length, split by lie_maximisation."""
    
    probe_info = data[['probe_question_idx', 'probe', 'probe_type']].drop_duplicates().sort_values('probe_question_idx')
    
    # Prepare data for plotting
    plot_data = []
    
    for i, (_, probe_row) in enumerate(probe_info.iterrows()):
        probe_idx = probe_row['probe_question_idx']
        result = discriminability_results['probe_results'][i]
        
        # Get context length and lie_maximisation from probe_df
        probe_row_df = probe_df[probe_df['probe_question_idx'] == probe_idx].iloc[0]
        context_length = probe_row_df['context_length']
        lie_maximisation = probe_row_df['lie_maximisation']
        
        # Mean effect size (absolute of average of signed)
        mean_effect_size = abs(result['average_effect_size'])
        
        # Max effect size (max of absolute values across append types)
        if len(result['effect_sizes_by_append']) >= 2:
            max_effect_size = max(abs(result['effect_sizes_by_append'][0]), 
                                 abs(result['effect_sizes_by_append'][1]))
        elif len(result['effect_sizes_by_append']) == 1:
            max_effect_size = abs(result['effect_sizes_by_append'][0])
        else:
            max_effect_size = 0.0
        
        plot_data.append({
            'context_length': context_length,
            'lie_maximisation': lie_maximisation,
            'effect_size_mean': mean_effect_size,
            'effect_size_max': max_effect_size,
            'category': f"Questions generated by {'lie' if lie_maximisation else 'truth'} marginal",
            'probe_idx': probe_idx
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Define colors for the two categories
    palette = {
        'Questions generated by lie marginal': '#d62728',     # red (for lie maximisation)
        'Questions generated by truth marginal': '#1f77b4'     # blue (for truth maximisation)
    }
    
    # Add scatterplots with offsets and jitter
    unique_contexts = sorted(plot_df['context_length'].unique())
    jitter_amount = 0.05

    # Collect mean lines for both metrics
    category_to_mean_line_x = {}
    category_to_mean_line_y = {}
    category_to_max_line_x = {}
    category_to_max_line_y = {}
    
    for category, offset in [('Questions generated by lie marginal', -0.1), ('Questions generated by truth marginal', 0.1)]:
        category_data = plot_df[plot_df['category'] == category]
        
        for context in unique_contexts:
            context_data = category_data[category_data['context_length'] == context]
            
            if len(context_data) > 0:
                # Use context length directly as base position
                base_position = context
                x_positions = base_position + offset + np.random.normal(0, jitter_amount, len(context_data))
                y_values = context_data['effect_size_mean'].values
                
                ax.scatter(x_positions, y_values, 
                          color=palette[category], alpha=0.3, s=30, 
                          edgecolors='black', linewidth=0.5)

                # Calculate means and stds for both metrics
                mean_y_mean = context_data['effect_size_mean'].mean()
                std_y_mean = context_data['effect_size_mean'].std()
                mean_y_max = context_data['effect_size_max'].mean()
                
                # Plot mean effect size (solid)
                ax.plot([base_position + offset], [mean_y_mean], marker='o', color=palette[category], 
                       markersize=10, label=f"{category} (mean across addendum)" if context == unique_contexts[0] else "")
                ax.errorbar([base_position + offset], [mean_y_mean], yerr=[std_y_mean], fmt='none', 
                           ecolor=palette[category], elinewidth=2, capsize=6)

                # Collect data for connecting lines
                category_to_mean_line_y[category] = category_to_mean_line_y.get(category, []) + [mean_y_mean]
                category_to_mean_line_x[category] = category_to_mean_line_x.get(category, []) + [base_position + offset]
                category_to_max_line_y[category] = category_to_max_line_y.get(category, []) + [mean_y_max]
                category_to_max_line_x[category] = category_to_max_line_x.get(category, []) + [base_position + offset]
    
    # Draw solid lines connecting the means for each category across context lengths
    for category in category_to_mean_line_y.keys():
        ax.plot(category_to_mean_line_x[category], category_to_mean_line_y[category], 
               color=palette[category], linewidth=2, linestyle='-', alpha=0.8)
    
    # Draw dotted lines for the max effect sizes
    for category in category_to_max_line_y.keys():
        ax.plot(category_to_max_line_x[category], category_to_max_line_y[category], 
               color=palette[category], linewidth=2, linestyle=':', alpha=0.8,
               label=f"{category} (max across addendum)")

    # Customize plot
    ax.set_title('Absolute Effect Size vs Context Length by Lie Maximisation', fontsize=16)
    ax.set_xlabel('Context Length (N)', fontsize=14)
    ax.set_ylabel('Absolute Effect Size |Cohen\'s d|', fontsize=14)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Set x-axis ticks to be the actual context lengths
    ax.set_xticks(unique_contexts)
    ax.set_xticklabels(unique_contexts)
    
    # Customize legend
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    return fig, ax


# Main execution
if __name__ == "__main__":
    
    import sys, os
    from util.util import YamlConfig

    config_path = sys.argv[1]
    args = YamlConfig(config_path)

    save_base = os.path.join('lie_detector_results/i_discriminative_tokens', args.args_name)

    # Configuration (modify these paths as needed)
    probe_responses_path = os.path.join(save_base, 'probe_response.csv')
    probe_questions_path = os.path.join(save_base, 'new_probe_questions.csv')
    output_path = save_base
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data, probe_df = load_and_preprocess_data_modified(probe_responses_path, probe_questions_path)
    
    # Check probability sums
    prob_sums = data['prob_yes'] + data['prob_no']
    print(f"\nProb_yes + prob_no range: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]")
    non_unity_probs = prob_sums[(prob_sums < 0.99) | (prob_sums > 1.01)]
    print(f"Non-unity probability sums: {len(non_unity_probs)} / {len(prob_sums)}")

    # # Filter out 'bad' answers
    # data = data[prob_sums > 0.7]

    # # Filter out probe questions that don't have both truth=0 and truth=1 because of this
    # truth_counts = data.groupby(['probe_question_idx', 'i_append_string', 'truth']).size().unstack(fill_value=0)
    # # Find probes that have both truth=0 and truth=1 for both append strings
    # valid_probes = []
    # for probe_idx in data['probe_question_idx'].unique():
    #     probe_data = data[data['probe_question_idx'] == probe_idx]
    #     has_both_truths = True
    #     for append_val in [0.0, 1.0]:
    #         append_data = probe_data[probe_data['i_append_string'] == append_val]
    #         if len(append_data[append_data['truth'] == 0]) == 0 or len(append_data[append_data['truth'] == 1]) == 0:
    #             has_both_truths = False
    #             break
    #     if has_both_truths:
    #         valid_probes.append(probe_idx)

    # original_len = len(data)
    # data = data[data['probe_question_idx'].isin(valid_probes)]
    # probe_df = probe_df[probe_df['probe_question_idx'].isin(valid_probes)]
    # print(f"Filtered out {len(set(data['probe_question_idx'].unique()) ^ set(valid_probes))} probe questions missing truth values")
    # print(f"Remaining data: {len(data)} rows, {len(valid_probes)} probe questions")
        
        
    # Perform analysis and create plots
    print("Performing discriminability analysis...")
    discriminability_results = plot_probe_type_analysis_modified(
        data, 
        # os.path.join(output_path, 'probe_type_analysis_modified.png'),
        None,
        enable_category_braces=True,   
    )
    
    # Create effect size vs context length plot
    print("Creating effect size vs context length plot...")
    plot_effect_size_vs_context_length(
        data, probe_df, discriminability_results,
        os.path.join(output_path, 'effect_size_vs_context_length.png')
    )
    
    # Save results
    results_filename = os.path.join(output_path, 'discriminability_results_modified.json')
    with open(results_filename, 'w') as f:
        json.dump(discriminability_results, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {output_path}")
    print(f"Plots: probe_type_analysis_modified.png, effect_size_vs_context_length.png")
    print(f"Data: discriminability_results_modified.json")