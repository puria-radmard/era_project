#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from scipy.stats import ttest_rel
import json
import sys
import os


import json

from tqdm import tqdm

from util.util import YamlConfig

import sys
import os

def load_and_preprocess_data_modified(probe_responses_path: str, probe_questions_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess data with probe categories and log odds calculation."""
    
    # Load probe responses
    probe_responses_data = pd.read_csv(probe_responses_path)
    
    # Calculate log odds
    epsilon = 1e-10
    probe_responses_data['prob_yes'] = np.clip(probe_responses_data['prob_yes'], epsilon, 1 - epsilon)
    probe_responses_data['prob_no'] = np.clip(probe_responses_data['prob_no'], epsilon, 1 - epsilon)
    probe_responses_data['log_odds'] = np.log(probe_responses_data['prob_yes'] / probe_responses_data['prob_no'])
    
    # Load probe questions and create categories
    probe_df = pd.read_csv(probe_questions_path)
    probe_df['probe_question_idx'] = probe_df.index

    # Merge probe categories into main data
    probe_responses_data = probe_responses_data.merge(
        probe_df,
        on='probe_question_idx'
    ).rename(columns={'generated_question': 'probe'})
    
    # probe_responses_data = probe_responses_data[probe_responses_data.question_idx < 11]
    # probe_responses_data = probe_responses_data[probe_responses_data['question_achieved'] == True]
    # probe_df = probe_df[probe_df['question_achieved'] == True]

    return probe_responses_data, probe_df




def compute_probe_discriminability_modified(data: pd.DataFrame) -> List[Dict]:
    """Compute discriminability statistics for each probe question with 4 conditions."""
    
    # probe_info = data[['probe_question_idx', 'probe', 'probe_type']].drop_duplicates().sort_values('probe_question_idx')
    probe_info = data.drop_duplicates().sort_values('probe_question_idx')
    
    probe_results = []

    for probe_idx in tqdm(probe_info.probe_question_idx.unique(), desc="Computing discriminability"):
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
            
            if len(pivot_data) == 0:
                continue
                
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
        
        if len(effect_sizes_by_append) < 2:
            continue
            
        # Check if effect sizes have same sign (robustness check)
        signs_match = np.sign(effect_sizes_by_append[0]) == np.sign(effect_sizes_by_append[1])
        average_effect_size = np.mean(effect_sizes_by_append)

        any_significant = any(p < 0.05 for p in p_values_by_append)
            
        probe_results.append({
            'probe_idx': int(probe_idx),
            # 'probe_type': probe_row.get('probe_type'),
            'effect_sizes_by_append': effect_sizes_by_append,
            'p_values_by_append': p_values_by_append,
            'stats_by_append': stats_by_append,
            'signs_match': bool(signs_match),
            'average_effect_size': float(average_effect_size),
            'abs_average_effect_size': float(abs(average_effect_size)),
            'significant': bool(any_significant),
        })
        
    return probe_results




def create_probe_boxplot_modified(data: pd.DataFrame, discriminability_results: List[Dict], ax: plt.Axes) -> plt.Axes:
    """Create boxplot with 4 hues based on truth and i_append_string."""
    
    # Create combined category for hue with new labels
    data = data.copy()
    data['truth_label'] = data['truth'].map({0: 'Narrowly misaligned', 1: 'Aligned'})
    data['append_label'] = data['i_append_string'].map({0: "'Answer with yes or no.'", 1: "'Answer with no or yes.'"})
    data['truth_append'] = data['truth_label'] + " | " + data['append_label']
    
    # Define explicit order: group by append, then aligned before misaligned
    hue_order = [
        "Aligned | 'Answer with yes or no.'",
        "Narrowly misaligned | 'Answer with yes or no.'", 
        "Aligned | 'Answer with no or yes.'",
        "Narrowly misaligned | 'Answer with no or yes.'"
    ]
    
    # Define colors: green family for aligned (truth=1), red family for misaligned (truth=0)  
    # dark for append=0, light for append=1
    palette = {
        "Narrowly misaligned | 'Answer with yes or no.'": '#d62728',    # dark red
        "Narrowly misaligned | 'Answer with no or yes.'": '#ff7f7f',    # light red  
        "Aligned | 'Answer with yes or no.'": '#2ca02c',                # dark green
        "Aligned | 'Answer with no or yes.'": '#98df8a'                 # light green
    }
    
    # Create box plot with more spacing
    sns.boxplot(
        data=data, 
        x='probe_question_idx', 
        y='log_odds', 
        hue='truth_append',
        hue_order=hue_order,
        palette=palette,
        width=1.0,
        ax=ax
    )

    # Add translucent vertical lines between probe groups
    probe_indices = sorted(data['probe_question_idx'].unique())
    for i in range(len(probe_indices) - 1):
        ax.axvline(x=i + 0.5, color='gray', alpha=0.3, linewidth=0.8, linestyle='-')
    
    return ax


def add_effect_direction_tiles(ax: plt.Axes, data: pd.DataFrame, discriminability_results: List[Dict]):
    """Add colored tiles at x-axis to show effect direction."""
    from matplotlib.patches import Rectangle
    
    probe_indices = sorted(data['probe_question_idx'].unique())
    
    # Get axis limits
    y_min, y_max = ax.get_ylim()
    tile_height = (y_max - y_min) * 0.02  # 5% of plot height
    tile_y = y_min - tile_height
    
    for i, result in enumerate(discriminability_results):
        if i >= len(probe_indices):
            continue
            
        # Determine color based on effect direction and robustness
        if not result['signs_match']:
            color = '#808080'  # Gray for non-robust
        elif result['average_effect_size'] > 0:
            color = '#2ca02c'  # Green for positive (aligned > misaligned)
        else:
            color = '#d62728'  # Red for negative (aligned < misaligned)
        
        # Add rectangle tile
        rect = Rectangle((i - 0.4, tile_y), 0.8, tile_height, 
                        facecolor=color, edgecolor='white', linewidth=0.5)
        ax.add_patch(rect)
    
    # Extend y-axis to show tiles
    ax.set_ylim(tile_y, y_max)


def add_significance_stars_modified(ax: plt.Axes, data: pd.DataFrame, 
                                   discriminability_results: List[Dict]):
    """Add significance stars to the plot."""
    
    probe_indices = sorted(data['probe_question_idx'].unique())
    
    for i, probe_result in enumerate(discriminability_results):
        if i >= len(probe_indices):
            continue
            
        probe_idx = probe_indices[i]
        max_y = data[data['probe_question_idx'] == probe_idx]['log_odds'].max()
        
        # Add significance star if significant
        if probe_result['significant']:
            ax.text(i, max_y + 0.15, '*', 
                   ha='center', va='bottom', fontsize=18, fontweight='bold')


def analyze_single_config(config_path: str, question_type: str) -> Tuple[pd.DataFrame, List[Dict]]:
    """Analyze a single configuration and return data and results."""
    
    # Load individual config
    args = YamlConfig(config_path)
    
    model_name = args.model_name
    probe_file_name = args.probe_file_name

    # Prepare for saving results
    save_directory = os.path.join('probe_generation_results/b_neurips_workshop_results', args.args_name)
    
    # Load data
    probe_responses_path = os.path.join(save_directory, 'b_original_probe_questions', 'probe_answers', 'probe_results.csv')
    probe_questions_path = f'data/probe_questions/{probe_file_name}.csv'
    
    try:
        probe_responses_data, probe_df = load_and_preprocess_data_modified(probe_responses_path, probe_questions_path)
        
        # Check prob_yes + prob_no sums
        prob_sums = probe_responses_data['prob_yes'] + probe_responses_data['prob_no']
        print(f"\n{question_type} - Prob_yes + prob_no range: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]")
        non_unity_probs = prob_sums[(prob_sums < 0.99) | (prob_sums > 1.01)]
        print(f"{question_type} - Non-unity probability sums: {len(non_unity_probs)} / {len(prob_sums)}")

        # Compute discriminability
        discriminability_results = compute_probe_discriminability_modified(probe_responses_data)
        
        return probe_responses_data, discriminability_results
        
    except Exception as e:
        print(f"Error processing {question_type}: {e}")
        return None, None


def plot_group_analysis(group_config_path: str):
    """Main function to create grouped analysis plots."""
    
    # Load group config
    group_config = YamlConfig(group_config_path)
    question_types = group_config.question_types
    savesubdir = group_config.savesubdir
    
    # Prepare output paths
    output_dir = os.path.join('probe_generation_results/b_neurips_workshop_results/bx_probe_analysis')
    os.makedirs(output_dir, exist_ok=True)
    output_path_png = os.path.join(output_dir, f'{savesubdir}_probe_analysis.png')
    output_path_svg = os.path.join(output_dir, f'{savesubdir}_probe_analysis.svg')
    
    # Analyze each configuration
    analyses = {}
    valid_question_types = []
    
    for question_type, config_path in question_types.__dict__.items():
        print(f"\nProcessing {question_type}...")
        data, results = analyze_single_config(config_path, question_type)
        if data is not None and results is not None:
            analyses[question_type] = (data, results)
            valid_question_types.append(question_type)
    
    if not valid_question_types:
        print("No valid analyses found!")
        return
    
    # Create figure with stacked subplots and shared x-axis
    n_plots = len(valid_question_types)
    fig, axes = plt.subplots(n_plots, 1, figsize=(25, 10 * n_plots), 
                           sharex=True, gridspec_kw={'hspace': 0.05})
    
    # Handle case where there's only one subplot
    if n_plots == 1:
        axes = [axes]
    
    # Create each subplot
    for i, question_type in enumerate(valid_question_types):
        data, discriminability_results = analyses[question_type]
        ax = axes[i]
        
        # Create the boxplot
        create_probe_boxplot_modified(data, discriminability_results, ax)
        add_significance_stars_modified(ax, data, discriminability_results)
        add_effect_direction_tiles(ax, data, discriminability_results)
        
        # Set labels - no title, bigger fonts
        ax.set_ylabel(f'{question_type} persona prompts', fontsize=30)
        ax.tick_params(axis='y', labelsize=18)
        
        # Only show x-axis elements on bottom subplot
        if i == len(valid_question_types) - 1:
            ax.set_xlabel(f'Probe Question Index', fontsize=30)
            ax.tick_params(axis='x', rotation=45, labelsize=18)
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelbottom=False)
        
        # Only show legend on first subplot to avoid repetition
        if i == 0:
            ax.legend(fontsize=25, title_fontsize=25)
            ax.set_title('Log odds (yes / no)', fontsize=30)
        else:
            ax.get_legend().remove()
        
        # Print results for this question type
        print(f"\n{question_type} Discriminability Analysis Results:")
        print("Probe | Append=0 P-val | Append=1 P-val | Avg Effect | Robust | Significant")
        print("-" * 80)
        
        for j, result in enumerate(discriminability_results):
            p0 = result['p_values_by_append'][0] if len(result['p_values_by_append']) > 0 else 1.0
            p1 = result['p_values_by_append'][1] if len(result['p_values_by_append']) > 1 else 1.0
            robust = "✓" if result['signs_match'] else "❌"
            sig = "*" if result['significant'] else ""
            
            print(f"{j:5d} | {p0:13.4f} | {p1:13.4f} | {result['average_effect_size']:10.3f} | {robust:6s} | {sig:11s}")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_path_svg, bbox_inches='tight')
    
    print(f"\nSaved analysis plots to:")
    print(f"  PNG: {output_path_png}")
    print(f"  SVG: {output_path_svg}")
    
    plt.show()




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <group_config_path>")
        sys.exit(1)
        
    group_config_path = sys.argv[1]
    plot_group_analysis(group_config_path)