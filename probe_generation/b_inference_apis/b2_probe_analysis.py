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




def compute_probe_discriminability_modified(data: pd.DataFrame) -> Dict:
    """Compute discriminability statistics for each probe question with 4 conditions."""
    
    # probe_info = data[['probe_question_idx', 'probe', 'probe_type']].drop_duplicates().sort_values('probe_question_idx')
    probe_info = data.drop_duplicates().sort_values('probe_question_idx')
    
    probe_results = []

    for probe_idx in tqdm(probe_info.probe_question_idx.unique()):
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




def create_probe_boxplot_modified(data: pd.DataFrame) -> plt.Axes:
    """Create boxplot with 4 hues based on truth and i_append_string."""
    
    # Create combined category for hue
    data = data.copy()
    data['truth_append'] = "Truth = " + data['truth'].astype(str) + " | Append = " + data['i_append_string'].astype(str)
    
    # Define colors: red family for truth=0, blue family for truth=1
    # light for appendtype 1.0, dark for append type 0.0
    palette = {
        'Truth = 0 | Append = 0': '#d62728',    # red
        'Truth = 0 | Append = 1': '#ff7f7f',    # light red  
        'Truth = 1 | Append = 0': '#1f77b4',    # blue
        'Truth = 1 | Append = 1': '#87ceeb'     # light blue
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
    
    ax.legend(fontsize=14)
    
    return ax


def add_significance_stars_and_crosses_modified(ax: plt.Axes, data: pd.DataFrame, 
                                               probe_info: pd.DataFrame, 
                                               discriminability_results: Dict):
    """Add significance stars and robustness crosses to the plot."""
    
    for i, probe_result in enumerate(discriminability_results):
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
        result = discriminability_results[i]
        
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
        probe_means[probe_idx] = {
            ias: (probe_data[probe_data['i_append_string'] == ias])['log_odds'].mean()
            for ias in probe_data['i_append_string'].unique()
        }
    
    # Create transformed dataset
    transformed_data = []
    for _, row in data.iterrows():
        probe_idx = row['probe_question_idx']
        ias = row['i_append_string']
        new_pos = probe_idx_to_new_pos[probe_idx]
        delta_quantity = row['log_odds'] - probe_means[probe_idx][ias]
        
        # Create combined category
        truth_append = f"Truth = {row['truth']} | Append = {row['i_append_string']}"
        
        transformed_data.append({
            'ordered_probe_idx': new_pos,
            'delta_log_odds': delta_quantity,
            'truth_append': truth_append,
            'question_idx': row['question_idx']
        })
    
    transformed_df = pd.DataFrame(transformed_data)
    
    # Define colors: red family for truth=0, blue family for truth=1
    # light for appendtype 1.0, dark for append type 0.0
    palette = {
        'Truth = 0 | Append = 0': '#d62728',    # red
        'Truth = 0 | Append = 1': '#ff7f7f',    # light red  
        'Truth = 1 | Append = 0': '#1f77b4',    # blue
        'Truth = 1 | Append = 1': '#87ceeb'     # light blue
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



def plot_probe_type_analysis_modified(data: pd.DataFrame, filepath: str) -> Dict:
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
        
        ax1.set_title('Log Odds Distribution by Probe Question\n(* = significant, ❌ = non-robust effect)', fontsize=20)
        ax1.set_xlabel('Probe Question Index', fontsize=18)
        ax1.set_ylabel('Log Odds', fontsize=18)
        ax1.tick_params(axis='x', rotation=45)
        
        
        # Second subplot: Ordered by discriminability
        plt.sca(ax2)
        ax2 = create_discriminability_ordered_plot_modified(data, discriminability_results)
        
        ax2.set_xlabel('Probe Index (ordered by discriminability)', fontsize=18)
        ax2.set_ylabel('Δ Log Odds (centered at probe mean)', fontsize=18)
        
        # Handle legend for second plot
        handles, _ = ax2.get_legend_handles_labels()
        # ax2.legend(handles[:4], legend_labels, fontsize=14)
        ax2.legend(fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    # Print results
    print("\nDiscriminability Analysis Results:")
    print("Probe | Append=0 P-val | Append=1 P-val | Avg Effect | Robust | Significant")
    print("-" * 80)
    
    for i, result in enumerate(discriminability_results):
        p0 = result['p_values_by_append'][0] if len(result['p_values_by_append']) > 0 else 1.0
        p1 = result['p_values_by_append'][1] if len(result['p_values_by_append']) > 1 else 1.0
        robust = "✓" if result['signs_match'] else "❌"
        sig = "*" if result['significant'] else ""
        
        print(f"{i:5d} | {p0:13.4f} | {p1:13.4f} | {result['average_effect_size']:10.3f} | {robust:6s} | {sig:11s}")
    
    return discriminability_results




if __name__ == "__main__":


    config_path = sys.argv[1]
    args = YamlConfig(config_path)
    
    model_name = args.model_name
    probe_file_name = args.probe_file_name

    # Prepare for saving results
    save_directory = os.path.join('probe_generation_results/b_neurips_workshop_results', args.args_name)
    
    # Load data
    probe_responses_path = os.path.join(save_directory, 'b_original_probe_questions', 'probe_answers', 'probe_results.csv')
    probe_questions_path = f'data/probe_questions/{probe_file_name}.csv'
    probe_responses_data, probe_df = load_and_preprocess_data_modified(probe_responses_path, probe_questions_path)
    
    # Check prob_yes + prob_no sums
    prob_sums = probe_responses_data['prob_yes'] + probe_responses_data['prob_no']
    print(f"\nProb_yes + prob_no range: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]")
    non_unity_probs = prob_sums[(prob_sums < 0.99) | (prob_sums > 1.01)]
    print(f"Non-unity probability sums: {len(non_unity_probs)} / {len(prob_sums)}")

    output_path_png = os.path.join(save_directory, 'b_original_probe_questions', 'discriminability_results.png')
    output_path_json = os.path.join(save_directory, 'b_original_probe_questions', 'discriminability_results.json')

    discriminability_results = plot_probe_type_analysis_modified(probe_responses_data, output_path_png)
    with open(output_path_json, 'w') as f:
        json.dump(discriminability_results, f, indent=2)
