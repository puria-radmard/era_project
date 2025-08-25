"""
Cross-Type Steering Analysis Script

This script loads steering results from multiple question-type-specific configs
to analyze selectivity in steering effects across different question types.
"""

import os
import sys
import yaml
import pandas as pd
from typing import Dict, Any, List, Tuple
from util.util import YamlConfig
from probe_generation.b_inference_apis.c_in_context_liar_v import load_aggregated_results
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap



def grouped_cohens_d(aligned_deltas: np.ndarray, misaligned_deltas: np.ndarray) -> float:
    """
    Calculate Cohen's d between aligned and misaligned deltas, averaged across questions.
    
    Args:
        aligned_deltas: Array of shape [n_questions, n_samples]
        misaligned_deltas: Array of shape [n_questions, n_samples]
        
    Returns:
        Single Cohen's d value (average across questions)
    """
    assert aligned_deltas.shape == misaligned_deltas.shape
    n_questions, n_samples = aligned_deltas.shape
    
    question_cohens_d = []
    
    for q_idx in range(n_questions):
        aligned_samples = aligned_deltas[q_idx, :]
        misaligned_samples = misaligned_deltas[q_idx, :]
        
        # Remove NaN values
        aligned_clean = aligned_samples[~np.isnan(aligned_samples)]
        misaligned_clean = misaligned_samples[~np.isnan(misaligned_samples)]
        
        if len(aligned_clean) == 0 or len(misaligned_clean) == 0:
            continue
            
        # Calculate Cohen's d for this question
        mean_diff = np.mean(aligned_clean) - np.mean(misaligned_clean)
        
        # Pooled standard deviation
        n1, n2 = len(aligned_clean), len(misaligned_clean)
        var1, var2 = np.var(aligned_clean, ddof=1), np.var(misaligned_clean, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std > 0:
            cohens_d = mean_diff / pooled_std
            question_cohens_d.append(cohens_d)
    
    return np.mean(question_cohens_d) if question_cohens_d else np.nan


def calculate_selectivity_matrix(
    all_results: Dict,
    question_metadata: Dict,
    context_length: int
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Calculate selectivity matrix for a given context length.
    
    Args:
        all_results: Cross-type steering data
        question_metadata: Question dataframes for each steering type
        context_length: Context length to analyze
        
    Returns:
        Tuple of (selectivity_matrix, steering_types, target_types)
    """
    # Get all steering context types
    steering_types = list(all_results.keys())
    
    # Get all possible target question types (from any of the question metadata)
    all_target_types = set()
    for df in question_metadata.values():
        all_target_types.update(df['type'].unique())
    target_types = sorted(list(all_target_types))
    
    # Initialize selectivity matrix
    selectivity_matrix = np.full((len(steering_types), len(target_types)), np.nan)
    
    for i, steering_type in enumerate(steering_types):
        if context_length not in all_results[steering_type]['aligned']:
            continue
            
        # Get the question indices for this steering type
        unique_questions = all_results[steering_type]['aligned'][context_length]['unique_questions']
        questions_df = question_metadata[steering_type]
        
        # Get deltas for this steering type
        aligned_deltas = (
            all_results[steering_type]['aligned'][context_length]['question_truth_lie_diffs_across_samples'] -
            all_results[steering_type]['random'][context_length]['question_truth_lie_diffs_across_samples']
        )
        misaligned_deltas = (
            all_results[steering_type]['misaligned'][context_length]['question_truth_lie_diffs_across_samples'] -
            all_results[steering_type]['random'][context_length]['question_truth_lie_diffs_across_samples']
        )
        
        for j, target_type in enumerate(target_types):
            # Find questions of this target type
            target_question_indices = questions_df[questions_df['type'] == target_type].index.tolist()
            
            # Find positions in unique_questions that correspond to target_type
            target_positions = [
                pos for pos, q_idx in enumerate(unique_questions) 
                if q_idx in target_question_indices
            ]
            
            if not target_positions:
                continue  # No questions of this type for this steering context
            
            # Extract deltas for target question type
            target_aligned = aligned_deltas[target_positions, :]
            target_misaligned = misaligned_deltas[target_positions, :]
            
            # Calculate Cohen's d for this (steering_type, target_type) pair
            cohens_d = grouped_cohens_d(target_aligned, target_misaligned)
            selectivity_matrix[i, j] = cohens_d
    
    return selectivity_matrix, steering_types, target_types


def plot_selectivity_heatmaps(
    all_results: Dict,
    question_metadata: Dict,
    context_lengths: List[int],
):
    """
    Plot selectivity heatmaps for multiple context lengths.
    """
    
    # Create custom red-green colormap (deeper green for negative, original red for positive)
    colors = ['darkgreen', 'white', '#d73027']  # Using the red from RdBu_r colormap
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('RedGreen', colors, N=n_bins)
    
    # Calculate number of subplots needed
    n_lengths = len(context_lengths)
    n_cols = n_lengths
    n_rows = 2  # Two rows: raw values and diagonal-normalized
    
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), squeeze=False)
    fig2, axes2 = plt.subplots(1, n_cols, figsize=(4*n_cols, 4), squeeze=False)
    
    # Pre-calculate all matrices to determine global color scales
    raw_matrices = []
    normalized_matrices = []
    
    for context_length in context_lengths:
        # Calculate selectivity matrix
        selectivity_matrix, steering_types, target_types = calculate_selectivity_matrix(
            all_results, question_metadata, context_length
        )
        raw_matrices.append(selectivity_matrix)
        
        # Create diagonal-normalized matrix
        normalized_matrix = selectivity_matrix.copy()
        for j, target_type in enumerate(target_types):
            # Find the diagonal element (where steering_type == target_type)
            diagonal_idx = None
            for i, steering_type in enumerate(steering_types):
                if steering_type == target_type:
                    diagonal_idx = i
                    break
            
            # If diagonal element exists and is non-zero, normalize the column by absolute value to preserve sign
            if diagonal_idx is not None and not np.isnan(selectivity_matrix[diagonal_idx, j]) and selectivity_matrix[diagonal_idx, j] != 0:
                normalized_matrix[:, j] = selectivity_matrix[:, j] / abs(selectivity_matrix[diagonal_idx, j])
        
        normalized_matrices.append(normalized_matrix)
    
    # Calculate global vmin/vmax for each row
    raw_values = np.concatenate([mat.flatten() for mat in raw_matrices])
    raw_values = raw_values[~np.isnan(raw_values)]
    raw_vmin, raw_vmax = np.percentile(raw_values, [5, 95]) if len(raw_values) > 0 else (-1, 1)
    raw_vmax = max(abs(raw_vmin), abs(raw_vmax))  # Make symmetric
    raw_vmin = -raw_vmax
    
    norm_values = np.concatenate([mat.flatten() for mat in normalized_matrices])
    norm_values = norm_values[~np.isnan(norm_values)]
    norm_vmin, norm_vmax = np.percentile(norm_values, [5, 95]) if len(norm_values) > 0 else (-1, 1)
    norm_vmax = max(abs(norm_vmin), abs(norm_vmax))  # Make symmetric
    norm_vmin = -norm_vmax
    
    # Plot heatmaps
    for idx, (context_length, raw_matrix, norm_matrix) in enumerate(zip(context_lengths, raw_matrices, normalized_matrices)):
        
        # First row: Raw values
        ax1_raw = axes1[0, idx]
        ax2_raw = axes2[0, idx]
        
        sns.heatmap(
            raw_matrix,
            xticklabels=target_types,
            yticklabels=steering_types if idx == 0 else False,
            cmap=cmap,
            center=0,
            vmin=raw_vmin,
            vmax=raw_vmax,
            annot=True,
            fmt='.2f',
            ax=ax1_raw,
            cbar=False,
            annot_kws={'size': 12}
        )

        sns.heatmap(
            raw_matrix,
            xticklabels=target_types,
            yticklabels=steering_types if idx == 0 else False,
            cmap=cmap,
            center=0,
            vmin=raw_vmin,
            vmax=raw_vmax,
            annot=True,
            fmt='.2f',
            ax=ax2_raw,
            cbar=False,
            annot_kws={'size': 12}
        )
        
        # Top row titles show context length
        if idx == 0:
            ax1_raw.set_title(f'N = {context_length}', fontsize=16)
            ax2_raw.set_title(f'N = {context_length}', fontsize=16)
        else:
            ax1_raw.set_title(f'N = {context_length}', fontsize=16)
            ax2_raw.set_title(f'N = {context_length}', fontsize=16)
        
        # Only leftmost plot gets metric label on x-axis
        if idx == 0:
            ax1_raw.set_ylabel('Raw Cohen\'s d', fontsize=16)
            ax2_raw.set_ylabel('Raw Cohen\'s d', fontsize=16)
        else:
            ax1_raw.set_xlabel('')
            ax2_raw.set_xlabel('')
        
        # Set label orientations
        ax1_raw.set_xticklabels(ax1_raw.get_xticklabels(), rotation=0, ha='center')
        ax2_raw.set_xticklabels(ax2_raw.get_xticklabels(), rotation=0, ha='center')
        if idx == 0:
            ax1_raw.set_yticklabels(ax1_raw.get_yticklabels(), rotation=90)
            ax2_raw.set_yticklabels(ax2_raw.get_yticklabels(), rotation=90)
        
        # Second row: Diagonal-normalized values
        ax_norm = axes1[1, idx]
        
        sns.heatmap(
            norm_matrix,
            xticklabels=target_types,
            yticklabels=steering_types if idx == 0 else False,
            cmap=cmap,
            center=0,
            vmin=norm_vmin,
            vmax=norm_vmax,
            annot=True,
            fmt='.2f',
            ax=ax_norm,
            cbar=False,
            annot_kws={'size': 12}
        )
        
        # No titles for bottom row
        ax_norm.set_title('')
        
        # Only leftmost plot gets metric label on x-axis
        if idx == 0:
            ax_norm.set_ylabel('Diagonal-Normalised', fontsize=16)
        else:
            ax_norm.set_xlabel('')
        
        # Set label orientations
        ax_norm.set_xticklabels(ax_norm.get_xticklabels(), rotation=0, ha='center')
        if idx == 0:
            ax_norm.set_yticklabels(ax_norm.get_yticklabels(), rotation=90)
    
    # Add shared labels
    fig1.text(0.5, 0.02, 'Target Question Type', ha='center', va='bottom', fontsize=16)
    fig2.text(0.5, 0.02, 'Target Question Type', ha='center', va='bottom', fontsize=16)
    
    fig1.text(0.02, 0.5, 'Steering Context Type', ha='center', va='center', rotation='vertical', fontsize=16)
    fig2.text(0.02, 0.5, 'Steering Context Type', ha='center', va='center', rotation='vertical', fontsize=16)
    
    # Adjust layout
    fig1.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.9, wspace=0.05, hspace=0.2)
    
    return fig1, fig2


def load_cross_type_steering_data(group_config_path: str, subdir_name: str) -> Dict[str, Dict[str, Dict[int, Dict[str, Any]]]]:
    """
    Load steering results for all question type configs.
    
    Args:
        group_config_path: Path to group YAML file
        
    Returns:
        Nested dictionary: all_results[steering_context_type][context_type][context_length] = data_dict
    """
    # Load group configuration
    group_config = YamlConfig(group_config_path)
    question_type_configs = group_config.question_types.__dict__
    
    print(f"Found {len(question_type_configs)} question type configs:")
    for qt, config_path in question_type_configs.items():
        print(f"  {qt}: {config_path}")
    
    all_results = {}
    
    # Load data for each question type
    for steering_context_type, config_path in question_type_configs.items():
        print(f"\nLoading data for steering context type: {steering_context_type}")
        
        # Load individual config
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        config = YamlConfig(config_path)
        
        # Determine save directory
        save_base = os.path.join('probe_generation_results/b_neurips_workshop_results', config.args_name)
        steering_dir = os.path.join(save_base, subdir_name)
        aggregated_dir = os.path.join(steering_dir, 'aggregated_outputs')
        
        # Check if aggregated results exist
        if not os.path.exists(aggregated_dir):
            raise FileNotFoundError(
                f"Aggregated results directory not found: {aggregated_dir}\n"
                f"Make sure you've run the collection script for config: {config_path}"
            )
        
        # Load aggregated results
        try:
            steering_results = load_aggregated_results(aggregated_dir)
            all_results[steering_context_type] = steering_results
            
            # Print summary
            context_types = list(steering_results.keys())
            context_lengths = set()
            for ct_results in steering_results.values():
                context_lengths.update(ct_results.keys())
            context_lengths = sorted(list(context_lengths))
            
            print(f"  Loaded context types: {context_types}")
            print(f"  Loaded context lengths: {context_lengths}")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load aggregated results for {steering_context_type}: {e}\n"
                f"Make sure the collection script completed successfully for: {config_path}"
            )
    
    return all_results


def load_question_metadata(group_config_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load initial questions dataframes for each question type config.
    
    Args:
        group_config_path: Path to group YAML file
        
    Returns:
        Dictionary mapping steering_context_type to initial_questions_df
    """
    group_config = YamlConfig(group_config_path)
    question_type_configs = group_config.question_types.__dict__
    question_metadata = {}
    
    for steering_context_type, config_path in question_type_configs.items():
        config = YamlConfig(config_path)
        questions_df = pd.read_csv(f'data/initial_questions/{config.questions_data_name}.csv')
        question_metadata[steering_context_type] = questions_df
    
    return question_metadata


def main(group_config_path: str, subdir_name: str):
    """
    Main function to load cross-type steering data and set up for analysis.
    """
    print("="*60)
    print("CROSS-TYPE STEERING ANALYSIS")
    print("="*60)
    
    # Load all steering results
    print("\nLoading cross-type steering data...")
    all_results = load_cross_type_steering_data(group_config_path, subdir_name)
    
    # Load question metadata
    print("\nLoading question metadata...")
    question_metadata = load_question_metadata(group_config_path)
    
    # Print summary
    print("\n" + "="*40)
    print("DATA LOADING COMPLETE")
    print("="*40)
    
    print(f"\nLoaded steering data for {len(all_results)} context types:")
    for steering_context_type in all_results.keys():
        print(f"  - {steering_context_type}")
    
    print(f"\nData structure: all_results[steering_context_type][context_type][context_length]")
    print(f"Available context types: {list(next(iter(all_results.values())).keys())}")
    
    # Show context lengths available
    all_context_lengths = set()
    for steering_results in all_results.values():
        for context_type_results in steering_results.values():
            all_context_lengths.update(context_type_results.keys())
    print(f"Available context lengths: {sorted(list(all_context_lengths))}")
    
    print(f"\nQuestion metadata available for each steering context type in 'question_metadata' dict")
    
    print("\n" + "="*40)
    print("READY FOR CUSTOM ANALYSIS")
    print("="*40)
    print("Variables available:")
    print("  - all_results: Cross-type steering data")
    print("  - question_metadata: Question dataframes for each type")
    print("  - group_config_path: Path to group config file")
    
    # Calculate selectivity for available context lengths
    available_lengths = set()
    for steering_results in all_results.values():
        for context_type_results in steering_results.values():
            available_lengths.update(context_type_results.keys())
    available_lengths = sorted(list(available_lengths))
    if 0 in available_lengths:
        available_lengths.remove(0)

    # Plot heatmaps
    fig1, fig2 = plot_selectivity_heatmaps(
        all_results=all_results,
        question_metadata=question_metadata, 
        context_lengths=available_lengths,
    )

    # Save both PNG and SVG
    os.makedirs(f'probe_generation_results/b_neurips_workshop_results/cx_steering_selectivity/{subdir_name}', exist_ok=True)
    
    fig1path_png = os.path.join(f'probe_generation_results/b_neurips_workshop_results/cx_steering_selectivity/{subdir_name}', f'{YamlConfig(group_config_path).savesubdir}_selectivity.png')
    fig1path_svg = os.path.join(f'probe_generation_results/b_neurips_workshop_results/cx_steering_selectivity/{subdir_name}', f'{YamlConfig(group_config_path).savesubdir}_selectivity.svg')

    fig2path_png = os.path.join(f'probe_generation_results/b_neurips_workshop_results/cx_steering_selectivity/{subdir_name}', f'simple_{YamlConfig(group_config_path).savesubdir}_selectivity.png')
    fig2path_svg = os.path.join(f'probe_generation_results/b_neurips_workshop_results/cx_steering_selectivity/{subdir_name}', f'simple_{YamlConfig(group_config_path).savesubdir}_selectivity.svg')

    fig1.savefig(fig1path_png, dpi=300, bbox_inches='tight')
    fig1.savefig(fig1path_svg, bbox_inches='tight')

    fig2.savefig(fig2path_png, dpi=300, bbox_inches='tight')
    fig2.savefig(fig2path_svg, bbox_inches='tight')

    print(f'Figure saved to {fig1path_png}')
    print(f'Figure saved to {fig1path_svg}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python cross_type_steering_analysis.py <group_config.yaml>")
        sys.exit(1)
    
    group_config_path = sys.argv[1]
    main(group_config_path, subdir_name = 'c2_ordered_in_context_liar')
    main(group_config_path, subdir_name = 'c3_ordered_tiled_in_context_liar')
