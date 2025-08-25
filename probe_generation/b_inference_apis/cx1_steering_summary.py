"""
Cross-Config Steering Summary Script

This script loads steering results from multiple question-type-specific configs
and creates summary plots showing within vs outside category performance.
"""

import os
import sys
import yaml
import pandas as pd
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from util.util import YamlConfig
from probe_generation.b_inference_apis.c_in_context_liar_v import load_aggregated_results


def load_cross_config_data(group_config_path: str, subdir_name: str) -> Tuple[Dict, Dict]:
    """
    Load steering results and question metadata for all configs.
    
    Args:
        group_config_path: Path to group YAML file
        subdir_name: Subdirectory name for results
        
    Returns:
        Tuple of (all_results, question_metadata)
    """
    # Load group configuration
    group_config = YamlConfig(group_config_path)
    question_type_configs = group_config.question_types.__dict__
    
    print(f"Found {len(question_type_configs)} question type configs:")
    for qt, config_path in question_type_configs.items():
        print(f"  {qt}: {config_path}")
    
    all_results = {}
    question_metadata = {}
    
    # Load data for each question type
    for config_key, config_path in question_type_configs.items():
        print(f"\nLoading data for config: {config_key}")
        
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
            all_results[config_key] = steering_results
            
            # Load question metadata
            questions_df = pd.read_csv(f'data/initial_questions/{config.questions_data_name}.csv')
            question_metadata[config_key] = questions_df
            
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
                f"Failed to load aggregated results for {config_key}: {e}\n"
                f"Make sure the collection script completed successfully for: {config_path}"
            )
    
    return all_results, question_metadata


def get_all_question_types(question_metadata: Dict) -> List[str]:
    """Get all unique question types across all configs."""
    all_types = set()
    for questions_df in question_metadata.values():
        all_types.update(questions_df['type'].unique())
    return sorted(list(all_types))


def create_color_palettes(question_types: List[str]) -> Dict[str, Dict[str, str]]:
    """Create consistent color palettes for all question types."""
    n_types = len(question_types)
    
    # Create green shades (from dark to light)
    green_colors = []
    for i in range(n_types):
        # Generate shades from dark green to light green
        intensity = 0.3 + 0.7 * (i / max(1, n_types - 1))  # From 0.3 to 1.0
        green_colors.append(mcolors.to_hex((0, intensity, 0)))
    
    # Create red shades (from dark to light)  
    red_colors = []
    for i in range(n_types):
        # Generate shades from dark red to light red
        intensity = 0.3 + 0.7 * (i / max(1, n_types - 1))  # From 0.3 to 1.0
        red_colors.append(mcolors.to_hex((intensity, 0, 0)))
    
    return {
        'aligned': {qtype: green_colors[i] for i, qtype in enumerate(question_types)},
        'misaligned': {qtype: red_colors[i] for i, qtype in enumerate(question_types)}
    }


def calculate_performance_by_question_type(
    results: Dict[str, Dict[int, Dict[str, Any]]],
    questions_df: pd.DataFrame,
    target_question_type: str,
    context_length: int,
    all_question_types: List[str]
) -> Dict[str, Dict[str, Tuple[List[int], List[float], List[float]]]]:
    """
    Calculate performance for each individual question type.
    
    Args:
        results: Results for one config {context_type: {context_length: data}}
        questions_df: Questions dataframe for this config
        target_question_type: The question type this config targets
        context_length: Context length to analyze
        all_question_types: All question types across all configs
        
    Returns:
        Dict with structure: {context_type: {question_type: (lengths, means, stds)}}
    """
    performance_data = {}
    
    for context_type in ['aligned', 'misaligned']:
        if context_length not in results[context_type]:
            continue
            
        data = results[context_type][context_length]
        unique_questions = data['unique_questions']
        
        # Calculate difference from random
        context_diffs = (
            data['question_truth_lie_diffs_across_samples'] -
            results['random'][context_length]['question_truth_lie_diffs_across_samples']
        )  # [n_questions, n_samples]
        
        performance_data[context_type] = {}
        
        # Calculate performance for each question type
        for question_type in all_question_types:
            type_positions = []
            
            for pos, q_idx in enumerate(unique_questions):
                if q_idx < len(questions_df):
                    if questions_df.iloc[q_idx]['type'] == question_type:
                        type_positions.append(pos)
            
            if type_positions:
                type_diffs = context_diffs[type_positions, :]  # [n_type, n_samples]
                type_means = np.nanmean(type_diffs, axis=0)  # [n_samples]
                
                performance_data[context_type][question_type] = (
                    [context_length],
                    [np.nanmean(type_means)],
                    [np.nanstd(type_means)]
                )
            else:
                performance_data[context_type][question_type] = ([], [], [])
    
    return performance_data


def plot_cross_config_summary(
    all_results: Dict,
    question_metadata: Dict,
    output_path: str,
    filename: str,
    skip_dots: bool
):
    """
    Create summary plots showing performance by individual question type for each config.
    """
    config_keys = list(all_results.keys())
    n_configs = len(config_keys)
    
    # Get all question types and create color palettes
    all_question_types = get_all_question_types(question_metadata)
    color_palettes = create_color_palettes(all_question_types)
    
    # Determine subplot layout
    n_cols = min(3, n_configs)  # Max 3 columns
    n_rows = (n_configs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), squeeze=False, sharey=True)
    
    # Get all available context lengths
    all_context_lengths = set()
    for config_results in all_results.values():
        for context_type_results in config_results.values():
            all_context_lengths.update(context_type_results.keys())
    context_lengths = sorted([cl for cl in all_context_lengths if cl > 0])  # Exclude 0
    
    # Get actual context lengths used across all configs
    actual_context_lengths = set([0])  # Always include 0
    actual_context_lengths.update(context_lengths)
    
    for config_idx, config_key in enumerate(config_keys):
        row = config_idx // n_cols
        col = config_idx % n_cols
        ax = axes[row, col]
        
        results = all_results[config_key]
        questions_df = question_metadata[config_key]
        
        # Collect data for all context lengths
        plot_data = {
            'aligned': {qtype: {'lengths': [], 'means': [], 'stds': []} for qtype in all_question_types},
            'misaligned': {qtype: {'lengths': [], 'means': [], 'stds': []} for qtype in all_question_types}
        }
        
        for context_length in context_lengths:
            perf_data = calculate_performance_by_question_type(
                results, questions_df, config_key, context_length, all_question_types
            )
            
            for context_type in ['aligned', 'misaligned']:
                if context_type in perf_data:
                    for question_type in all_question_types:
                        if question_type in perf_data[context_type]:
                            lengths, means, stds = perf_data[context_type][question_type]
                            plot_data[context_type][question_type]['lengths'].extend(lengths)
                            plot_data[context_type][question_type]['means'].extend(means)
                            plot_data[context_type][question_type]['stds'].extend(stds)
        
        # Plot lines with offsets
        line_idx = 0
        target_color = None
        for context_type in ['aligned', 'misaligned']:
            for question_type in all_question_types:
                data = plot_data[context_type][question_type]
                if not data['lengths']:
                    continue
                
                color = color_palettes[context_type][question_type]
                
                # Store target color for indicator circle
                if question_type == config_key and context_type == 'aligned':
                    target_color = color
                
                # Determine line style (solid for target type, dashed for others)
                linestyle = '-' if question_type == config_key else '--'
                marker = 'o' if question_type == config_key else 's'
                linewidth = 3 if question_type == config_key else 2
                alpha = 1.0 if question_type == config_key else 0.7
                
                # Add zero point and apply offset
                x_vals = np.array([0] + data['lengths'])
                y_vals = [0] + data['means']
                y_errs = [0] + data['stds']
                
                # Apply small offset to avoid overlapping error bars
                offset = (line_idx - len(all_question_types)) * 0.1
                x_vals_offset = x_vals + offset
                
                ax.errorbar(x_vals_offset, y_vals, yerr=y_errs,
                           color=color, linestyle=linestyle, marker=marker,
                           capsize=3, capthick=1, linewidth=linewidth, 
                           markersize=6, alpha=alpha)
                
                line_idx += 1
        
        # Add colored circle indicators in corners
        if target_color:
            if not skip_dots:
                # Green circle (aligned) in top right
                ax.scatter(0.95, 0.95, s=100, c=target_color, transform=ax.transAxes, zorder=10)
                # Red circle (misaligned) in bottom right  
                misaligned_color = color_palettes['misaligned'][config_key]
                ax.scatter(0.95, 0.05, s=100, c=misaligned_color, transform=ax.transAxes, zorder=10)
        
        # Add horizontal line at y=0 for reference
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        
        # Clean up axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if col > 0:  # Remove left spine for all but leftmost
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False)
        
        # Remove grid
        ax.grid(False)
        
        # Set x ticks to actual context lengths only
        sorted_lengths = sorted(list(actual_context_lengths))
        ax.set_xticks(sorted_lengths)
        ax.set_xticklabels([str(x) for x in sorted_lengths])
        
        # Set y ticks to round numbers that span the axis
        if col == 0:  # Only set for leftmost (sharey will apply to others)
            y_min, y_max = ax.get_ylim()
            # Find nice round numbers that span the range with finer granularity
            max_abs = max(abs(y_min), abs(y_max))
            if max_abs < 0.001:
                tick_step = 0.00001
            elif max_abs < 0.001:
                tick_step = 0.0001
            elif max_abs < 0.01:
                tick_step = 0.005
            elif max_abs < 0.05:
                tick_step = 0.01
            elif max_abs < 0.1:
                tick_step = 0.05
            elif max_abs < 1:
                tick_step = 0.1
            elif max_abs < 2:
                tick_step = 0.2
            elif max_abs < 5:
                tick_step = 0.5
            else:
                tick_step = 1.0
            
            # Create ticks from negative to positive, ensuring 0 is included
            n_ticks = int(max_abs / tick_step) + 1
            ticks = []
            for i in range(-n_ticks, n_ticks + 1):
                tick = i * tick_step
                if y_min <= tick <= y_max:
                    ticks.append(tick)
            ax.set_yticks(ticks)
        
        # Formatting
        ax.set_title(f'{config_key.replace("_", " ").title()}', fontsize=18)
        ax.set_xlabel('Context Length (N)', fontsize=16)
        
        # Make tick labels bigger
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Add arrow annotations outside the leftmost subplot
    leftmost_ax = axes[0, 0]
    
    # Position arrows to the left of the leftmost plot in axes fraction coordinates
    arrow_x = -0.2  # Further outside to accommodate vertical text
    arrow_length = 0.1  # In axes fraction units
    
    # Up arrow for "more aligned than random"
    leftmost_ax.annotate('', xy=(arrow_x, 0.5 + arrow_length), xytext=(arrow_x, 0.5),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='green', lw=3))
    leftmost_ax.text(arrow_x - 0.05, 0.5 + arrow_length + 0.05, 'More aligned\nthan random', 
                     transform=leftmost_ax.transAxes, rotation=90,
                     ha='center', va='bottom', fontsize=14, color='green')
    
    # Down arrow for "more misaligned than random"  
    leftmost_ax.annotate('', xy=(arrow_x, 0.5 - arrow_length), xytext=(arrow_x, 0.5),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='red', lw=3))
    leftmost_ax.text(arrow_x - 0.05, 0.5 - arrow_length - 0.05, 'More misaligned\nthan random', 
                     transform=leftmost_ax.transAxes, rotation=90,
                     ha='center', va='top', fontsize=14, color='red')
    
    # Hide empty subplots
    for config_idx in range(n_configs, n_rows * n_cols):
        row = config_idx // n_cols
        col = config_idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"Saved plot: {filepath}")


def main(group_config_path: str, subdir_name: str, skip_dots: bool):
    """
    Main function to create cross-config steering summary plots.
    """
    print("="*60)
    print("CROSS-CONFIG STEERING SUMMARY")
    print("="*60)
    
    # Load all data
    print("\nLoading cross-config data...")
    all_results, question_metadata = load_cross_config_data(group_config_path, subdir_name)
    
    # Create output directory and filename
    group_config = YamlConfig(group_config_path)
    output_dir = f'probe_generation_results/b_neurips_workshop_results/cx1_steering_summary/{subdir_name}'
    filename = f'{group_config.savesubdir}.svg'
    
    # Generate plot
    print("\nGenerating summary plots...")
    plot_cross_config_summary(
        all_results=all_results,
        question_metadata=question_metadata,
        output_path=output_dir,
        filename=filename,
        skip_dots=skip_dots
    )
    
    print("\n" + "="*60)
    print("CROSS-CONFIG SUMMARY COMPLETE!")
    print("="*60)
    print(f"Plot saved to: {os.path.join(output_dir, filename)}")


if __name__ == '__main__':
    if len(sys.argv) not in [2, 3]:
        print("Usage: python cross_config_summary.py <group_config.yaml>")
        sys.exit(1)
    
    group_config_path = sys.argv[1]
    main(group_config_path, 'c3_ordered_tiled_in_context_liar', len(sys.argv)==3)