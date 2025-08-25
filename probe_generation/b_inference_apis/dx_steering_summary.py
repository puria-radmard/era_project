"""
Multi-config normalized steering analysis script.

This script loads multiple configs from a YAML specification file and creates 
side-by-side comparisons of normalized differential steering effects.
"""

import numpy as np
import os
import sys
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
from util.util import YamlConfig


def load_prompted_activations(base_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load prompted truth and lie activations.
    
    Args:
        base_path: Base experiment directory path
        
    Returns:
        Tuple of (truth_activations, lie_activations) with shape [num_questions, num_layers, hidden_size]
    """
    prompted_dir = os.path.join(base_path, 'd2_prompted_liar_activations')
    
    truth_path = os.path.join(prompted_dir, 'all_truth_residual_with_question.npy')
    lie_path = os.path.join(prompted_dir, 'all_lie_residual_with_question.npy')
    
    truth_activations = np.load(truth_path)
    lie_activations = np.load(lie_path)
    
    print(f"Loaded prompted activations from {base_path}:")
    print(f"  Truth shape: {truth_activations.shape}")
    print(f"  Lie shape: {lie_activations.shape}")
    
    return truth_activations, lie_activations


def calculate_alignment_vectors_and_normalization(
    truth_activations: np.ndarray, 
    lie_activations: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate alignment vectors and normalization parameters.
    
    Args:
        truth_activations: Shape [num_questions, num_layers, hidden_size]
        lie_activations: Shape [num_questions, num_layers, hidden_size]
        
    Returns:
        Tuple of (alignment_vectors, scale_factors, shift_factors)
        - alignment_vectors: Shape [num_layers, hidden_size], unit normalized
        - scale_factors: Shape [num_layers], scaling to make truth→+1, lie→-1
        - shift_factors: Shape [num_layers], centering offset
    """
    # Calculate mean activations across questions for each layer
    truth_means = np.mean(truth_activations, axis=0)  # [num_layers, hidden_size]
    lie_means = np.mean(lie_activations, axis=0)      # [num_layers, hidden_size]
    
    # Calculate alignment direction (truth - lie)
    alignment_vectors = truth_means - lie_means  # [num_layers, hidden_size]
    
    # Unit normalize each layer's vector
    norms = np.linalg.norm(alignment_vectors, axis=1, keepdims=True)  # [num_layers, 1]
    alignment_vectors = alignment_vectors / norms  # [num_layers, hidden_size]
    
    # Project prompted activations onto alignment vectors
    num_questions, num_layers, hidden_size = truth_activations.shape
    truth_projections = np.zeros((num_questions, num_layers))
    lie_projections = np.zeros((num_questions, num_layers))
    
    for layer in range(num_layers):
        alignment_vec = alignment_vectors[layer]  # [hidden_size]
        truth_projections[:, layer] = np.dot(truth_activations[:, layer, :], alignment_vec)
        lie_projections[:, layer] = np.dot(lie_activations[:, layer, :], alignment_vec)
    
    # Calculate normalization parameters for each layer
    truth_proj_means = np.mean(truth_projections, axis=0)  # [num_layers]
    lie_proj_means = np.mean(lie_projections, axis=0)      # [num_layers]
    
    # Scale so that truth → +1 and lie → -1
    midpoints = (truth_proj_means + lie_proj_means) / 2    # [num_layers]
    ranges = truth_proj_means - lie_proj_means             # [num_layers]
    scale_factors = 2.0 / ranges                          # [num_layers]
    shift_factors = midpoints                              # [num_layers]

    print(f"  Calculated normalization parameters")
    
    return alignment_vectors, scale_factors, shift_factors


def load_batch_context_activations(activation_results_path: str) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load batch context activations that were extracted earlier.
    
    Args:
        activation_results_path: Path to the activation results directory
        
    Returns:
        Dict mapping context_length -> context_type -> activations [num_questions, num_contexts, num_layers, hidden_size]
    """
    activations_data = {}
    
    # Find all .npy files in the directory
    for filename in os.listdir(activation_results_path):
        if filename.startswith('batch_contextual_activations_N') and filename.endswith('.npy'):
            # Parse filename: batch_contextual_activations_N5_contextaligned.npy
            parts = filename.replace('.npy', '').split('_')
            context_length = int(parts[3][1:])  # Remove 'N' from 'N5'
            context_type = parts[4].replace('context', '')  # Remove 'context' prefix
            
            filepath = os.path.join(activation_results_path, filename)
            activations = np.load(filepath)
            
            if context_length not in activations_data:
                activations_data[context_length] = {}
            activations_data[context_length][context_type] = activations
            
            print(f"  Loaded {filename}: shape {activations.shape}")
    
    return activations_data


def project_and_normalize_activations(
    activations_data: Dict[int, Dict[str, np.ndarray]], 
    alignment_vectors: np.ndarray,
    scale_factors: np.ndarray,
    shift_factors: np.ndarray
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Project batch context activations onto alignment vectors and apply normalization.
    Average over contexts, keeping questions for error bars.
    
    Args:
        activations_data: Context activations [num_questions, num_contexts, num_layers, hidden_size]
        alignment_vectors: Alignment vectors [num_layers, hidden_size]
        scale_factors: Scaling factors [num_layers]
        shift_factors: Shift factors [num_layers]
        
    Returns:
        Dict mapping context_length -> context_type -> normalized_projections [num_questions, num_layers]
    """
    projections_data = {}
    
    for context_length in activations_data:
        projections_data[context_length] = {}
        
        for context_type in activations_data[context_length]:
            activations = activations_data[context_length][context_type]  # [num_questions, num_contexts, num_layers, hidden_size]
            num_questions, num_contexts, num_layers, hidden_size = activations.shape
            
            # Average over contexts to get [num_questions, num_layers, hidden_size]
            averaged_activations = np.mean(activations, axis=1)
            
            # Project each question's averaged activations onto alignment vectors
            raw_projections = np.zeros((num_questions, num_layers))
            
            for layer in range(num_layers):
                alignment_vec = alignment_vectors[layer]  # [hidden_size]
                layer_activations = averaged_activations[:, layer, :]  # [num_questions, hidden_size]
                raw_projections[:, layer] = np.dot(layer_activations, alignment_vec)  # [num_questions]
            
            # Apply normalization: (projection - shift) * scale
            normalized_projections = np.zeros_like(raw_projections)
            for layer in range(num_layers):
                normalized_projections[:, layer] = (raw_projections[:, layer] - shift_factors[layer]) * scale_factors[layer]
            
            projections_data[context_length][context_type] = normalized_projections
    
    return projections_data


def load_yaml_config(yaml_path: str) -> Tuple[str, Dict[str, str]]:
    """
    Load YAML configuration file.
    
    Args:
        yaml_path: Path to YAML config file
        
    Returns:
        Tuple of (filename, configs_dict)
    """
    with open(yaml_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    filename = config_data['filename']
    configs = config_data['configs']
    
    print(f"Loaded YAML config: {yaml_path}")
    print(f"  Output filename: {filename}")
    print(f"  Configs: {list(configs.keys())}")
    
    return filename, configs


def load_single_config_data(config_path: str) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load and process data for a single config.
    
    Args:
        config_path: Path to the config file
        
    Returns:
        projections_data: Normalized projections data
    """
    print(f"\nProcessing config: {config_path}")
    
    # Load configuration
    config = YamlConfig(config_path)
    
    # Setup paths
    base_experiment_path = os.path.join(
        'probe_generation_results/b_neurips_workshop_results', 
        config.args_name
    )
    
    activation_results_path = os.path.join(
        'probe_generation_results/b_neurips_workshop_results', 
        config.args_name, 
        'd_ordered_in_context_liar'
    )
    
    # Load prompted activations and calculate alignment vectors + normalization
    truth_activations, lie_activations = load_prompted_activations(base_experiment_path)
    alignment_vectors, scale_factors, shift_factors = calculate_alignment_vectors_and_normalization(
        truth_activations, lie_activations
    )
    
    # Load batch context activations
    print("  Loading batch context activations...")
    activations_data = load_batch_context_activations(activation_results_path)
    
    # Project and normalize activations
    print("  Projecting and normalizing activations...")
    projections_data = project_and_normalize_activations(
        activations_data, alignment_vectors, scale_factors, shift_factors
    )
    
    return projections_data


def plot_multi_config_comparison(
    all_projections_data: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    output_path: str,
    filename: str
) -> None:
    """
    Create side-by-side comparison of normalized differential steering effects.
    
    Args:
        all_projections_data: Dict mapping config_title -> projections_data
        output_path: Directory to save the plot
        filename: Output filename (without extension)
    """
    config_titles = list(all_projections_data.keys())
    n_configs = len(config_titles)
    
    # Determine subplot layout
    n_cols = min(4, n_configs)  # Max 4 columns
    n_rows = (n_configs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), squeeze=False, sharey=True)
    
    for config_idx, config_title in enumerate(config_titles):
        row = config_idx // n_cols
        col = config_idx % n_cols
        ax = axes[row, col]
        
        projections_data = all_projections_data[config_title]
        
        # Get available context lengths and layers
        context_lengths = sorted([cl for cl in projections_data.keys() if cl > 0])  # Skip N=0
        
        if not context_lengths:
            ax.text(0.5, 0.5, f'No data for\n{config_title}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title(config_title, fontsize=20)
            continue
        
        # Determine number of layers from first available data
        first_context_length = context_lengths[0]
        num_layers = projections_data[first_context_length]['aligned'].shape[1]
        
        # Color gradients: light to dark as N increases
        greens = plt.cm.Greens(np.linspace(0.4, 1.0, len(context_lengths)))  # Aligned
        reds = plt.cm.Reds(np.linspace(0.4, 1.0, len(context_lengths)))      # Misaligned
        
        layer_indices = np.arange(num_layers)
        
        for cl_idx, context_length in enumerate(context_lengths):
            # Check if we have all required context types
            required_types = ['aligned', 'misaligned', 'random']
            if not all(ct in projections_data[context_length] for ct in required_types):
                print(f"Warning: Missing context types for N={context_length} in {config_title}")
                continue
            
            # Get normalized projections for this context length
            aligned_proj = projections_data[context_length]['aligned']      # [num_questions, num_layers]
            misaligned_proj = projections_data[context_length]['misaligned']  # [num_questions, num_layers]
            random_proj = projections_data[context_length]['random']        # [num_questions, num_layers]
            
            # Calculate differences relative to random
            aligned_diff = aligned_proj - random_proj      # [num_questions, num_layers]
            misaligned_diff = misaligned_proj - random_proj  # [num_questions, num_layers]
            
            # Calculate mean and std across questions for all layers
            aligned_means = np.mean(aligned_diff, axis=0)  # [num_layers]
            aligned_stds = np.std(aligned_diff, axis=0)    # [num_layers]
            misaligned_means = np.mean(misaligned_diff, axis=0)  # [num_layers]
            misaligned_stds = np.std(misaligned_diff, axis=0)    # [num_layers]
            
            # Slight x-offset for each context length to prevent overlap
            offset = (cl_idx - len(context_lengths)/2) * 0.15
            x_aligned = layer_indices + offset
            x_misaligned = layer_indices + offset
            
            # Plot with error bars (now representing variance across questions)
            ax.errorbar(x_aligned, aligned_means, yerr=aligned_stds, 
                       color=greens[cl_idx], marker='o', capsize=3, markersize=4,
                       linewidth=1.5, alpha=0.8)
            ax.errorbar(x_misaligned, misaligned_means, yerr=misaligned_stds,
                       color=reds[cl_idx], marker='s', capsize=3, markersize=4,
                       linewidth=1.5, alpha=0.8)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
        
        # Clean up axes - same style as cross-config summary
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if col > 0:  # Remove left spine for all but leftmost
            ax.spines['left'].set_visible(False)
            ax.tick_params(left=False)
        
        # Remove grid
        ax.grid(False)
        
        # Set y ticks to round numbers that span the axis
        if col == 0:  # Only set for leftmost (sharey will apply to others)
            y_min, y_max = ax.get_ylim()
            # Find nice round numbers that span the range with finer granularity
            y_range = y_max - y_min
            if y_range < 0.001:
                tick_step = 0.0001
            elif y_range < 0.01:
                tick_step = 0.001
            elif y_range < 0.1:
                tick_step = 0.01
            elif y_range < 1:
                tick_step = 0.1
            elif y_range < 2:
                tick_step = 0.2
            elif y_range < 5:
                tick_step = 0.5
            else:
                tick_step = 1.0
            
            # Create ticks from negative to positive, ensuring 0 is included
            max_abs = max(abs(y_min), abs(y_max))
            n_ticks = int(max_abs / tick_step) + 1
            ticks = []
            for i in range(-n_ticks, n_ticks + 1):
                tick = i * tick_step
                if y_min <= tick <= y_max:
                    ticks.append(tick)
            ax.set_yticks(ticks)
        
        # Formatting
        ax.set_title(config_title, fontsize=20)
        ax.set_xlabel('Layer Index', fontsize=18)
        ax.set_xlim(-1, num_layers)
        
        # Make tick labels bigger
        ax.tick_params(axis='both', which='major', labelsize=16)
    
    # Add colored dot legend in the first subplot
    if n_configs > 0:
        first_ax = axes[0, 0]
        
        # Get context lengths from first config for legend
        first_config_data = list(all_projections_data.values())[0]
        legend_context_lengths = sorted([cl for cl in first_config_data.keys() if cl > 0])
        
        if legend_context_lengths:
            # Create color gradients matching the plot
            legend_greens = plt.cm.Greens(np.linspace(0.4, 1.0, len(legend_context_lengths)))
            legend_reds = plt.cm.Reds(np.linspace(0.4, 1.0, len(legend_context_lengths)))
            
            # Position dots in bottom-right area of first subplot
            dot_x_start = 0.65
            dot_spacing = 0.08
            
            # Green dots on top row
            green_y = 0.30
            # Red dots on bottom row - 0.2 spacing as requested
            red_y = 0.10
            
            # Add "N=" label to the left of the dots
            first_ax.text(dot_x_start - 0.05, (green_y + red_y) / 2, 'N=', 
                        transform=first_ax.transAxes, ha='center', va='center',
                        fontsize=14)
            
            # Add dots and numerical labels
            for i, context_length in enumerate(legend_context_lengths):
                dot_x = dot_x_start + i * dot_spacing
                
                # Green dot (aligned)
                first_ax.scatter(dot_x, green_y, s=100, c=legend_greens[i], 
                               transform=first_ax.transAxes, zorder=10)
                
                # Red dot (misaligned)
                first_ax.scatter(dot_x, red_y, s=100, c=legend_reds[i], 
                               transform=first_ax.transAxes, zorder=10)
                
                # Numerical value between the two dots
                first_ax.text(dot_x, (green_y + red_y) / 2, str(context_length), 
                            transform=first_ax.transAxes, ha='center', va='center',
                            fontsize=13)
            
            # Add descriptive text using figure coordinates so it can overlap other axes
            # Get the position of the first axes in figure coordinates
            axes_pos = first_ax.get_position()
            
            # Calculate figure coordinates for the text
            text_fig_x = axes_pos.x0 + axes_pos.width * (dot_x_start + len(legend_context_lengths) * dot_spacing + 0.02)
            green_fig_y = axes_pos.y0 + axes_pos.height * green_y  
            red_fig_y = axes_pos.y0 + axes_pos.height * red_y
            
            # Place text on figure to allow overlap with other subplots
            fig.text(text_fig_x, green_fig_y, 'binary answers after aligned behaviour', 
                    ha='left', va='center', fontsize=9, color='green')
            #fig.text(text_fig_x, red_fig_y, 'binary answers after misaligned behaviour', 
            #        ha='left', va='center', fontsize=9, color='red')
    
    # Add arrow annotations outside the leftmost subplot
    if n_configs > 0:
        leftmost_ax = axes[0, 0]
        
        # Position arrows closer to the leftmost plot
        arrow_x = -0.12  # Closer to the plot
        arrow_length = 0.1  # In axes fraction units
        
        # Up arrow - grey color
        leftmost_ax.annotate('', xy=(arrow_x, 0.5 + arrow_length), xytext=(arrow_x, 0.5),
                            xycoords='axes fraction', textcoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', color='grey', lw=3))
        leftmost_ax.text(arrow_x - 0.05, 0.5 + arrow_length + 0.05, 'Higher projection\nthan random', 
                         transform=leftmost_ax.transAxes, rotation=90,
                         ha='center', va='bottom', fontsize=16, color='grey')
        
        # Down arrow - grey color
        leftmost_ax.annotate('', xy=(arrow_x, 0.5 - arrow_length), xytext=(arrow_x, 0.5),
                            xycoords='axes fraction', textcoords='axes fraction',
                            arrowprops=dict(arrowstyle='->', color='grey', lw=3))
        leftmost_ax.text(arrow_x - 0.05, 0.5 - arrow_length - 0.05, 'Lower projection\nthan random', 
                         transform=leftmost_ax.transAxes, rotation=90,
                         ha='center', va='top', fontsize=16, color='grey')
    
    # Hide empty subplots
    for config_idx in range(n_configs, n_rows * n_cols):
        row = config_idx // n_cols
        col = config_idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, f'{filename}.svg')
    plt.savefig(filepath, dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print(f"Saved multi-config comparison plot: {filepath}")


def main():
    """Main function to create multi-config normalized steering comparison."""
    
    if len(sys.argv) != 2:
        print("Usage: python multi_config_projections.py <config.yaml>")
        print("\nExample config.yaml:")
        print("filename: mistral-small-24b-instruct-2501")
        print("configs:")
        print("  Prompt present: probe_generation/b_inference_apis/config/medical_prompted_mistral-small-24b-instruct-2501.yaml")
        print("  Prompt albated: probe_generation/b_inference_apis/config/medical_noprompt_mistral-small-24b-instruct-2501.yaml")
        sys.exit(1)
    
    yaml_config_path = sys.argv[1]
    
    # Load YAML configuration
    output_filename, configs_dict = load_yaml_config(yaml_config_path)
    
    print(f"Processing {len(configs_dict)} configs:")
    for title, config_path in configs_dict.items():
        print(f"  {title}: {config_path}")
    
    # Load data for all configs
    all_projections_data = {}
    
    for config_title, config_path in configs_dict.items():
        print(f"\nProcessing: {config_title}")
        try:
            projections_data = load_single_config_data(config_path)
            all_projections_data[config_title] = projections_data
        except Exception as e:
            print(f"ERROR: Failed to process {config_path}: {e}")
            continue
    
    if not all_projections_data:
        print("ERROR: No configs were successfully processed.")
        sys.exit(1)
    
    # Create output directory
    output_path = os.path.join(
        'probe_generation_results/b_neurips_workshop_results',
        'dx_steering_summary'
    )
    
    # Generate comparison plot
    print("\nGenerating multi-config comparison plot...")
    plot_multi_config_comparison(all_projections_data, output_path, output_filename)
    
    print(f"\n{'='*80}")
    print("MULTI-CONFIG NORMALIZED STEERING COMPARISON COMPLETE!")
    print(f"{'='*80}")
    print(f"Compared {len(all_projections_data)} configs")
    print(f"Results saved to: {output_path}")
    print(f"Output file: {output_filename}.svg")


if __name__ == '__main__':
    main()