"""
Normalized steering analysis script.

This script calculates alignment projections normalized by prompted truth/lie differences,
then analyzes differential steering effects relative to random context.
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
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
    
    print(f"Loaded prompted activations:")
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

    print(f"Calculated alignment vectors and normalization:")
    print(f"  Alignment vectors shape: {alignment_vectors.shape}")
    print(f"  Scale factors shape: {scale_factors.shape}")
    print(f"  Prompted truth proj range: [{truth_proj_means.min():.3f}, {truth_proj_means.max():.3f}]")
    print(f"  Prompted lie proj range: [{lie_proj_means.min():.3f}, {lie_proj_means.max():.3f}]")
    
    return alignment_vectors, scale_factors, shift_factors


def load_batch_context_activations(activation_results_path: str) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load batch context activations that were extracted earlier.
    
    Args:
        activation_results_path: Path to the activation results directory
        
    Returns:
        Dict mapping context_length -> context_type -> activations [num_contexts, num_layers, hidden_size]
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
            
            print(f"Loaded {filename}: shape {activations.shape}")
    
    return activations_data


def project_and_normalize_activations(
    activations_data: Dict[int, Dict[str, np.ndarray]], 
    alignment_vectors: np.ndarray,
    scale_factors: np.ndarray,
    shift_factors: np.ndarray
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Project batch context activations onto alignment vectors and apply normalization.
    
    Args:
        activations_data: Context activations [num_contexts, num_layers, hidden_size]
        alignment_vectors: Alignment vectors [num_layers, hidden_size]
        scale_factors: Scaling factors [num_layers]
        shift_factors: Shift factors [num_layers]
        
    Returns:
        Dict mapping context_length -> context_type -> normalized_projections [num_contexts, num_layers]
    """
    projections_data = {}
    
    for context_length in activations_data:
        projections_data[context_length] = {}
        
        for context_type in activations_data[context_length]:
            activations = activations_data[context_length][context_type]  # [num_contexts, num_layers, hidden_size]
            num_contexts, num_layers, hidden_size = activations.shape
            
            # Project each context's activations onto alignment vectors
            raw_projections = np.zeros((num_contexts, num_layers))
            
            for layer in range(num_layers):
                alignment_vec = alignment_vectors[layer]  # [hidden_size]
                layer_activations = activations[:, layer, :]  # [num_contexts, hidden_size]
                raw_projections[:, layer] = np.dot(layer_activations, alignment_vec)  # [num_contexts]
            
            # Apply normalization: (projection - shift) * scale
            normalized_projections = np.zeros_like(raw_projections)
            for layer in range(num_layers):
                normalized_projections[:, layer] = (raw_projections[:, layer] - shift_factors[layer]) * scale_factors[layer]
            
            projections_data[context_length][context_type] = normalized_projections
            print(f"N{context_length} {context_type}: normalized projections shape {normalized_projections.shape}")
    
    return projections_data


def save_normalized_projections(projections_data: Dict[int, Dict[str, np.ndarray]], output_path: str) -> None:
    """
    Save normalized projection results to disk.
    
    Args:
        projections_data: Projections organized by context_length and context_type
        output_path: Directory to save results
    """
    os.makedirs(output_path, exist_ok=True)
    
    for context_length in projections_data:
        for context_type in projections_data[context_length]:
            filename = f'normalized_projections_N{context_length}_context{context_type}.npy'
            filepath = os.path.join(output_path, filename)
            np.save(filepath, projections_data[context_length][context_type])
            print(f"Saved {filepath}")


def plot_normalized_differential_steering_effects(
    projections_data: Dict[int, Dict[str, np.ndarray]], 
    output_path: str
) -> None:
    """
    Plot normalized differential steering effects across all layers.
    
    Args:
        projections_data: Normalized projections [num_contexts, num_layers]
        output_path: Directory to save plots
    """
    context_lengths = sorted([cl for cl in projections_data.keys() if cl > 0])  # Skip N=0
    
    # Determine number of layers from first available data
    first_context_length = context_lengths[0]
    num_layers = projections_data[first_context_length]['aligned'].shape[1]
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    # Color gradients: light to dark as N increases
    greens = plt.cm.Greens(np.linspace(0.4, 1.0, len(context_lengths)))  # Aligned
    reds = plt.cm.Reds(np.linspace(0.4, 1.0, len(context_lengths)))      # Misaligned
    
    layer_indices = np.arange(num_layers)
    
    for cl_idx, context_length in enumerate(context_lengths):
        # Check if we have all required context types
        required_types = ['aligned', 'misaligned', 'random']
        if not all(ct in projections_data[context_length] for ct in required_types):
            continue
        
        # Get normalized projections for this context length
        aligned_proj = projections_data[context_length]['aligned']      # [num_contexts, num_layers]
        misaligned_proj = projections_data[context_length]['misaligned']  # [num_contexts, num_layers]
        random_proj = projections_data[context_length]['random']        # [num_contexts, num_layers]
        
        # Calculate differences relative to random
        aligned_diff = aligned_proj - random_proj      # [num_contexts, num_layers]
        misaligned_diff = misaligned_proj - random_proj  # [num_contexts, num_layers]
        
        # Calculate mean and std across contexts for all layers
        aligned_means = np.mean(aligned_diff, axis=0)  # [num_layers]
        aligned_stds = np.std(aligned_diff, axis=0)    # [num_layers]
        misaligned_means = np.mean(misaligned_diff, axis=0)  # [num_layers]
        misaligned_stds = np.std(misaligned_diff, axis=0)    # [num_layers]
        
        # Slight x-offset for each context length to prevent overlap
        offset = (cl_idx - len(context_lengths)/2) * 0.15
        x_aligned = layer_indices + offset
        x_misaligned = layer_indices + offset
        
        # Plot with error bars
        ax.errorbar(x_aligned, aligned_means, yerr=aligned_stds, 
                   color=greens[cl_idx], marker='o', capsize=3, markersize=4,
                   label=f'Aligned N={context_length}', linewidth=1.5, alpha=0.8)
        ax.errorbar(x_misaligned, misaligned_means, yerr=misaligned_stds,
                   color=reds[cl_idx], marker='s', capsize=3, markersize=4,
                   label=f'Misaligned N={context_length}', linewidth=1.5, alpha=0.8)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Formatting
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Δ Normalized Alignment Projection\n(vs Random Context)', fontsize=12)
    ax.set_title('Normalized Differential Steering Effects: Aligned/Misaligned vs Random Context', 
                fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlim(-1, num_layers)
    
    # Add reference lines at ±1 (where prompted truth/lie would be)
    #ax.axhline(y=1, color='green', linestyle=':', alpha=0.7, linewidth=1, label='Prompted Truth Level')
    #ax.axhline(y=-1, color='red', linestyle=':', alpha=0.7, linewidth=1, label='Prompted Lie Level')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'normalized_differential_steering_effects.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved normalized differential steering effects plot")


def main():
    """Main function to calculate normalized alignment projections and create visualizations."""
    
    # Load configuration
    config_path = sys.argv[1]
    config = YamlConfig(config_path)
    
    print("Configuration loaded:")
    print(f"  Experiment: {config.args_name}")
    
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
    
    output_path = os.path.join(base_experiment_path, 'd3_ordered_in_context_liar_projections')
    
    # Load prompted activations and calculate alignment vectors + normalization
    print("Loading prompted activations...")
    truth_activations, lie_activations = load_prompted_activations(base_experiment_path)
    
    print("Calculating alignment vectors and normalization parameters...")
    alignment_vectors, scale_factors, shift_factors = calculate_alignment_vectors_and_normalization(
        truth_activations, lie_activations
    )
    
    # Load batch context activations
    print("Loading batch context activations...")
    activations_data = load_batch_context_activations(activation_results_path)
    
    # Project and normalize activations
    print("Projecting and normalizing activations...")
    projections_data = project_and_normalize_activations(
        activations_data, alignment_vectors, scale_factors, shift_factors
    )
    
    # Save normalized projections
    print("Saving normalized projection results...")
    save_normalized_projections(projections_data, output_path)
    
    # Create visualization
    print("Creating normalized differential steering effects plot...")
    plot_normalized_differential_steering_effects(projections_data, output_path)
    
    print(f"\n{'='*80}")
    print("NORMALIZED STEERING ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()