import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def plot_prompted_projections(truth_projections, lie_projections, results, save_path):
    """
    Plot histograms of truth vs lie projections for each layer.
    
    Args:
        truth_projections: array of shape (n_questions, n_layers)
        lie_projections: array of shape (n_questions, n_layers) 
        results: dict with layer keys containing cohens_d values
        save_path: path to save the figure
    """
    n_questions, n_layers = truth_projections.shape
    
    fig, axes = plt.subplots(7, 5, figsize=(9, 12))
    axes = axes.flatten()

    for layer in range(n_layers):
        ax = axes[layer]
        
        # Histogram both distributions
        ax.hist(truth_projections[:, layer], alpha=0.5, label='Truth', bins=20)
        ax.hist(lie_projections[:, layer], alpha=0.5, label='Lie', bins=20)
        
        ax.set_title(f'Layer {layer} (d={results[layer]["cohens_d"]:.2f})')
        ax.legend()
        ax.set_xlabel('Projection onto Lie Direction')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_prompted_control_projections(truth_projections_control, lie_projections_control, results, save_path):
    """
    Plot histograms of truth vs lie projections onto control directions for each layer.
    
    Args:
        truth_projections_control: array of shape (n_questions, num_controls, n_layers)
        lie_projections_control: array of shape (n_questions, num_controls, n_layers)
        results: dict with layer keys containing cohens_d values
        save_path: path to save the figure
    """
    n_questions, num_controls, n_layers = truth_projections_control.shape
    
    fig, axes = plt.subplots(7, 5, figsize=(9, 12))
    axes = axes.flatten()

    for layer in range(n_layers):
        ax = axes[layer]
        
        # Histogram both distributions (using first control direction)
        ax.hist(truth_projections_control[:, 0, layer], alpha=0.5, label='Truth', bins=20)
        ax.hist(lie_projections_control[:, 0, layer], alpha=0.5, label='Lie', bins=20)
        
        ax.set_title(f'Layer {layer} (d={results[layer]["cohens_d"]:.2f})')
        ax.legend()
        ax.set_xlabel('Projection onto example control direction')
        ax.set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_contextual_projections(all_contextual_data, context_lengths, save_path):
    """
    Plot histograms of contextual projections across different context types and lengths.
    
    Args:
        all_contextual_data: list of dicts containing contextual projection data
        context_lengths: list of context lengths used
        save_path: path to save the figure
    """
    # Determine number of layers from first entry
    first_entry = all_contextual_data[0]
    n_layers = first_entry['projections'].shape[1]
    num_context_lengths = len(context_lengths)
    
    fig, contextual_axes = plt.subplots(num_context_lengths, n_layers, figsize=(5 * n_layers, 5 * num_context_lengths))
    
    # Group data by context length
    data_by_length = {}
    for entry in all_contextual_data:
        length = entry['context_length']
        if length not in data_by_length:
            data_by_length[length] = []
        data_by_length[length].append(entry)
    
    for cli, context_length in enumerate(context_lengths):
        contextual_axes[cli, 0].set_ylabel(f'N = {context_length}')
        
        if context_length in data_by_length:
            for entry in data_by_length[context_length]:
                context_type = entry['context_type']
                projections = entry['projections']
                
                for layer in range(n_layers):
                    contextual_axes[0, layer].set_title(f'Layer = {layer + 1}')
                    contextual_axes[cli, layer].hist(projections[:, layer], alpha=0.5, label=context_type, bins=20)
        
        contextual_axes[cli, -1].legend(title='Context type')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_contextual_control_projections(all_contextual_data, context_lengths, save_path):
    """
    Plot histograms of contextual control projections across different context types and lengths.
    
    Args:
        all_contextual_data: list of dicts containing contextual projection data
        context_lengths: list of context lengths used
        save_path: path to save the figure
    """
    # Determine number of layers from first entry
    first_entry = all_contextual_data[0]
    n_layers = first_entry['control_projections'].shape[2]
    num_context_lengths = len(context_lengths)
    
    fig_control, contextual_axes_control = plt.subplots(num_context_lengths, n_layers, figsize=(5 * n_layers, 5 * num_context_lengths))
    
    # Group data by context length
    data_by_length = {}
    for entry in all_contextual_data:
        length = entry['context_length']
        if length not in data_by_length:
            data_by_length[length] = []
        data_by_length[length].append(entry)
    
    for cli, context_length in enumerate(context_lengths):
        contextual_axes_control[cli, 0].set_ylabel(f'N = {context_length}')
        
        if context_length in data_by_length:
            for entry in data_by_length[context_length]:
                context_type = entry['context_type']
                control_projections = entry['control_projections']
                
                for layer in range(n_layers):
                    contextual_axes_control[0, layer].set_title(f'Layer = {layer + 1}')
                    # Use first control direction (index 0)
                    contextual_axes_control[cli, layer].hist(control_projections[:, 0, layer], alpha=0.5, label=context_type, bins=20)
        
        contextual_axes_control[cli, -1].legend(title='Context type')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def reconstruct_projection_arrays(results):
    """
    Reconstruct the projection arrays from the saved results dict.
    
    Args:
        results: dict with layer keys containing projection data
        
    Returns:
        truth_projections: array of shape (n_questions, n_layers)
        lie_projections: array of shape (n_questions, n_layers)
    """
    n_layers = len(results)
    n_questions = len(results[0]['truth_projs'])
    
    truth_projections = np.zeros((n_questions, n_layers))
    lie_projections = np.zeros((n_questions, n_layers))
    
    for layer in range(n_layers):
        truth_projections[:, layer] = results[layer]['truth_projs']
        lie_projections[:, layer] = results[layer]['lie_projs']
    
    return truth_projections, lie_projections


def reconstruct_control_projection_arrays(truth_acts, lie_acts, results):
    """
    Reconstruct control projection arrays by re-projecting original activations.
    Note: This requires the original activation tensors which may not be saved.
    
    Args:
        truth_acts: array of shape (n_questions, n_layers, n_neurons)
        lie_acts: array of shape (n_questions, n_layers, n_neurons)
        results: dict with layer keys containing control_directions
        
    Returns:
        truth_projections_control: array of shape (n_questions, num_controls, n_layers)
        lie_projections_control: array of shape (n_questions, num_controls, n_layers)
    """
    n_questions, n_layers, n_neurons = truth_acts.shape
    num_controls = results[0]['control_directions'].shape[0]
    
    truth_projections_control = np.zeros((n_questions, num_controls, n_layers))
    lie_projections_control = np.zeros((n_questions, num_controls, n_layers))
    
    for layer in range(n_layers):
        control_directions = results[layer]['control_directions']
        
        for q_idx in range(n_questions):
            for c_idx in range(num_controls):
                control_direction = control_directions[c_idx]
                truth_projections_control[q_idx, c_idx, layer] = np.dot(truth_acts[q_idx, layer, :], control_direction)
                lie_projections_control[q_idx, c_idx, layer] = np.dot(lie_acts[q_idx, layer, :], control_direction)
    
    return truth_projections_control, lie_projections_control


def plot_single_layer_histogram(truth_projections, lie_projections, results, layer_idx, save_path, truth_label: str = 'Truth', lie_label: str = 'Lie', vector_label: str = 'lie'):
    """
    Plot histogram of truth vs lie projections for a single layer.
    
    Args:
        truth_projections: array of shape (n_questions, n_layers)
        lie_projections: array of shape (n_questions, n_layers) 
        results: dict with layer keys containing cohens_d values
        layer_idx: int, index of layer to plot
        save_path: path to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Histogram both distributions for the specified layer
    ax.hist(lie_projections[:, layer_idx], alpha=0.5, label=f'{lie_label} prompt', bins=20)
    ax.hist(truth_projections[:, layer_idx], alpha=0.5, label=f'{truth_label} prompt', bins=20)
    
    ax.set_title(f'Layer {layer_idx + 1} (Cohen\'s d={results[layer_idx]["cohens_d"]:.3f})')
    ax.legend()
    ax.set_xlabel(f'Projection onto {vector_label} Direction')
    ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
