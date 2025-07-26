import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec


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




def plot_behavioral_context_effects(context_lengths, prob_diff_ttest_stats, prob_diff_mean_diff, 
                                   prob_diff_p_values, more_pairs_to_compare, projection_results_path):
    """
    Plot behavioral context effects showing t-test statistics and mean differences.
    
    Parameters:
    - context_lengths: list of context lengths
    - prob_diff_ttest_stats: array of shape [len(more_pairs_to_compare), len(context_lengths)]
    - prob_diff_mean_diff: array of shape [len(more_pairs_to_compare), len(context_lengths)]
    - prob_diff_p_values: array of shape [len(more_pairs_to_compare), len(context_lengths)]
    - more_pairs_to_compare: list of tuples with context type pairs
    - projection_results_path: path to save the figure
    """
    fig_beh, ax_beh = plt.subplots(1, 2, figsize=(15, 10))

    # Colors for different pairs
    colors = plt.cm.tab10(np.linspace(0, 1, len(more_pairs_to_compare)))

    for i_pair, (pair_context_1, pair_context_2) in enumerate(more_pairs_to_compare):
        # Line style: first pair solid, others dotted
        linestyle = '-' if i_pair == 0 else '--'
        
        # Plot ttest statistics
        ax_beh[0].plot(context_lengths, prob_diff_ttest_stats[i_pair, :], 
                    color=colors[i_pair], linestyle=linestyle, marker='o', 
                    label=f'{pair_context_1} vs {pair_context_2}', linewidth=2, markersize=6)

        # Plot mean diff
        ax_beh[1].plot(context_lengths, prob_diff_mean_diff[i_pair, :], 
                    color=colors[i_pair], linestyle=linestyle, marker='o', 
                    label=f'{pair_context_1} vs {pair_context_2}', linewidth=2, markersize=6)
        
        # Add significance stars
        for i_N, (N, p_val) in enumerate(zip(context_lengths, prob_diff_p_values[i_pair, :])):
            if p_val < 0.05 / 3 and not np.isnan(p_val):
                y_pos = prob_diff_ttest_stats[i_pair, i_N]
                ax_beh[0].scatter(N, y_pos, marker='*', s=100, color=colors[i_pair], 
                              edgecolors='black', linewidth=0.5, zorder=10)

    ax_beh[0].set_xlabel('Context Length (N)')
    ax_beh[0].set_ylabel('t-test Statistic')
    ax_beh[0].set_title('p(truth)\nt-test statistics across context lengths')
    ax_beh[1].legend()
    ax_beh[0].grid(True, alpha=0.3)

    ax_beh[1].set_xlabel('Context Length (N)')
    ax_beh[1].set_ylabel('Mean diff')
    ax_beh[1].set_title('p(truth)\nmean diff across context lengths')
    ax_beh[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(projection_results_path, 'context_effect_results.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_activation_context_effects(context_lengths_activations, activations_ttest_stats, activations_mean_diffs,
                                  activations_p_values, control_activations_ttest_stats, control_activations_mean_diffs,
                                  pairs_to_compare, n_layers, projection_results_path):
    """
    Plot activation context effects across all layers.
    
    Parameters:
    - context_lengths_activations: list of context lengths for activations
    - activations_ttest_stats: array of shape [len(pairs_to_compare), len(context_lengths_activations)]
    - activations_mean_diffs: array of shape [len(pairs_to_compare), len(context_lengths_activations)]
    - activations_p_values: array of shape [len(pairs_to_compare), len(context_lengths_activations)]
    - control_activations_ttest_stats: array of shape [len(pairs_to_compare), len(context_lengths_activations), num_control_directions]
    - control_activations_mean_diffs: array of shape [len(pairs_to_compare), len(context_lengths_activations), num_control_directions]
    - pairs_to_compare: list of tuples with context type pairs
    - n_layers: number of layers (should be 32)
    - projection_results_path: path to save the figure
    """
    fig_act = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(8, 8, figure=fig_act, wspace=0.15, hspace=0.3)

    # Create subplots with custom spacing - smaller gaps within pairs
    axes_act = []
    for row in range(8):
        row_axes = []
        for col_pair in range(4):  # 4 pairs of columns
            # Left subplot of pair (t-stat)
            ax_left = fig_act.add_subplot(gs[row, col_pair*2])
            # Right subplot of pair (mean difference) 
            ax_right = fig_act.add_subplot(gs[row, col_pair*2 + 1])
            row_axes.extend([ax_left, ax_right])
        axes_act.append(row_axes)

    axes_act = np.array(axes_act)

    # Reshape axes for easier indexing: [layer, stat_type] where stat_type 0=t-stat, 1=mean difference
    axes_act = axes_act.reshape(n_layers, 2)

    colors_act = plt.cm.tab10(np.linspace(0, 1, len(pairs_to_compare)))

    for layer in range(n_layers):
        # Left subplot: t-test statistics
        ax_tstat = axes_act[layer, 0]
        # Right subplot: mean difference
        ax_mean_diffs = axes_act[layer, 1]
        
        for i_pair, (pair_context_1, pair_context_2) in enumerate(pairs_to_compare):
            # Line style: first pair solid, others dotted
            linestyle = '-' if i_pair == 0 else '--'
            color = colors_act[i_pair]
            
            # Plot t-statistics (left subplot)
            ax_tstat.plot(context_lengths_activations, activations_ttest_stats[i_pair, :], 
                         color=color, linestyle=linestyle, marker='o', 
                         linewidth=1.5, markersize=4)
            
            # Plot mean difference (right subplot)
            ax_mean_diffs.plot(context_lengths_activations, activations_mean_diffs[i_pair, :], 
                          color=color, linestyle=linestyle, marker='o', 
                          linewidth=1.5, markersize=4)
            
            # Add error bars for control data
            # Control stats shape: [len(pairs_to_compare), len(context_lengths), num_control_directions]
            control_tstat_mean = np.mean(control_activations_ttest_stats[i_pair, :, :], axis=1)
            control_tstat_std = np.std(control_activations_ttest_stats[i_pair, :, :], axis=1)
            control_mean_diffs_mean = np.mean(control_activations_mean_diffs[i_pair, :, :], axis=1)
            control_mean_diffs_std = np.std(control_activations_mean_diffs[i_pair, :, :], axis=1)
            
            # Add control error bars (slightly offset x for visibility)
            x_offset = (i_pair - len(pairs_to_compare)/2) * 0.05
            x_positions = np.array(context_lengths_activations) + x_offset
            
            ax_tstat.errorbar(x_positions, control_tstat_mean, yerr=control_tstat_std,
                             color=color, linestyle=':', alpha=0.6, capsize=2, 
                             capthick=1, linewidth=1)
            ax_mean_diffs.errorbar(x_positions, control_mean_diffs_mean, yerr=control_mean_diffs_std,
                              color=color, linestyle=':', alpha=0.6, capsize=2, 
                              capthick=1, linewidth=1)
            
            # Add significance stars to t-stat plot only
            for i_N, (N, p_val) in enumerate(zip(context_lengths_activations, activations_p_values[i_pair, :])):
                if p_val < 0.05 and not np.isnan(p_val):
                    y_pos = activations_ttest_stats[i_pair, i_N]
                    ax_tstat.scatter(N, y_pos, marker='*', s=50, color=color, 
                                   edgecolors='black', linewidth=0.3, zorder=10)
        
        # Formatting
        ax_tstat.set_title(f'Layer {layer}: T-statistics', fontsize=10)
        ax_mean_diffs.set_title(f'Layer {layer}: Mean difference', fontsize=10)
        
        if layer == n_layers - 1:  # Bottom row
            ax_tstat.set_xlabel('Context Length (N)')
            ax_mean_diffs.set_xlabel('Context Length (N)')
        
        ax_tstat.grid(True, alpha=0.3)
        ax_mean_diffs.grid(True, alpha=0.3)
        
        # Smaller font for tick labels
        ax_tstat.tick_params(labelsize=8)
        ax_mean_diffs.tick_params(labelsize=8)

    # Add legend to the figure
    legend_elements = []
    for i_pair, (pair_context_1, pair_context_2) in enumerate(pairs_to_compare):
        linestyle = '-' if i_pair == 0 else '--'
        legend_elements.append(plt.Line2D([0], [0], color=colors_act[i_pair], 
                                         linestyle=linestyle, linewidth=2,
                                         label=f'{pair_context_1} vs {pair_context_2}'))

    fig_act.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
                   ncol=len(pairs_to_compare), fontsize=10)

    plt.suptitle('Activation Context Effects Across Layers', fontsize=14, y=0.98)
    plt.subplots_adjust(bottom=0.08, top=0.94, hspace=0.3, wspace=0.15)
    plt.savefig(os.path.join(projection_results_path, 'activation_context_effects.png'), dpi=300, bbox_inches='tight')
    plt.close()





def plot_absolute_activation_context_effects(layer_idx, context_lengths_activations, contextual_activations_data,
                                           control_contextual_activations_data, context_type_aliases, 
                                           projection_results_path):
    """
    Plot absolute activation context effects for a single layer.
    
    Parameters:
    - layer_idx: int, the layer index to plot
    - context_lengths_activations: list of context lengths for activations
    - contextual_activations_data: dict with context types as keys, projection arrays as values
    - control_contextual_activations_data: dict with context types as keys, control projection arrays as values
    - context_type_aliases: dict mapping context type keys to legend names
    - projection_results_path: path to save the figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Colors for different context types
    colors = plt.cm.tab10(np.linspace(0, 1, len(context_type_aliases)))

    for i, (context_type, alias) in enumerate(context_type_aliases.items()):
        color = colors[i]
        
        # Get projection data for this context type
        projections = contextual_activations_data[context_type]
        control_projections = control_contextual_activations_data[context_type]
        
        # Plot main projection means (solid line)
        ax.plot(context_lengths_activations, projections, 
               color=color, linestyle='-', marker='o', 
               label=alias, linewidth=2, markersize=6)
        
        # Add control data (dotted lines with error bars)
        control_mean = np.mean(control_projections, axis=1)
        control_std = np.std(control_projections, axis=1)
        
        # Slightly offset x positions for visibility
        x_offset = (i - len(context_type_aliases)/2) * 0.05
        x_positions = np.array(context_lengths_activations) + x_offset

        ax.errorbar(x_positions, control_mean, yerr=control_std,
                   color=color, linestyle=':', alpha=0.6, capsize=3,
                   capthick=1, linewidth=1.5, label=f'{alias} (control vectors)')

    # Formatting
    ax.set_xlabel('Context Length (N)', fontsize=12)
    ax.set_ylabel('Absolute Projection Mean', fontsize=12)
    ax.set_title(f'Layer {layer_idx + 1}: Absolute Projection Means', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(projection_results_path, f'context_absolute_effect_results_layer_{layer_idx + 1}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()



import matplotlib.pyplot as plt
import numpy as np
import os


def plot_differential_activation_context_effects(layer_idx, context_lengths_activations, contextual_activations_data,
                                                control_contextual_activations_data, control_context_type,
                                                context_type_aliases, projection_results_path):
    """
    Plot differential activation context effects for a single layer relative to a control context.
    
    Parameters:
    - layer_idx: int, the layer index to plot
    - context_lengths_activations: list of context lengths for activations
    - contextual_activations_data: dict with context types as keys, raw projection arrays as values [num_context_lengths, questions]
    - control_contextual_activations_data: dict with context types as keys, control projection arrays as values [num_context_lengths, num_control_directions, questions]
    - control_context_type: str, the context type to use as baseline/control
    - context_type_aliases: dict mapping context type keys to legend names
    - projection_results_path: path to save the figure
    """
    # Filter out the control context from plotting
    plot_context_types = [ct for ct in context_type_aliases.keys() if ct != control_context_type]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Colors for different context types
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_context_types)))

    control_display_name = context_type_aliases.get(control_context_type, control_context_type.replace("_", " ").title())
    
    for i, context_type in enumerate(plot_context_types):
        if context_type not in contextual_activations_data or control_context_type not in contextual_activations_data:
            continue
            
        color = colors[i]
        alias = context_type_aliases.get(context_type, context_type.replace("_", " ").title())
        
        # Get projection data for this context type and control
        context_projections = contextual_activations_data[context_type]  # [num_context_lengths, questions]
        control_projections = contextual_activations_data[control_context_type]  # [num_context_lengths, questions]
        
        # Calculate differences for each context length
        diff_means = []
        diff_stds = []
        
        for length_idx, length in enumerate(context_lengths_activations):
            # Compute difference across questions dimension
            question_diffs = context_projections[length_idx] - control_projections[length_idx]  # [questions]
            
            diff_means.append(np.mean(question_diffs))
            diff_stds.append(np.std(question_diffs))
        
        # Add small jitter to x-values for visibility
        x_offset = (i - len(plot_context_types)/2) * 0.05
        x_positions = np.array(context_lengths_activations) + x_offset
        
        # Plot main differences (solid line)
        ax.errorbar(x_positions, diff_means, yerr=diff_stds,
                   color=color, linestyle='-', marker='o', 
                   label=alias, linewidth=2, markersize=6,
                   capsize=3, capthick=1)
        
        # Add individual control vector comparisons (dotted lines)
        if (context_type in control_contextual_activations_data and 
            control_context_type in control_contextual_activations_data):
            
            context_control_data = control_contextual_activations_data[context_type]  # [num_context_lengths, num_control_directions, questions]
            control_control_data = control_contextual_activations_data[control_context_type]  # [num_context_lengths, num_control_directions, questions]
            
            num_control_directions = context_control_data.shape[1]  # Should be 16
            
            # Plot each of the 16 control vectors individually
            for control_dir in range(num_control_directions):
                control_diff_means = []
                control_diff_stds = []
                
                for length_idx, length in enumerate(context_lengths_activations):
                    context_control_projs = context_control_data[length_idx, control_dir, :]  # [questions]
                    control_control_projs = control_control_data[length_idx, control_dir, :]  # [questions]
                    
                    # Difference across questions for this control direction
                    control_question_diffs = context_control_projs - control_control_projs  # [questions]
                    control_diff_means.append(np.mean(control_question_diffs))
                    control_diff_stds.append(np.std(control_question_diffs))
                
                # Plot this individual control vector (dotted line)
                # Only add label for the first control vector to avoid legend clutter
                control_label = f'{alias} (control vectors)' if control_dir == 0 else None
                
                ax.errorbar(x_positions, control_diff_means, yerr=control_diff_stds,
                           color=color, linestyle=':', alpha=0.3, capsize=1,
                           capthick=0.5, linewidth=1, label=control_label)

    # Add horizontal reference line at y=0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Formatting
    ax.set_xlabel('Context Length (N)', fontsize=12)
    ax.set_ylabel(f'Δ Projection vs {control_display_name}', fontsize=12)
    ax.set_title(f'Layer {layer_idx + 1}: Projection Differences Relative to {control_display_name}', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(projection_results_path, f'context_differential_effect_results_layer_{layer_idx + 1}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
