import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import ttest_rel
import os


def plot_context_effect_analysis(all_results, context_types, context_lengths, n_samples, output_path, filename_prefix=""):
    """
    Plot truth vs lie log probability differences by context composition.
    
    Args:
        all_results: Dictionary with results for each context type
        context_types: List of context type names
        context_lengths: List of context lengths tested
        n_samples: Number of samples per context length
        output_path: Directory to save the plot
        filename_prefix: Prefix to add to filename (e.g., "posthoc_")
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(context_types)))
    
    num_context_lengths = len(context_lengths)

    for i, context_type in enumerate(context_types):
        results = all_results[context_type]

        context_lengths_plot = results['context_length']
        mean_diffs = results['mean_truth_lie_diff']
        std_diffs = results['std_truth_lie_diff']
        
        # Add small jitter to x-values to separate overlapping points
        jitter = (i - len(context_types)/2) * 0.05
        x_values = np.array(context_lengths_plot) + jitter
        
        axes.errorbar(x_values, mean_diffs, yerr=std_diffs, 
                    label=f'{context_type.replace("_", " ").title()}',
                    marker='o', capsize=3, capthick=1, linewidth=2, markersize=6,
                    color=colors[i], alpha=0.8)

    # Add significance testing between lie and truth contexts
    if 'top_lie_shuffled_together' in all_results and 'top_truth_shuffled_together' in all_results:
        lie_results = all_results['top_lie_shuffled_together']
        truth_results = all_results['top_truth_shuffled_together']
        
        for length_idx in range(num_context_lengths):
            if not np.isnan(lie_results['context_length'][length_idx]) and not np.isnan(truth_results['context_length'][length_idx]):
                N_current = lie_results['context_length'][length_idx]
                n_samples_eff = min(n_samples, math.perm(int(N_current), int(N_current)))
                
                # Get question means across samples for both contexts
                lie_question_means = np.mean(lie_results['question_truth_lie_diffs_across_samples'][length_idx, :, :n_samples_eff], axis=1)
                truth_question_means = np.mean(truth_results['question_truth_lie_diffs_across_samples'][length_idx, :, :n_samples_eff], axis=1)
                
                if len(lie_question_means) > 1 and len(truth_question_means) > 1:
                    stat, p_value = ttest_rel(lie_question_means, truth_question_means)
                    
                    if p_value < 0.05:
                        max_y = max(
                            lie_results['mean_truth_lie_diff'][length_idx] + lie_results['std_truth_lie_diff'][length_idx],
                            truth_results['mean_truth_lie_diff'][length_idx] + truth_results['std_truth_lie_diff'][length_idx]
                        )
                        axes.text(N_current, max_y + 0.01, '*', 
                                ha='center', va='bottom', fontsize=16, fontweight='bold')

    axes.set_xlabel('Context Length (N)')
    axes.set_ylabel('Mean Log P(Truth) - Log P(Lie)')
    axes.set_title('Truth vs Lie Log Probability Differences by Context Composition\n(Yes/No Context Format, * indicates p<0.05 for lie vs truth contexts)')
    axes.legend()
    axes.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{filename_prefix}context_effect_analysis.png'
    plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_context_effect_by_question_type(all_results, context_types, context_lengths, n_samples, 
                                        unique_questions, initial_questions_df, output_path, filename_prefix=""):
    """
    Plot truth vs lie log probability differences by question type and context composition.
    
    Args:
        all_results: Dictionary with results for each context type
        context_types: List of context type names  
        context_lengths: List of context lengths tested
        n_samples: Number of samples per context length
        unique_questions: Array of unique question indices
        initial_questions_df: DataFrame with question data including 'type' column
        output_path: Directory to save the plot
        filename_prefix: Prefix to add to filename (e.g., "posthoc_")
    """
    question_types = initial_questions_df['type'].unique()
    num_initial_question_types = len(question_types)
    num_context_lengths = len(context_lengths)
    
    fig, axes = plt.subplots(num_initial_question_types, 1, figsize=(10, 5*num_initial_question_types))
    if num_initial_question_types == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(context_types)))

    for type_idx, question_type in enumerate(question_types):
        # Get questions of this type
        type_question_indices = initial_questions_df[initial_questions_df['type'] == question_type].index.tolist()
        # Find which positions in unique_questions correspond to this type
        type_positions = [i for i, q_idx in enumerate(unique_questions) if q_idx in type_question_indices]
        
        for i, context_type in enumerate(context_types):
            results = all_results[context_type]
            
            # Extract data for completed context lengths
            completed_lengths = []
            type_means = []
            type_stds = []
            individual_question_data = []
            
            for length_idx in range(num_context_lengths):
                if not np.isnan(results['context_length'][length_idx]):
                    N_current = results['context_length'][length_idx]
                    completed_lengths.append(N_current)
                    
                    # Get data for this question type
                    n_samples_eff = min(n_samples, math.perm(int(N_current), int(N_current)))
                    type_data = results['question_truth_lie_diffs_across_samples'][length_idx][type_positions, :n_samples_eff]
                    question_means = np.mean(type_data, axis=1)  # Mean across samples for each question
                    
                    # Store individual question means for plotting
                    individual_question_data.append(question_means)
                    
                    # Calculate mean and std across questions of this type
                    type_means.append(np.mean(question_means))
                    type_stds.append(np.std(question_means))
            
            if len(completed_lengths) > 0:
                # Add small jitter to x-values
                jitter = (i - len(context_types)/2) * 0.05
                x_values = np.array(completed_lengths) + jitter
                
                # Plot mean line with error bars (normal alpha)
                axes[type_idx].errorbar(x_values, type_means, yerr=type_stds,
                                    label=f'{context_type.replace("_", " ").title()}',
                                    marker='o', capsize=3, capthick=1, linewidth=2, markersize=6,
                                    color=colors[i], alpha=0.8)
                
                # Plot individual question lines (low alpha)
                for q_pos in range(len(type_positions)):
                    individual_means = [individual_question_data[length_idx][q_pos] for length_idx in range(len(completed_lengths))]
                    axes[type_idx].plot(x_values, individual_means, 
                                    color=colors[i], alpha=0.2, linewidth=1)
        
        axes[type_idx].set_xlabel('Context Length (N)')
        axes[type_idx].set_ylabel('Mean Log P(Truth) - Log P(Lie)')
        axes[type_idx].set_title(f'{question_type} Questions')
        axes[type_idx].legend()
        axes[type_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    filename = f'{filename_prefix}context_effect_by_question_type.png'
    plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_context_diff_analysis(all_results, context_types, control_context_type, context_lengths, n_samples, 
                               output_path, truth_answer_label, lie_answer_label, context_aliases=None, filename_prefix=""):
    """
    Plot differences from control context for truth vs lie log probability differences.
    
    Args:
        all_results: Dictionary with results for each context type
        context_types: List of context type names
        control_context_type: Name of the context type to use as control/baseline
        context_lengths: List of context lengths tested
        n_samples: Number of samples per context length
        output_path: Directory to save the plot
        truth_answer_label: Label for truth answers in y-axis
        lie_answer_label: Label for lie answers in y-axis
        context_aliases: Dictionary mapping context type names to display names
        filename_prefix: Prefix to add to filename (e.g., "posthoc_")
    """
    if context_aliases is None:
        context_aliases = {}
    
    # Filter out the control context from plotting
    plot_context_types = [ct for ct in context_types if ct != control_context_type]
    
    fig, axes = plt.subplots(1, 1, figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_context_types)))
    
    num_context_lengths = len(context_lengths)
    control_results = all_results[control_context_type]

    for i, context_type in enumerate(plot_context_types):
        if context_type not in all_results:
            continue
            
        results = all_results[context_type]
        
        # Calculate differences from control for each context length
        diff_means = []
        diff_stds = []
        valid_lengths = []
        
        for length_idx in range(num_context_lengths):
            if (not np.isnan(results['context_length'][length_idx]) and 
                not np.isnan(control_results['context_length'][length_idx])):
                
                N_current = results['context_length'][length_idx]
                n_samples_eff = min(n_samples, math.perm(int(N_current), int(N_current)))
                
                # Get question means across samples for both contexts
                other_question_means = np.mean(results['question_truth_lie_diffs_across_samples'][length_idx, :, :n_samples_eff], axis=1)
                control_question_means = np.mean(control_results['question_truth_lie_diffs_across_samples'][length_idx, :, :n_samples_eff], axis=1)
                
                # Calculate differences
                question_diffs = other_question_means - control_question_means
                
                diff_means.append(np.mean(question_diffs))
                diff_stds.append(np.std(question_diffs))
                valid_lengths.append(N_current)
        
        if len(valid_lengths) > 0:
            # Add small jitter to x-values to separate overlapping points
            jitter = (i - len(plot_context_types)/2) * 0.05
            x_values = np.array(valid_lengths) + jitter
            
            # Get display name for legend
            display_name = context_aliases.get(context_type, context_type.replace("_", " ").title())
            
            axes.errorbar(x_values, diff_means, yerr=diff_stds,
                        label=display_name,
                        marker='o', capsize=3, capthick=1, linewidth=2, markersize=6,
                        color=colors[i], alpha=0.8)

    # Add horizontal line at y=0 for reference
    axes.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    axes.set_xlabel('Context Length (N)')
    axes.set_ylabel(f'Difference in Log P({truth_answer_label}) - Log P({lie_answer_label})')
    
    control_display_name = context_aliases.get(control_context_type, control_context_type.replace("_", " ").title())
    axes.set_title(f'Context Effect Differences Relative to {control_display_name}\n(Positive = More {truth_answer_label} Bias Than Control)')
    axes.legend()
    axes.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{filename_prefix}context_diff_analysis.png'
    plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_context_diff_by_question_type(all_results, context_types, control_context_type, context_lengths, n_samples,
                                      unique_questions, initial_questions_df, output_path, truth_answer_label, 
                                      lie_answer_label, context_aliases=None, aligned_misaligned_keys=None, filename_prefix=""):
    """
    Plot differences from control context by question type.
    
    Args:
        all_results: Dictionary with results for each context type
        context_types: List of context type names
        control_context_type: Name of the context type to use as control/baseline
        context_lengths: List of context lengths tested
        n_samples: Number of samples per context length
        unique_questions: Array of unique question indices
        initial_questions_df: DataFrame with question data including 'type' column
        output_path: Directory to save the plot
        truth_answer_label: Label for truth answers in y-axis
        lie_answer_label: Label for lie answers in y-axis
        context_aliases: Dictionary mapping context type names to display names
        aligned_misaligned_keys: Optional tuple of (aligned_key, misaligned_key) for SNR analysis
        filename_prefix: Prefix to add to filename (e.g., "posthoc_")
    """
    if context_aliases is None:
        context_aliases = {}
    
    # Filter out the control context from plotting
    plot_context_types = [ct for ct in context_types if ct != control_context_type]
    
    question_types = initial_questions_df['type'].unique()
    num_initial_question_types = len(question_types)
    num_context_lengths = len(context_lengths)
    control_results = all_results[control_context_type]
    
    # Pre-calculate question type positions
    type_positions_map = {}
    for question_type in question_types:
        type_question_indices = initial_questions_df[initial_questions_df['type'] == question_type].index.tolist()
        type_positions_map[question_type] = [i for i, q_idx in enumerate(unique_questions) if q_idx in type_question_indices]
    
    # Data collection: collect all needed data in one pass
    plot_data = {}  # {context_type: {question_type: {'lengths': [], 'means': [], 'stds': [], 'individual_diffs': []}}}
    snr_data = {}   # {question_type: {'lengths': [], 'snr_values': []}}
    
    # Initialize data structures
    for context_type in plot_context_types:
        if context_type in all_results:
            plot_data[context_type] = {qt: {'lengths': [], 'means': [], 'stds': [], 'individual_diffs': []} 
                                     for qt in question_types}
    
    if aligned_misaligned_keys is not None:
        aligned_key, misaligned_key = aligned_misaligned_keys
        if aligned_key in all_results and misaligned_key in all_results:
            snr_data = {qt: {'lengths': [], 'snr_values': []} for qt in question_types}
            aligned_results = all_results[aligned_key]
            misaligned_results = all_results[misaligned_key]
    
    # Single data collection loop
    for length_idx in range(num_context_lengths):
        # Check if this length is valid for control
        if np.isnan(control_results['context_length'][length_idx]):
            continue
            
        N_current = control_results['context_length'][length_idx]
        n_samples_eff = min(n_samples, math.perm(int(N_current), int(N_current)))
        
        for question_type in question_types:
            type_positions = type_positions_map[question_type]
            
            # Get control data for this question type
            control_type_data = control_results['question_truth_lie_diffs_across_samples'][length_idx][type_positions, :n_samples_eff]
            control_question_means = np.mean(control_type_data, axis=1)
            
            # Process main plot data for each context type
            for context_type in plot_context_types:
                if context_type not in all_results:
                    continue
                    
                results = all_results[context_type]
                if np.isnan(results['context_length'][length_idx]):
                    continue
                
                # Get data for this question type and context
                other_type_data = results['question_truth_lie_diffs_across_samples'][length_idx][type_positions, :n_samples_eff]
                other_question_means = np.mean(other_type_data, axis=1)
                
                # Calculate differences from control
                question_diffs = other_question_means - control_question_means
                
                # Store plot data
                plot_data[context_type][question_type]['lengths'].append(N_current)
                plot_data[context_type][question_type]['means'].append(np.mean(question_diffs))
                plot_data[context_type][question_type]['stds'].append(np.std(question_diffs))
                plot_data[context_type][question_type]['individual_diffs'].append(question_diffs)
            
            # Process SNR data if needed
            if (aligned_misaligned_keys is not None and 
                aligned_key in all_results and misaligned_key in all_results and
                not np.isnan(aligned_results['context_length'][length_idx]) and 
                not np.isnan(misaligned_results['context_length'][length_idx])):
                
                # Get aligned and misaligned data
                aligned_type_data = aligned_results['question_truth_lie_diffs_across_samples'][length_idx][type_positions, :n_samples_eff]
                misaligned_type_data = misaligned_results['question_truth_lie_diffs_across_samples'][length_idx][type_positions, :n_samples_eff]
                
                aligned_question_means = np.mean(aligned_type_data, axis=1)
                misaligned_question_means = np.mean(misaligned_type_data, axis=1)
                
                # Calculate SNR (paired Cohen's d)
                question_diffs_snr = aligned_question_means - misaligned_question_means
                if len(question_diffs_snr) > 1 and np.std(question_diffs_snr) > 0:
                    snr = abs(np.mean(question_diffs_snr)) / np.std(question_diffs_snr)
                    snr_data[question_type]['lengths'].append(N_current)
                    snr_data[question_type]['snr_values'].append(snr)
    
    # Generate main difference plot
    fig, axes = plt.subplots(num_initial_question_types, 1, figsize=(14, 5*num_initial_question_types))
    if num_initial_question_types == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(plot_context_types)))

    for type_idx, question_type in enumerate(question_types):
        for i, context_type in enumerate(plot_context_types):
            if context_type not in plot_data:
                continue
            
            data = plot_data[context_type][question_type]
            if not data['lengths']:
                continue
            
            # Add small jitter to x-values
            jitter = (i - len(plot_context_types)/2) * 0.05
            x_values = np.array(data['lengths']) + jitter
            
            # Get display name for legend
            display_name = context_aliases.get(context_type, context_type.replace("_", " ").title())
            
            # Plot mean line with error bars
            axes[type_idx].errorbar(x_values, data['means'], yerr=data['stds'],
                                label=display_name,
                                marker='o', capsize=3, capthick=1, linewidth=2, markersize=6,
                                color=colors[i], alpha=0.8)
            
            # Plot individual question lines (low alpha)
            for q_pos in range(len(type_positions_map[question_type])):
                individual_diffs = [data['individual_diffs'][length_idx][q_pos] for length_idx in range(len(data['lengths']))]
                axes[type_idx].plot(x_values, individual_diffs, 
                                color=colors[i], alpha=0.2, linewidth=1)
        
        # Add horizontal line at y=0 for reference
        axes[type_idx].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        axes[type_idx].set_xlabel('Context Length (N)')
        axes[type_idx].set_ylabel(f'Difference in Log P({truth_answer_label}) - Log P({lie_answer_label})')
        axes[type_idx].set_title(f'{question_type} Questions')
        axes[type_idx].legend()
        axes[type_idx].grid(True, alpha=0.3)

    control_display_name = context_aliases.get(control_context_type, control_context_type.replace("_", " ").title())
    plt.suptitle(f'Context Effect Differences by Question Type Relative to {control_display_name}')
    plt.tight_layout()
    filename = f'{filename_prefix}context_diff_by_question_type.png'
    plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate SNR plot if data was collected
    if snr_data and any(snr_data[qt]['lengths'] for qt in question_types):
        fig_snr, axes_snr = plt.subplots(1, 1, figsize=(14, 5))
        
        for type_idx, question_type in enumerate(question_types):
            data = snr_data[question_type]
            if data['lengths']:
                axes_snr.plot(data['lengths'], data['snr_values'], 
                                      marker='o', linewidth=2, markersize=6, label = question_type)
            
            axes_snr.set_xlabel('Context Length (N)')
            axes_snr.set_ylabel('SNR (Paired Cohen\'s d)')
            axes_snr.set_title(f'{question_type} Questions')
            axes_snr.grid(True, alpha=0.3)
        
        aligned_display = context_aliases.get(aligned_key, aligned_key.replace("_", " ").title())
        misaligned_display = context_aliases.get(misaligned_key, misaligned_key.replace("_", " ").title())
        axes_snr.legend()
        plt.suptitle(f'SNR Between {aligned_display} and {misaligned_display} by Question Type')
        plt.tight_layout()
        
        snr_filename = f'{filename_prefix}context_diff_SNR_by_question_type.png'
        plt.savefig(os.path.join(output_path, snr_filename), dpi=300, bbox_inches='tight')
        plt.close()
