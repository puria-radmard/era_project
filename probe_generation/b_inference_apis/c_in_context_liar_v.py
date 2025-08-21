"""
Collect In-Context Steering Results Script

This script checks the status of submitted in-context steering batch jobs,
downloads completed results, extracts logprobs, aggregates data, and generates visualizations.
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from typing import Dict, List, Tuple, Optional, Any
from model.fireworks import (
    FireworksBatchWrapper,
    load_batch_metadata
)
from util.util import YamlConfig


def parse_custom_id(custom_id: str) -> Tuple[int, int, str, int, str, int]:
    """
    Parse custom ID to extract experimental parameters.
    
    Args:
        custom_id: String like "N5_s2_ctaligned_q42_rtruth3"
        
    Returns:
        Tuple of (context_length, sample_idx, context_type, question_idx, response_type, resp_idx)
    """
    parts = custom_id.split('_')
    context_length = int(parts[0][1:])  # Remove 'N'
    sample_idx = int(parts[1][1:])      # Remove 's'
    context_type = parts[2][2:]         # Remove 'ct'
    question_idx = int(parts[3][1:])    # Remove 'q'
    
    # Parse response part: "rtruth3" -> response_type="truth", resp_idx=3
    response_part = parts[4][1:]        # Remove 'r'
    if response_part.startswith('truth'):
        response_type = 'truth'
        resp_idx = int(response_part[5:])  # Remove 'truth'
    elif response_part.startswith('lie'):
        response_type = 'lie'
        resp_idx = int(response_part[3:])   # Remove 'lie'
    else:
        raise ValueError(f"Unknown response type in custom_id: {custom_id}")
    
    return context_length, sample_idx, context_type, question_idx, response_type, resp_idx


def extract_prefilled_logprobs(response_data: dict) -> Tuple[float, str]:
    """
    Extract the average logprob of the prefilled response tokens.
    
    Args:
        response_data: The response data from the API
        
    Returns:
        Average logprob of the prefilled response tokens
    """
    logprobs_data = response_data['choices'][0]['logprobs']
    tokens = logprobs_data['tokens']
    token_logprobs = logprobs_data['token_logprobs']

    model_name = response_data['model']
    
    # Find the last two assistant header occurrences
    assistant_starts = []
    for i, token in enumerate(tokens):
        
        if 'llama' in model_name.lower():
            if token == '<|start_header_id|>':
                # Check if next tokens form assistant header
                if (i + 2 < len(tokens) and 
                    tokens[i + 1] == 'assistant' and 
                    tokens[i + 2] == '<|end_header_id|>'):
                    assistant_starts.append(i)
        
        elif 'mistral' in model_name.lower():
            if token == '[/INST]':
                assistant_starts.append(i)
    
    if 'llama' in model_name:
        if True:
            start_idx = assistant_starts[-1] + 3  # Skip <|start_header_id|>assistant<|end_header_id|>
            end_idx = len(tokens)

        else:
            # Get the span between the final two assistant headers
            # Content starts after the last assistant header + newline tokens
            start_idx = assistant_starts[-2] + 3  # Skip <|start_header_id|>assistant<|end_header_id|>
            end_idx = assistant_starts[-1]
            tokens[assistant_starts[-1] + 3: len(tokens)]
            
    else:
        start_idx = assistant_starts[-1] + 1  # Skip [/INST]
        end_idx = len(tokens)
    
    # Skip initial whitespace/newline tokens
    while start_idx < end_idx and tokens[start_idx].strip() == '':
        start_idx += 1
    
    # Extract logprobs for the content tokens
    if start_idx >= end_idx:
        raise ValueError("No content tokens found in prefilled response")
    
    content_logprobs = token_logprobs[start_idx:end_idx]
    content_tokens = tokens[start_idx:end_idx]      # This is just for debugging

    # Filter out any remaining special tokens or empty tokens
    filtered_logprobs = []
    filtered_tokens = []
    for i in range(len(content_logprobs)):
        token = tokens[start_idx + i]
        if (not token.startswith('<|') and 
            not token.endswith('|>') and 
            token.strip() != ''):
            filtered_logprobs.append(content_logprobs[i])
            filtered_tokens.append(content_tokens[i])
    
    if not filtered_logprobs:
        raise ValueError("No valid content tokens found after filtering")

    return np.mean(filtered_logprobs), "".join(content_tokens)


def check_batch_statuses(batch_wrapper: FireworksBatchWrapper, batch_ids: Dict[int, str]) -> Dict[int, str]:
    """
    Check the status of all batches.
    
    Returns:
        Dict mapping context_length to status
    """
    statuses = {}
    for context_length, batch_id in batch_ids.items():
        status = batch_wrapper.get_batch_status(batch_id)
        statuses[context_length] = status['status']
        print(f"Context length {context_length}: {status['status']}")
    
    return statuses


def download_and_process_batch(
    batch_wrapper: FireworksBatchWrapper,
    context_length: int,
    batch_id: str,
    raw_outputs_dir: str
) -> Optional[Dict[str, Any]]:
    """
    Download and process results from a single batch.
    
    Returns:
        Dictionary with processed results or None if failed
    """
    # Download results
    results_path = os.path.join(raw_outputs_dir, f'steering_results_N{context_length}.jsonl')

    if os.path.exists(results_path):
        print(f'Results for context length {context_length} {results_path} already exists, not downloading again.')
    else:
        batch_wrapper.download_batch_results(batch_id, results_path)
        print(f"Downloaded results for context length {context_length}")
    
    # Process results
    results = {}
    example_rollouts = {}
    with open(results_path, 'r') as f:
        for line in f:
            result = json.loads(line.strip())
            custom_id = result['custom_id']
            
            # Parse custom ID
            ctx_len, sample_idx, context_type, question_idx, response_type, resp_idx = parse_custom_id(custom_id)
            
            # Extract logprob
            logprob, rollout = extract_prefilled_logprobs(result['response'])
            
            # Store result
            key = (sample_idx, context_type, question_idx, response_type, resp_idx)
            results[key] = logprob

            example_rollouts[key] = rollout
    
    print(f"Processed {len(results)} results for context length {context_length}")
    return results
        


def aggregate_and_save_results(
    processed_results: Dict[int, Dict[str, Any]],
    metadata: Dict[str, Any],
    aggregated_dir: str
) -> None:
    """
    Aggregate processed results and save to .npy files.
    """
    context_types = ['aligned', 'misaligned', 'random']
    n_samples = metadata['n_samples']
    
    for context_length, results in processed_results.items():
        print(f"Aggregating results for context length {context_length}")
        
        # Find dimensions
        unique_questions = set()
        max_resp_idx = {'truth': 0, 'lie': 0}
        
        for (sample_idx, context_type, question_idx, response_type, resp_idx) in results.keys():
            unique_questions.add(question_idx)
            max_resp_idx[response_type] = max(max_resp_idx[response_type], resp_idx)
        
        unique_questions = sorted(list(unique_questions))
        n_questions = len(unique_questions)
        n_truth_rollouts = max_resp_idx['truth'] + 1
        n_lie_rollouts = max_resp_idx['lie'] + 1
        
        # Create mapping from question_idx to position
        question_to_pos = {q_idx: pos for pos, q_idx in enumerate(unique_questions)}
        
        for context_type in context_types:
            # Initialize arrays
            truth_logprobs = np.full((n_samples, n_questions, n_truth_rollouts), np.nan)
            lie_logprobs = np.full((n_samples, n_questions, n_lie_rollouts), np.nan)
            
            # Fill arrays
            for (sample_idx, ct, question_idx, response_type, resp_idx), logprob in results.items():
                if ct != context_type:
                    continue
                
                if question_idx not in question_to_pos:
                    continue
                
                q_pos = question_to_pos[question_idx]
                
                if response_type == 'truth' and resp_idx < n_truth_rollouts:
                    truth_logprobs[sample_idx, q_pos, resp_idx] = logprob
                elif response_type == 'lie' and resp_idx < n_lie_rollouts:
                    lie_logprobs[sample_idx, q_pos, resp_idx] = logprob
            
            # Save results
            result_data = {
                'question_truth_log_probs': truth_logprobs,
                'question_lie_log_probs': lie_logprobs,
                'unique_questions': unique_questions,
                'context_length': context_length,
                'n_samples': n_samples
            }
            
            filename = f'steering_results_{context_type}_N{context_length}.npy'
            filepath = os.path.join(aggregated_dir, filename)
            np.save(filepath, result_data)
            
            print(f"Saved {context_type} results for N={context_length}: {truth_logprobs.shape}")


def load_aggregated_results(aggregated_dir: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Load aggregated results from .npy files.
    
    Returns:
        Dict structure: {context_type: {context_length: data_dict}}
    """
    all_results = {}
    context_types = ['aligned', 'misaligned', 'random']
    
    for context_type in context_types:
        all_results[context_type] = {}
        
        # Find all files for this context type
        for filename in os.listdir(aggregated_dir):
            if filename.startswith(f'steering_results_{context_type}_N') and filename.endswith('.npy'):
                # Extract context length from filename
                context_length_str = filename.split('_N')[1].split('.')[0]
                context_length = int(context_length_str)
                
                # Load data
                filepath = os.path.join(aggregated_dir, filename)
                data = np.load(filepath, allow_pickle=True).item()
                
                # Calculate truth-lie differences for compatibility with viz functions
                truth_probs = data['question_truth_log_probs']  # [n_samples, n_questions, n_rollouts]
                lie_probs = data['question_lie_log_probs']
                
                # Average across rollouts, then calculate difference
                truth_means = np.nanmean(truth_probs, axis=2)  # [n_samples, n_questions]
                lie_means = np.nanmean(lie_probs, axis=2)
                truth_lie_diffs = truth_means - lie_means  # [n_samples, n_questions]
                
                # Store in format expected by visualization functions
                all_results[context_type][context_length] = {
                    'question_truth_lie_diffs_across_samples': truth_lie_diffs.T,  # [n_questions, n_samples]
                    'context_length': context_length,
                    'unique_questions': data['unique_questions'],
                    'n_samples': data['n_samples']
                }
    
    return all_results


def plot_context_diff_by_question_type(
    all_results: Dict[str, Dict[int, Dict[str, Any]]],
    context_types: List[str],
    control_context_type: str,
    initial_questions_df: pd.DataFrame,
    output_path: str,
    truth_answer_label: str = "Aligned",
    lie_answer_label: str = "Narrowly misaligned",
    context_aliases: Optional[Dict[str, str]] = None,
    filename_prefix: str = ""
):
    """
    Plot differences from control context by question type.
    Adapted from the original visualization function.
    """
    if context_aliases is None:
        context_aliases = {}
    
    # Filter out the control context from plotting
    plot_context_types = [ct for ct in context_types if ct != control_context_type]
    
    # Get question types and context lengths
    question_types = initial_questions_df['type'].unique()
    num_question_types = len(question_types)
    
    # Get all context lengths across all context types
    all_context_lengths = set()
    for context_type in context_types:
        if context_type in all_results:
            all_context_lengths.update(all_results[context_type].keys())
    context_lengths = sorted(list(all_context_lengths))
    
    if control_context_type not in all_results:
        print(f"Warning: Control context type '{control_context_type}' not found in results")
        return
    
    control_results = all_results[control_context_type]
    
    # Pre-calculate question type positions for each context length
    type_positions_map = {}
    for context_length in context_lengths:
        if context_length in control_results:
            unique_questions = control_results[context_length]['unique_questions']
            type_positions_map[context_length] = {}
            
            for question_type in question_types:
                type_question_indices = initial_questions_df[initial_questions_df['type'] == question_type].index.tolist()
                type_positions_map[context_length][question_type] = [
                    i for i, q_idx in enumerate(unique_questions) if q_idx in type_question_indices
                ]
    
    # Data collection
    plot_data = {}  # {context_type: {question_type: {'lengths': [], 'means': [], 'stds': [], 'individual_diffs': []}}}
    
    # Initialize data structures
    for context_type in plot_context_types:
        if context_type in all_results:
            plot_data[context_type] = {qt: {'lengths': [], 'means': [], 'stds': [], 'individual_diffs': []} 
                                     for qt in question_types}
    
    # Collect data
    for context_length in context_lengths:
        if context_length not in control_results:
            continue
        
        control_data = control_results[context_length]
        control_diffs = control_data['question_truth_lie_diffs_across_samples']  # [n_questions, n_samples]
        n_samples = control_data['n_samples']
        
        for question_type in question_types:
            if context_length not in type_positions_map or question_type not in type_positions_map[context_length]:
                continue
                
            type_positions = type_positions_map[context_length][question_type]
            if not type_positions:
                continue
            
            # Get control data for this question type
            control_type_diffs = control_diffs[type_positions, :]  # [n_type_questions, n_samples]
            control_question_means = np.nanmean(control_type_diffs, axis=1)  # [n_type_questions]
            
            # Process each other context type
            for context_type in plot_context_types:
                if context_type not in all_results or context_length not in all_results[context_type]:
                    continue
                
                other_data = all_results[context_type][context_length]
                other_diffs = other_data['question_truth_lie_diffs_across_samples']  # [n_questions, n_samples]
                
                # Get data for this question type
                other_type_diffs = other_diffs[type_positions, :]  # [n_type_questions, n_samples]
                other_question_means = np.nanmean(other_type_diffs, axis=1)  # [n_type_questions]
                
                # Calculate differences from control
                question_diffs = other_question_means - control_question_means
                
                # Store plot data
                plot_data[context_type][question_type]['lengths'].append(context_length)
                plot_data[context_type][question_type]['means'].append(np.nanmean(question_diffs))
                plot_data[context_type][question_type]['stds'].append(np.nanstd(question_diffs))
                plot_data[context_type][question_type]['individual_diffs'].append(question_diffs)
    
    # Generate plot
    fig, axes = plt.subplots(num_question_types, 1, figsize=(10, 4*num_question_types))
    if num_question_types == 1:
        axes = [axes]

    
    colors = {
        'aligned': 'green',
        'misaligned': 'crimson'
    }

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
                                color=colors[context_type], alpha=0.8)
            
            # Plot individual question lines (low alpha)
            if len(data['individual_diffs']) > 0 and len(data['individual_diffs'][0]) > 0:
                n_questions_this_type = len(data['individual_diffs'][0])
                for q_pos in range(n_questions_this_type):
                    individual_diffs = [data['individual_diffs'][length_idx][q_pos] 
                                      for length_idx in range(len(data['lengths']))]
                    axes[type_idx].plot(x_values, individual_diffs, 
                                    color=colors[context_type], alpha=0.2, linewidth=1)
        
        # Add horizontal line at y=0 for reference
        axes[type_idx].axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        axes[type_idx].set_title(f'{question_type} Questions', fontsize=15)
        axes[type_idx].grid(True, alpha=0.3)

    # Labels and legend
    control_display_name = context_aliases.get(control_context_type, control_context_type.replace("_", " ").title())
    fig.text(0.0, 0.5, f'Difference in Log P({truth_answer_label}) - Log P({lie_answer_label})\nrelative to {control_display_name}', 
             fontsize=15, ha='center', va='center', rotation='vertical')
    axes[-1].set_xlabel('Context Length (N)', fontsize=15)
    axes[-1].legend(fontsize=15)

    plt.tight_layout()
    filename = f'{filename_prefix}context_diff_by_question_type.png'
    plt.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {filename}")


def main(save_base: str, subdir_name: str):
    """Main function to collect and process in-context steering results."""
    
    steering_dir = os.path.join(save_base, subdir_name)
    
    if not os.path.exists(steering_dir):
        print(f"ERROR: Steering directory {steering_dir} does not exist.")
        print("Make sure you've run the b script first.")
        sys.exit(1)
    
    # Load batch metadata
    print("Loading batch metadata...")
    metadata = load_batch_metadata(steering_dir)
    
    batch_ids = metadata['steering_batch_ids_by_context']
    model_name = metadata['model_name']
    context_lengths = metadata['context_lengths']
    n_samples = metadata['n_samples']
    questions_data_name = metadata['questions_data_name']
    
    print(f"Found batches for context lengths: {list(batch_ids.keys())}")
    print(f"Model: {model_name}")
    
    # Initialize batch wrapper
    batch_wrapper = FireworksBatchWrapper(model_name)
    
    # Check batch statuses
    print("\nChecking batch statuses...")
    statuses = check_batch_statuses(batch_wrapper, batch_ids)
    
    # Find completed batches
    completed_batches = {ctx_len: batch_id for ctx_len, batch_id in batch_ids.items() 
                        if statuses[ctx_len] == batch_wrapper.success_code}
    
    if not completed_batches:
        print("\nNo batches completed yet. Current statuses:")
        for ctx_len, status in statuses.items():
            print(f"  Context length {ctx_len}: {status}")
        print("\nPlease check back later when batches are completed.")
        return
    
    print(f"\nFound {len(completed_batches)} completed batches")
    
    # Setup directories
    raw_outputs_dir = os.path.join(steering_dir, 'raw_outputs')
    aggregated_dir = os.path.join(steering_dir, 'aggregated_outputs')
    plots_dir = os.path.join(steering_dir, 'plots')
    
    for directory in [raw_outputs_dir, aggregated_dir, plots_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Download and process completed batches
    print("\nDownloading and processing results...")
    processed_results = {}
    
    for context_length, batch_id in completed_batches.items():
        print(f"\nProcessing context length {context_length}...")
        results = download_and_process_batch(batch_wrapper, context_length, batch_id, raw_outputs_dir)
        if results is not None:
            processed_results[context_length] = results
    
    if not processed_results:
        print("ERROR: No results were successfully processed.")
        sys.exit(1)
    
    # Aggregate and save results
    print("\nAggregating and saving results...")
    aggregate_and_save_results(processed_results, metadata, aggregated_dir)
    
    # Load results for visualization
    print("\nLoading results for visualization...")
    all_results = load_aggregated_results(aggregated_dir)
    
    # Load question data for visualization
    initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')
    
    # Generate visualization
    print("\nGenerating visualization...")
    context_types = ['aligned', 'misaligned', 'random']
    context_aliases = {
        'aligned': 'Aligned context',
        'misaligned': 'Narrowly misaligned context', 
        'random': 'Random Context'
    }
    
    plot_context_diff_by_question_type(
        all_results=all_results,
        context_types=context_types,
        control_context_type='random',
        initial_questions_df=initial_questions_df,
        output_path=plots_dir,
        truth_answer_label="Truth",
        lie_answer_label="Lie",
        context_aliases=context_aliases,
        filename_prefix=""
    )
    
    print("\n" + "="*60)
    print("IN-CONTEXT STEERING COLLECTION COMPLETE!")
    print("="*60)
    print(f"Processed context lengths: {list(processed_results.keys())}")
    print(f"Aggregated results saved to: {aggregated_dir}")
    print(f"Plots saved to: {plots_dir}")
    
    # Print summary statistics
    total_processed = sum(len(results) for results in processed_results.values())
    print(f"Total requests processed: {total_processed}")
    
    # Show which context lengths are still pending
    pending_batches = {ctx_len: statuses[ctx_len] for ctx_len in batch_ids.keys() 
                        if ctx_len not in completed_batches}
    if pending_batches:
        print(f"\nPending batches:")
        for ctx_len, status in pending_batches.items():
            print(f"  Context length {ctx_len}: {status}")


if __name__ == '__main__':
    config_path = sys.argv[1]
    args = YamlConfig(config_path)
    
    # Setup directories
    save_base = os.path.join('probe_generation_results/b_neurips_workshop_results', args.args_name)
    
    main(save_base, 'c_in_context_liar')