import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
from typing import List, Dict, Optional
import random
from scipy.stats import ttest_rel
import torch.nn.functional as F

from model.load import load_model
from util.elicit import elicit_sequence_log_probs
from util.lying_context import get_context_yn

from util.util import YamlConfig

import sys
import os
import copy
from tqdm import tqdm

config_path = sys.argv[1]
args = YamlConfig(config_path)

batch_size = args.batch_size
system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name
question_instruction = args.question_instruction

initial_answers_args_name = args.initial_answers_args_name
probe_responses_args_name = args.probe_responses_args_name
probe_analysis_args_name = args.probe_analysis_args_name

probe_file_name = args.probe_file_name
probe_response_type = args.probe_response_type

assert probe_response_type == 'yn'

prompt_index = args.prompt_idx

context_lengths = args.context_lengths
n_samples = args.samples_per_context_length

excluded_probe_types = args.excluded_probe_types
sorting_key = args.icl_sorting_key

# Prepare for saving results
output_path = os.path.join('lie_detector_results/d_in_context_lying', args.args_name)
os.makedirs(output_path, exist_ok=True)
args.save(output_path)

# Load stochastic answers dataframe
print("Loading stochastic answers...")
stochastic_answers_path = os.path.join('lie_detector_results/a_initial_answers', initial_answers_args_name, 'initial_answers_stochastic.csv')
stochastic_df = pd.read_csv(stochastic_answers_path)

# Filter to the specified prompt_idx
stochastic_df = stochastic_df[stochastic_df['prompt_idx'] == prompt_index].copy()
print(f"Filtered to {len(stochastic_df)} rows for prompt_idx {prompt_index}")

# Load probe results dataframe  
probe_responses_path = os.path.join('lie_detector_results/b_probe_answers', probe_responses_args_name, 'probe_response.csv')
probe_results_df = pd.read_csv(probe_responses_path)

# Filter to valid questions (those that appear in probe data)
valid_question_ids = set(probe_results_df['question_idx'].unique())
stochastic_df = stochastic_df[stochastic_df['question_idx'].isin(valid_question_ids)].copy()
print(f"Filtered to {len(stochastic_df)} rows with valid question IDs")

# Load questions data and create QA pairs
print("Loading test questions...")
with open(f'data/{questions_data_name}.json', 'r') as f:
    question_data = json.load(f)

qa_pairs = [(question_data['question'][f'{idx}'].strip(), question_data['answer'][f'{idx}']) 
            for idx in range(len(question_data['question']))]

# Load probe questions for context
probes_df_original = pd.read_csv(f'data/{probe_file_name}.csv')
probes_df = probes_df_original[~probes_df_original['probe_type'].isin(excluded_probe_types)]
probe_questions = probes_df['probe'].tolist()
print(f"Using {len(probe_questions)} probe questions (excluded: {excluded_probe_types})")

# Load discriminability results
discriminability_data_path = os.path.join('lie_detector_results/c_probe_discrimination', probe_analysis_args_name, 'discriminability_results.json')
with open(discriminability_data_path, 'r') as f:
    discriminability_data = json.load(f)

# Get top discriminative probes (excluding filtered probe types)
probe_results = discriminability_data['probe_results']

# Filter to only include non-excluded probe types
valid_probe_results = [(i, result) for i, result in enumerate(probe_results) 
                      if result['probe_type'] not in excluded_probe_types]

# Sort by discriminability
if sorting_key == 'random':
    valid_probe_results.sort(key=lambda x: np.random.random())
else:
    valid_probe_results.sort(key=lambda x: abs(x[1][sorting_key]), reverse=True)

print(f"Found {len(valid_probe_results)} valid discriminative probes")

# Load model
chat_wrapper = load_model(model_name, device='auto')

# Get unique questions from stochastic data
unique_questions = stochastic_df['question_idx'].unique()
print(f"Processing {len(unique_questions)} unique questions")

# Context composition types
context_types = [
    'top_lie_shuffled_together',
    'top_truth_shuffled_together',
    #'top_lie_questions_shuffled',
    #'top_truth_questions_shuffled',
    'top_questions_random_answers',
]

num_context_lengths = len(context_lengths)

# Results storage
all_results = {context_type: {
    'context_length': np.full((num_context_lengths, ), np.nan),
    'context_type': [None] * num_context_lengths,
    'mean_truth_lie_diff': np.full((num_context_lengths, ), np.nan),
    'std_truth_lie_diff': np.full((num_context_lengths, ), np.nan),
    'question_truth_lie_diffs_across_samples': np.full((num_context_lengths, len(unique_questions), n_samples), np.nan),
    } for context_type in context_types
}

# Process each context length
context_lengths_desc = sorted(context_lengths, reverse=True)

for iN, N in enumerate(context_lengths_desc):
    print(f"\n{'='*80}")
    print(f"TESTING CONTEXT LENGTH N={N}")
    print(f"{'='*80}")

    # Generate all context materials for this sample
    all_context_materials = [get_context_yn(N, valid_probe_results, probes_df_original) for _ in range(n_samples)]
    
    for context_type in context_types:
        print(f"\n\tTesting context type: {context_type}")
        
        question_truth_lie_diffs_across_samples = np.full((len(unique_questions), n_samples), np.nan)
        
        for sample_idx in range(n_samples):

            print(f"\n\t\tSample {sample_idx} for this context type. Iterating over questions")
            
            context_materials = all_context_materials[sample_idx]
            
            # Get the specific questions and answers for this context type
            if N == 0:
                shared_in_context_questions = None
                shared_in_context_answers = None
            else:
                shared_in_context_questions, shared_in_context_answers = context_materials[context_type]
            
            # Create base cache with system prompt and in-context examples
            base_cache_info = chat_wrapper.create_prompt_cache(
                system_prompt=system_prompt,
                in_context_questions=shared_in_context_questions,
                in_context_answers=shared_in_context_answers,
                prefiller=None
            )
            
            # Process each question individually
            for q_idx_pos, question_idx in tqdm(enumerate(unique_questions)):
                question = qa_pairs[question_idx][0]
                
                # Clone base cache for this question
                question_cache = copy.deepcopy(base_cache_info)
                
                # Extend cache with the question
                full_question = question + f' {question_instruction}'
                question_inputs = chat_wrapper.tokenizer(
                    full_question,
                    return_tensors="pt",
                    add_special_tokens=False
                ).to(chat_wrapper.device)
                
                # Get cache length and create proper attention mask
                cache_length = question_cache["cache"].get_seq_length()
                full_attention_mask = torch.cat([
                    torch.ones(1, cache_length, device=chat_wrapper.device),
                    question_inputs.attention_mask
                ], dim=1)
                
                # Extend the cache with the question
                with torch.no_grad():
                    question_outputs = chat_wrapper.model(
                        input_ids=question_inputs.input_ids,
                        attention_mask=full_attention_mask,
                        past_key_values=question_cache["cache"],
                        use_cache=True,
                        return_dict=True
                    )
                
                # Update the cache with the new key-value states
                question_cache["cache"] = question_outputs.past_key_values
                
                # Get all truth and lie responses for this question
                question_data = stochastic_df[stochastic_df['question_idx'] == question_idx]
                truth_responses = question_data['truth_answer'].tolist()
                lie_responses = question_data['lie_answer'].tolist()
                
                # Calculate sequence log probabilities
                truth_log_probs = elicit_sequence_log_probs(chat_wrapper, question_cache, truth_responses)
                lie_log_probs = elicit_sequence_log_probs(chat_wrapper, question_cache, lie_responses)
                
                # Calculate average truth prob - average lie prob
                avg_truth_log_prob = truth_log_probs.mean().item() if len(truth_log_probs) > 0 else 0.0
                avg_lie_log_prob = lie_log_probs.mean().item() if len(lie_log_probs) > 0 else 0.0
                truth_lie_diff = avg_truth_log_prob - avg_lie_log_prob
                
                question_truth_lie_diffs_across_samples[q_idx_pos, sample_idx] = truth_lie_diff
        
        # Store results
        all_results[context_type]['context_length'][iN] = N
        all_results[context_type]['context_type'][iN] = context_type
        all_results[context_type]['mean_truth_lie_diff'][iN] = np.mean(question_truth_lie_diffs_across_samples)
        all_results[context_type]['std_truth_lie_diff'][iN] = np.std(question_truth_lie_diffs_across_samples.mean(-1))
        all_results[context_type]['question_truth_lie_diffs_across_samples'][iN] = question_truth_lie_diffs_across_samples

        print(f"{context_type} results for {len(question_truth_lie_diffs_across_samples)} questions:")
        print(f"  Mean truth-lie log prob diff: {all_results[context_type]['mean_truth_lie_diff'][iN]:.4f} ± {all_results[context_type]['std_truth_lie_diff'][iN]:.4f}")

    # Plot results after completing all context types for this N
    print(f"\nPlotting results after N={N}...")

    fig, axes = plt.subplots(1, 1, figsize=(14, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(context_types)))

    for i, context_type in enumerate(context_types):
        results = all_results[context_type]
        
        context_lengths_plot = [r['context_length'] for r in results]
        mean_diffs = [r['mean_truth_lie_diff'] for r in results]
        std_diffs = [r['std_truth_lie_diff'] for r in results]
        
        # Add small jitter to x-values to separate overlapping points
        jitter = (i - len(context_types)/2) * 0.05
        x_values = np.array(context_lengths_plot) + jitter
        
        axes.errorbar(x_values, mean_diffs, yerr=std_diffs, 
                    label=f'{context_type.replace("_", " ").title()}',
                    marker='o', capsize=3, capthick=1, linewidth=2, markersize=6,
                    color=colors[i], alpha=0.8)

    # Add significance testing between lie and truth contexts
    if len(all_results['top_lie_shuffled_together']) > 0 and len(all_results['top_truth_shuffled_together']) > 0:
        lie_results = all_results['top_lie_shuffled_together']
        truth_results = all_results['top_truth_shuffled_together']
        
        for lie_result, truth_result in zip(lie_results, truth_results):
            if lie_result['context_length'] == truth_result['context_length']:
                N_current = lie_result['context_length']
                
                lie_diffs = lie_result['question_truth_lie_diffs']
                truth_diffs = truth_result['question_truth_lie_diffs']
                
                if len(lie_diffs) > 1 and len(truth_diffs) > 1:
                    stat, p_value = ttest_rel(lie_diffs, truth_diffs)
                    
                    if p_value < 0.05:
                        max_y = max(
                            lie_result['mean_truth_lie_diff'] + lie_result['std_truth_lie_diff'],
                            truth_result['mean_truth_lie_diff'] + truth_result['std_truth_lie_diff']
                        )
                        axes.text(N_current, max_y + 0.01, '*', 
                                ha='center', va='bottom', fontsize=16, fontweight='bold')

    axes.set_xlabel('Context Length (N)')
    axes.set_ylabel('Mean Log P(Truth) - Log P(Lie)')
    axes.set_title('Truth vs Lie Log Probability Differences by Context Composition\n(Yes/No Context Format, * indicates p<0.05 for lie vs truth contexts)')
    axes.legend()
    axes.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'context_effect_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save detailed results so far
    for context_type in context_types:
        context_type_npy_path = os.path.join(output_path, f'context_effect_results_{context_type}.npy')
        np.save(context_type_npy_path, all_results[context_type])

    print(f"Results updated and saved after N={N}: e.g. {context_type_npy_path}")
