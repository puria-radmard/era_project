import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
from typing import List
import random
from scipy.stats import ttest_rel

from model.load import load_model
from util.elicit import elicit_next_token_probs
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
args.save_args(output_path)

# 1. Load original answers dataframe
print("Loading original answers...")
initial_answers_path = os.path.join('lie_detector_results/a_initial_answers', initial_answers_args_name, 'initial_answers.csv')
initial_answers_df = pd.read_csv(initial_answers_path)

# 2. Load probe results dataframe  
probe_responses_path = os.path.join('lie_detector_results/b_probe_answers', probe_responses_args_name, 'probe_response.csv')
probe_results_df = pd.read_csv(probe_responses_path)

# 3. Filter initial_answers_df to prompt_idx and question_idx in probe data
valid_question_ids = set(probe_results_df['question_idx'].unique())
filtered_answers_df = initial_answers_df[
    (initial_answers_df['prompt_idx'] == prompt_index) & 
    (initial_answers_df['question_idx'].isin(valid_question_ids))
].copy()

print(f"Filtered to {len(filtered_answers_df)} questions")

# 4. Load questions data and create QA pairs
print("Loading test questions...")
with open(f'data/{questions_data_name}.json', 'r') as f:
    question_data = json.load(f)

qa_pairs = [(question_data['question'][f'{idx}'].strip(), question_data['answer'][f'{idx}']) 
            for idx in range(len(question_data['question']))]

# Filter QA pairs to only include questions we have in filtered_answers_df
valid_qa_pairs = [qa_pairs[idx] for idx in filtered_answers_df['question_idx']]
print(f"Using {len(valid_qa_pairs)} test questions")

# Load probe questions for context and keep unfiltered version for indexing
probes_df_original = pd.read_csv(f'data/{probe_file_name}.csv')
probes_df = probes_df_original[~probes_df_original['probe_type'].isin(excluded_probe_types)]
probe_questions = probes_df['probe'].tolist()
print(f"Using {len(probe_questions)} probe questions (excluded: {excluded_probe_types})")

# Load discriminability results
discriminability_data_path = os.path.join('lie_detector_results/c_probe_discimination', probe_analysis_args_name, 'discriminability_results.json')
with open(discriminability_data_path, 'r') as f:
    discriminability_data = json.load(f)

# Get top discriminative probes (excluding filtered probe types)
probe_results = discriminability_data['probe_results']
# Filter to only include non-excluded probe types
valid_probe_results = [(i, result) for i, result in enumerate(probe_results) 
                      if result['probe_type'] not in excluded_probe_types]

# Sort by absolute effect size (most discriminative first)
if sorting_key == 'random':
    valid_probe_results.sort(key=lambda x: np.random.random())
else:
    valid_probe_results.sort(key=lambda x: abs(x[1][sorting_key]), reverse=True)

print(f"Found {len(valid_probe_results)} valid discriminative probes")

chat_wrapper = load_model(model_name, device='auto')

# FILTERING STAGE 1: Filter by single-token truth answers
print("Filtering for single-token truth answers...")
truth_answers = filtered_answers_df['truth_answer'].tolist()
lie_answers = filtered_answers_df['lie_answer'].tolist()

single_token_truth_indices = []
for i, truth_ans in enumerate(truth_answers):
    # Remove trailing periods if present
    truth_clean = truth_ans.rstrip('.') if truth_ans.endswith('.') else truth_ans
    truth_tokens = chat_wrapper.tokenizer.encode(truth_clean, add_special_tokens=False)
    if len(truth_tokens) == 1:
        single_token_truth_indices.append(i)

print(f"Questions with single-token truth answers: {len(single_token_truth_indices)}")

# FILTERING STAGE 2: From those, filter by single-token lie answers
print("Further filtering for single-token lie answers...")
valid_indices = []
truth_token_ids = []
lie_token_ids = []

for i in single_token_truth_indices:
    truth_ans = truth_answers[i]
    lie_ans = lie_answers[i]
    
    # Clean and tokenize truth answer
    truth_clean = truth_ans.rstrip('.') if truth_ans.endswith('.') else truth_ans
    truth_tokens = chat_wrapper.tokenizer.encode(truth_clean, add_special_tokens=False)
    
    # Clean and tokenize lie answer
    lie_clean = lie_ans.rstrip('.') if lie_ans.endswith('.') else lie_ans

    lie_tokens = chat_wrapper.tokenizer.encode(lie_clean, add_special_tokens=False)
    
    # Only keep if both are single tokens
    if len(truth_tokens) == 1 and len(lie_tokens) == 1:
        valid_indices.append(i)
        truth_token_ids.append(truth_tokens[0])
        lie_token_ids.append(lie_tokens[0])

no_token_id = chat_wrapper.tokenizer.encode("No", add_special_tokens=False)[0]
yes_token_id = chat_wrapper.tokenizer.encode("Yes", add_special_tokens=False)[0]

print(f"Questions with both single-token truth and lie answers: {len(valid_indices)}")

# FILTERING STAGE 3: Limit to first 96 for speed
# print("LIMITING TO FIRST 96 VALID QUESTIONS")
# valid_indices = valid_indices[:96]
# truth_token_ids = truth_token_ids[:96]
# lie_token_ids = lie_token_ids[:96]

# Apply final filtering to QA pairs
valid_qa_pairs = [valid_qa_pairs[i] for i in valid_indices]

print(f"Final dataset size: {len(valid_qa_pairs)} questions")
print(f"Truth token IDs: {truth_token_ids}")
print(f"Lie token IDs: {lie_token_ids}")

# Context composition types
context_types = [
    'top_lie_shuffled_together',      # 1. Top questions + lie answers, shuffled together
    'top_truth_shuffled_together',    # 2. Top questions + truth answers, shuffled together  
    'top_lie_questions_shuffled',     # 3. Top questions shuffled + lie answers fixed order
    'top_truth_questions_shuffled',   # 4. Top questions shuffled + truth answers fixed order
    # 'random_lie_answers',             # 5. Random questions + lie answers fixed order
    # 'random_truth_answers'            # 6. Random questions + truth answers fixed order
    'top_questions_random_answers',     # 5alt. Top questions + 50/50 random answers
]

# Results storage
all_results = {context_type: [] for context_type in context_types}

# Go down in context size (largest to smallest)
context_lengths_desc = sorted(context_lengths, reverse=True)

for N in context_lengths_desc:
    print(f"\n{'='*80}")
    print(f"TESTING CONTEXT LENGTH N={N}")
    print(f"{'='*80}")

    # Generate all context materials for this sample
    all_context_materials = [get_context_yn(N, valid_probe_results, probes_df_original) for _ in range(n_samples)]
    
    for context_type in context_types:
        print(f"\nTesting context type: {context_type}")
        
        question_truth_probs_across_samples = np.full((len(valid_qa_pairs), n_samples), np.nan)
        question_lie_probs_across_samples = np.full((len(valid_qa_pairs), n_samples), np.nan)
        question_yes_probs_across_samples = np.full((len(valid_qa_pairs), n_samples), np.nan)
        question_no_probs_across_samples = np.full((len(valid_qa_pairs), n_samples), np.nan)
        question_truth_probs_across_samples_excl_yn = np.full((len(valid_qa_pairs), n_samples), np.nan)
        question_lie_probs_across_samples_excl_yn = np.full((len(valid_qa_pairs), n_samples), np.nan)
        
        for sample_idx in tqdm(range(n_samples)):

            context_materials = all_context_materials[sample_idx]
            
            # Get the specific questions and answers for this context type
            if N == 0:
                shared_in_context_questions = None
                shared_in_context_answers = None
            else:
                shared_in_context_questions, shared_in_context_answers = context_materials[context_type]
            
            # Extract just the questions for elicit_next_token_probs
            test_questions = [qa[0] for qa in valid_qa_pairs]
            
            # Process in batches
            all_probs = []

            # Prompt cache
            context_cache_info = chat_wrapper.create_prompt_cache(
                system_prompt=system_prompt,
                in_context_questions=shared_in_context_questions,
                in_context_answers=shared_in_context_answers,
                prefiller=None
            )
            
            for i in range(0, len(test_questions), batch_size):
                
                batch_questions = test_questions[i:i+batch_size]

                import pdb; pdb.set_trace()

                batch_questions = [bq + f' {question_instruction}' for bq in batch_questions]
                
                # Call elicit_next_token_probs
                result = elicit_next_token_probs(
                    chat_wrapper=chat_wrapper,
                    questions=batch_questions,
                    cache_data=copy.deepcopy(context_cache_info),

                    # system_prompt=system_prompt,
                    # shared_in_context_questions=shared_in_context_questions,
                    # shared_in_context_answers=shared_in_context_answers,

                    prefiller=""
                )
                
                batch_probs = result["probs"]  # [batch_size, vocab_size]

                all_probs.append(batch_probs)
            
            # Concatenate all batch results
            all_probs = torch.cat(all_probs, dim=0)  # [num_questions, vocab_size]

            
            # Extract probabilities for truth and lie answer tokens
            for q_idx in range(len(valid_qa_pairs)):
                truth_token_id = truth_token_ids[q_idx]
                lie_token_id = lie_token_ids[q_idx]
                
                truth_prob = all_probs[q_idx, truth_token_id].item()
                lie_prob = all_probs[q_idx, lie_token_id].item()

                yes_prob = all_probs[q_idx, yes_token_id].item()
                no_prob = all_probs[q_idx, no_token_id].item()
                
                question_truth_probs_across_samples[q_idx,sample_idx] = truth_prob
                question_lie_probs_across_samples[q_idx,sample_idx] = lie_prob

                question_yes_probs_across_samples[q_idx,sample_idx] = yes_prob
                question_no_probs_across_samples[q_idx,sample_idx] = no_prob

                question_truth_probs_across_samples_excl_yn[q_idx,sample_idx] = truth_prob / (1.0 - yes_prob - no_prob + 1e-10)
                question_lie_probs_across_samples_excl_yn[q_idx,sample_idx] = lie_prob / (1.0 - yes_prob - no_prob + 1e-10)

        
        # Average probabilities across samples for each question
        question_avg_truth_probs = question_truth_probs_across_samples.mean(-1)
        question_avg_lie_probs = question_lie_probs_across_samples.mean(-1)
        question_avg_yes_probs = question_yes_probs_across_samples.mean(-1)
        question_avg_no_probs = question_no_probs_across_samples.mean(-1)
        question_avg_truth_probs_excl_yn = question_truth_probs_across_samples_excl_yn.mean(-1)
        question_avg_lie_probs_excl_yn = question_lie_probs_across_samples_excl_yn.mean(-1)
        
        # Store results
        all_results[context_type].append({
            'context_length': N,
            'context_type': context_type,
            'mean_truth_prob': float(np.mean(question_avg_truth_probs)),
            'std_truth_prob': float(np.std(question_avg_truth_probs)),
            'mean_lie_prob': float(np.mean(question_avg_lie_probs)),
            'std_lie_prob': float(np.std(question_avg_lie_probs)),
            'mean_yes_prob': float(np.mean(question_avg_yes_probs)),
            'std_yes_prob': float(np.std(question_avg_yes_probs)),
            'mean_no_prob': float(np.mean(question_avg_no_probs)),
            'std_no_prob': float(np.std(question_avg_no_probs)),
            'mean_truth_prob_excl_yn': float(np.mean(question_avg_truth_probs_excl_yn)),
            'std_truth_prob_excl_yn': float(np.std(question_avg_truth_probs_excl_yn)),
            'mean_lie_prob_excl_yn': float(np.mean(question_avg_lie_probs_excl_yn)),
            'std_lie_prob_excl_yn': float(np.std(question_avg_lie_probs_excl_yn)),
            'question_truth_probs': [float(q) for q in question_avg_truth_probs],
            'question_lie_probs': [float(q) for q in question_avg_lie_probs],
            'question_truth_probs_excl_yn': [float(q) for q in question_avg_truth_probs_excl_yn],
            'question_lie_probs_excl_yn': [float(q) for q in question_avg_lie_probs_excl_yn],
        })
        
        print(f"{context_type} results for {len(question_avg_truth_probs)} questions:")
        print(f"  Mean truth prob: {np.mean(question_avg_truth_probs):.4f} ± {np.std(question_avg_truth_probs):.4f}")
        print(f"  Mean lie prob: {np.mean(question_avg_lie_probs):.4f} ± {np.std(question_avg_lie_probs):.4f}")
        print(f"  Mean yes prob: {np.mean(question_avg_yes_probs):.4f} ± {np.std(question_avg_yes_probs):.4f}")
        print(f"  Mean no prob: {np.mean(question_avg_no_probs):.4f} ± {np.std(question_avg_no_probs):.4f}")

        print(f"  Mean truth prob excluding y/n: {np.mean(question_avg_truth_probs_excl_yn):.4f} ± {np.std(question_avg_truth_probs_excl_yn):.4f}")
        print(f"  Mean lie prob excluding y/n: {np.mean(question_avg_lie_probs_excl_yn):.4f} ± {np.std(question_avg_lie_probs_excl_yn):.4f}")

    # Plot results after completing all context types for this N
    print(f"\nPlotting results after N={N}...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 16))

    # Colors for different context types
    colors = plt.cm.tab10(np.linspace(0, 1, len(context_types)))

    for i, context_type in enumerate(context_types):
        results = all_results[context_type]
            
        context_lengths_plot = [r['context_length'] for r in results]
        
        mean_truth_probs = [r['mean_truth_prob'] for r in results]
        std_truth_probs = [r['std_truth_prob'] for r in results]
        
        mean_lie_probs = [r['mean_lie_prob'] for r in results]
        std_lie_probs = [r['std_lie_prob'] for r in results]

        mean_yes_probs = [r['mean_yes_prob'] for r in results]
        std_yes_probs = [r['std_yes_prob'] for r in results]

        mean_no_probs = [r['mean_no_prob'] for r in results]
        std_no_probs = [r['std_no_prob'] for r in results]

        mean_truth_probs_excl_yn = [r['mean_truth_prob_excl_yn'] for r in results]
        std_truth_probs_excl_yn = [r['std_truth_prob_excl_yn'] for r in results]
        
        mean_lie_probs_excl_yn = [r['mean_lie_prob_excl_yn'] for r in results]
        std_lie_probs_excl_yn = [r['std_lie_prob_excl_yn'] for r in results]
        
        
        # Add small jitter to x-values to separate overlapping points
        jitter = (i - len(context_types)/2) * 0.05
        x_values = np.array(context_lengths_plot) + jitter
        
        # Plot truth probabilities (solid lines)
        axes[0].errorbar(x_values, mean_truth_probs, yerr=std_truth_probs, 
                    label=f'{context_type.replace("_", " ").title()} (truth)',
                    marker='x', capsize=3, capthick=1, linewidth=2, markersize=6,
                    color=colors[i], alpha=0.8, linestyle='-')
        
        # Plot lie probabilities (dashed lines)
        axes[0].errorbar(x_values, mean_lie_probs, yerr=std_lie_probs, 
                    label=f'{context_type.replace("_", " ").title()} (previous lie)',
                    marker='o', capsize=3, capthick=1, linewidth=2, markersize=6,
                    color=colors[i], alpha=0.8, linestyle='--')

        # Plot Yes probabilities (solid lines)
        axes[1].errorbar(x_values, mean_yes_probs, yerr=std_yes_probs, 
                    label=f'{context_type.replace("_", " ").title()} ("Yes")',
                    marker='x', capsize=3, capthick=1, linewidth=2, markersize=6,
                    color=colors[i], alpha=0.8, linestyle='-')
        
        # Plot No probabilities (dashed lines)
        axes[1].errorbar(x_values, mean_no_probs, yerr=std_no_probs, 
                    label=f'{context_type.replace("_", " ").title()} ("No")',
                    marker='o', capsize=3, capthick=1, linewidth=2, markersize=6,
                    color=colors[i], alpha=0.8, linestyle='--')
        
        # Plot truth probabilities (solid lines) excluding yes and no probs
        axes[2].errorbar(x_values, mean_truth_probs_excl_yn, yerr=std_truth_probs_excl_yn, 
                    label=f'{context_type.replace("_", " ").title()} (truth) - excluding yes/no prob',
                    marker='x', capsize=3, capthick=1, linewidth=2, markersize=6,
                    color=colors[i], alpha=0.8, linestyle='-')
        
        # Plot lie probabilities (dashed lines) excluding yes and no probs
        axes[2].errorbar(x_values, mean_lie_probs_excl_yn, yerr=std_lie_probs_excl_yn, 
                    label=f'{context_type.replace("_", " ").title()} (previous lie) - excluding yes/no prob',
                    marker='o', capsize=3, capthick=1, linewidth=2, markersize=6,
                    color=colors[i], alpha=0.8, linestyle='--')

    # Add significance stars for top_lie_shuffled_together vs top_truth_shuffled_together
    lie_results = all_results['top_lie_shuffled_together']
    truth_results = all_results['top_truth_shuffled_together']
    
    for lie_result, truth_result in zip(lie_results, truth_results):
        if lie_result['context_length'] == truth_result['context_length']:
            N_current = lie_result['context_length']
            
            # Get question-level truth probabilities for both context types
            lie_question_truth_probs = lie_result['question_truth_probs']
            truth_question_truth_probs = truth_result['question_truth_probs']
            
            # Paired t-test comparing truth probability under lie vs truth contexts
            if len(lie_question_truth_probs) > 1 and len(truth_question_truth_probs) > 1:
                stat, p_value = ttest_rel(lie_question_truth_probs, truth_question_truth_probs)
                
                if p_value < 0.05 and stat < 0.0:
                    # Add star above the highest point at this N
                    max_y = max(
                        lie_result['mean_truth_prob'] + lie_result['std_truth_prob'],
                        truth_result['mean_truth_prob'] + truth_result['std_truth_prob']
                    )
                    axes[0].text(N_current - 0.15, max_y + 0.02, '*', 
                            ha='center', va='bottom', fontsize=16, fontweight='bold')


    # Add significance stars for top_lie_shuffled_together vs top_truth_shuffled_together for case with Yes and No probability mass removed
    lie_results = all_results['top_lie_shuffled_together']
    truth_results = all_results['top_truth_shuffled_together']
    
    for lie_result, truth_result in zip(lie_results, truth_results):
        if lie_result['context_length'] == truth_result['context_length']:
            N_current = lie_result['context_length']
            
            # Get question-level truth probabilities for both context types
            lie_question_truth_probs = lie_result['question_truth_probs_excl_yn']
            truth_question_truth_probs = truth_result['question_truth_probs_excl_yn']
            
            # Paired t-test comparing truth probability under lie vs truth contexts
            if len(lie_question_truth_probs) > 1 and len(truth_question_truth_probs) > 1:
                stat, p_value = ttest_rel(lie_question_truth_probs, truth_question_truth_probs)
                
                if p_value < 0.05:
                    # Add star above the highest point at this N
                    max_y = max(
                        lie_result['mean_truth_prob'] + lie_result['std_truth_prob'],
                        truth_result['mean_truth_prob'] + truth_result['std_truth_prob']
                    )
                    axes[2].text(N_current - 0.15, max_y + 0.02, '*', 
                            ha='center', va='bottom', fontsize=16, fontweight='bold')
                    

    axes[0].set_xlabel('Context Length (N)')
    axes[0].set_ylabel('Average Token Probability')
    axes[0].set_title('Truth vs (previous) Lie Token Probabilities by Context Composition\n(* indicates p<0.05 for lie vs truth contexts)')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    axes[1].set_xlabel('Context Length (N)')
    axes[1].set_ylabel('Average Token Probability')
    axes[1].set_title('Yes vs No Token Probabilities by Context Composition')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    axes[2].set_xlabel('Context Length (N)')
    axes[2].set_ylabel('Average Token Probability')
    axes[2].set_title('Truth vs (previous) Lie Token Probabilities by Context Composition\nExcluding Yes and No probability mass\n(* indicates p<0.05 for lie vs truth contexts)')
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'context_effect_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close to save memory
    
    # Save detailed results so far
    with open(os.path.join(output_path, 'context_effect_results.json'), 'w') as f:
        json.dump(all_results, f)

    print(f"Results updated and saved after N={N}")


print("Analysis complete!")
print("Results saved to:")
print(f"  - {output_path}/context_effect_results.csv")
print(f"  - Individual CSV files for each context type")

