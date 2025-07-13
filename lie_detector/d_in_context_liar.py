#!/usr/bin/env python3

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

from util.util import YamlConfig

import sys
import os
from tqdm import tqdm

config_path = sys.argv[1]
args = YamlConfig(config_path)

batch_size = args.batch_size
system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name
save_path = args.save_path

prompt_index = args.prompt_idx

context_lengths = args.context_lengths
n_samples = args.samples_per_context_length

# Probe types to exclude
excluded_probe_types = args.excluded_probe_types

chat_wrapper = load_model(model_name, device='auto')

# 1. Load original answers dataframe
print("Loading original answers...")
original_answers_df = pd.read_csv(f'{save_path}/initial_answers/{questions_data_name}.csv')

# 2. Load probe results dataframe  
probe_results_df = pd.read_csv(f'{save_path}/probe_answers/{questions_data_name}_probe_prompt{prompt_index}.csv')

# 3. Filter original_answers_df to prompt_idx and question_idx in probe data
valid_question_ids = set(probe_results_df['question_idx'].unique())
filtered_answers_df = original_answers_df[
    (original_answers_df['prompt_idx'] == prompt_index) & 
    (original_answers_df['question_idx'].isin(valid_question_ids))
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
probes_df_original = pd.read_csv('data/probes_with_yn.csv')
probes_df = probes_df_original[~probes_df_original['probe_type'].isin(excluded_probe_types)]
probe_questions = probes_df['probe'].tolist()
print(f"Using {len(probe_questions)} probe questions (excluded: {excluded_probe_types})")

# Load discriminability results
with open(f'{save_path}/probe_analysis/prompt{prompt_index}/discriminability_results.json', 'r') as f:
    discriminability_data = json.load(f)

# Get top discriminative probes (excluding filtered probe types)
probe_results = discriminability_data['probe_results']
# Filter to only include non-excluded probe types
valid_probe_results = [(i, result) for i, result in enumerate(probe_results) 
                      if result['probe_type'] not in excluded_probe_types]

# Sort by absolute effect size (most discriminative first)
valid_probe_results.sort(key=lambda x: abs(x[1]['effect_size']), reverse=True)

print(f"Found {len(valid_probe_results)} valid discriminative probes")

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

print(f"Questions with both single-token truth and lie answers: {len(valid_indices)}")

# # FILTERING STAGE 3: Limit to first 64 for speed
# print("LIMITING TO FIRST 64 VALID QUESTIONS")
# valid_indices = valid_indices[:64]
# truth_token_ids = truth_token_ids[:64]
# lie_token_ids = lie_token_ids[:64]

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
    'random_lie_answers',             # 5. Random questions + lie answers fixed order
    'random_truth_answers'            # 6. Random questions + truth answers fixed order
]

def get_context(N: int):
    """Generate all base context materials for one sample."""
    
    if N == 0:
        return {}
    
    # Get top N discriminative probes
    top_n_probe_data = valid_probe_results[:N]
    top_n_questions = [probes_df_original.iloc[original_idx]['probe'] for original_idx, _ in top_n_probe_data]
    
    # Generate truth-inducing answers (opposite of lie-inducing based on effect size)
    truth_inducing_answers = []
    for original_idx, result in top_n_probe_data:
        effect_size = result['effect_size']
        if effect_size > 0:
            # Positive effect: truth=1 higher than truth=0, so "Yes" induces truth, "No" induces lies
            truth_answer = "Yes."
        else:
            # Negative effect: truth=0 higher than truth=1, so "No" induces truth, "Yes" induces lies  
            truth_answer = "No."
        truth_inducing_answers.append(truth_answer)
    
    # Generate lie-inducing answers (opposite of truth-inducing)
    lie_inducing_answers = ["No." if ans == "Yes." else "Yes." for ans in truth_inducing_answers]
    
    # Shuffle top questions and answers together for types 1 & 2
    top_questions_lie_shuffled = top_n_questions.copy()
    lie_answers_shuffled = lie_inducing_answers.copy()
    combined_lie = list(zip(top_questions_lie_shuffled, lie_answers_shuffled))
    random.shuffle(combined_lie)
    top_questions_lie_shuffled, lie_answers_shuffled = zip(*combined_lie)
    
    top_questions_truth_shuffled = top_n_questions.copy()
    truth_answers_shuffled = truth_inducing_answers.copy()
    combined_truth = list(zip(top_questions_truth_shuffled, truth_answers_shuffled))
    random.shuffle(combined_truth)
    top_questions_truth_shuffled, truth_answers_shuffled = zip(*combined_truth)
    
    # Shuffle top questions independently for types 3 & 4
    top_questions_shuffled = top_n_questions.copy()
    random.shuffle(top_questions_shuffled)
    
    # Random questions for types 5 & 6
    random_questions = random.sample(probe_questions, min(N, len(probe_questions)))
    
    return {
        # Type 1: Top questions + lie answers, shuffled together
        'top_lie_shuffled_together': (list(top_questions_lie_shuffled), list(lie_answers_shuffled)),
        
        # Type 2: Top questions + truth answers, shuffled together  
        'top_truth_shuffled_together': (list(top_questions_truth_shuffled), list(truth_answers_shuffled)),
        
        # Type 3: Top questions shuffled independently + lie answers in same order as type 1
        'top_lie_questions_shuffled': (top_questions_shuffled, list(lie_answers_shuffled)),
        
        # Type 4: Top questions shuffled independently + truth answers in same order as type 2
        'top_truth_questions_shuffled': (top_questions_shuffled, list(truth_answers_shuffled)),
        
        # Type 5: Random questions + lie answers in same order as type 1
        'random_lie_answers': (random_questions, list(lie_answers_shuffled)),
        
        # Type 6: Random questions + truth answers in same order as type 2
        'random_truth_answers': (random_questions, list(truth_answers_shuffled))
    }

# Results storage
all_results = {context_type: [] for context_type in context_types}

# Go down in context size (largest to smallest)
context_lengths_desc = sorted(context_lengths, reverse=True)

for N in context_lengths_desc:
    print(f"\n{'='*80}")
    print(f"TESTING CONTEXT LENGTH N={N}")
    print(f"{'='*80}")

    # Generate all context materials for this sample
    context_materials = get_context(N)
    
    for context_type in context_types:
        print(f"\nTesting context type: {context_type}")
        
        question_truth_probs_across_samples = [[] for _ in range(len(valid_qa_pairs))]
        question_lie_probs_across_samples = [[] for _ in range(len(valid_qa_pairs))]
        
        for sample_idx in range(n_samples):
            print(f"  Sample {sample_idx + 1}/{n_samples}")
            
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
            
            for i in tqdm(range(0, len(test_questions), batch_size)):
                batch_questions = test_questions[i:i+batch_size]
                
                # Call elicit_next_token_probs
                result = elicit_next_token_probs(
                    chat_wrapper=chat_wrapper,
                    questions=batch_questions,
                    cache_data=context_cache_info,
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
                
                question_truth_probs_across_samples[q_idx].append(truth_prob)
                question_lie_probs_across_samples[q_idx].append(lie_prob)
        
        # Average probabilities across samples for each question
        question_avg_truth_probs = [np.mean(probs) for probs in question_truth_probs_across_samples]
        question_avg_lie_probs = [np.mean(probs) for probs in question_lie_probs_across_samples]
        
        # Store results
        all_results[context_type].append({
            'context_length': N,
            'context_type': context_type,
            'mean_truth_prob': np.mean(question_avg_truth_probs),
            'std_truth_prob': np.std(question_avg_truth_probs),
            'mean_lie_prob': np.mean(question_avg_lie_probs),
            'std_lie_prob': np.std(question_avg_lie_probs),
            'question_truth_probs': question_avg_truth_probs,
            'question_lie_probs': question_avg_lie_probs
        })
        
        print(f"  Mean truth prob: {np.mean(question_avg_truth_probs):.4f} ± {np.std(question_avg_truth_probs):.4f}")
        print(f"  Mean lie prob: {np.mean(question_avg_lie_probs):.4f} ± {np.std(question_avg_lie_probs):.4f}")

    # Plot results after completing all context types for this N
    print(f"\nPlotting results after N={N}...")
    
    # Prepare for saving results
    os.makedirs(f"{save_path}/in_context_learning/prompt{prompt_index}", exist_ok=True)

    plt.figure(figsize=(14, 8))

    # Colors for different context types
    colors = plt.cm.tab10(np.linspace(0, 1, len(context_types)))
    markers = ['o', 's', '^', 'v', 'D', 'p']

    for i, context_type in enumerate(context_types):
        results = all_results[context_type]
        if not results:  # Skip if no results yet
            continue
            
        context_lengths_plot = [r['context_length'] for r in results]
        mean_truth_probs = [r['mean_truth_prob'] for r in results]
        std_truth_probs = [r['std_truth_prob'] for r in results]
        mean_lie_probs = [r['mean_lie_prob'] for r in results]
        std_lie_probs = [r['std_lie_prob'] for r in results]
        
        # Add small jitter to x-values to separate overlapping points
        jitter = (i - len(context_types)/2) * 0.05
        x_values = np.array(context_lengths_plot) + jitter
        
        # Plot truth probabilities (solid lines)
        plt.errorbar(x_values, mean_truth_probs, yerr=std_truth_probs, 
                    label=f'{context_type.replace("_", " ").title()} (truth)',
                    marker=markers[i], capsize=3, capthick=1, linewidth=2, markersize=6,
                    color=colors[i], alpha=0.8, linestyle='-')
        
        # Plot lie probabilities (dashed lines)
        plt.errorbar(x_values, mean_lie_probs, yerr=std_lie_probs, 
                    label=f'{context_type.replace("_", " ").title()} (previous lie)',
                    marker=markers[i], capsize=3, capthick=1, linewidth=2, markersize=6,
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
                
                if p_value < 0.05:
                    # Add star above the highest point at this N
                    max_y = max(
                        lie_result['mean_truth_prob'] + lie_result['std_truth_prob'],
                        truth_result['mean_truth_prob'] + truth_result['std_truth_prob']
                    )
                    plt.text(N_current - 0.15, max_y + 0.02, '*', 
                            ha='center', va='bottom', fontsize=16, fontweight='bold')

    plt.xlabel('Context Length (N)')
    plt.ylabel('Average Token Probability')
    plt.title('Truth vs Lie Token Probabilities by Context Composition\n(* indicates p<0.05 for lie vs truth contexts)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/in_context_learning/prompt{prompt_index}/context_effect_analysis_comprehensive.png', 
               dpi=300, bbox_inches='tight')
    plt.close()  # Close to save memory
    
    # Save detailed results so far
    all_results_flat = []
    for context_type in context_types:
        for result in all_results[context_type]:
            all_results_flat.append(result)

    if all_results_flat:  # Only save if we have results
        results_df = pd.DataFrame(all_results_flat)
        results_df.to_csv(f'{save_path}/in_context_learning/prompt{prompt_index}/context_effect_results_comprehensive.csv', 
                         index=False)

    print(f"Results updated and saved after N={N}")

# Save individual results for each context type at the end
for context_type in context_types:
    context_results = all_results[context_type]
    if context_results:
        context_df = pd.DataFrame(context_results)
        context_df.to_csv(f'{save_path}/in_context_learning/prompt{prompt_index}/context_effect_results_{context_type}.csv', 
                         index=False)

print("Analysis complete!")
print("Results saved to:")
print(f"  - {save_path}/in_context_learning/prompt{prompt_index}/context_effect_analysis_comprehensive.png")
print(f"  - {save_path}/in_context_learning/prompt{prompt_index}/context_effect_results_comprehensive.csv")
print(f"  - Individual CSV files for each context type")