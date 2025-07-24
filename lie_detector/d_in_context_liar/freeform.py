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
from util.lying_context import get_context_freeform

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
probe_response_type: str = args.probe_response_type

assert probe_response_type.endswith('_words')

prompt_index = args.prompt_idx

context_lengths = args.context_lengths
n_samples = args.samples_per_context_length

excluded_probe_types = args.excluded_probe_types
sorting_key = args.icl_sorting_key

# Prepare for saving results
output_path = os.path.join('lie_detector_results/d_in_context_lying', args.args_name)
os.makedirs(output_path, exist_ok=True)
args.save(output_path)

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
import pdb; pdb.set_trace(header = 'check qa_pairs works here!')
initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')
qa_pairs = [(initial_questions_df['question'][idx].strip(), str(initial_questions_df['answer'][idx])) for idx in range(len(initial_questions_df['question']))]


# Filter QA pairs to only include questions we have in filtered_answers_df
valid_qa_pairs = [qa_pairs[idx] for idx in filtered_answers_df['question_idx']]
print(f"Using {len(valid_qa_pairs)} test questions")

# Load probe questions for context and keep unfiltered version for indexing
probes_df_original = pd.read_csv(f'data/probe_questions/{probe_file_name}.csv')
probes_df = probes_df_original[~probes_df_original['probe_type'].isin(excluded_probe_types)]
probe_questions = probes_df['probe'].tolist()
print(f"Using {len(probe_questions)} probe questions (excluded: {excluded_probe_types})")

# Load discriminability results
discriminability_data_path = os.path.join('lie_detector_results/c_probe_discrimination', probe_analysis_args_name, 'discriminability_results.json')
with open(discriminability_data_path, 'r') as f:
    discriminability_data = json.load(f)

# Load also the BERT projections for each answer
bert_lie_proj_path = os.path.join('lie_detector_results/c_probe_discrimination', probe_analysis_args_name, 'probe_response_bert_proj.npy')
bert_lie_proj = np.load(bert_lie_proj_path)
assert len(bert_lie_proj) == len(probe_results_df)

most_truth_answers = {}
most_lie_answers = {}

excluded_probe_indices = []     # XXX: excluding some probes on the basis of bad BERT discriminability

for _probe_idx in probe_results_df.probe_question_idx.unique():
    _rows = probe_results_df[(probe_results_df.probe_question_idx == _probe_idx)]
    _indices = _rows.index
    _relevant_projs = bert_lie_proj[_indices]

    _most_truth_row_index = _indices[_relevant_projs.argmin()]
    _most_truth_row = probe_results_df.iloc[_most_truth_row_index]

    _most_lie_row_index = _indices[_relevant_projs.argmax()]
    _most_lie_row = probe_results_df.iloc[_most_lie_row_index]
    
    if (_most_lie_row.truth != 0) or (_most_truth_row.truth != 1):
        excluded_probe_indices.append(_probe_idx)
    
    most_truth_answers[int(_probe_idx)] = _most_truth_row.resp
    most_lie_answers[int(_probe_idx)] = _most_lie_row.resp
        

# Get top discriminative probes (excluding filtered probe types)
probe_results = discriminability_data['probe_results']


# Filter to only include non-excluded probe types
valid_probe_results = [(i, result) for i, result in enumerate(probe_results) 
                      if result['probe_type'] not in excluded_probe_types
                      and i not in excluded_probe_indices]

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

print(f"Questions with both single-token truth and lie answers: {len(valid_indices)}")


# Apply final filtering to QA pairs
valid_qa_pairs = [valid_qa_pairs[i] for i in valid_indices]

print(f"Final dataset size: {len(valid_qa_pairs)} questions")
print(f"Truth token IDs: {truth_token_ids}")
print(f"Lie token IDs: {lie_token_ids}")

# Context composition types
context_types = [
    'top_lie_shuffled_together',      # 1. Top questions + lie answers, shuffled together
    'top_truth_shuffled_together',    # 2. Top questions + truth answers, shuffled together  
    # 'top_lie_questions_shuffled',     # 3. Top questions shuffled + lie answers fixed order
    # 'top_truth_questions_shuffled',   # 4. Top questions shuffled + truth answers fixed order
    # 'random_lie_answers',             # 5. Random questions + lie answers fixed order
    # 'random_truth_answers'            # 6. Random questions + truth answers fixed order
    'top_questions_random_answers',     # 5alt. Top questions + 50/50 random answers
]

num_context_lengths = len(context_lengths)

# Results storage
all_results = {context_type: {
    'context_length': np.full((num_context_lengths, ), np.nan),
    'context_type': [None] * num_context_lengths,
    'mean_truth_prob': np.full((num_context_lengths, ), np.nan),
    'std_truth_prob': np.full((num_context_lengths, ), np.nan),
    'mean_lie_prob': np.full((num_context_lengths, ), np.nan),
    'std_lie_prob': np.full((num_context_lengths, ), np.nan),
    'question_truth_probs_across_samples': np.full((num_context_lengths, len(valid_qa_pairs), n_samples), np.nan),
    'question_lie_probs_across_samples': np.full((num_context_lengths, len(valid_qa_pairs), n_samples), np.nan),
    } for context_type in context_types
}

# Go down in context size (largest to smallest)
context_lengths_desc = sorted(context_lengths, reverse=False)

for iN, N in context_lengths_desc:

    print(f"\n{'='*80}")
    print(f"TESTING CONTEXT LENGTH N={N}")
    print(f"{'='*80}")

    # Generate all context materials for this sample
    all_context_materials = [get_context_freeform(N, valid_probe_results, probes_df_original, most_truth_answers, most_lie_answers) for _ in range(n_samples)]
    
    for context_type in context_types:
        print(f"\nTesting context type: {context_type}")

        question_truth_probs_across_samples = np.full((len(valid_qa_pairs), n_samples), np.nan)
        question_lie_probs_across_samples = np.full((len(valid_qa_pairs), n_samples), np.nan)
        
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

                question_truth_probs_across_samples[q_idx,sample_idx] = truth_prob
                question_lie_probs_across_samples[q_idx,sample_idx] = lie_prob


        # Store results
        all_results[context_type]['context_length'][iN] = N
        all_results[context_type]['context_type'][iN] = context_type
        all_results[context_type]['mean_truth_prob'][iN] = np.mean(question_truth_probs_across_samples)
        all_results[context_type]['std_truth_prob'][iN] = np.std(question_truth_probs_across_samples.mean(-1))
        all_results[context_type]['mean_lie_prob'][iN] = np.mean(question_lie_probs_across_samples)
        all_results[context_type]['std_lie_prob'][iN] = np.std(question_lie_probs_across_samples.mean(-1))
        all_results[context_type]['question_truth_probs_across_samples'][iN] = question_truth_probs_across_samples
        all_results[context_type]['question_lie_probs_across_samples'][iN] = question_lie_probs_across_samples
        
        print(f"{context_type} results for {len(question_truth_probs_across_samples)} questions:")
        print(f"  Mean truth prob: {all_results[context_type]['mean_truth_prob'][iN]:.4f} ± {all_results[context_type]['std_truth_prob'][iN]:.4f}")
        print(f"  Mean lie prob: {all_results[context_type]['mean_lie_prob'][iN]:.4f} ± {all_results[context_type]['std_lie_prob'][iN]:.4f}")

    # Plot results after completing all context types for this N
    print(f"\nPlotting results after N={N}...")

    fig, axes = plt.subplots(1, 1, figsize=(14, 7))

    # Colors for different context types
    colors = plt.cm.tab10(np.linspace(0, 1, len(context_types)))

    for i, context_type in enumerate(context_types):
            
        context_lengths_plot = all_results[context_type]['context_length']
        
        mean_truth_probs = all_results[context_type]['mean_truth_prob']
        std_truth_probs = all_results[context_type]['std_truth_prob']
        
        mean_lie_probs = all_results[context_type]['mean_lie_prob']
        std_lie_probs = all_results[context_type]['std_lie_prob']

        
        # Add small jitter to x-values to separate overlapping points
        jitter = (i - len(context_types)/2) * 0.05
        x_values = np.array(context_lengths_plot) + jitter
        
        # Plot truth probabilities (solid lines)
        axes.errorbar(x_values, mean_truth_probs, yerr=std_truth_probs, 
                    label=f'{context_type.replace("_", " ").title()} (truth)',
                    marker='x', capsize=3, capthick=1, linewidth=2, markersize=6,
                    color=colors[i], alpha=0.8, linestyle='-')
        
        # Plot lie probabilities (dashed lines)
        axes.errorbar(x_values, mean_lie_probs, yerr=std_lie_probs, 
                    label=f'{context_type.replace("_", " ").title()} (previous lie)',
                    marker='o', capsize=3, capthick=1, linewidth=2, markersize=6,
                    color=colors[i], alpha=0.8, linestyle='--')

    # Add significance stars for top_lie_shuffled_together vs top_truth_shuffled_together
    for lie_result, truth_result in zip(all_results['top_lie_shuffled_together'], all_results['top_truth_shuffled_together']):
        N_current = lie_result['context_length']
        
        # Get question-level truth probabilities for both context types
        lie_question_truth_probs = lie_result['question_lie_probs_across_samples'].mean(-1)
        truth_question_truth_probs = truth_result['question_truth_probs_across_samples'].mean(-1)
        
        # Paired t-test comparing truth probability under lie vs truth contexts
        if len(lie_question_truth_probs) > 1 and len(truth_question_truth_probs) > 1:
            stat, p_value = ttest_rel(lie_question_truth_probs, truth_question_truth_probs)
            
            if p_value < 0.05 and stat < 0.0:
                # Add star above the highest point at this N
                max_y = max(
                    lie_result['mean_truth_prob'] + lie_result['std_truth_prob'],
                    truth_result['mean_truth_prob'] + truth_result['std_truth_prob']
                )
                axes.text(N_current - 0.15, max_y + 0.02, '*', 
                        ha='center', va='bottom', fontsize=16, fontweight='bold')
                    

    axes.set_xlabel('Context Length (N)')
    axes.set_ylabel('Average Token Probability')
    axes.set_title('Truth vs (previous) Lie Token Probabilities by Context Composition\n(* indicates p<0.05 for lie vs truth contexts)')
    axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    axes.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'context_effect_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close to save memory
    
    # Save detailed results so far
    for context_type in context_types:
        context_type_npy_path = os.path.join(output_path, f'context_effect_results_{context_type}.npy')
        np.save(context_type_npy_path, all_results[context_type])

    print(f"Results updated and saved after N={N}: e.g. {context_type_npy_path}")


print("Analysis complete!")
