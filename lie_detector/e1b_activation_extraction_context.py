#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import random

from model.load import load_model

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
save_path = args.save_path

probe_file_name = args.probe_file_name
probe_response_type = args.probe_response_type

if probe_response_type == 'yn':
    from 

prompt_index = args.prompt_idx

context_lengths = args.context_lengths_activations
n_samples = args.samples_per_context_length_activations

excluded_probe_types = args.excluded_probe_types
sorting_key = args.icl_sorting_key

chat_wrapper = load_model(model_name, device='auto')

# Prepare for saving results
save_target = f"{save_path}/activation_discovery/contextual/{probe_file_name}/{probe_response_type}/{questions_data_name}/prompt{prompt_index}"
os.makedirs(save_target, exist_ok=True)

# Load probe questions for context and keep unfiltered version for indexing
probes_df_original = pd.read_csv(f'data/{probe_file_name}.csv')
probes_df = probes_df_original[~probes_df_original['probe_type'].isin(excluded_probe_types)]

# Load discriminability results
with open(f'{save_path}/probe_analysis/{probe_file_name}/{probe_response_type}/{questions_data_name}/prompt{prompt_index}/discriminability_results.json', 'r') as f:
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



# Context composition types
context_types = [
    'top_lie_shuffled_together',      # 1. Top questions + lie answers, shuffled together
    'top_truth_shuffled_together',    # 2. Top questions + truth answers, shuffled together 
    'top_questions_random_answers',     # 5alt. Top questions + 50/50 random answers
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

    # Signal truth and lie
    top_questions_shuffled = top_n_questions.copy()
    top_answers_shuffled = truth_inducing_answers.copy()
    combined_truth_qa = list(zip(top_questions_shuffled, top_answers_shuffled))
    random.shuffle(combined_truth_qa)
    top_questions_shuffled, truth_answers_shuffled = zip(*combined_truth_qa)
    top_questions_shuffled = list(top_questions_shuffled)
    truth_answers_shuffled = list(truth_answers_shuffled)

    lie_answers_shuffled = ["No." if ans == "Yes." else "Yes." for ans in truth_answers_shuffled]

    # Shuffle only the questions
    top_questions_shuffled_control = top_questions_shuffled.copy()
    random.shuffle(top_questions_shuffled_control)
    top_questions_shuffled_control = list(top_questions_shuffled_control)

    # Random questions for types 5 and 6
    # random_questions = random.sample(probe_questions, min(N, len(probe_questions)))

    # Random answers for type 5alt
    random50_answers = ["Yes." if random.random() < 0.5 else "No." for _ in truth_answers_shuffled]
    
    return {
        # Type 1: Top questions + lie answers, shuffled together
        'top_lie_shuffled_together': (top_questions_shuffled, lie_answers_shuffled),
        
        # Type 2: Top questions + truth answers, shuffled together  
        'top_truth_shuffled_together': (top_questions_shuffled, truth_answers_shuffled),

        # Type 5alt. Top questions + 50/50 random answers
        'top_questions_random_answers': (top_questions_shuffled, random50_answers)
    }




response_data = pd.read_csv(f'{save_path}/initial_answers/{questions_data_name}.csv')
with open(f'data/{questions_data_name}.json', 'r') as f:
    question_data = json.load(f)

# Calculate the average of 'told_lie' where 'knows_answer' and 'told_truth' are True, grouped by 'prompt_idx'
knows_answer = response_data[(response_data['knows_answer']) & (response_data['told_truth'])]

# Get questions which reliably truthed and lied
trainable_answers = knows_answer[(knows_answer['prompt_idx'] == prompt_index) & (knows_answer['told_lie'])]
trainable_questions_idxs = trainable_answers['question_idx']



# 1. Load original answers dataframe
print("Loading original answers...")
original_answers_df = pd.read_csv(f'{save_path}/initial_answers/{questions_data_name}.csv')

# 2. Load probe results dataframe  
probe_results_df = pd.read_csv(f'{save_path}/probe_answers/{probe_file_name}/{probe_response_type}/{questions_data_name}_probe_prompt{prompt_index}.csv')

# 3. Filter original_answers_df to prompt_idx and question_idx in probe data
valid_question_ids = set(probe_results_df['question_idx'].unique())
filtered_answers_df = original_answers_df[
    (original_answers_df['prompt_idx'] == prompt_index) & 
    (original_answers_df['question_idx'].isin(valid_question_ids))
].copy()

# 4. Load questions data and create QA pairs
print("Loading test questions...")
with open(f'data/{questions_data_name}.json', 'r') as f:
    question_data = json.load(f)

qa_pairs = [(question_data['question'][f'{idx}'].strip(), question_data['answer'][f'{idx}']) 
            for idx in range(len(question_data['question']))]

# Filter QA pairs to only include questions we have in filtered_answers_df
valid_qa_pairs = [qa_pairs[idx] for idx in filtered_answers_df['question_idx']]

import pdb; pdb.set_trace(header = 'valid_qa_pairs same length as trainable_questions_idxs??')

test_questions = [qa[0] for qa in valid_qa_pairs]


candidate_layers = list(range(32))

num_context_lengths = len(context_lengths)
num_candidate_layers = len(candidate_layers)
num_questions = len(trainable_questions_idxs)
residual_stream_size = 4096


# Go down in context size (largest to smallest)
context_lengths_desc = sorted(context_lengths, reverse=True)

for iN, N in enumerate(context_lengths_desc):

    print(f"\n{'='*80}")
    print(f"TESTING CONTEXT LENGTH N={N}")
    print(f"{'='*80}")

    # Generate all context materials for this sample
    all_context_materials = [get_context(N) for _ in range(n_samples)]
    
    for context_type in context_types:
        print(f"\nTesting context type: {context_type}")

        residuals_per_N_per_context = np.zeros([num_questions, n_samples, num_candidate_layers, residual_stream_size])
        
        for sample_idx in tqdm(range(n_samples)):

            context_materials = all_context_materials[sample_idx]
            
            # Get the specific questions and answers for this context type
            if N == 0:
                shared_in_context_questions = None
                shared_in_context_answers = None
            else:
                shared_in_context_questions, shared_in_context_answers = context_materials[context_type]

            context_cache_info = chat_wrapper.create_prompt_cache(
                system_prompt=system_prompt,
                in_context_questions=shared_in_context_questions,
                in_context_answers=shared_in_context_answers,
                prefiller=None
            )
            
            for i in range(0, num_questions, batch_size):
                
                batch_questions = test_questions[i:i+batch_size]

                question_chats = [chat_wrapper.format_chat(
                    system_prompt=None,
                    user_message=question,
                    prefiller = ''
                ) for question in batch_questions]

                context_outputs = chat_wrapper.forward(
                    chats = question_chats,
                    past_key_values=copy.deepcopy(context_cache_info['cache']),
                    use_cache = True,
                    output_hidden_states = True
                )
                context_hs = context_outputs.hidden_states

                for cli, layer_idx in enumerate(candidate_layers):
                    residuals_per_N_per_context[i:i+batch_size, sample_idx, cli, :] = context_hs[layer_idx + 1].cpu().numpy()[:,-1,:]

        torch.save(residuals_per_N_per_context, os.path.join(save_target, f'all_contextual_residual_without_question_N{N}_context{context_type}.pt'))
