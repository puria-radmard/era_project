import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import random

from model.load import load_model

from util.lying_context import get_context_freeform, get_context_yn
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

prompt_index = args.prompt_idx
limit_to_lying = args.limit_to_lying

context_lengths = args.context_lengths_activations
n_samples = args.samples_per_context_length_activations

excluded_probe_types = args.excluded_probe_types
sorting_key = args.icl_sorting_key

chat_wrapper = load_model(model_name, device='auto')

# Prepare for saving results
output_path = os.path.join('lie_detector_results/e_activation_analysis', args.args_name, 'contextual')
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
with open(f'data/{questions_data_name}.json', 'r') as f:
    question_data = json.load(f)

qa_pairs = [(question_data['question'][f'{idx}'].strip(), question_data['answer'][f'{idx}']) 
            for idx in range(len(question_data['question']))]


# Filter QA pairs to only include questions we have in filtered_answers_df
valid_qa_pairs = [qa_pairs[idx] for idx in filtered_answers_df['question_idx']]

test_questions = [qa[0] for qa in valid_qa_pairs]


# Load probe questions for context and keep unfiltered version for indexing
probes_df_original = pd.read_csv(f'data/{probe_file_name}.csv')
probes_df = probes_df_original[~probes_df_original['probe_type'].isin(excluded_probe_types)]

# Load discriminability results
discriminability_data_path = os.path.join('lie_detector_results/c_probe_discrimination', probe_analysis_args_name, 'discriminability_results.json')
with open(discriminability_data_path, 'r') as f:
    discriminability_data = json.load(f)


if probe_response_type.endswith('_words'):

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
else:
    excluded_probe_indices = []

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



# Context composition types
context_types = [
    'top_lie_shuffled_together',      # 1. Top questions + lie answers, shuffled together
    'top_truth_shuffled_together',    # 2. Top questions + truth answers, shuffled together 
    'top_questions_random_answers',     # 5alt. Top questions + 50/50 random answers
]



# Get questions which reliably truthed and lied
if limit_to_lying:
    knows_answer = initial_answers_df[(initial_answers_df['knows_answer']) & (initial_answers_df['told_truth'])]
    trainable_answers = knows_answer[(knows_answer['prompt_idx'] == prompt_index) & (knows_answer['told_lie'])]
else:
    trainable_answers = initial_answers_df
trainable_questions_idxs = trainable_answers['question_idx']

assert len(trainable_questions_idxs) == len(valid_qa_pairs)

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
    if probe_response_type == 'yn':
        all_context_materials = [get_context_yn(N, valid_probe_results, probes_df_original) for _ in range(n_samples)]
    elif probe_response_type.endswith('_words'):
        all_context_materials = [get_context_freeform(N, valid_probe_results, probes_df_original, most_truth_answers, most_lie_answers) for _ in range(n_samples)]

    
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
                    user_message=question + f' {question_instruction}',
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

        torch.save(residuals_per_N_per_context, os.path.join(output_path, f'all_contextual_residual_without_question_N{N}_context{context_type}.pt'))
