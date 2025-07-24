import math
import pandas as pd
import numpy as np
import json
import torch

from lie_detector.d_in_context_liar.viz import plot_context_effect_analysis, plot_context_effect_by_question_type


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

system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name
question_instruction = args.question_instruction

num_initial_generation_samples = args.num_initial_generation_samples

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
initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')
qa_pairs = [(initial_questions_df['question'][idx].strip(), str(initial_questions_df['answer'][idx])) for idx in range(len(initial_questions_df['question']))]


# Load probe questions for context
probes_df_original = pd.read_csv(f'data/probe_questions/{probe_file_name}.csv')
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
    'question_truth_log_probs_across_samples': np.full((num_context_lengths, len(unique_questions), n_samples, num_initial_generation_samples), np.nan),
    'question_lie_log_probs_across_samples': np.full((num_context_lengths, len(unique_questions), n_samples, num_initial_generation_samples), np.nan),
    } for context_type in context_types
}


question_types = initial_questions_df['type'].unique()
num_initial_question_types = len(question_types)

# Process each context length
context_lengths_desc = sorted(context_lengths, reverse=False)

for iN, N in enumerate(context_lengths_desc):
    print(f"\n{'='*80}")
    print(f"TESTING CONTEXT LENGTH N={N}")
    print(f"{'='*80}")

    # Generate all context materials for this sample
    all_context_materials = [get_context_yn(N, valid_probe_results, probes_df_original) for _ in range(n_samples)]
    
    for context_type in context_types:
        n_samples_eff = min(n_samples, math.perm(N, N))
        print(f"\n\tTesting context type: {context_type} - taking {n_samples_eff} samples")
        
        question_truth_lie_diffs_across_samples = np.full((len(unique_questions), n_samples_eff), np.nan)
        question_truth_log_probs_across_samples = np.full((len(unique_questions), n_samples_eff, num_initial_generation_samples), np.nan)
        question_lie_log_probs_across_samples = np.full((len(unique_questions), n_samples_eff, num_initial_generation_samples), np.nan)
        
        for sample_idx in range(n_samples_eff):

            print(f"\n\t\tSample {sample_idx + 1} of {n_samples_eff} for this context type. Iterating over questions")
            
            context_materials = all_context_materials[sample_idx]
            
            # Get the specific questions and answers for this context type
            if N > 0:
                shared_in_context_questions, shared_in_context_answers = context_materials[context_type]
            
                # Create base cache with system prompt and in-context examples
                base_cache_info = chat_wrapper.create_prompt_cache(
                    system_prompt=system_prompt,
                    in_context_questions=shared_in_context_questions,
                    in_context_answers=shared_in_context_answers,
                    prefiller=None
                )
            
            # Process each question individually
            for q_idx_pos, question_idx in tqdm(enumerate(unique_questions), total = len(unique_questions)):
                question = qa_pairs[question_idx][0]
                
                # Extend cache with the question
                full_question = question + f' {question_instruction}'
                question_inputs = chat_wrapper.tokenizer(
                    full_question,
                    return_tensors="pt",
                    add_special_tokens=False
                ).to(chat_wrapper.device)

                if N > 0:
                    # Clone base cache for this question
                    question_cache = copy.deepcopy(base_cache_info)
                    
                    # Get cache length and create proper attention mask
                    cache_length = question_cache["cache"].get_seq_length()
                    full_attention_mask = torch.cat([
                        torch.ones(1, cache_length, device=chat_wrapper.device),
                        question_inputs.attention_mask
                    ], dim=1)
                
                else:
                    question_cache = {}
                    full_attention_mask = question_inputs.attention_mask

                # Extend the cache with the question
                with torch.no_grad():
                    question_outputs = chat_wrapper.model(
                        input_ids=question_inputs.input_ids,
                        attention_mask=full_attention_mask,
                        past_key_values=question_cache["cache"] if N > 0 else None,
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


                question_truth_log_probs_across_samples[q_idx_pos, sample_idx] = truth_log_probs
                question_lie_log_probs_across_samples[q_idx_pos, sample_idx] = lie_log_probs
                question_truth_lie_diffs_across_samples[q_idx_pos, sample_idx] = truth_lie_diff
        
        # Store results
        all_results[context_type]['context_length'][iN] = N
        all_results[context_type]['context_type'][iN] = context_type
        all_results[context_type]['mean_truth_lie_diff'][iN] = np.mean(question_truth_lie_diffs_across_samples)
        all_results[context_type]['std_truth_lie_diff'][iN] = np.std(question_truth_lie_diffs_across_samples.mean(-1))
        all_results[context_type]['question_truth_lie_diffs_across_samples'][iN, :, :n_samples_eff] = question_truth_lie_diffs_across_samples
        all_results[context_type]['question_lie_log_probs_across_samples'][iN, :, :n_samples_eff] = question_lie_log_probs_across_samples
        all_results[context_type]['question_truth_log_probs_across_samples'][iN, :, :n_samples_eff] = question_truth_log_probs_across_samples

        print(f"{context_type} results for {len(question_truth_lie_diffs_across_samples)} questions:")
        print(f"  Mean truth-lie log prob diff: {all_results[context_type]['mean_truth_lie_diff'][iN]:.4f} ± {all_results[context_type]['std_truth_lie_diff'][iN]:.4f}")

    # Plot results after completing all context types for this N
    print(f"\nPlotting results after N={N}...")
    print(f"\nPlotting results after N={N}...")
    plot_context_effect_analysis(all_results, context_types, context_lengths, n_samples, output_path)
    plot_context_effect_by_question_type(all_results, context_types, context_lengths, n_samples, 
                                        unique_questions, initial_questions_df, output_path)

    # Save detailed results so far
    for context_type in context_types:
        context_type_npy_path = os.path.join(output_path, f'context_effect_results_{context_type}.npy')
        np.save(context_type_npy_path, all_results[context_type])

    print(f"Results updated and saved after N={N}: e.g. {context_type_npy_path}")
