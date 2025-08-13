import numpy as np
import pandas as pd
import json
import torch
import copy
import random
from tqdm import tqdm
import os
import sys

from model.load import load_model
from util.elicit import elicit_sequence_log_probs
from util.util import YamlConfig

# Setup
config_path = sys.argv[1]
args = YamlConfig(config_path)

questions_data_name = args.questions_data_name
model_name = args.model_name
question_instruction = args.question_instruction
banned_words = args.banned_words
probe_file_name = args.probe_file_name

# Experimental parameters
n_in_context = args.context_length  # fixed context length
n_probe_samples = args.n_probe_samples  # number of random probe sets per generation type
n_order_samples = args.n_order_samples_icl  # number of random orderings per probe set

# persona_types = ['aligned', 'misaligned', 'random']
persona_types = ['aligned', 'misaligned']

# Prepare for saving results
save_base = os.path.join('probe_generation_results/a_introspective_probe_generation', args.args_name)
output_path = os.path.join(save_base, 'd_original_in_context_steering')
os.makedirs(output_path, exist_ok=True)
args.save(output_path)

# Load discriminability results
with open(os.path.join(save_base, 'b_discriminability_results.json'), 'r') as f:
    original_discriminability_results_list = json.load(f)
discriminability_results_idxs = [item['probe_idx'] for item in original_discriminability_results_list]
assert len(discriminability_results_idxs) == len(set(discriminability_results_idxs))

# Load the linking dataset between original and truncated probes
original_probe_questions_df = pd.read_csv(f'data/probe_questions/{probe_file_name}.csv')

# Filter out banned words -- shouldn't be a problem!
probe_questions = original_probe_questions_df[~original_probe_questions_df['probe'].str.lower().apply(lambda x: any(word in x for word in banned_words))]

# Load in the original questions and answer rollouts
print("Loading initial questions and stochastic answers...")
initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')
stochastic_answers_path = os.path.join(save_base, 'a_stochastic_initial_answers.csv')
stochastic_df = pd.read_csv(stochastic_answers_path)

# Get unique evaluation questions and number of stochastic samples per question
unique_eval_questions = stochastic_df['question_idx'].unique()
n_stochastic_samples = len(stochastic_df[stochastic_df['question_idx'] == unique_eval_questions[0]])
print(f"Found {len(unique_eval_questions)} evaluation questions with {n_stochastic_samples} stochastic samples per question")

append_strings = ["Answer with yes or no.", "Answer with no or yes."]
n_append_types = len(append_strings)

# Setup numpy arrays for results
print("Setting up result arrays...")
all_results = {}
for persona_type in persona_types:
    all_results[persona_type] = {
        'ordered_context_indices': np.full((n_probe_samples, n_order_samples, n_in_context), np.iinfo(np.int32).min, dtype=int),
        'context_append_indices': np.full((n_probe_samples, n_order_samples, n_in_context), np.iinfo(np.int32).min, dtype=int),
        'question_truth_log_probs': np.full((n_probe_samples, n_order_samples, len(unique_eval_questions), n_stochastic_samples), np.nan),
        'question_lie_log_probs': np.full((n_probe_samples, n_order_samples, len(unique_eval_questions), n_stochastic_samples), np.nan),
    }



# Filter discriminability results to entries where effect_sizes_by_append share a sign
def shared_sign(effect_sizes):
    signs = [1 if x > 0 else -1 if x < 0 else 0 for x in effect_sizes]
    unique_signs = set(signs)
    if len(unique_signs) == 1 and 0 not in unique_signs:
        return signs[0]
    return None

shared_sign_results = [
    item for item in original_discriminability_results_list
    if shared_sign(item['effect_sizes_by_append']) is not None
]

positive_sign_results_idxs = [
    item['probe_idx'] for item in shared_sign_results
    if shared_sign(item['effect_sizes_by_append']) == 1
]

negative_sign_results_idxs = [
    item['probe_idx'] for item in shared_sign_results
    if shared_sign(item['effect_sizes_by_append']) == -1
]


# Load model
print("Loading model...")
chat_wrapper = load_model(model_name, device='auto')


# Sample many random sets of augmented probes for this generation type
for probe_sample_idx in range(n_probe_samples):

    print(f"  Processing context sample {probe_sample_idx + 1}/{n_probe_samples}")
            
    # Randomly and evenly sample effect signs
    intended_effect_signs = [random.choice([1, -1]) for _ in range(n_in_context)]

    # Select context probes with balanced effect signs
    context_probe_idxs = []
    persona_answers = {pt: [] for pt in persona_types}
    ptrs = {+1: 0, -1: 0}
    shuffled_idxs = {
        +1: random.sample(positive_sign_results_idxs, len(positive_sign_results_idxs)),
        -1: random.sample(negative_sign_results_idxs, len(negative_sign_results_idxs))
    }
    for i_ies in range(len(intended_effect_signs)):
        ies = intended_effect_signs[i_ies]
        try:
            # Try to get next probe with intended effect sign
            next_item = shuffled_idxs[ies][ptrs[ies]]
            ptrs[ies] += 1
        except IndexError:
            # If not enough probes of intended sign, fallback to opposite sign
            next_item = shuffled_idxs[-ies][ptrs[-ies]]
            ptrs[-ies] += 1
            intended_effect_signs[i_ies:] = [-ies] * (len(intended_effect_signs) - i_ies)
            ies = - ies
        context_probe_idxs.append(next_item)
        persona_answers['aligned'].append('Yes.' if ies == 1 else 'No.')
        persona_answers['misaligned'].append('No.' if ies == 1 else 'Yes.')

    
    # Test multiple random orderings of these questions
    for order_idx in range(n_order_samples):

        print(f"    Processing order sample: {order_idx + 1} / {n_order_samples}")
            
        shuffle_indices = list(range(n_in_context))
        random.shuffle(shuffle_indices)

        context_append_indices = [random.randint(0, n_append_types-1) for _ in range(n_in_context)]

        # Apply append types to questions
        shuffled_questions_with_append = []
        ordered_context_indices = []
        for i, shuffle_idx in enumerate(shuffle_indices):
            base_question = original_probe_questions_df.iloc[context_probe_idxs[shuffle_idx]]['probe']
            append_idx = context_append_indices[i]
            final_question = base_question + " " + append_strings[append_idx]
            shuffled_questions_with_append.append(final_question)
            ordered_context_indices.append(context_probe_idxs[shuffle_idx])


        for persona_type in persona_types:
            
            # Store context append indices
            all_results[persona_type]['context_append_indices'][probe_sample_idx, order_idx, :n_in_context] = context_append_indices
            all_results[persona_type]['ordered_context_indices'][probe_sample_idx, order_idx, :n_in_context] = ordered_context_indices


            # Persona specific answers
            shuffled_answers = [persona_answers[persona_type][i] for i in shuffle_indices]
                
            # Create base cache with system prompt and context
            base_cache_info = chat_wrapper.create_prompt_cache(
                system_prompt="",
                in_context_questions=shuffled_questions_with_append,
                in_context_answers=shuffled_answers,
                prefiller=None
            )
                

            # Test on all evaluation questions
            for q_idx_pos, eval_question_idx in enumerate(
                tqdm(unique_eval_questions, desc=f"Persona: {persona_type} - Questions", leave=False)
            ):
                
                # Get question text
                question_data = stochastic_df[stochastic_df['question_idx'] == eval_question_idx]
                question_text = initial_questions_df.iloc[eval_question_idx].question
                full_question = f'{question_text} {question_instruction}'
                
                # Extend cache with evaluation question
                question_inputs = chat_wrapper.tokenizer(
                    full_question,
                    return_tensors="pt",
                    add_special_tokens=False
                ).to(chat_wrapper.device)

                # Clone base cache for this question
                question_cache = copy.deepcopy(base_cache_info)
                
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
                question_cache["cache"] = question_outputs.past_key_values

                # Get truth and lie responses for this question
                truth_responses = question_data['truth_answer'].tolist()
                lie_responses = question_data['lie_answer'].tolist()
                
                # Calculate log probabilities
                truth_log_probs = elicit_sequence_log_probs(chat_wrapper, question_cache, truth_responses)
                lie_log_probs = elicit_sequence_log_probs(chat_wrapper, question_cache, lie_responses)

                # Store results
                all_results[persona_type]['question_truth_log_probs'][probe_sample_idx, order_idx, q_idx_pos] = truth_log_probs.cpu().numpy()
                all_results[persona_type]['question_lie_log_probs'][probe_sample_idx, order_idx, q_idx_pos] = lie_log_probs.cpu().numpy()

        # Save intermediate results after every ordering sample
        print(f"Completed {probe_sample_idx + 1}/{n_probe_samples} samples")
        for persona_type in persona_types:
            results_path = os.path.join(output_path, f'steering_results_{persona_type}.npy')
            np.save(results_path, all_results[persona_type])

print(f"\nExperiment completed! Results saved to: {output_path}")
