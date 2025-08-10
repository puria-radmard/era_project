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

# Experimental parameters
N = args.context_length  # fixed context length
n_probe_samples = args.n_probe_samples  # number of random probe sets per generation type
n_order_samples = args.n_order_samples  # number of random orderings per probe set

persona_types = ['aligned', 'misaligned', 'random']
generation_types = ['lie-truth_contrastive', 'truth-lie_contrastive', 'lie-only', 'truth-only']

# Prepare for saving results
save_base = os.path.join('probe_generation_results/a_introspective_probe_generation', args.args_name)
output_path = os.path.join(save_base, 'd_in_context_steering')
os.makedirs(output_path, exist_ok=True)
args.save(output_path)

# Load discriminability results
with open(os.path.join(save_base, 'c_truncated_discriminability_results.json'), 'r') as f:
    truncated_discriminability_results_list = json.load(f)

# Convert to dictionary keyed by probe_idx for easy lookup
truncated_discriminability_results = {}
for item in truncated_discriminability_results_list:
    probe_idx = item['probe_idx']
    assert probe_idx not in truncated_discriminability_results, f"Duplicate probe_idx found: {probe_idx}"
    truncated_discriminability_results[probe_idx] = item

# Load the linking dataset between original and truncated probes
truncated_probe_questions_df = pd.read_csv(os.path.join(save_base, "c_truncated_probe_completions.csv"))

# Filter out banned words
probe_questions = truncated_probe_questions_df[~truncated_probe_questions_df['generated_sequence'].str.lower().apply(lambda x: any(word in x for word in banned_words))]

# Get the original probes that were augmented (these should be the 10 least discriminative)
original_probe_idxs = probe_questions['probe_idx'].unique()
print(f"Found {len(original_probe_idxs)} original probes with augmentations")

# Load in the original questions
initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')

# Load stochastic answers dataframe
print("Loading stochastic answers...")
stochastic_answers_path = os.path.join(save_base, 'a_stochastic_initial_answers.csv')
stochastic_df = pd.read_csv(stochastic_answers_path)

# Get unique evaluation questions
unique_eval_questions = stochastic_df['question_idx'].unique()
print(f"Found {len(unique_eval_questions)} evaluation questions")

# Get number of stochastic samples per question
n_stochastic_samples = len(stochastic_df[stochastic_df['question_idx'] == unique_eval_questions[0]])
print(f"Found {n_stochastic_samples} stochastic samples per question")

# Load model
print("Loading model...")
chat_wrapper = load_model(model_name, device='auto')

print(f"Starting experiment with {len(generation_types)} generation types...")

# Main experimental loops
for gen_idx, generation_type in enumerate(generation_types):

    # Setup numpy arrays for results
    print("Setting up result arrays...")
    generation_type_results = {}
    for persona_type in persona_types:
        generation_type_results[persona_type] = {
            'context_contents_indices': np.full((n_probe_samples, N), -1, dtype=int),
            'question_truth_log_probs': np.full((n_probe_samples, n_order_samples, len(unique_eval_questions), n_stochastic_samples), np.nan),
            'question_lie_log_probs': np.full((n_probe_samples, n_order_samples, len(unique_eval_questions), n_stochastic_samples), np.nan),
        }

    print(f"\n{'='*80}")
    print(f"PROCESSING GENERATION TYPE: {generation_type} ({gen_idx+1}/{len(generation_types)})")
    print(f"{'='*80}")
    
    # Sample many random sets of augmented probes for this generation type
    for probe_sample_idx in range(n_probe_samples):

        print(f"  Processing probe sample {probe_sample_idx + 1}/{n_probe_samples} for generation type '{generation_type}'...")
        
        # Randomly sample 1 augmentation per original probe (across prefix lengths)
        context_contents_indices = []
        effect_sizes = []
        probe_questions_text = []
        
        for orig_probe_idx in original_probe_idxs:

            # Get available augmentations for this original probe and generation type
            available_augmentations = probe_questions[
                (probe_questions['probe_idx'] == orig_probe_idx) & 
                (probe_questions['generation_type'] == generation_type)
            ]
            
            # Filter to augmentations with consistent effect sign and discriminability results
            valid_augmentations = []
            for _, aug_row in available_augmentations.iterrows():
                aug_idx = aug_row.name
                if aug_idx in truncated_discriminability_results:
                    aug_result = truncated_discriminability_results[aug_idx]
                    effect_sizes_by_append = aug_result['effect_sizes_by_append']
                    # Check if effect signs are consistent across append types
                    if len(effect_sizes_by_append) >= 2:
                        signs = [1 if x > 0 else -1 if x < 0 else 0 for x in effect_sizes_by_append]
                        if len(set(signs)) == 1 and signs[0] != 0:  # All same sign and non-zero
                            valid_augmentations.append((aug_idx, aug_result, aug_row))
            
            if len(valid_augmentations) == 0:
                message = f"Warning: No valid augmentations found for probe {orig_probe_idx}, generation {generation_type}"
                print(message)
                with open(os.path.join(output_path, "bug.log"), "a") as bug_file:
                    bug_file.write(message + "\n")
                continue
                
            # Randomly sample one valid augmentation
            sampled_aug_idx, sampled_result, sampled_row = random.choice(valid_augmentations)
            context_contents_indices.append(sampled_aug_idx)
            
            # Get average effect size for this augmentation
            avg_effect_size = np.mean(sampled_result['effect_sizes_by_append'])
            effect_sizes.append(avg_effect_size)
            
            # Get probe question text
            probe_questions_text.append(sampled_row['generated_sequence'])
            
        # Store context contents indices
        for persona_type in persona_types:
            generation_type_results[persona_type]['context_contents_indices'][probe_sample_idx] = context_contents_indices
        
        # Generate yes/no answers for each persona type
        persona_answers = {}
        for persona_type in persona_types:
            if persona_type == 'aligned':
                answers = ['Yes.' if effect > 0 else 'No.' for effect in effect_sizes]
            elif persona_type == 'misaligned': 
                answers = ['No.' if effect > 0 else 'Yes.' for effect in effect_sizes]
            else:  # random
                answers = ['Yes.' if random.random() < 0.5 else 'No.' for _ in effect_sizes]
            persona_answers[persona_type] = answers
        
        # Test multiple random orderings of these questions
        for order_idx in range(n_order_samples):

            print(f"    Processing order sample: {order_idx + 1} / {n_order_samples}")
            
            # Create shuffle order (shared across personas)
            shuffle_indices = list(range(N))
            random.shuffle(shuffle_indices)
            
            shuffled_questions = [probe_questions_text[i] for i in shuffle_indices]
            
            # Test each persona with this ordering
            for persona_type in persona_types:
                
                shuffled_answers = [persona_answers[persona_type][i] for i in shuffle_indices]
                
                # Create base cache with system prompt and context
                base_cache_info = chat_wrapper.create_prompt_cache(
                    system_prompt="",
                    in_context_questions=shuffled_questions,
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
                    generation_type_results[persona_type]['question_truth_log_probs'][probe_sample_idx, order_idx, q_idx_pos] = truth_log_probs.cpu().numpy()
                    generation_type_results[persona_type]['question_lie_log_probs'][probe_sample_idx, order_idx, q_idx_pos] = lie_log_probs.cpu().numpy()

        # Save intermediate results after every ordering sample
        print(f"Completed {probe_sample_idx + 1}/{n_probe_samples} samples for {generation_type}")
        for persona_type in persona_types:
            results_path = os.path.join(output_path, f'steering_results_{generation_type}_{persona_type}.npy')
            np.save(results_path, generation_type_results[persona_type])

print(f"\nExperiment completed! Results saved to: {output_path}")
