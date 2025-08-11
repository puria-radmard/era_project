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
from util.elicit import get_choice_token_logits_from_token_ids
from util.util import YamlConfig

# Setup
config_path = sys.argv[1]
args = YamlConfig(config_path)

questions_data_name = args.questions_data_name
model_name = args.model_name
question_instruction = args.question_instruction
banned_words = args.banned_words

# Experimental parameters
context_lengths = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
n_probe_samples = args.n_probe_samples  # number of random probe selections per generation type
n_order_samples = args.n_order_samples  # number of random orderings per probe sample
generation_types = ['lie-truth_contrastive', 'truth-lie_contrastive', 'lie-only', 'truth-only']
persona_types = ['aligned', 'misaligned']

# Prepare for saving results
save_base = os.path.join('probe_generation_results/a_introspective_probe_generation', args.args_name)
output_path = os.path.join(save_base, 'e_probe_consistency')
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

# Get the original probe indices that were augmented
original_probe_indices = sorted(probe_questions['probe_idx'].unique())
print(f"Found {len(original_probe_indices)} original probe indices with augmentations")

# Load model
print("Loading model...")
chat_wrapper = load_model(model_name, device='auto')

# Get yes/no tokens
yesno_strings = [
    ['Yes', 'yes'],
    ['No', 'no'],
]
yesno_tokens = []
for option_str_list in yesno_strings:
    option_tokens = []
    for option_str in option_str_list:
        token_ids = chat_wrapper.tokenizer.encode(option_str, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Choice token variation '{option_str}'"
                f"produces {len(token_ids)} tokens: {token_ids}. "
                f"All choice tokens must be exactly one token."
            )
        option_tokens.extend(token_ids)
    yesno_tokens.append(option_tokens)

append_strings = ["Answer with yes or no.", "Answer with no or yes."]
n_append_types = len(append_strings)

print(f"Starting experiment with {len(generation_types)} generation types...")

# Setup nested dictionary structure for results (same as steering script)
max_context_length = max(context_lengths)
n_test_original_probes = len(original_probe_indices)

all_results = {}
for generation_type in generation_types:
    all_results[generation_type] = {}
    for persona_type in persona_types:
        all_results[generation_type][persona_type] = {
            # Dimensions: [test_original_probe_idx, context_length, probe_sample_idx, order_sample_idx]
            'test_probe_indices': np.full((n_test_original_probes, max_context_length+1, n_probe_samples, n_order_samples), -1, dtype=int),
            'test_append_indices': np.full((n_test_original_probes, max_context_length+1, n_probe_samples, n_order_samples), -1, dtype=int),
            # Context dimensions: [test_original_probe_idx, context_length, probe_sample_idx, order_sample_idx, context_position]
            'context_probe_indices': np.full((n_test_original_probes, max_context_length+1, n_probe_samples, n_order_samples, max_context_length), -1, dtype=int),
            'context_append_indices': np.full((n_test_original_probes, max_context_length+1, n_probe_samples, n_order_samples, max_context_length), -1, dtype=int),
            'context_generation_types': np.full((n_test_original_probes, max_context_length+1, n_probe_samples, n_order_samples, max_context_length), -1, dtype=int),
            'test_prob_yes': np.full((n_test_original_probes, max_context_length+1, n_probe_samples, n_order_samples), np.nan),
            'test_prob_no': np.full((n_test_original_probes, max_context_length+1, n_probe_samples, n_order_samples), np.nan),
        }

# Create mapping from generation type to index for storage
generation_type_to_idx = {gen_type: idx for idx, gen_type in enumerate(generation_types)}

# Main experimental loops (follow same structure as steering script)
for gen_idx, generation_type in enumerate(generation_types):
    print(f"\n{'='*80}")
    print(f"PROCESSING GENERATION TYPE: {generation_type} ({gen_idx+1}/{len(generation_types)})")
    print(f"{'='*80}")
    
    for test_original_probe_idx_pos, test_original_probe_idx in enumerate(original_probe_indices):
        print(f"\n  Testing original probe {test_original_probe_idx} ({test_original_probe_idx_pos+1}/{len(original_probe_indices)})")
        
        # Get remaining original probes for context
        context_original_probe_indices = [p for p in original_probe_indices if p != test_original_probe_idx]
        
        for context_length in context_lengths:
            print(f"    Context length: {context_length}")
            
            for probe_sample_idx in tqdm(range(n_probe_samples), desc=f"Probe samples N={context_length}"):
                
                # Sample test probe from test original probe (same for all order samples)
                test_probe_candidates = probe_questions[
                    (probe_questions['probe_idx'] == test_original_probe_idx) & 
                    (probe_questions['generation_type'] == generation_type)
                ]
                
                # Filter for valid test probes (consistent effect signs)
                valid_test_probes = []
                for _, test_row in test_probe_candidates.iterrows():
                    test_idx = test_row.name
                    if test_idx in truncated_discriminability_results:
                        test_result = truncated_discriminability_results[test_idx]
                        effect_sizes_by_append = test_result['effect_sizes_by_append']
                        if len(effect_sizes_by_append) >= 2:
                            signs = [1 if x > 0 else -1 if x < 0 else 0 for x in effect_sizes_by_append]
                            if len(set(signs)) == 1 and signs[0] != 0:
                                valid_test_probes.append((test_idx, test_result, test_row))
                
                if len(valid_test_probes) == 0:
                    print(f"Warning: No valid test probes for original probe {test_original_probe_idx}, generation {generation_type}")
                    continue
                
                # Randomly sample one test probe for this probe sample
                test_probe_index, test_result, test_row = random.choice(valid_test_probes)
                test_avg_effect_size = np.mean(test_result['effect_sizes_by_append'])
                test_question_text = test_row['generated_sequence']
                
                # Sample context probes if context_length > 0 (same for all order samples)
                context_probe_data = []
                if context_length > 0:
                    # Sample context original probes (without replacement)
                    context_original_indices = random.sample(context_original_probe_indices, min(context_length, len(context_original_probe_indices)))
                    
                    for context_original_probe_idx in context_original_indices:
                        # Sample one augmented probe from this original probe
                        context_probe_candidates = probe_questions[
                            (probe_questions['probe_idx'] == context_original_probe_idx) & 
                            (probe_questions['generation_type'] == generation_type)
                        ]
                        
                        # Filter for valid context probes
                        valid_context_probes = []
                        for _, context_row in context_probe_candidates.iterrows():
                            context_idx = context_row.name
                            if context_idx in truncated_discriminability_results:
                                context_result = truncated_discriminability_results[context_idx]
                                effect_sizes_by_append = context_result['effect_sizes_by_append']
                                if len(effect_sizes_by_append) >= 2:
                                    signs = [1 if x > 0 else -1 if x < 0 else 0 for x in effect_sizes_by_append]
                                    if len(set(signs)) == 1 and signs[0] != 0:
                                        valid_context_probes.append((context_idx, context_result, context_row))
                        
                        if len(valid_context_probes) == 0:
                            continue
                        
                        # Randomly sample one context probe
                        context_probe_index, context_result, context_row = random.choice(valid_context_probes)
                        context_avg_effect_size = np.mean(context_result['effect_sizes_by_append'])
                        context_question_text = context_row['generated_sequence']
                        
                        context_probe_data.append({
                            'probe_index': context_probe_index,
                            'effect_size': context_avg_effect_size,
                            'question_text': context_question_text
                        })
                
                # Test multiple random orderings of these probes
                for order_idx in range(n_order_samples):
                    
                    # Sample append indices for test and context probes
                    test_append_idx = random.randint(0, n_append_types-1)
                    
                    context_append_indices = []
                    shuffled_context_data = []
                    
                    if context_length > 0 and len(context_probe_data) > 0:
                        # Sample append indices for context probes
                        context_append_indices = [random.randint(0, n_append_types-1) for _ in range(len(context_probe_data))]
                        
                        # Shuffle context probes and their append indices together
                        context_with_appends = list(zip(context_probe_data, context_append_indices))
                        random.shuffle(context_with_appends)
                        shuffled_context_data, context_append_indices = zip(*context_with_appends) if context_with_appends else ([], [])
                        shuffled_context_data = list(shuffled_context_data)
                        context_append_indices = list(context_append_indices)
                    
                    # Test both persona types
                    for persona_type in persona_types:
                        
                        # Create context if context_length > 0
                        cache = None
                        if context_length > 0 and len(shuffled_context_data) > 0:
                            # Generate persona-specific answers for context
                            context_questions_with_append = []
                            context_answers = []
                            
                            for context_data, append_idx in zip(shuffled_context_data, context_append_indices):
                                # Add append to question
                                question_with_append = f"{context_data['question_text']} {append_strings[append_idx]}"
                                context_questions_with_append.append(question_with_append)
                                
                                # Generate persona-specific answer
                                if persona_type == 'aligned':
                                    answer = 'Yes.' if context_data['effect_size'] > 0 else 'No.'
                                else:  # misaligned
                                    answer = 'No.' if context_data['effect_size'] > 0 else 'Yes.'
                                context_answers.append(answer)
                            
                            # Create cache with context
                            cache_info = chat_wrapper.create_prompt_cache(
                                system_prompt="",
                                in_context_questions=context_questions_with_append,
                                in_context_answers=context_answers,
                                prefiller=None
                            )
                            cache = cache_info["cache"]
                        
                        # Test the probe with this context
                        test_question_with_append = f"{test_question_text} {append_strings[test_append_idx]}"
                        
                        test_chats = [chat_wrapper.format_chat(
                            system_prompt="",
                            user_message=test_question_with_append,
                            prefiller='',
                        )]
                        
                        # Get yes/no probabilities
                        forward_result = chat_wrapper.forward(
                            chats=test_chats,
                            past_key_values=copy.deepcopy(cache) if cache is not None else None,
                        )
                        yes_no_probs = get_choice_token_logits_from_token_ids(forward_result['logits'], yesno_tokens)
                        
                        # Store results
                        all_results[generation_type][persona_type]['test_probe_indices'][test_original_probe_idx_pos, context_length, probe_sample_idx, order_idx] = test_probe_index
                        all_results[generation_type][persona_type]['test_append_indices'][test_original_probe_idx_pos, context_length, probe_sample_idx, order_idx] = test_append_idx
                        
                        # Store context information (pad with -1 for shorter contexts)
                        if context_length > 0 and len(shuffled_context_data) > 0:
                            for ctx_pos, (context_data, append_idx) in enumerate(zip(shuffled_context_data, context_append_indices)):
                                all_results[generation_type][persona_type]['context_probe_indices'][test_original_probe_idx_pos, context_length, probe_sample_idx, order_idx, ctx_pos] = context_data['probe_index']
                                all_results[generation_type][persona_type]['context_append_indices'][test_original_probe_idx_pos, context_length, probe_sample_idx, order_idx, ctx_pos] = append_idx
                        
                        all_results[generation_type][persona_type]['test_prob_yes'][test_original_probe_idx_pos, context_length, probe_sample_idx, order_idx] = yes_no_probs[0, 0].item()
                        all_results[generation_type][persona_type]['test_prob_no'][test_original_probe_idx_pos, context_length, probe_sample_idx, order_idx] = yes_no_probs[0, 1].item()
                        
                        # Clean up
                        del forward_result
            
            # Save intermediate results after each context length
            print(f"    Completed context length {context_length} for test probe {test_original_probe_idx}")
            for persona_type in persona_types:
                results_path = os.path.join(output_path, f'consistency_results_{generation_type}_{persona_type}.npy')
                np.save(results_path, all_results[generation_type][persona_type])

print(f"\nExperiment completed! Results saved to: {output_path}")

# Save generation type mapping
generation_type_mapping_path = os.path.join(output_path, 'generation_type_mapping.json')
with open(generation_type_mapping_path, 'w') as f:
    json.dump(generation_type_to_idx, f)

# Save original probe indices mapping
original_probe_mapping_path = os.path.join(output_path, 'original_probe_indices.json')
with open(original_probe_mapping_path, 'w') as f:
    json.dump({str(pos): idx for pos, idx in enumerate(original_probe_indices)}, f)

print(f"Mappings saved to: {output_path}")