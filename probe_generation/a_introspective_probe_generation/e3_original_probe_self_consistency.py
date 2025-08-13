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
probe_file_name = args.probe_file_name

# Experimental parameters
context_lengths = list(range(11))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
n_probe_samples = args.n_probe_samples_sc  # number of random probe selections per test probe
n_order_samples = args.n_order_samples_sc  # number of random orderings per probe sample
persona_types = ['aligned', 'misaligned']

# Prepare for saving results
save_base = os.path.join('probe_generation_results/a_introspective_probe_generation', args.args_name)
output_path = os.path.join(save_base, 'e_original_probe_consistency')
os.makedirs(output_path, exist_ok=True)
args.save(output_path)

# Load discriminability results
with open(os.path.join(save_base, 'b_discriminability_results.json'), 'r') as f:
    original_discriminability_results_list = json.load(f)

# Convert to dictionary keyed by probe_idx for easy lookup
original_discriminability_results = {}
for item in original_discriminability_results_list:
    probe_idx = item['probe_idx']
    original_discriminability_results[probe_idx] = item

# Load original probe questions
original_probe_questions_df = pd.read_csv(f'data/probe_questions/{probe_file_name}.csv')

# Filter out banned words
probe_questions = original_probe_questions_df[~original_probe_questions_df['probe'].str.lower().apply(lambda x: any(word in x for word in banned_words))]

# Filter to probes that have discriminability results
probe_questions_with_results = probe_questions[probe_questions.index.isin(original_discriminability_results.keys())]

print(f"Found {len(probe_questions_with_results)} probes with discriminability results")

# Calculate mean absolute effect size for each probe and determine effect sign
probe_stats = []
for probe_idx, row in probe_questions_with_results.iterrows():
    if probe_idx in original_discriminability_results:
        result = original_discriminability_results[probe_idx]
        effect_sizes_by_append = result['effect_sizes_by_append']
        mean_abs_effect = np.mean(np.abs(effect_sizes_by_append))
        mean_effect = np.mean(effect_sizes_by_append)
        effect_sign = 1 if mean_effect > 0 else -1
        if len(set(esa > 0.0 for esa in effect_sizes_by_append)) == 1:
            probe_stats.append({
                'probe_idx': probe_idx,
                'probe_text': row['probe'],
                'mean_abs_effect': mean_abs_effect,
                'mean_effect': mean_effect,
                'effect_sign': effect_sign
            })

# Sort by mean absolute effect size, descending
probe_stats.sort(key=lambda x: x['mean_abs_effect'], reverse=True)

# Get top 9 positive and top 9 negative effect probes
positive_probes = [p for p in probe_stats if p['effect_sign'] == 1]
negative_probes = [p for p in probe_stats if p['effect_sign'] == -1]

# Get as many as possible of each type, up to 9 each
top_positive = positive_probes[:min(9, len(positive_probes))]
top_negative = negative_probes[:min(9, len(negative_probes))]

# If we need more to reach 18 total, take from remaining highest-ranked probes
total_selected = len(top_positive) + len(top_negative)
if total_selected < 18:
    remaining_needed = 18 - total_selected
    remaining_probes = positive_probes[len(top_positive):] + negative_probes[len(top_negative):]
    remaining_probes.sort(key=lambda x: x['mean_abs_effect'], reverse=True)
    
    for probe in remaining_probes[:remaining_needed]:
        if probe['effect_sign'] == 1:
            top_positive.append(probe)
        else:
            top_negative.append(probe)

if len(top_positive) + len(top_negative) < 18:
    raise Exception(f"Could not find 18 total probes. Found {len(top_positive)} positive + {len(top_negative)} negative")

print(f"Selected top 9 positive and 9 negative probes (18 total)")

# Randomly split into context and test pools while maintaining balance
random.shuffle(top_positive)
random.shuffle(top_negative)

# Split each type proportionally to maintain balance as much as possible
pos_for_context = min(5, len(top_positive) // 2 + len(top_positive) % 2)  # Round up half
neg_for_context = min(5, len(top_negative) // 2 + len(top_negative) % 2)

# If we don't have enough for 10 context total, take more from the other type
total_context = pos_for_context + neg_for_context
if total_context < 10:
    if len(top_positive) - pos_for_context > 0:
        pos_for_context = min(len(top_positive), pos_for_context + (10 - total_context))
    elif len(top_negative) - neg_for_context > 0:
        neg_for_context = min(len(top_negative), neg_for_context + (10 - total_context))

context_probes = top_positive[:pos_for_context] + top_negative[:neg_for_context]
test_probes = top_positive[pos_for_context:] + top_negative[neg_for_context:]

# Shuffle the pools
random.shuffle(context_probes)
random.shuffle(test_probes)

print(f"Context pool: {len([p for p in context_probes if p['effect_sign'] == 1])} positive + {len([p for p in context_probes if p['effect_sign'] == -1])} negative = {len(context_probes)} total")
print(f"Test pool: {len([p for p in test_probes if p['effect_sign'] == 1])} positive + {len([p for p in test_probes if p['effect_sign'] == -1])} negative = {len(test_probes)} total")

# Save probe mappings immediately
context_probes_path = os.path.join(output_path, 'context_probes.json')
with open(context_probes_path, 'w') as f:
    json.dump(context_probes, f, indent=2)

test_probes_path = os.path.join(output_path, 'test_probes.json')
with open(test_probes_path, 'w') as f:
    json.dump(test_probes, f, indent=2)

print(f"Probe mappings saved early to: {output_path}")

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

# Setup result arrays
max_context_length = max(context_lengths)
n_test_probes = len(test_probes)

all_results = {}
for persona_type in persona_types:
    all_results[persona_type] = {
        # Dimensions: [test_probe_idx, context_length, probe_sample_idx, order_sample_idx]
        'test_probe_indices': np.full((n_test_probes, len(context_lengths), n_probe_samples, n_order_samples), -1, dtype=int),
        'test_append_indices': np.full((n_test_probes, len(context_lengths), n_probe_samples, n_order_samples), -1, dtype=int),
        # Context dimensions: [test_probe_idx, context_length, probe_sample_idx, order_sample_idx, context_position]
        'context_probe_indices': np.full((n_test_probes, len(context_lengths), n_probe_samples, n_order_samples, max_context_length), -1, dtype=int),
        'context_append_indices': np.full((n_test_probes, len(context_lengths), n_probe_samples, n_order_samples, max_context_length), -1, dtype=int),
        'test_prob_yes': np.full((n_test_probes, len(context_lengths), n_probe_samples, n_order_samples), np.nan),
        'test_prob_no': np.full((n_test_probes, len(context_lengths), n_probe_samples, n_order_samples), np.nan),
    }

print(f"Starting experiment...")

# Main experimental loops
for test_probe_idx, test_probe in enumerate(test_probes):
    print(f"\n{'='*80}")
    print(f"TESTING PROBE {test_probe_idx+1}/{n_test_probes}: {test_probe['probe_text'][:50]}...")
    print(f"Effect sign: {test_probe['effect_sign']}, Mean abs effect: {test_probe['mean_abs_effect']:.3f}")
    print(f"{'='*80}")
    
    for i_context_length, context_length in enumerate(context_lengths):
        print(f"  Context length: {context_length}")
        
        for probe_sample_idx in tqdm(range(n_probe_samples), desc=f"Probe samples N={context_length}"):
            
            # Sample context probes if context_length > 0
            if context_length > 0:
                # Create available context pool (exclude current test probe if it's in context_probes)
                available_context_probes = [p for p in context_probes if p['probe_idx'] != test_probe['probe_idx']]
                
                if len(available_context_probes) < context_length:
                    raise Exception(f"Not enough context probes available. Need {context_length}, have {len(available_context_probes)}")
                
                # Sample without replacement
                sampled_context_probes = random.sample(available_context_probes, context_length)
            else:
                sampled_context_probes = []
            
            # Test multiple random orderings
            for order_idx in range(n_order_samples):
                
                # Sample append indices
                test_append_idx = random.randint(0, n_append_types-1)
                
                context_append_indices = []
                shuffled_context_probes = []
                
                if context_length > 0:
                    # Sample append indices for context probes
                    context_append_indices = [random.randint(0, n_append_types-1) for _ in range(len(sampled_context_probes))]
                    
                    # Shuffle context probes and their append indices together
                    context_with_appends = list(zip(sampled_context_probes, context_append_indices))
                    random.shuffle(context_with_appends)
                    shuffled_context_probes, context_append_indices = zip(*context_with_appends) if context_with_appends else ([], [])
                    shuffled_context_probes = list(shuffled_context_probes)
                    context_append_indices = list(context_append_indices)
                
                # Test both persona types
                for persona_type in persona_types:
                    
                    # Create context if context_length > 0
                    context_questions_with_append = []
                    context_answers = []
                    
                    if context_length > 0:
                        for context_probe, append_idx in zip(shuffled_context_probes, context_append_indices):
                            # Add append to question
                            question_with_append = f"{context_probe['probe_text']} {append_strings[append_idx]}"
                            context_questions_with_append.append(question_with_append)
                            
                            # Generate persona-specific answer based on effect sign
                            if persona_type == 'aligned':
                                answer = 'Yes.' if context_probe['effect_sign'] > 0 else 'No.'
                            else:  # misaligned
                                answer = 'No.' if context_probe['effect_sign'] > 0 else 'Yes.'
                            context_answers.append(answer)
                    
                    # Test the probe with this context
                    test_question_with_append = f"{test_probe['probe_text']} {append_strings[test_append_idx]}"
                    
                    test_chats = [chat_wrapper.format_chat(
                        system_prompt="",
                        in_context_questions = context_questions_with_append if context_length > 0 else None,
                        in_context_answers = context_answers if context_length > 0 else None,
                        user_message=test_question_with_append,
                        prefiller='',
                    )]
                    
                    # Get yes/no probabilities
                    forward_result = chat_wrapper.forward(
                        chats=test_chats,
                    )
                    yes_no_probs = get_choice_token_logits_from_token_ids(forward_result['logits'], yesno_tokens)
                    
                    # Store results
                    all_results[persona_type]['test_probe_indices'][test_probe_idx, i_context_length, probe_sample_idx, order_idx] = test_probe['probe_idx']
                    all_results[persona_type]['test_append_indices'][test_probe_idx, i_context_length, probe_sample_idx, order_idx] = test_append_idx
                    
                    # Store context information
                    if context_length > 0:
                        for ctx_pos, (context_probe, append_idx) in enumerate(zip(shuffled_context_probes, context_append_indices)):
                            all_results[persona_type]['context_probe_indices'][test_probe_idx, i_context_length, probe_sample_idx, order_idx, ctx_pos] = context_probe['probe_idx']
                            all_results[persona_type]['context_append_indices'][test_probe_idx, i_context_length, probe_sample_idx, order_idx, ctx_pos] = append_idx
                    
                    all_results[persona_type]['test_prob_yes'][test_probe_idx, i_context_length, probe_sample_idx, order_idx] = yes_no_probs[0, 0].item()
                    all_results[persona_type]['test_prob_no'][test_probe_idx, i_context_length, probe_sample_idx, order_idx] = yes_no_probs[0, 1].item()
                    
                    # Clean up
                    del forward_result
        
        # Save intermediate results after each context length
        print(f"  Completed context length {context_length} for test probe {test_probe_idx}")
        for persona_type in persona_types:
            results_path = os.path.join(output_path, f'consistency_results_{persona_type}.npy')
            np.save(results_path, all_results[persona_type])

print(f"\nExperiment completed! Results saved to: {output_path}")