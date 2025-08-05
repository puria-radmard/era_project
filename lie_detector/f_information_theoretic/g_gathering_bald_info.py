import json, copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
import sys

from steering_vectors import ablation_then_addition_operator
from util.steering import LayerSpecificMultipliersSteeringVector
from model.load import load_model
from util.util import YamlConfig

# Load configuration
config_path = sys.argv[1]
args = YamlConfig(config_path)

# Extract parameters
system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name
initial_answers_args_name = args.initial_answers_args_name
question_instruction = args.question_instruction
probe_file_name = args.probe_file_name
prompt_index = args.prompt_idx
limit_to_lying = args.limit_to_lying

# Steering parameters
chosen_layers = args.chosen_layers
multipliers = args.multipliers
multipliers = sorted(list(set(multipliers + list(map(lambda x: -x, multipliers)))))

# Set up paths
activation_analysis_path = os.path.join('lie_detector_results/e_activation_analysis', args.args_name, 'projection_results')
save_base = os.path.join('lie_detector_results/f_information_theoretic', args.args_name, 'bald_estimation')
os.makedirs(save_base, exist_ok=True)
args.save(save_base)

output_path = os.path.join(save_base, 'steering_probe_results.npy')

print("Loading initial answers and filtering...")
initial_answers_path = os.path.join('lie_detector_results/a_initial_answers', initial_answers_args_name, 'initial_answers.csv')
response_data = pd.read_csv(initial_answers_path)

# Load questions and prompts
initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')

with open('data/all_prompts.json', 'r') as f:
    prompts = json.load(f)
    lie_prompt = prompts['lie_prompts'][prompt_index]
    truth_prompt = prompts['truth_prompts'][prompt_index]

# Filter trainable questions
if limit_to_lying:
    knows_answer = response_data[(response_data['knows_answer']) & (response_data['told_truth'])]
    trainable_answers = knows_answer[(knows_answer['prompt_idx'] == prompt_index) & (knows_answer['told_lie'])]
else:
    trainable_answers = response_data
trainable_questions_idxs = trainable_answers['question_idx'].unique()

print(f"Found {len(trainable_questions_idxs)} trainable questions")

# Load probe questions
probe_questions = pd.read_csv(f'data/probe_questions/{probe_file_name}.csv')['probe'].tolist()
print(f"Loaded {len(probe_questions)} probe questions")

# Load model and tokenizer
print("Loading model...")
chat_wrapper = load_model(model_name, device='auto')

# Tokenize probe questions to get lengths
tokenized_probes = []
probe_lengths = []
for probe in probe_questions:
    formatted_probe = chat_wrapper.format_chat(
        system_prompt="",
        user_message=probe,
        prefiller='',
    )
    tokens = chat_wrapper.tokenizer.encode(formatted_probe, add_special_tokens=False)
    tokenized_probes.append(tokens)
    probe_lengths.append(len(tokens))

max_probe_length = max(probe_lengths)
print(f"Max probe length: {max_probe_length} tokens")

# Load steering vector information
print("Loading steering vector...")
pre_answer_vector_info = np.load(os.path.join(activation_analysis_path, 'prompted_projection_along_average_lie_vector.npy'), allow_pickle=True).item()

# Set up steering vector components
gt_misaligned_cluster_mean = {layer_num: pre_answer_vector_info[layer_num]['lie_mean'] for layer_num in chosen_layers}
gt_aligned_cluster_mean = {layer_num: pre_answer_vector_info[layer_num]['truth_mean'] for layer_num in chosen_layers}
midpoints = {cl: (gt_misaligned_cluster_mean[cl] + gt_aligned_cluster_mean[cl]) / 2.0 for cl in chosen_layers}
half_lengths = {cl: (gt_misaligned_cluster_mean[cl] - gt_aligned_cluster_mean[cl]) / 2.0 for cl in chosen_layers}

layer_direction = {layer_num: torch.tensor(pre_answer_vector_info[layer_num]['direction']).cuda().to(torch.float16) 
                  for layer_num in chosen_layers}
steering_vector = LayerSpecificMultipliersSteeringVector(layer_direction, "decoder_block")

# Initialize results structure
num_questions = len(trainable_questions_idxs)
num_probe_questions = len(probe_questions)
num_multipliers = len(multipliers)
num_layers = len(chosen_layers)

all_results = {
    'multipliers': np.array(multipliers),
    'layers': np.array(chosen_layers),
    
    # With steering applied
    'truth_log_probs': np.full((num_questions, num_probe_questions, num_multipliers, max_probe_length), np.nan),
    'lie_log_probs': np.full((num_questions, num_probe_questions, num_multipliers, max_probe_length), np.nan),
    
    'truth_conditioned_entropy': np.full((num_questions, num_probe_questions, num_multipliers, max_probe_length), np.nan),
    'lie_conditioned_entropy': np.full((num_questions, num_probe_questions, num_multipliers, max_probe_length), np.nan),
    
    # No steering baseline
    'truth_log_probs_no_steering': np.full((num_questions, num_probe_questions, max_probe_length), np.nan),
    'lie_log_probs_no_steering': np.full((num_questions, num_probe_questions, max_probe_length), np.nan),
    'truth_conditioned_entropy_no_steering': np.full((num_questions, num_probe_questions, max_probe_length), np.nan),
    'lie_conditioned_entropy_no_steering': np.full((num_questions, num_probe_questions, max_probe_length), np.nan),
    'truth_projections_no_steering': np.full((num_questions, num_probe_questions, num_layers, max_probe_length), np.nan),
    'lie_projections_no_steering': np.full((num_questions, num_probe_questions, num_layers, max_probe_length), np.nan),
    
    # Metadata
    'question_indices': np.array(trainable_questions_idxs),
    'probe_lengths': np.array(probe_lengths)
}

print(f"Initialized results arrays: {num_questions} questions × {num_probe_questions} probes × {num_multipliers} multipliers")
print(f"Multipliers: {multipliers}")

# Main processing loop
print(f"\n{'='*80}")
print(f"PROCESSING {num_questions} QUESTIONS")
print(f"Each question: {num_probe_questions} probes × (1 no-steering + {num_multipliers} multipliers) = {num_probe_questions * (1 + num_multipliers)} forward passes")
print(f"Total forward passes: {num_questions * num_probe_questions * (1 + num_multipliers)}")
print(f"{'='*80}")

for q_idx, question_idx in enumerate(trainable_questions_idxs):

    print(f'Processing question {q_idx + 1} of {len(trainable_questions_idxs)}. Iterating probe questions...')
    
    question = initial_questions_df['question'][question_idx].strip()
    response_row = trainable_answers[trainable_answers['question_idx'] == question_idx].iloc[0]

    # Create truth and lie caches
    truth_cache_info = chat_wrapper.create_prompt_cache(
        system_prompt=system_prompt,
        in_context_questions=[f'{truth_prompt} {question} {question_instruction}'],
        in_context_answers=[response_row.truth_answer]
    )
    truth_cache = truth_cache_info["cache"]

    lie_cache_info = chat_wrapper.create_prompt_cache(
        system_prompt=system_prompt,
        in_context_questions=[f'{lie_prompt} {question} {question_instruction}'],
        in_context_answers=[response_row.lie_answer]
    )
    lie_cache = lie_cache_info["cache"]

    # Process each probe question
    for probe_idx, probe_tokens in tqdm(enumerate(tokenized_probes), total = len(tokenized_probes)):
        probe_length = len(probe_tokens)
        
        # Format probe for model input
        formatted_probe = chat_wrapper.format_chat(
            system_prompt="",
            user_message=probe_questions[probe_idx],
            prefiller='',
        )
        
        # Process no-steering baseline for both caches
        for cache_name, cache in [("truth", truth_cache), ("lie", lie_cache)]:
            
            # Run probe through this cache without steering
            with torch.no_grad():
                probe_inputs = chat_wrapper.tokenizer(
                    formatted_probe,
                    return_tensors="pt",
                    add_special_tokens=False
                ).to(chat_wrapper.device)
                
                probe_outputs = chat_wrapper.model(
                    input_ids=probe_inputs.input_ids,
                    attention_mask=probe_inputs.attention_mask,
                    past_key_values=copy.deepcopy(cache),
                    use_cache=False,
                    return_dict=True,
                    output_hidden_states=True
                )
                
                # Extract log probabilities (offset by 1 token)
                logits = probe_outputs.logits[0]  # Remove batch dimension
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Get log-probs for actual tokens (shift by 1)
                if probe_length > 1:
                    target_tokens = probe_inputs.input_ids[0, 1:]  # Skip first token
                    token_log_probs = log_probs[:-1].gather(1, target_tokens.unsqueeze(1)).squeeze(1)
                    
                    # Store log probs results
                    actual_length = min(probe_length - 1, max_probe_length)
                    all_results[f'{cache_name}_log_probs_no_steering'][q_idx, probe_idx, :actual_length] = token_log_probs[:actual_length].cpu().numpy()
                    
                    import pdb; pdb.set_trace()
                    all_results[f'{cache_name}_conditioned_entropy_no_steering'][q_idx, probe_idx, mult_idx, :actual_length] = steered_token_entropies.cpu().numpy()

                    # Calculate projections onto lie-truth direction
                    # Using the primary chosen layer for projection
                    for i_layer, layer in enumerate(chosen_layers):
                        hidden_states = probe_outputs.hidden_states[layer + 1][0]  # Remove batch dim
                        direction_vec = layer_direction[layer]
                        projections = torch.matmul(hidden_states[:-1], direction_vec)  # Skip last position
                        all_results[f'{cache_name}_projections_no_steering'][q_idx, probe_idx, i_layer, :actual_length] = projections[:actual_length].cpu().numpy()

        # Process with steering for each multiplier
        for mult_idx, multiplier in enumerate(multipliers):
            # Calculate cluster mean for this multiplier
            cluster_mean = {cl: midpoints[cl] + multiplier * half_lengths[cl] for cl in chosen_layers}
            
            # Apply steering and process both caches
            with steering_vector.apply(chat_wrapper.model, multiplier=cluster_mean, min_token_index=0, 
                                     operator=ablation_then_addition_operator()):
                
                for cache_name, cache in [("truth", truth_cache), ("lie", lie_cache)]:
                    
                    with torch.no_grad():
                        probe_inputs = chat_wrapper.tokenizer(
                            formatted_probe,
                            return_tensors="pt",
                            add_special_tokens=False
                        ).to(chat_wrapper.device)
                        
                        probe_outputs = chat_wrapper.model(
                            input_ids=probe_inputs.input_ids,
                            attention_mask=probe_inputs.attention_mask,
                            past_key_values=copy.deepcopy(cache),
                            use_cache=False,
                            return_dict=True,
                            output_hidden_states=True
                        )
                        
                        # Extract log probabilities and projections
                        logits = probe_outputs.logits[0]
                        log_probs = torch.log_softmax(logits, dim=-1)
                        
                        if probe_length > 1:
                            target_tokens = probe_inputs.input_ids[0, 1:]
                            token_log_probs = log_probs[:-1].gather(1, target_tokens.unsqueeze(1)).squeeze(1)
                            
                            # Store logprob results
                            actual_length = min(probe_length - 1, max_probe_length)
                            all_results[f'{cache_name}_log_probs'][q_idx, probe_idx, mult_idx, :actual_length] = token_log_probs[:actual_length].cpu().numpy()
                            
                            all_results[f'{cache_name}_conditioned_entropy'][q_idx, probe_idx, mult_idx, :actual_length] = token_entropies.cpu().numpy()

                            # Calculate projections
                            for i_layer, layer in enumerate(chosen_layers):
                                hidden_states = probe_outputs.hidden_states[layer + 1][0]
                                direction_vec = layer_direction[layer]
                                projections = torch.matmul(hidden_states[:-1], direction_vec)
                                
                                if not torch.isclose(projections[0], projections, rtol = 1e-2).all():
                                    print(f'STEERING NOT CLOSE - question {q_idx}, probe {probe_idx}, layer {i_layer + 1}')
                                    print(projections.tolist())

        # Clean up GPU memory
        torch.cuda.empty_cache()
    
    np.save(output_path, all_results)
    print(f"Results so far saved to: {output_path}")
        
print(f"\n{'='*80}")
print(f"PROCESSING COMPLETE - COMPUTING SUMMARY STATISTICS")
print(f"{'='*80}")

# Calculate summary statistics
total_measurements = 0
for q_idx in range(num_questions):
    for probe_idx in range(num_probe_questions):
        probe_length = all_results['probe_lengths'][probe_idx]
        if probe_length > 1:  # Only count probes with actual tokens to predict
            total_measurements += (probe_length - 1) * 2  # truth + lie
            total_measurements += (probe_length - 1) * 2 * num_multipliers  # steering cases

print(f"Total token predictions recorded: {total_measurements}")
print(f"Data completeness check:")

# Check data completeness
no_steering_truth_valid = (~np.isnan(all_results['truth_log_probs_no_steering'])).sum()
no_steering_lie_valid = (~np.isnan(all_results['lie_log_probs_no_steering'])).sum()
steering_truth_valid = (~np.isnan(all_results['truth_log_probs'])).sum()
steering_lie_valid = (~np.isnan(all_results['lie_log_probs'])).sum()

print(f"  No-steering truth log-probs: {no_steering_truth_valid} valid entries")
print(f"  No-steering lie log-probs: {no_steering_lie_valid} valid entries") 
print(f"  Steering truth log-probs: {steering_truth_valid} valid entries")
print(f"  Steering lie log-probs: {steering_lie_valid} valid entries")

# Sample analysis: average effect of steering
if steering_truth_valid > 0 and steering_lie_valid > 0:
    print(f"\nSample steering effects (averaged across all valid measurements):")
    
    # Compare no-steering vs steering at different multipliers
    for mult_idx, multiplier in enumerate(multipliers):
        if multiplier == 0:  # Should match no-steering baseline
            continue
            
        truth_no_steer = all_results['truth_log_probs_no_steering']
        truth_steer = all_results['truth_log_probs'][:, :, mult_idx, :]
        
        # Find positions where both are valid
        valid_mask = ~(np.isnan(truth_no_steer) | np.isnan(truth_steer))
        if valid_mask.sum() > 0:
            truth_diff = (truth_steer - truth_no_steer)[valid_mask].mean()
            print(f"  Multiplier {multiplier:+.1f}: Truth log-prob change = {truth_diff:+.4f}")
        
        if mult_idx >= 3:  # Just show first few examples
            break

print(f"\n{'='*80}")
print(f"PROCESSING COMPLETE")
print(f"{'='*80}")

# Save results
np.save(output_path, all_results)
print(f"Results saved to: {output_path}")