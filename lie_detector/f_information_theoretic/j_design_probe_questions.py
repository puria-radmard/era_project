import json, copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
import sys
import random

from steering_vectors import ablation_then_addition_operator
from util.steering import LayerSpecificMultipliersSteeringVector
from model.load import load_model
from util.util import YamlConfig

from lie_detector.f_information_theoretic.z_util import prob_mode

# Configuration
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
chosen_multiplier = args.chosen_multiplier
chosen_temperature = args.chosen_temperature
num_inits = args.num_inits_optimisation

# Optimization parameters
template = "Do {adjective} {noun1} {verb} {noun2} {adverb}? Answer with yes or no."
optimisation_target = "truth"  # Pipeline to optimize for
max_cycles = 3
word_positions = ['adjective', 'noun1', 'verb', 'noun2', 'adverb']

aux_target = "truth" if optimisation_target == "lie" else "lie"

print(f"Template: {template}")
print(f"Optimizing for: {optimisation_target} pipeline")

# Set up paths
activation_analysis_path = os.path.join('lie_detector_results/e_activation_analysis', args.args_name, 'projection_results')
save_base = os.path.join('lie_detector_results/f_information_theoretic', args.args_name, 'bald_estimation')
os.makedirs(save_base, exist_ok=True)
args.save(save_base)

# Load word table
word_table_path = 'data/probe_question_design/word_table.csv'  # Adjust path as needed
word_df = pd.read_csv(word_table_path)
word_table = {}
for col in word_positions:
    # Remove NaN values and convert to list
    words = word_df[col].dropna().tolist()
    word_table[col] = words
    print(f"{col}: {len(words)} words")

print("Loading initial answers and filtering...")
initial_answers_path = os.path.join('lie_detector_results/a_initial_answers', initial_answers_args_name, 'initial_answers.csv')
response_data = pd.read_csv(initial_answers_path)

# Load questions and prompts
initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')

with open('data/all_prompts.json', 'r') as f:
    prompts = json.load(f)
    optimisation_prompt = prompts[f'{optimisation_target}_prompts'][prompt_index]
    aux_prompt = prompts[f'{aux_target}_prompts'][prompt_index]

# Filter trainable questions
if limit_to_lying:
    knows_answer = response_data[(response_data['knows_answer']) & (response_data['told_truth'])]
    trainable_answers = knows_answer[(knows_answer['prompt_idx'] == prompt_index) & (knows_answer['told_lie'])]
else:
    trainable_answers = response_data
trainable_questions_idxs = trainable_answers['question_idx'].unique()

# Subsample questions
trainable_questions_idxs = trainable_questions_idxs[:10]            
print('REDUCING trainable_questions_idxs TO JUST SPORTS QUESTIONS!!')
print(f"Using {len(trainable_questions_idxs)} initial questions")

# Load model and tokenizer
print("Loading model...")
chat_wrapper = load_model(model_name, device='auto')

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

# Prepare caches for optimization target pipeline only
print(f"Preparing {optimisation_target} caches...")
truth_caches = []
lie_caches = []

for question_idx in tqdm(trainable_questions_idxs):

    question = initial_questions_df['question'][question_idx].strip()
    response_row = trainable_answers[trainable_answers['question_idx'] == question_idx].iloc[0]
    
    truth_cache_info = chat_wrapper.create_prompt_cache(
        system_prompt=system_prompt,
        in_context_questions=[f'{optimisation_prompt} {question} {question_instruction}'],
        in_context_answers=[response_row.truth_answer]
    )
    truth_caches.append(truth_cache_info["cache"])

    lie_cache_info = chat_wrapper.create_prompt_cache(
        system_prompt=system_prompt,
        in_context_questions=[f'{aux_prompt} {question} {question_instruction}'],
        in_context_answers=[response_row.lie_answer]
    )
    lie_caches.append(lie_cache_info["cache"])

print(f"Created {len(truth_caches)} caches")

# A-score evaluation function
def evaluate_a_score(probe_question):
    """Evaluate A-score for a probe question using the truth pipeline"""
    
    # Format probe for model input
    formatted_probe = chat_wrapper.format_chat(
        system_prompt="",
        user_message=probe_question,
        prefiller='',
    )
    
    # Get tokenized length
    probe_tokens = chat_wrapper.tokenizer.encode(formatted_probe, add_special_tokens=False)
    
    all_unsteered_truth_projections_for_probe = np.full((len(truth_caches), len(chosen_layers), len(probe_tokens)-1), np.nan)
    all_unsteered_lie_projections_for_probe = np.full((len(truth_caches), len(chosen_layers), len(probe_tokens)-1), np.nan)

    all_unsteered_entropies = np.full((len(truth_caches), len(probe_tokens)-1), np.nan)
    steered_entropies = {
        +1: np.full((len(truth_caches), len(probe_tokens)-1), np.nan),
        -1: np.full((len(truth_caches), len(probe_tokens)-1), np.nan),
    }
    
    a_values = []
    
    # Loop over initial questions first to generate the relevant logprobs/projections
    for i_cache, (truth_cache, lie_cache) in enumerate(zip(truth_caches, lie_caches)):

        # Get unsteered probabilities
        with torch.no_grad():
            probe_inputs = chat_wrapper.tokenizer(
                formatted_probe,
                return_tensors="pt", 
                add_special_tokens=False
            ).to(chat_wrapper.device)
            
            unsteered_truth_outputs = chat_wrapper.model(
                input_ids=probe_inputs.input_ids,
                attention_mask=probe_inputs.attention_mask,
                past_key_values=copy.deepcopy(truth_cache),
                use_cache=False,
                return_dict=True,
                output_hidden_states=True
            )
            
            unsteered_lie_outputs = chat_wrapper.model(
                input_ids=probe_inputs.input_ids,
                attention_mask=probe_inputs.attention_mask,
                past_key_values=copy.deepcopy(lie_cache),
                use_cache=False,
                return_dict=True,
                output_hidden_states=True
            )

            # Extract unsteered log probabilities
            logits = (unsteered_truth_outputs if optimisation_target == 'truth' else unsteered_lie_outputs).logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = log_probs.exp()
            unsteered_entropy = -(probs[:-1] * log_probs[:-1]).sum(-1).cpu().numpy()
            all_unsteered_entropies[i_cache] = unsteered_entropy
            
            for i_layer, layer in enumerate(chosen_layers):
                
                truth_hidden_states = unsteered_truth_outputs.hidden_states[layer + 1][0]  # Remove batch dim
                direction_vec = layer_direction[layer]
                projections = torch.matmul(truth_hidden_states[:-1], direction_vec)  # Skip last position
                all_unsteered_truth_projections_for_probe[i_cache, i_layer] = projections.cpu().numpy()

                lie_hidden_states = unsteered_lie_outputs.hidden_states[layer + 1][0]  # Remove batch dim
                direction_vec = layer_direction[layer]
                projections = torch.matmul(lie_hidden_states[:-1], direction_vec)  # Skip last position
                all_unsteered_lie_projections_for_probe[i_cache, i_layer] = projections.cpu().numpy()
        
        # Get steered probabilities for both directions
        for mult_dir in [-1, +1]:
            
            cluster_mean = {cl: midpoints[cl] + mult_dir * chosen_multiplier * half_lengths[cl] for cl in chosen_layers}
            
            with steering_vector.apply(chat_wrapper.model, multiplier=cluster_mean, min_token_index=0,
                                        operator=ablation_then_addition_operator()):
                with torch.no_grad():
                    steered_outputs = chat_wrapper.model(
                        input_ids=probe_inputs.input_ids,
                        attention_mask=probe_inputs.attention_mask,
                        past_key_values=copy.deepcopy(truth_cache if optimisation_target == 'truth' else lie_cache),
                        use_cache=False,
                        return_dict=True
                    )
                    
                    steered_logits = steered_outputs.logits[0]
                    steered_log_probs_tensor = torch.log_softmax(steered_logits, dim=-1)
                    steered_probs = steered_log_probs_tensor.exp()

                    steered_entropy = -(steered_probs[:-1] * steered_log_probs_tensor[:-1]).sum(-1).cpu().numpy()
                    steered_entropies[mult_dir][i_cache] = steered_entropy


    # Calculate p(z | x_{<t}) for all initial questions here...
    truth_means = all_unsteered_truth_projections_for_probe.mean(0)
    truth_stds = all_unsteered_truth_projections_for_probe.std(0)
    lie_means = all_unsteered_lie_projections_for_probe.mean(0)
    lie_stds = all_unsteered_lie_projections_for_probe.std(0)

    all_posterior_probs = [
        prob_mode(
            all_unsteered_truth_projections_for_probe[..., tpos],
            truth_means[..., tpos], truth_stds[..., tpos],
            lie_means[..., tpos], lie_stds[..., tpos]
        ) for tpos in range(len(probe_tokens) - 1)
    ]

    # Loop over initial questions again to actually calculate the BALD value
    for i_cache in range(len(truth_caches)):
        
        # Compute A_t values for each token position
        a_t_values = []
        for token_pos in range(len(probe_tokens) - 1):
            
            # Get entropy values for MI computation
            entropy_unsteered = all_unsteered_entropies[i_cache, token_pos].item()
            
            # Second term: mixture of steered probs per layer
            # Choose 
            p_truth = all_posterior_probs[token_pos][i_cache].max() if optimisation_target == 'truth' else all_posterior_probs[token_pos][i_cache].min()

            temp = chosen_temperature
            p_truth_temp = np.power(p_truth, 1.0 / temp)
            p_lie_temp = np.power(1 - p_truth, 1.0 / temp)
            p_truth_tempered = p_truth_temp / (p_truth_temp + p_lie_temp)
            p_lie_tempered = 1 - p_truth_tempered
            
            truth_steered_entropy = steered_entropies[-1][i_cache, token_pos].item()  # Steered toward truth  
            lie_steered_entropy = steered_entropies[+1][i_cache, token_pos].item()    # Steered toward lie
            
            entropy_steered = p_truth_tempered * truth_steered_entropy + p_lie_tempered * lie_steered_entropy
            
            # Mutual information: I(x_t; z | x_{<t}) = H(x_t | x_{<t}) - H(x_t | z, x_{<t})
            a_t = entropy_unsteered - entropy_steered
            a_t_values.append(a_t)
        
        # Sum across tokens for this cache
        cache_a_score = sum(a_t_values)
        a_values.append(cache_a_score)

    # Clean up GPU memory
    torch.cuda.empty_cache()
    
    # Average across caches
    return np.mean(a_values), np.std(a_values)

# Initialize optimization
print("Starting coordinate ascent optimization...")

def construct_probe(words):
    """Construct probe question from template and word list"""
    word_dict = {pos: word for pos, word in zip(word_positions, words)}
    return template.format(**word_dict)


trajectory_probes = []
trajectory_scores = []
trajectory_scores_stds = []
trajectory_words = []
trajectory_init_ids = []


for initialisation_idx in range(num_inits * 2):

    current_words = []
    for pos in word_positions:
        current_words.append(random.choice(word_table[pos]))

    initial_probe = construct_probe(current_words)
    print(f"Initial probe: {initial_probe}")

    # Evaluate initial probe
    best_score, best_score_std = evaluate_a_score(initial_probe)
    print(f"Initial A-score: {best_score:.6f}")

    # Initialize trajectory tracking
    trajectory_probes.append(initial_probe)  
    trajectory_scores.append(best_score)
    trajectory_scores_stds.append(best_score_std)
    trajectory_words.append(current_words.copy())
    trajectory_init_ids.append(initialisation_idx)

    # Coordinate ascent optimization loop
    for cycle in range(max_cycles):

        print(f"\n--- Cycle {cycle + 1}/{max_cycles} ---")
        changes_made = False
        
        for position_idx, position in enumerate(word_positions):
            print(f"Optimizing position {position_idx + 1}/{len(word_positions)}: {position}")
            
            # Try all alternatives for this position
            alternatives = word_table[position]
            print(f"Testing {len(alternatives)} alternatives...")

            alt_idx_s = list(range(len(alternatives)))
            random.shuffle(alt_idx_s)
            
            for alt_idx in tqdm(alt_idx_s, desc=f"Testing {position}"):

                alternative_word = alternatives[alt_idx]
                
                if alternative_word == current_words[position_idx]:
                    continue  # Skip current word
                
                # Create test probe
                test_words = current_words.copy()
                test_words[position_idx] = alternative_word
                test_probe = construct_probe(test_words)
                
                # Evaluate
                test_score, test_score_std = evaluate_a_score(test_probe)

                replacement_condition = test_score > best_score if initialisation_idx % 2 == 0 else test_score < best_score
                if replacement_condition:
                    
                    old_word = current_words[position_idx]

                    best_score = test_score
                    current_words[position_idx] = alternative_word
                    print(f"  New best: {old_word} → {alternative_word} (score: {test_score:.6f})")

                    trajectory_probes.append(construct_probe(current_words))
                    trajectory_scores.append(best_score)
                    trajectory_scores_stds.append(test_score_std)
                    trajectory_words.append(current_words.copy())
                    trajectory_init_ids.append(initialisation_idx)

                    changes_made = True
                    
                    # Live save to df
                    probes_df = pd.DataFrame({
                        'probe_type': [f'Optimised_{init_idx}' for init_idx in trajectory_init_ids],
                        'probe': trajectory_probes,
                        'a_score': trajectory_scores,
                        'a_score_std': trajectory_scores_stds
                    })
                    probes_csv_path = os.path.join(save_base, 'optimized_probe_questions.csv')
                    probes_df.to_csv(probes_csv_path, index=False)

                    break   # Move onto next position
        
        if not changes_made:
            print("No changes made in this cycle. Early stopping.")
            break

        print(f"\nOptimization complete!")
        current_probe = construct_probe(current_words)
        print(f"Cycle final probe: {current_probe}")
        print(f"Cycle final A-score: {best_score:.6f}")
        print(f"Trajectory length: {len(trajectory_probes)}")

        # Save results
        trajectory_data = {
            'probes': np.array(trajectory_probes),
            'scores': np.array(trajectory_scores),
            'words': np.array(trajectory_words), 
            'word_positions': np.array(word_positions),
            'template': template,
            'optimisation_target': optimisation_target,
            'initial_questions_used': np.array(trainable_questions_idxs)
        }

        # Save trajectory data
        np.save(os.path.join(save_base, 'coordinate_ascent_trajectory.npy'), trajectory_data)
        print(f"Trajectory data saved to: {os.path.join(save_base, 'coordinate_ascent_trajectory.npy')}")

        # Save optimized probes to CSV
        probes_df = pd.DataFrame({
            'probe_type': [f'Optimised_{init_idx}' for init_idx in trajectory_init_ids],
            'probe': trajectory_probes,
            'a_score': trajectory_scores,
            'a_score_std': trajectory_scores_stds
        })
        probes_csv_path = os.path.join(save_base, 'optimized_probe_questions.csv')
        probes_df.to_csv(probes_csv_path, index=False)
        print(f"Optimized probes saved to: {probes_csv_path}")

        print("\nCycle trajectory summary:")
        for i, (probe, score) in enumerate(zip(trajectory_probes, trajectory_scores)):
            print(f"{i}: {score:.6f} - {probe}")