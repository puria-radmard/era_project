import numpy as np
import pandas as pd
import json
import torch
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

from steering_vectors import ablation_then_addition_operator
from util.steering import LayerSpecificMultipliersSteeringVector
from model.load import load_model
from util.elicit import elicit_sequence_log_probs
from util.util import YamlConfig

# Load configuration
config_path = sys.argv[1]
args = YamlConfig(config_path)

# Extract parameters
system_prompt = args.system_prompt
question_instruction = args.question_instruction
questions_data_name = args.questions_data_name
model_name = args.model_name
prompt_idx = args.prompt_idx

chosen_layers = args.chosen_layers
multipliers = args.multipliers  # Should only include non-negative

multipliers = sorted(list(set(multipliers + list(map(lambda x: -x, multipliers)))))

initial_answers_args_name = args.initial_answers_args_name
probe_responses_args_name = args.probe_responses_args_name

# Set up paths
activation_analysis_path = os.path.join('lie_detector_results/e_activation_analysis', args.args_name, 'projection_results')
output_path = os.path.join('lie_detector_results/f_information_theoretic', args.args_name, 'clamped_prob_diffs')
os.makedirs(output_path, exist_ok=True)
args.save(output_path)

print("Loading stochastic answers...")
stochastic_answers_path = os.path.join('lie_detector_results/a_initial_answers', initial_answers_args_name, 'initial_answers_stochastic.csv')
stochastic_df = pd.read_csv(stochastic_answers_path)

# Filter to the specified prompt_idx
stochastic_df = stochastic_df[stochastic_df['prompt_idx'] == prompt_idx].copy()
print(f"Filtered to {len(stochastic_df)} rows for prompt_idx {prompt_idx}")

# Load probe results to get valid question IDs
probe_responses_path = os.path.join('lie_detector_results/b_probe_answers', probe_responses_args_name, 'probe_response.csv')
probe_results_df = pd.read_csv(probe_responses_path)

# Filter to valid questions (those that appear in probe data)
valid_question_ids = set(probe_results_df['question_idx'].unique())
stochastic_df = stochastic_df[stochastic_df['question_idx'].isin(valid_question_ids)].copy()
print(f"Filtered to {len(stochastic_df)} rows with valid question IDs")

# Load questions data
print("Loading test questions...")
initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')
qa_pairs = [(initial_questions_df['question'][idx].strip(), str(initial_questions_df['answer'][idx])) 
           for idx in range(len(initial_questions_df['question']))]

# Load steering vector information
print("Loading steering vector...")
pre_answer_vector_info = np.load(os.path.join(activation_analysis_path, 'prompted_projection_along_average_lie_vector.npy'), 
                                allow_pickle=True).item()

# Set up steering vector
gt_misaligned_cluster_mean = {layer_num: pre_answer_vector_info[layer_num]['lie_mean'] for layer_num in chosen_layers}
gt_aligned_cluster_mean = {layer_num: pre_answer_vector_info[layer_num]['truth_mean'] for layer_num in chosen_layers}
midpoints = {cl: (gt_misaligned_cluster_mean[cl] + gt_aligned_cluster_mean[cl]) / 2.0 for cl in chosen_layers}
half_lengths = {cl: (gt_misaligned_cluster_mean[cl] - gt_aligned_cluster_mean[cl]) / 2.0 for cl in chosen_layers}

layer_direction = {layer_num: torch.tensor(pre_answer_vector_info[layer_num]['direction']).cuda().to(torch.float16) 
                  for layer_num in chosen_layers}
steering_vector = LayerSpecificMultipliersSteeringVector(layer_direction, "decoder_block")

# Load model
print("Loading model...")
chat_wrapper = load_model(model_name, device='auto')

# Get unique questions and determine max response count for array sizing
unique_questions = stochastic_df['question_idx'].unique()
max_responses_per_question = max([
    len(stochastic_df[stochastic_df['question_idx'] == qid]['truth_answer'].tolist()) 
    for qid in unique_questions
])

print(f"Processing {len(unique_questions)} unique questions with {len(multipliers)} multipliers")
print(f"Max responses per question: {max_responses_per_question}")

# Initialize results structure
num_multipliers = len(multipliers)
num_questions = len(unique_questions)

all_results = {
    'multiplier': np.array(multipliers),
    'mean_truth_lie_diff': np.full(num_multipliers, np.nan),
    'std_truth_lie_diff': np.full(num_multipliers, np.nan),
    'question_truth_lie_diffs': np.full((num_multipliers, num_questions), np.nan),
    'question_truth_log_probs': np.full((num_multipliers, num_questions, max_responses_per_question), np.nan),
    'question_lie_log_probs': np.full((num_multipliers, num_questions, max_responses_per_question), np.nan),
}

print(f"\n{'='*80}")
print(f"PROCESSING {len(unique_questions)} QUESTIONS ACROSS {len(multipliers)} MULTIPLIERS")
print(f"Multipliers: {multipliers}")
print(f"{'='*80}")

# Main processing loop: questions outer, multipliers inner (for efficient caching)
for q_idx, question_idx in enumerate(unique_questions):

    print(f'Processing question {q_idx + 1} of {len(unique_questions)}. Iterating multipliers...')

    question = qa_pairs[question_idx][0]
    full_question = question + f' {question_instruction}'
    
    # Get truth and lie responses for this question
    question_data = stochastic_df[stochastic_df['question_idx'] == question_idx]
    truth_responses = question_data['truth_answer'].tolist()
    lie_responses = question_data['lie_answer'].tolist()
    
    # Create base cache for this question (system prompt + question)
    formatted_chat = chat_wrapper.format_chat(
        system_prompt=system_prompt,
        user_message=full_question,
        prefiller=""
    )
    
    # Tokenize the full question
    question_inputs = chat_wrapper.tokenizer(
        formatted_chat,
        return_tensors="pt",
        add_special_tokens=False
    ).to(chat_wrapper.device)
    
    # Create cache with the question
    with torch.no_grad():
        question_outputs = chat_wrapper.model(
            input_ids=question_inputs.input_ids,
            attention_mask=question_inputs.attention_mask,
            past_key_values=None,
            use_cache=True,
            return_dict=True
        )
    
    base_cache_info = {"cache": question_outputs.past_key_values}
    
    # Test each multiplier for this question
    for m_idx, multiplier in tqdm(enumerate(multipliers), total=len(multipliers)):
        # Calculate cluster mean for this multiplier
        cluster_mean = {cl: midpoints[cl] + multiplier * half_lengths[cl] for cl in chosen_layers}
        
        # Apply steering vector
        with steering_vector.apply(chat_wrapper.model, multiplier=cluster_mean, min_token_index=0, 
                                 operator=ablation_then_addition_operator()):
            
            # Calculate sequence log probabilities
            truth_log_probs = elicit_sequence_log_probs(chat_wrapper, base_cache_info, truth_responses)
            lie_log_probs = elicit_sequence_log_probs(chat_wrapper, base_cache_info, lie_responses)
            
            # Calculate truth-lie difference
            avg_truth_log_prob = truth_log_probs.mean().item() if len(truth_log_probs) > 0 else 0.0
            avg_lie_log_prob = lie_log_probs.mean().item() if len(lie_log_probs) > 0 else 0.0
            truth_lie_diff = avg_truth_log_prob - avg_lie_log_prob
            
            # Store results
            all_results['question_truth_lie_diffs'][m_idx, q_idx] = truth_lie_diff
            
            # Store individual log probs (pad with NaN if needed)
            n_truth = len(truth_log_probs)
            n_lie = len(lie_log_probs)
            all_results['question_truth_log_probs'][m_idx, q_idx, :n_truth] = truth_log_probs.cpu().numpy()
            all_results['question_lie_log_probs'][m_idx, q_idx, :n_lie] = lie_log_probs.cpu().numpy()
    
    # Log progress for this question
    if (q_idx + 1) % 10 == 0 or q_idx == len(unique_questions) - 1:
        print(f"\nCompleted question {q_idx + 1}/{len(unique_questions)} (idx={question_idx})")
        sample_diffs = all_results['question_truth_lie_diffs'][:, q_idx]
        print(f"  Truth-lie diffs across multipliers: min={np.nanmin(sample_diffs):.3f}, max={np.nanmax(sample_diffs):.3f}")

print(f"\n{'='*80}")
print(f"QUESTION PROCESSING COMPLETE - COMPUTING AGGREGATE STATISTICS")
print(f"{'='*80}")

# Calculate aggregate statistics across all questions
for m_idx, multiplier in enumerate(multipliers):
    question_diffs = all_results['question_truth_lie_diffs'][m_idx, :]
    valid_diffs = question_diffs[~np.isnan(question_diffs)]
    
    all_results['mean_truth_lie_diff'][m_idx] = np.mean(valid_diffs)
    all_results['std_truth_lie_diff'][m_idx] = np.std(valid_diffs)

# Create mapping from question_idx to question type
question_idx_to_type = {}
for i, question_idx in enumerate(unique_questions):
    question_idx_to_type[question_idx] = initial_questions_df.loc[question_idx, 'type']

# Get unique question types and create type-specific results
question_types = initial_questions_df['type'].unique()
type_results = {}

for question_type in question_types:
    # Find questions of this type
    type_question_indices = [i for i, qid in enumerate(unique_questions) 
                           if question_idx_to_type[qid] == question_type]
    
    if len(type_question_indices) == 0:
        continue
        
    # Calculate mean truth-lie diff for this type across multipliers
    type_means = []
    for m_idx in range(len(multipliers)):
        type_diffs = all_results['question_truth_lie_diffs'][m_idx, type_question_indices]
        valid_type_diffs = type_diffs[~np.isnan(type_diffs)]
        if len(valid_type_diffs) > 0:
            type_means.append(np.mean(valid_type_diffs))
        else:
            type_means.append(np.nan)
    
    type_results[question_type] = {
        'means': np.array(type_means),
        'n_questions': len(type_question_indices)
    }

print(f"Found {len(question_types)} question types:")
for qtype in question_types:
    if qtype in type_results:
        print(f"  {qtype}: {type_results[qtype]['n_questions']} questions")

# Create final plot with question type breakdowns
plt.figure(figsize=(14, 8))

# Plot fainter lines for each question type
colors = plt.cm.Set3(np.linspace(0, 1, len(question_types)))
for i, question_type in enumerate(question_types):
    if question_type in type_results:
        type_data = type_results[question_type]
        plt.plot(all_results['multiplier'], type_data['means'], 
                color=colors[i], linewidth=1, alpha=0.6, marker='o', markersize=4,
                label=f'{question_type} (n={type_data["n_questions"]})')

# Plot main aggregate line on top (bold)
plt.plot(all_results['multiplier'], all_results['mean_truth_lie_diff'], 
         'k-o', linewidth=3, markersize=8, label=f'Overall (n={len(unique_questions)})')

plt.xlabel('Steering Multiplier', fontsize=12)
plt.ylabel('Truth-Lie Log Probability Difference', fontsize=12)
plt.title('Effect of Steering Vector Magnitude on Truth vs Lie Probability by Question Type', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
plt.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_path, 'steering_effects.png'), dpi=300, bbox_inches='tight')
plt.show()

# Save final results
np.save(os.path.join(output_path, 'steering_results_final.npy'), all_results)
np.save(os.path.join(output_path, 'steering_results_by_type.npy'), type_results)

print(f"\n{'='*80}")
print(f"ANALYSIS COMPLETE")
print(f"{'='*80}")
print(f"Results saved to: {output_path}")
print(f"Processed {len(unique_questions)} questions × {len(multipliers)} multipliers = {len(unique_questions) * len(multipliers)} total measurements")
print("\nSummary Results:")
