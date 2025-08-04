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

# Get unique questions and determine max response count for array sizing
unique_questions = stochastic_df['question_idx'].unique()
max_responses_per_question = max([
    len(stochastic_df[stochastic_df['question_idx'] == qid]['truth_answer'].tolist()) 
    for qid in unique_questions
])


# Save final results
all_results = np.load(os.path.join(output_path, 'steering_results_final.npy'), allow_pickle=True).item()

# Create mapping from question_idx to question type
question_idx_to_type = {}
for i, question_idx in enumerate(unique_questions):
    question_idx_to_type[question_idx] = initial_questions_df.loc[question_idx, 'type']

# Get unique question types and create type-specific results
question_types = initial_questions_df['type'].unique()

try:
    type_results = np.load(os.path.join(output_path, 'steering_results_by_type.npy'), allow_pickle=True).item()

except FileNotFoundError:

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



print(f"\n{'='*80}")
print(f"QUESTION PROCESSING COMPLETE - COMPUTING AGGREGATE STATISTICS")
print(f"{'='*80}")

# Calculate aggregate statistics across all questions
for m_idx, multiplier in enumerate(multipliers):
    question_diffs = all_results['question_truth_lie_diffs'][m_idx, :]
    valid_diffs = question_diffs[~np.isnan(question_diffs)]
    
    all_results['mean_truth_lie_diff'][m_idx] = np.mean(valid_diffs)
    all_results['std_truth_lie_diff'][m_idx] = np.std(valid_diffs)




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
                color=colors[i], linewidth=1, marker='o', markersize=4,
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
plt.savefig(os.path.join(output_path, 'steering_effects_posthoc.png'), dpi=300, bbox_inches='tight')
plt.show()


print(f"\n{'='*80}")
print(f"ANALYSIS COMPLETE")
print(f"{'='*80}")
print(f"Results saved to: {output_path}")
print(f"Processed {len(unique_questions)} questions × {len(multipliers)} multipliers = {len(unique_questions) * len(multipliers)} total measurements")
print("\nSummary Results:")
