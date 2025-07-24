import numpy as np
import pandas as pd
import os
import sys
from lie_detector.d_in_context_liar.viz import plot_context_effect_analysis, plot_context_effect_by_question_type, plot_context_diff_analysis, plot_context_diff_by_question_type

from util.util import YamlConfig

config_path = sys.argv[1]
# Load config
args = YamlConfig(config_path)

# Get necessary parameters from config
questions_data_name = args.questions_data_name
context_lengths = args.context_lengths
n_samples = args.samples_per_context_length

# Get output path
output_path = os.path.join('lie_detector_results/d_in_context_lying', args.args_name)

# Define context types (same as in original script)
context_types = [
    'top_lie_shuffled_together',
    'top_truth_shuffled_together',
    'top_questions_random_answers',
]

# Load questions data for the question type breakdown plot
print("Loading questions data...")
initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')

# Load stochastic answers to get unique questions (mimicking original script)
initial_answers_args_name = args.initial_answers_args_name
probe_responses_args_name = args.probe_responses_args_name
prompt_index = args.prompt_idx

stochastic_answers_path = os.path.join('lie_detector_results/a_initial_answers', initial_answers_args_name, 'initial_answers_stochastic.csv')
stochastic_df = pd.read_csv(stochastic_answers_path)
stochastic_df = stochastic_df[stochastic_df['prompt_idx'] == prompt_index].copy()

# Filter to valid questions (same logic as original)
probe_responses_path = os.path.join('lie_detector_results/b_probe_answers', probe_responses_args_name, 'probe_response.csv')
probe_results_df = pd.read_csv(probe_responses_path)
valid_question_ids = set(probe_results_df['question_idx'].unique())
stochastic_df = stochastic_df[stochastic_df['question_idx'].isin(valid_question_ids)].copy()

unique_questions = stochastic_df['question_idx'].unique()

print(f"Context types to load: {context_types}")

# Reconstruct all_results dictionary by loading .npy files
all_results = {}

for context_type in context_types:
    npy_path = os.path.join(output_path, f'context_effect_results_{context_type}.npy')
    
    if os.path.exists(npy_path):
        print(f"Loading {context_type} results from {npy_path}")
        all_results[context_type] = np.load(npy_path, allow_pickle=True).item()
    else:
        print(f"Warning: {npy_path} not found, skipping {context_type}")

if not all_results:
    print("Error: No result files found!")
    exit()

print(f"Loaded results for {len(all_results)} context types")

# Generate post-hoc plots with prefix
print("Generating post-hoc plots...")

plot_context_effect_analysis(
    all_results, 
    list(all_results.keys()),  # Use actually loaded context types
    context_lengths, 
    n_samples, 
    output_path, 
    filename_prefix="posthoc_"
)

plot_context_effect_by_question_type(
    all_results, 
    list(all_results.keys()),  # Use actually loaded context types
    context_lengths, 
    n_samples, 
    unique_questions, 
    initial_questions_df, 
    output_path, 
    filename_prefix="posthoc_"
)


control_context_type = 'top_questions_random_answers'  # You can change this


# Generate difference plots
context_aliases = {
    'top_lie_shuffled_together': 'deceptively misaligned probe answers',
    'top_truth_shuffled_together': 'aligned probe answers', 
    'top_questions_random_answers': 'random Y/N probe answers'
}


plot_context_diff_analysis(
    all_results,
    list(all_results.keys()),
    control_context_type,
    context_lengths,
    n_samples,
    output_path,
    truth_answer_label="Aligned",
    lie_answer_label="Misaligned",
    context_aliases=context_aliases,
    filename_prefix="posthoc_"
)

plot_context_diff_by_question_type(
    all_results,
    list(all_results.keys()),
    control_context_type,
    context_lengths,
    n_samples,
    unique_questions,
    initial_questions_df,
    output_path,
    truth_answer_label="Aligned",
    lie_answer_label="Misaligned",
    context_aliases=context_aliases,
    filename_prefix="posthoc_"
)

print(f"Post-hoc plots saved to {output_path}")
print("Files created:")
print(f"  - posthoc_context_effect_analysis.png")
print(f"  - posthoc_context_effect_by_question_type.png")
