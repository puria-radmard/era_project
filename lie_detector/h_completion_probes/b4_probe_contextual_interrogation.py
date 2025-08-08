import json, copy
import numpy as np
import pandas as pd
import torch

from model.load import load_model
from util.util import YamlConfig
from util.lying_context import get_context_yn
from util.elicit import generate_discriminative_sequence
from steering_vectors import addition_operator, SteeringVector
import sys, os


# Main script logic
config_path = sys.argv[1]
args = YamlConfig(config_path)

model_name = args.model_name
max_generation_tokens = args.max_generation_tokens
question_steering_magnitudes = args.question_steering_magnitudes
chosen_layers = args.chosen_layers

# Probe-related args (from the attached script pattern)
probe_responses_args_name = args.probe_responses_args_name
probe_analysis_args_name = args.probe_analysis_args_name
probe_file_name = args.probe_file_name
excluded_probe_types = args.excluded_probe_types
sorting_key = args.icl_sorting_key

# Save to new subdirectory
save_base = os.path.join('lie_detector_results/i_discriminative_tokens', args.args_name)
os.makedirs(save_base, exist_ok=True)
args.save(save_base)

discriminative_sequences_path = os.path.join(save_base, 'probe_contextual_discriminative_sequences.csv')

# Load probe questions and discriminability data (like the attached script)
print("Loading probe discriminability data...")
probes_df_original = pd.read_csv(f'data/probe_questions/{probe_file_name}.csv')
probes_df = probes_df_original[~probes_df_original['probe_type'].isin(excluded_probe_types)]
probe_questions = probes_df['probe'].tolist()
print(f"Using {len(probe_questions)} probe questions (excluded: {excluded_probe_types})")

# Load discriminability results
discriminability_data_path = os.path.join('lie_detector_results/c_probe_discrimination', probe_analysis_args_name, 'discriminability_results.json')
with open(discriminability_data_path, 'r') as f:
    discriminability_data = json.load(f)

# Get top discriminative probes
probe_results = discriminability_data['probe_results']
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

# Load steering vector
question_directions_path = os.path.join('lie_detector_results/i_discriminative_tokens', args.args_name, 'question_expectation_directions.npy')
print(f"Loading steering vectors from: {question_directions_path}")

question_steering_vector = np.load(question_directions_path)
layer_direction = {layer_num: torch.tensor(question_steering_vector[layer_num + 1]).cuda().to(torch.float16) for layer_num in chosen_layers}
steering_vector = SteeringVector(layer_direction, "decoder_block")

# Use maximum available N (no iteration over context lengths)
max_N = len(valid_probe_results)  # Use all available discriminative probes
print(f"Using maximum context length N={max_N}")

# Get context materials (one sample only)
print("Getting probe context materials...")
context_materials = get_context_yn(max_N, valid_probe_results, probes_df_original)

# Initialize results storage
discriminative_df_rows = []

print("Starting probe-context discriminative generation...")

print(f"\n{'='*60}")
print(f"CREATING DISCRIMINATIVE CONTEXT CACHES")
print(f"{'='*60}")

# Get truth-favoring context 
truth_questions, truth_answers = context_materials['top_truth_shuffled_together']
print(f"Truth context has {len(truth_questions)} probe Q&A pairs")
print(f"Sample truth context - Q: {truth_questions[0][:50]}...")
print(f"Sample truth context - A: {truth_answers[0]}")

# Get lie-favoring context
lie_questions, lie_answers = context_materials['top_lie_shuffled_together']  
print(f"Lie context has {len(lie_questions)} probe Q&A pairs")
print(f"Sample lie context - Q: {lie_questions[0][:50]}...")
print(f"Sample lie context - A: {lie_answers[0]}")


# Create discriminative caches
truth_cache = chat_wrapper.create_prompt_cache(
    system_prompt="",
    in_context_questions=[f"Here's a question: {q}" for q in truth_questions],
    in_context_answers=truth_answers
)

lie_cache = chat_wrapper.create_prompt_cache(
    system_prompt="",
    in_context_questions=[f"Here's a question: {q}" for q in lie_questions],
    in_context_answers=lie_answers
)
print("Created discriminative caches with probe contexts")

# Test different steering and maximization combinations
for lie_maximisation in [True, False]:
    print(f"\n  Lie maximisation: {lie_maximisation}")

    for steering_magnitude in question_steering_magnitudes:
        print(f"    Question steering magnitude: {steering_magnitude}")

        with steering_vector.apply(chat_wrapper.model, multiplier=steering_magnitude, min_token_index=0, operator=addition_operator()):

            # Generate discriminative sequence with probe context
            generated_sequence, sequence_progression, discrimination_scores = generate_discriminative_sequence(
                chat_wrapper=chat_wrapper,
                truth_cache=copy.deepcopy(truth_cache),
                lie_cache=copy.deepcopy(lie_cache),
                max_tokens=max_generation_tokens,
                lie_maximise=lie_maximisation,
                initial_text="Here's a question: ",
                stopping_string = "?"
            )
            
            print("Generated sequence: '"+ generated_sequence.replace(chr(10), '\\n') + "'")
            print(f"    Average discrimination: {sum(discrimination_scores)/len(discrimination_scores) if discrimination_scores else 0:.4f}")
            
            # Add rows to dataframe
            for token_pos, score in enumerate(discrimination_scores):
                discriminative_df_rows.append({
                    'context_length': max_N,
                    'lie_maximisation': lie_maximisation,
                    'steering_magnitude': steering_magnitude,
                    'token_position': token_pos + 1,
                    'discrimination_value': score,
                    'sequence_so_far': sequence_progression[token_pos].replace('\n', '\\n'),
                })

            # Save after each generation
            discriminative_df = pd.DataFrame(discriminative_df_rows)
            discriminative_df.to_csv(discriminative_sequences_path, index=False)

print("\nProbe-context steering complete!")
print(f"Results saved to: {discriminative_sequences_path}")
print(f"Generated {len(discriminative_df_rows)} token-level results")
print(f"Used context length of {max_N} discriminative probe examples")