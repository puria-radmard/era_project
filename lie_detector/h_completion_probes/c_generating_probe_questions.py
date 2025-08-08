import json, copy
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model.load import load_model
from util.util import YamlConfig
from util.lying_context import get_context_yn
from util.elicit import generate_discriminative_sequence
from steering_vectors import addition_operator, SteeringVector
import sys, os, math


# Main script logic
config_path = sys.argv[1]
args = YamlConfig(config_path)

model_name = args.model_name
max_generation_tokens = args.max_generation_tokens
question_steering_magnitudes = args.question_steering_magnitudes
chosen_layers = args.chosen_layers

# Probe-related args
probe_responses_args_name = args.probe_responses_args_name
probe_analysis_args_name = args.probe_analysis_args_name
probe_file_name = args.probe_file_name
excluded_probe_types = args.excluded_probe_types

excluded_probe_indices = args.excluded_probe_indices

# Context iteration args (like original script)
context_lengths = args.context_lengths
n_samples = args.samples_per_context_length

# Save to subdirectory
save_base = os.path.join('lie_detector_results/i_discriminative_tokens', args.args_name)
os.makedirs(save_base, exist_ok=True)
args.save(save_base)

discriminative_sequences_path = os.path.join(save_base, 'new_probe_questions.csv')

# Load probe questions and discriminability data
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
                      if result['probe_type'] not in excluded_probe_types
                      and i not in excluded_probe_indices]

valid_probe_results.sort(key=lambda x: abs(x[1]['effect_size']), reverse=True)
valid_probe_results = valid_probe_results[:max(context_lengths)]

# Not sorting by anything and randomly selecting on each context
print(f"Found {len(valid_probe_results)} valid discriminative probes")

# Load model
chat_wrapper = load_model(model_name, device='auto')

# Load steering vector
question_directions_path = os.path.join('lie_detector_results/i_discriminative_tokens', args.args_name, 'question_expectation_directions.npy')
print(f"Loading steering vectors from: {question_directions_path}")

question_steering_vector = np.load(question_directions_path)
layer_direction = {layer_num: torch.tensor(question_steering_vector[layer_num + 1]).cuda().to(torch.float16) for layer_num in chosen_layers}
steering_vector = SteeringVector(layer_direction, "decoder_block")

# Context types to test
context_types = [
    'top_lie_shuffled_together',
    'top_truth_shuffled_together',
    'top_questions_random_answers'
]

# Initialize results storage
discriminative_df_rows = []

# Process each context length
context_lengths_desc = sorted(context_lengths, reverse=False)

for iN, N in enumerate(context_lengths_desc):
    print(f"\n{'='*80}")
    print(f"TESTING CONTEXT LENGTH N={N}")
    print(f"{'='*80}")

    # Generate all context materials for this sample
    n_samples_eff = min(n_samples, math.perm(context_lengths[-1], int(1)))
    all_context_materials = [get_context_yn(N, valid_probe_results, probes_df_original, randomly_select=True) for _ in range(n_samples_eff)]
    
    for sample_idx in tqdm(range(n_samples_eff)):
        
        context_materials = all_context_materials[sample_idx]

        assert N > 0
        
        # Get truth-favoring and lie-favoring contexts
        truth_questions, truth_answers = context_materials['top_truth_shuffled_together']
        lie_questions, lie_answers = context_materials['top_lie_shuffled_together']
        
        print(f"Truth context has {len(truth_questions)} probe Q&A pairs")
        print(f"Lie context has {len(lie_questions)} probe Q&A pairs")
        
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
        
        # Test different steering and maximization combinations
        for lie_maximisation in [True, False]:
            print(f"\tLie maximisation: {lie_maximisation}")

            for steering_magnitude in question_steering_magnitudes:
                print(f"\t\tQuestion steering magnitude: {steering_magnitude}")

                with steering_vector.apply(chat_wrapper.model, multiplier=steering_magnitude, min_token_index=0, operator=addition_operator()):

                    # Generate discriminative sequence with probe context
                    generated_sequence, sequence_progression, discrimination_scores, full_token_count = generate_discriminative_sequence(
                        chat_wrapper=chat_wrapper,
                        truth_cache=copy.deepcopy(truth_cache),
                        lie_cache=copy.deepcopy(lie_cache),
                        max_tokens=max_generation_tokens,
                        lie_maximise=lie_maximisation,
                        initial_text="Here's a question: ",
                        stopping_string="?",
                        do_discriminative=True
                    )

                    print("\t\t\tGenerated: '" + generated_sequence.replace(chr(10), '\\n') + "'")
                    avg_discrimination = sum(discrimination_scores)/len(discrimination_scores) if discrimination_scores else 0.0
                    print(f"\t\t\tAvg discrimination: {avg_discrimination:.4f}")
                    
                    # Add one row per generation
                    discriminative_df_rows.append({
                        'context_length': N,
                        'sample_idx': sample_idx,
                        'lie_maximisation': lie_maximisation,
                        'steering_magnitude': steering_magnitude,
                        'generated_question': generated_sequence.replace('\n', '\\n'),
                        'question_achieved': not full_token_count
                    })

                    # Save after each generation
                    discriminative_df = pd.DataFrame(discriminative_df_rows)
                    discriminative_df.to_csv(discriminative_sequences_path, index=False)

print("\nIterative probe-context steering complete!")
print(f"Results saved to: {discriminative_sequences_path}")
print(f"Generated {len(discriminative_df_rows)} new probe questions")
print(f"Tested {len(context_lengths)} context lengths with {n_samples} samples each")