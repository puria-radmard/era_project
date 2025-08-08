import json, copy
import numpy as np
import pandas as pd
import torch

from model.load import load_model
from util.util import YamlConfig
import sys, os

from util.elicit import generate_discriminative_sequence

from steering_vectors import addition_operator, SteeringVector


# Main script logic
config_path = sys.argv[1]
args = YamlConfig(config_path)

model_name = args.model_name
max_generation_tokens = args.max_generation_tokens
question_steering_magnitudes = args.question_steering_magnitudes
chosen_layers = args.chosen_layers
prompt_index = args.prompt_idx

# Save to new subdirectory
save_base = os.path.join('lie_detector_results/i_discriminative_tokens', args.args_name)
os.makedirs(save_base, exist_ok=True)
args.save(save_base)

discriminative_sequences_path = os.path.join(save_base, 'interr_discriminative_sequences.csv')

# Get the prompts which most reliably cause lies 
with open('data/all_prompts.json', 'r') as f:
    prompts = json.load(f)
    lie_system_prompt = prompts['lie_prompts'][prompt_index]
    truth_system_prompt = prompts['truth_prompts'][prompt_index]

print(f"Truth system prompt: {truth_system_prompt}")
print(f"Lie system prompt: {lie_system_prompt}")

# Load model
chat_wrapper = load_model(model_name, device='auto')

# Load steering vector from the i_discriminative_tokens directory
# Note: This assumes the question direction script was run with the same args_name
question_directions_path = os.path.join('lie_detector_results/i_discriminative_tokens', args.args_name, 'question_expectation_directions.npy')
print(f"Loading steering vectors from: {question_directions_path}")

question_steering_vector = np.load(question_directions_path)
layer_direction = {layer_num: torch.tensor(question_steering_vector[layer_num + 1]).cuda().to(torch.float16) for layer_num in chosen_layers}
steering_vector = SteeringVector(layer_direction, "decoder_block")

# Create caches with just system prompts
print("Creating persona caches...")
truth_cache = chat_wrapper.create_prompt_cache(
    system_prompt=truth_system_prompt,
    in_context_questions=[],
    in_context_answers=[]
)

lie_cache = chat_wrapper.create_prompt_cache(
    system_prompt=lie_system_prompt,
    in_context_questions=[],
    in_context_answers=[]
)

# Initial user text for discriminative generation
initial_user_text = " "
print(f"Initial user text: '{initial_user_text}'")

# Initialize results storage
discriminative_df_rows = []

print("Starting discriminative generation...")

for lie_maximisation in [True, False]:
    print(f"\tLie maximisation: {lie_maximisation}")

    for steering_magnitude in question_steering_magnitudes:
        print(f"\t\tQuestion steering magnitude: {steering_magnitude}")

        with steering_vector.apply(chat_wrapper.model, multiplier=steering_magnitude, min_token_index=0, operator=addition_operator()):

            # Generate discriminative sequence for this persona context
            generated_sequence, sequence_progression, discrimination_scores = generate_discriminative_sequence(
                chat_wrapper=chat_wrapper,
                truth_cache=copy.deepcopy(truth_cache),
                lie_cache=copy.deepcopy(lie_cache),
                max_tokens=max_generation_tokens,
                lie_maximise=lie_maximisation,
                initial_text=initial_user_text
            )
            
            print("Generated sequence: '"+ generated_sequence.replace(chr(10), '\\n') + "'")
            print(f"Average discrimination: {sum(discrimination_scores)/len(discrimination_scores) if discrimination_scores else 0:.4f}")
            
            # Add rows to dataframe (one row per generated token)
            for token_pos, score in enumerate(discrimination_scores):
                discriminative_df_rows.append({
                    'lie_maximisation': lie_maximisation,
                    'steering_magnitude': steering_magnitude,
                    'token_position': token_pos + 1,
                    'discrimination_value': score,
                    'sequence_so_far': sequence_progression[token_pos].replace('\n', '\\n'),
                })

            # Create and save discriminative sequences dataframe
            discriminative_df = pd.DataFrame(discriminative_df_rows)
            discriminative_df.to_csv(discriminative_sequences_path, index=False)

print("Direct persona steering complete!")
print(f"Results saved to: {discriminative_sequences_path}")