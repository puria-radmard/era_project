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

system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name
initial_answers_args_name = args.initial_answers_args_name
prompt_index = args.prompt_idx
limit_to_lying = args.limit_to_lying

max_generation_tokens = args.max_generation_tokens
question_steering_magnitudes = args.question_steering_magnitudes
chosen_layers = args.chosen_layers


# Save to new subdirectory
save_base = os.path.join('lie_detector_results/i_discriminative_tokens', args.args_name)
os.makedirs(save_base, exist_ok=True)
args.save(save_base)

initial_answers_path = os.path.join('lie_detector_results/a_initial_answers', initial_answers_args_name, 'initial_answers.csv')
discriminative_sequences_path = os.path.join(save_base, 'discriminative_sequences.csv')
analysis_path = os.path.join(save_base, 'discriminative_analysis.json')

response_data = pd.read_csv(initial_answers_path)
initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')

# Get the prompts which most reliably cause lies 
with open('data/all_prompts.json', 'r') as f:
    prompts = json.load(f)
    lie_prompt = prompts['lie_prompts'][prompt_index]
    truth_prompt = prompts['truth_prompts'][prompt_index]

# Get questions which reliably truthed and lied
if limit_to_lying:
    knows_answer = response_data[(response_data['knows_answer']) & (response_data['told_truth'])]
    trainable_answers = knows_answer[(knows_answer['prompt_idx'] == prompt_index) & (knows_answer['told_lie'])]
else:
    trainable_answers = response_data
trainable_questions_idxs = trainable_answers['question_idx']

# Subsample questions
trainable_questions_idxs = trainable_questions_idxs[:10]         
print('REDUCING trainable_questions_idxs TO JUST 10 QUESTIONS!!')
print(f"Using {len(trainable_questions_idxs)} initial questions")

# Load model
chat_wrapper = load_model(model_name, device='auto')

# Load steering vector
question_steering_vector = np.load(os.path.join(save_base, 'question_expectation_directions.npy'))
layer_direction = {layer_num: torch.tensor(question_steering_vector[layer_num + 1]).cuda().to(torch.float16) for layer_num in chosen_layers}
steering_vector = SteeringVector(layer_direction, "decoder_block")

# Initialize results storage
discriminative_df_rows = []

# Loop over training questions
for i_qai, qai in enumerate(trainable_questions_idxs):
        
    question = initial_questions_df['question'][qai].strip()
    response_row = trainable_answers[trainable_answers['question_idx'] == qai]

    # Create truth cache
    truth_cache = chat_wrapper.create_prompt_cache(
        system_prompt=truth_prompt,
        in_context_questions=[question],
        in_context_answers=[response_row.truth_answer.item()]
    )

    # Create lie cache
    lie_cache = chat_wrapper.create_prompt_cache(
        system_prompt=lie_prompt,
        in_context_questions=[question],
        in_context_answers=[response_row.lie_answer.item()]
    )

    print(f"Processing question {i_qai + 1}/{len(trainable_questions_idxs)}: {question[:50]}...")

    for lie_maximisation in [True, False]:
        print(f"\t\tLie maximisation: {lie_maximisation}")

        for steering_magnitude in question_steering_magnitudes:

            with steering_vector.apply(chat_wrapper.model, multiplier=steering_magnitude, min_token_index=0, operator=addition_operator()):
                print(f"\t\tQuestion steering: {steering_magnitude}")

                # Generate discriminative sequence for this question context
                generated_sequence, sequence_progression, discrimination_scores = generate_discriminative_sequence(
                    chat_wrapper=chat_wrapper,
                    truth_cache=copy.deepcopy(truth_cache),
                    lie_cache=copy.deepcopy(lie_cache),
                    max_tokens=max_generation_tokens,
                    lie_maximise=lie_maximisation
                )
                
                print("Generated sequence: '"+ generated_sequence.replace(chr(10), '\\n') + "'")
                print(f"Average discrimination: {sum(discrimination_scores)/len(discrimination_scores) if discrimination_scores else 0:.4f}")
                
                # Add rows to dataframe (one row per generated token)
                for token_pos, score in enumerate(discrimination_scores):

                    discriminative_df_rows.append({
                        'question_idx': qai,
                        'lie_maximisation': lie_maximisation,
                        'steering_magnitude': steering_magnitude,
                        'token_position': token_pos + 1,
                        'discrimination_value': score,
                        'sequence_so_far': sequence_progression[token_pos].replace('\n', '\\n')
                    })

                # Create and save discriminative sequences dataframe
                discriminative_df = pd.DataFrame(discriminative_df_rows)
                discriminative_df.to_csv(discriminative_sequences_path, index=False)

