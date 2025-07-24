import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from util.util import random_sample_excluding_indices
from model.load import load_model
from util.question import QuestionConfig
from util.experiment import ExperimentConfig

# Import our new utility functions
from util.icl_mcq import (
    elicit_mcq_batch,
    create_icl_plot
)

# Load data and configuration
ocean_questions_df = pd.read_csv('results/p2_mcq_probs.csv')
ocean_questions_df = ocean_questions_df.reset_index()

config_path = 'scripts/icl_mcq/p2_icl_mcq.yaml'
config = ExperimentConfig.from_yaml(config_path)
print(f"Model: {config.model_name}, Num repeats per context length: {config.minibatch_size}")

# Setup model and configs
chat_wrapper = load_model(config.model_name, device='auto')
question_config = QuestionConfig(config).initialize_choices(chat_wrapper.tokenizer)

# Experimental parameters
repeats_per_context_length = 16
context_lengths = [0, 1, 2, 5, 10]
num_context_lengths = len(context_lengths)

# Constants
chosen_trait_to_ocean_and_direction = {
    'an agreeable': ('A', +1),
    'an extraversive': ('E', +1),
    'a conscientious': ('C', +1),
    'a neurotic': ('N', +1),
    'an open': ('O', +1),
    'an introversive': ('E', -1),
    'a disagreeable': ('A', -1),
    'an unconscientious': ('C', -1),
    'a stable': ('N', -1),
    'a closed': ('O', -1),
}

key_positive_scores = np.array([5, 4, 3, 2, 1])
key_negative_scores = np.array([1, 2, 3, 4, 5])
prob_cols = ['pA', 'pB', 'pC', 'pD', 'pE']

# Prepare control questions
all_probs = ocean_questions_df[prob_cols].values
all_normalized_probs = all_probs / all_probs.sum(axis=1, keepdims=True)
all_answer_indices = np.argmax(all_normalized_probs, axis=1)
all_answer_letters = np.array(['A', 'B', 'C', 'D', 'E'])[all_answer_indices].tolist()
all_questions_and_answers = [
    {'index': row['index'], 'question': row['text'].lower(), 'answer': answer, 'key': row['key']}
    for row, answer in zip(ocean_questions_df.to_dict(orient="records"), all_answer_letters)
]

# Main experimental loop
for chosen_trait, (ocean_key, ocean_direction) in chosen_trait_to_ocean_and_direction.items():
    try:
        print(f'#### Beginning ICL for {chosen_trait}')

        # Prepare trait-specific questions
        chosen_trait_ocean_questions_df = ocean_questions_df[
            ocean_questions_df['chosen_trait'] == chosen_trait
        ]
        chosen_trait_matching_ocean_questions_df = chosen_trait_ocean_questions_df[
            chosen_trait_ocean_questions_df['label_ocean'] == ocean_key
        ]

        # Process probabilities and answers
        relevant_probs = chosen_trait_matching_ocean_questions_df[prob_cols].values
        relevant_normalized_probs = relevant_probs / relevant_probs.sum(axis=1, keepdims=True)
        relevant_answer_indices = np.argmax(relevant_normalized_probs, axis=1)
        relevant_answer_letters = np.array(['A', 'B', 'C', 'D', 'E'])[relevant_answer_indices].tolist()

        relevant_questions_and_answers = [
            {'index': row['index'], 'question': row['text'].lower(), 'answer': answer, 'key': row['key']}
            for row, answer in zip(
                chosen_trait_matching_ocean_questions_df.to_dict(orient="records"), 
                relevant_answer_letters
            )
        ]

        # Initialize data storage and load existing data if available
        log_file_path = f'results/icl_mcq/{chosen_trait.split()[1]}.npy'
        existing_data = {}
        
        try:
            existing_data = np.load(log_file_path, allow_pickle=True).item()
            print(f"Loaded existing data with keys: {list(existing_data.keys())}")
        except FileNotFoundError:
            print("No existing data file found, starting fresh")
        
        log_data = {
            'all_data': np.full([repeats_per_context_length, num_context_lengths, len(relevant_questions_and_answers), 5], np.nan),
            'control_all_data': np.full([repeats_per_context_length, num_context_lengths, len(relevant_questions_and_answers), 5], np.nan),
            'random_all_data': np.full([repeats_per_context_length, num_context_lengths, len(relevant_questions_and_answers), 5], np.nan),
            'random_question_all_data': np.full([repeats_per_context_length, num_context_lengths, len(relevant_questions_and_answers), 5], np.nan),
            'in_context_questions_indices': np.full([repeats_per_context_length, num_context_lengths, len(relevant_questions_and_answers), max(context_lengths)], np.nan),
            'control_in_context_questions_indices': np.full([repeats_per_context_length, num_context_lengths, len(relevant_questions_and_answers), max(context_lengths)], np.nan),
        }
        
        # Copy over existing data
        for key, data in existing_data.items():
            if key in log_data:
                log_data[key] = data
                print(f"Loaded existing data for {key}")
        
        # Determine which elicitation types need to be run
        elicitation_types = {
            'all_data': 'Main ICL',
            'control_all_data': 'Control ICL', 
            'random_all_data': 'Random answers',
            'random_question_all_data': 'Random questions',
        }
        
        # Check which data is missing (all NaN means not computed)
        missing_elicitations = []
        for elicit_key, elicit_name in elicitation_types.items():
            if elicit_key not in log_data:
                missing_elicitations.append(elicit_key)
                print(f"Will compute {elicit_name} ({elicit_key})")
            elif np.isnan(log_data[elicit_key]).all():
                missing_elicitations.append(elicit_key)
                print(f"Will compute {elicit_name} ({elicit_key})")
            else:
                print(f"Skipping {elicit_name} ({elicit_key}) - already exists")
        
        # Define data configs for plotting
        data_configs = [
            ('all_data', ('blue', 'red'), 'Relevant questions and answers'),
            ('control_all_data', ('gray', 'gray'), 'Other questions and answers'),
            ('random_all_data', ('green', 'green'), 'Relevant questions, other answers'),
            ('random_question_all_data', ('purple', 'purple'), 'Other questions, relevant answers'),
        ]

        # Process each context length
        for cl_idx, context_length in enumerate(context_lengths):
            print(f'### Beginning ICL with {context_length} examples - repeating {repeats_per_context_length} times')

            # Only run elicitation if there are missing data types
            if missing_elicitations:
                for rep_idx in range(repeats_per_context_length):
                    print(f'## Beginning {rep_idx+1}th repeat')
                    
                    # Process all batches for this context length and repetition
                    num_minibatches = (len(relevant_questions_and_answers) // config.minibatch_size + 
                                      bool(len(relevant_questions_and_answers) % config.minibatch_size != 0))
                    
                    for batch_idx in tqdm(range(num_minibatches), desc=f"Processing batches"):
                        torch.cuda.empty_cache()
                        
                        # Select questions for this batch
                        batch_upper_index = min((batch_idx + 1) * config.minibatch_size, 
                                               len(relevant_questions_and_answers))
                        asked_questions_idx = list(range(batch_idx * config.minibatch_size, 
                                                       batch_upper_index))
                        asked_questions = [relevant_questions_and_answers[i]['question'] 
                                          for i in asked_questions_idx]
                        actual_asked_questions_idx = [relevant_questions_and_answers[i]['index'] 
                                                    for i in asked_questions_idx]
                        
                        # Prepare in-context examples (needed for multiple elicitation types)
                        if not np.isnan(log_data['in_context_questions_indices'][rep_idx, cl_idx, asked_questions_idx[0], 0]):
                            saved_indices = log_data['in_context_questions_indices'][rep_idx, cl_idx, asked_questions_idx[0], :context_length].astype(int)
                            ic_qa_batch = [qa for qa in relevant_questions_and_answers if qa['index'] in saved_indices]
                            print(f"Reusing {len(ic_qa_batch)} saved in-context examples")
                        else:
                            ic_qa_batch = random_sample_excluding_indices(
                                relevant_questions_and_answers, context_length, asked_questions_idx
                            )
                        in_context_questions = [icqa['question'] for icqa in ic_qa_batch]
                        in_context_indices = [icqa['index'] for icqa in ic_qa_batch]
                        in_context_answers = [f"Answer: {icqa['answer']}" for icqa in ic_qa_batch]
                        
                        if not np.isnan(log_data['control_in_context_questions_indices'][rep_idx, cl_idx, asked_questions_idx[0], 0]):
                            control_saved_indices = log_data['control_in_context_questions_indices'][rep_idx, cl_idx, asked_questions_idx[0], :context_length].astype(int)
                            control_ic_qa_batch = [qa for qa in all_questions_and_answers if qa['index'] in control_saved_indices]
                            print(f"Reusing {len(control_ic_qa_batch)} saved control in-context examples")
                        else:
                            control_ic_qa_batch = random_sample_excluding_indices(
                                all_questions_and_answers, context_length, actual_asked_questions_idx
                            )
                        control_in_context_questions = [icqa['question'] for icqa in control_ic_qa_batch]
                        control_in_context_indices = [icqa['index'] for icqa in control_ic_qa_batch]
                        control_in_context_answers = [f"Answer: {icqa['answer']}" for icqa in control_ic_qa_batch]
                        
                        # 1. Main ICL with signal
                        if 'all_data' in missing_elicitations:
                            if context_length > 0:
                                log_data['in_context_questions_indices'][rep_idx, cl_idx, asked_questions_idx, :context_length] = in_context_indices
                            
                            choice_probs = elicit_mcq_batch(
                                chat_wrapper, asked_questions, question_config, config,
                                in_context_questions, in_context_answers
                            )
                            log_data['all_data'][rep_idx, cl_idx, asked_questions_idx, :] = choice_probs.cpu().numpy()
                        
                        # 2. Control ICL with noise
                        if 'control_all_data' in missing_elicitations:
                            if context_length > 0:
                                log_data['control_in_context_questions_indices'][rep_idx, cl_idx, asked_questions_idx, :context_length] = control_in_context_indices
                            
                            control_choice_probs = elicit_mcq_batch(
                                chat_wrapper, asked_questions, question_config, config,
                                control_in_context_questions, control_in_context_answers
                            )
                            log_data['control_all_data'][rep_idx, cl_idx, asked_questions_idx, :] = control_choice_probs.cpu().numpy()
                        
                        # 3. Random answer ICL (same questions, random answers)
                        if 'random_all_data' in missing_elicitations:
                            random_choice_probs = elicit_mcq_batch(
                                chat_wrapper, asked_questions, question_config, config,
                                in_context_questions, control_in_context_answers  # Mixed: signal questions, noise answers
                            )
                            log_data['random_all_data'][rep_idx, cl_idx, asked_questions_idx, :] = random_choice_probs.cpu().numpy()

                        # 4. Random question ICL (random questions, same answers)
                        if 'random_question_all_data' in missing_elicitations:
                            random_question_choice_probs = elicit_mcq_batch(
                                chat_wrapper, asked_questions, question_config, config,
                                control_in_context_questions, in_context_answers  # Mixed: random questions, signal answers
                            )
                            log_data['random_question_all_data'][rep_idx, cl_idx, asked_questions_idx, :] = random_question_choice_probs.cpu().numpy()

                    # Save progress
                    np.save(log_file_path, log_data)
            else:
                print("### All data already exists, skipping elicitation")

            # Create and save plot (always do this to show current results)
            fig = create_icl_plot(
                log_data, context_lengths, cl_idx, 
                relevant_questions_and_answers, key_positive_scores, 
                key_negative_scores, chosen_trait, ocean_direction,
                data_configs
            )
            fig.savefig(f'results/icl_mcq/{chosen_trait.split()[1]}.png')

    except RuntimeError:
        continue