import numpy as np
from steering_vectors import ablation_then_addition_operator, train_steering_vector
import torch
from util.steering import LayerSpecificMultipliersSteeringVector

import json
import pandas as pd
import os
from model.load import load_model

from tqdm import tqdm
from util.util import YamlConfig

import sys

config_path = sys.argv[1]
args = YamlConfig(config_path)

batch_size = args.batch_size
system_prompt = args.system_prompt
question_instruction = args.question_instruction
questions_data_name = args.questions_data_name
model_name = args.model_name
prompt_idx = args.prompt_idx

chosen_layers = args.chosen_layers
multipliers = args.multipliers


activation_analysis_path = os.path.join('lie_detector_results/e_activation_analysis', args.args_name, 'projection_results')

save_base = os.path.join('lie_detector_results/f_information_theoretic', args.args_name)
os.makedirs(save_base, exist_ok=True)
args.save(save_base)
initial_answers_df_path = os.path.join(save_base, "clamped_initial_answers.csv")
print(initial_answers_df_path)


# Load prompts
with open('data/all_prompts.json', 'r') as f:
    prompts = json.load(f)

truth_prompts = prompts['truth_prompts']
lie_prompts = prompts['lie_prompts']

# Load questions
initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')
qa_pairs = [(initial_questions_df['question'][idx].strip(), str(initial_questions_df['answer'][idx])) for idx in range(len(initial_questions_df['question']))]
assert set([len(qa[1].split()) for qa in qa_pairs]) == {1}

# Initialize results list
results = []

# Initialise the steering vector
pre_answer_vector_info = np.load(os.path.join(activation_analysis_path, 'prompted_projection_along_average_lie_vector.npy'), allow_pickle = True).item()

gt_misaligned_cluster_mean = {layer_num: pre_answer_vector_info[layer_num]['lie_mean'] for layer_num in chosen_layers}
gt_aligned_cluster_mean = {layer_num: pre_answer_vector_info[layer_num]['truth_mean'] for layer_num in chosen_layers}
midpoints = {cl: (gt_misaligned_cluster_mean[cl] + gt_aligned_cluster_mean[cl]) / 2.0 for cl in chosen_layers}
half_lengths = {cl: (gt_misaligned_cluster_mean[cl] - gt_aligned_cluster_mean[cl]) / 2.0 for cl in chosen_layers}

all_misaligned_cluster_mean = [{cl: midpoints[cl] + multiplier * half_lengths[cl] for cl in chosen_layers} for multiplier in multipliers]
all_aligned_cluster_mean = [{cl: midpoints[cl] - multiplier * half_lengths[cl] for cl in chosen_layers} for multiplier in multipliers]

layer_direction = {layer_num: torch.tensor(pre_answer_vector_info[layer_num]['direction']).cuda().to(torch.float16) for layer_num in chosen_layers}

steering_vector = LayerSpecificMultipliersSteeringVector(layer_direction, "decoder_block")


truth_prompt = prompts['truth_prompts'][prompt_idx]
lie_prompt = prompts['lie_prompts'][prompt_idx]


# Load model
chat_wrapper = load_model(model_name, device='auto')


for i_mult, multiplier in enumerate(multipliers):

    cluster_means = [all_misaligned_cluster_mean[i_mult], all_aligned_cluster_mean[i_mult]]

    # Generate with aligned clamping, then misaligned clamping
    for i_cm, cluster_mean in enumerate(cluster_means):

        with steering_vector.apply(chat_wrapper.model, multiplier=cluster_mean, min_token_index=0, operator=ablation_then_addition_operator()):

            print(f"Multiplier: {multiplier}. Aligned clamp: {i_cm == 1}. Iterating batches...")

            # Process in batches
            for batch_start in tqdm(range(0, len(qa_pairs), batch_size)):

                batch_end = min(batch_start + batch_size, len(qa_pairs))
                batch_qa_pairs = qa_pairs[batch_start:batch_end]
                batch_indices = list(range(batch_start, batch_end))
                
                # print(f"Processing batch {batch_start//batch_size + 1}/{(len(qa_pairs) + batch_size - 1)//batch_size}")

                # Generate truth responses
                formatted_truth_chats = [
                    chat_wrapper.format_chat(
                        system_prompt=system_prompt,
                        user_message=f'{truth_prompt} {qa_pair[0]} {question_instruction}',
                        prefiller=""
                    ) for qa_pair in batch_qa_pairs
                ]

                truth_answers = chat_wrapper.generate_parallel(
                    chats=formatted_truth_chats,
                    max_new_tokens=100,
                    temperature=None,
                    do_sample=False,
                    top_p = None,
                    max_length=None,
                    output_hidden_states = True
                )
                truth_responses = truth_answers['generated_texts']

                import pdb; pdb.set_trace()

                # Generate lie responses
                formatted_lie_chats = [
                    chat_wrapper.format_chat(
                        system_prompt=system_prompt,
                        user_message=f'{lie_prompt} {qa_pair[0]} {question_instruction}',
                        prefiller=""
                    ) for qa_pair in batch_qa_pairs
                ]

                lie_answers = chat_wrapper.generate_parallel(
                    chats=formatted_lie_chats,
                    max_new_tokens=100,
                    temperature=None,
                    do_sample=False,
                    top_p = None,
                    max_length=None
                )
                lie_responses = lie_answers['generated_texts']

                for i, (qa_pair, truth_resp, lie_resp) in enumerate(zip(
                    batch_qa_pairs, truth_responses, lie_responses
                )):
                    correct_answer = qa_pair[1].lower()

                    truth_resp = truth_resp.removesuffix(".")
                    lie_resp = lie_resp.removesuffix(".")
                    
                    told_truth = correct_answer in truth_resp.lower()
                    told_lie = correct_answer not in lie_resp.lower()  #True if it's actually a lie
                    
                    results.append({
                        'question_idx': batch_indices[i],
                        'clamping_cluster_aligned': i_cm,
                        'clamping_multiplier': multiplier,
                        'prompt_idx': prompt_idx,
                        'truth_answer': truth_resp,
                        'lie_answer': lie_resp,
                        'told_truth': told_truth,
                        'told_lie': told_lie
                    })

                df = pd.DataFrame(results)
                df.to_csv(initial_answers_df_path, index=False)