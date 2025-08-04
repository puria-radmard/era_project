import json, copy
import pandas as pd
from tqdm import tqdm
import numpy as np

import torch

from lie_detector.f_information_theoretic.z_util import find_most_discriminated_question
from model.load import load_model
from util.util import YamlConfig
import matplotlib.pyplot as plt

import sys, os

config_path = sys.argv[1]
args = YamlConfig(config_path)

system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name

initial_answers_args_name = args.initial_answers_args_name
probe_responses_args_name = args.probe_responses_args_name
probe_analysis_args_name = args.probe_analysis_args_name

hidden_state_size = args.hidden_state_size

try:
    batch_size = args.batch_size
except AttributeError:
    batch_size = None

probe_file_name = args.probe_file_name
probe_response_type: str = args.probe_response_type
question_instruction = args.question_instruction

num_layers = args.num_layers

assert probe_response_type == 'yn'

prompt_index = args.prompt_idx

limit_to_lying = args.limit_to_lying


save_base = os.path.join('lie_detector_results/f_information_theoretic', args.args_name)
os.makedirs(save_base, exist_ok=True)
args.save(save_base)

initial_answers_path = os.path.join('lie_detector_results/a_initial_answers', initial_answers_args_name, 'initial_answers.csv')

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


# Load in the probe questions and batchize them
probe_questions = pd.read_csv(f'data/probe_questions/{probe_file_name}.csv')['probe']
num_probe_questions = len(probe_questions)
probe_index_batches = [
    list(range(i, min(i + batch_size, num_probe_questions)))
    for i in range(0, num_probe_questions, batch_size)
]


####################################################################################################
####################################################################################################

# Select single question which is most discriminated
probe_responses_base_path = os.path.join('lie_detector_results/b_probe_answers', probe_responses_args_name)
probe_responses_path = os.path.join(probe_responses_base_path, 'probe_response.csv')
probe_responses_df = pd.read_csv(probe_responses_path)
probe_responses_df['log_odds'] = np.log(probe_responses_df['prob_yes'] / probe_responses_df['prob_no'])

most_discriminated_q, scores, detailed_results = find_most_discriminated_question(probe_responses_df)
ordered_question_idx = scores.question_idx.tolist()

####################################################################################################
####################################################################################################

activation_analysis_path = os.path.join('lie_detector_results/e_activation_analysis', args.args_name, 'projection_results')
projection_lie_directions_normalized = np.load(os.path.join(activation_analysis_path, 'prompted_projection_along_average_lie_vector.npy'), allow_pickle=True).item()
lie_directions = np.stack([projection_lie_directions_normalized[idx]['direction'] for idx in range(len(projection_lie_directions_normalized))])


####################################################################################################
####################################################################################################

# Get yes/no tokens which we'd like to search for
chat_wrapper = load_model(model_name, device='auto')


all_lie_projections = np.nan * np.zeros([len(trainable_questions_idxs), num_probe_questions, num_layers, 50])
all_truth_projections = np.nan * np.zeros([len(trainable_questions_idxs), num_probe_questions, num_layers, 50])

dedicated_directions = np.zeros([num_probe_questions, num_layers, 50, hidden_state_size])


# Have to iterate twice! once for getting the dedicated lie vectors, once for projecting onto these vectors!

# First iteration
for i_qai, qai in enumerate(tqdm(trainable_questions_idxs)):

    # 50 for new tokens!
    this_question_lie_hidden_states = np.nan * np.zeros([num_probe_questions, num_layers, 50, hidden_state_size])
    this_question_truth_hidden_states = np.nan * np.zeros([num_probe_questions, num_layers, 50, hidden_state_size])

    question = initial_questions_df['question'][qai].strip()
    response_row = trainable_answers[trainable_answers['question_idx'] == qai]

    truth_cache_info = chat_wrapper.create_prompt_cache(
        system_prompt=system_prompt,
        in_context_questions=[f'{truth_prompt} {question} {question_instruction}'],
        in_context_answers=[response_row.truth_answer.item()]
    )
    truth_cache = truth_cache_info["cache"]

    lie_cache_info = chat_wrapper.create_prompt_cache(
        system_prompt=system_prompt,
        in_context_questions=[f'{lie_prompt} {question} {question_instruction}'],
        in_context_answers=[response_row.lie_answer.item()]
    )
    lie_cache = lie_cache_info["cache"]

    for probe_index_batch in probe_index_batches:

        truth_followup_chats = [
            chat_wrapper.format_chat(
                system_prompt="",
                user_message=probe_questions[pi],
                prefiller='',
            ) for pi in probe_index_batch
        ]
        truth_forward = chat_wrapper.forward(
            chats = truth_followup_chats,
            past_key_values = copy.deepcopy(truth_cache),
            output_hidden_states = True
        )


        lie_followup_chats = [
            chat_wrapper.format_chat(
                system_prompt="",
                user_message=probe_questions[pi],
                prefiller='',
            ) for pi in probe_index_batch
        ]
        lie_forward = chat_wrapper.forward(
            chats = lie_followup_chats,
            past_key_values = copy.deepcopy(lie_cache),
            output_hidden_states = True
        )

        # [layers, new tokens, hidden size]
        lie_hidden_states = torch.concat(lie_forward['hidden_states'][1:])
        truth_hidden_states = torch.concat(truth_forward['hidden_states'][1:])

        this_question_lie_hidden_states[probe_index_batch,:,:lie_hidden_states.shape[1]] = lie_hidden_states.cpu().numpy()
        this_question_truth_hidden_states[probe_index_batch,:,:truth_hidden_states.shape[1]] = truth_hidden_states.cpu().numpy()

        del truth_forward
        del lie_forward
        torch.cuda.empty_cache()

    all_lie_projections[i_qai] = (this_question_lie_hidden_states * lie_directions[None,:,None,:]).sum(-1)
    all_truth_projections[i_qai] = (this_question_truth_hidden_states * lie_directions[None,:,None,:]).sum(-1)

    dedicated_directions += (this_question_lie_hidden_states - this_question_truth_hidden_states) / len(trainable_questions_idxs)


dedicated_directions_normalized = dedicated_directions / np.sqrt(np.square(dedicated_directions).sum(-1, keepdims=True))


all_lie_dedicated_projections = np.nan * np.zeros([len(trainable_questions_idxs), num_probe_questions, num_layers, 50])
all_truth_dedicated_projections = np.nan * np.zeros([len(trainable_questions_idxs), num_probe_questions, num_layers, 50])


for i_qai, qai in enumerate(tqdm(trainable_questions_idxs)):

    # 50 for new tokens!
    this_question_lie_hidden_states = np.nan * np.zeros([num_probe_questions, num_layers, 50, hidden_state_size])
    this_question_truth_hidden_states = np.nan * np.zeros([num_probe_questions, num_layers, 50, hidden_state_size])

    question = initial_questions_df['question'][qai].strip()
    response_row = trainable_answers[trainable_answers['question_idx'] == qai]

    truth_cache_info = chat_wrapper.create_prompt_cache(
        system_prompt=system_prompt,
        in_context_questions=[f'{truth_prompt} {question} {question_instruction}'],
        in_context_answers=[response_row.truth_answer.item()]
    )
    truth_cache = truth_cache_info["cache"]

    lie_cache_info = chat_wrapper.create_prompt_cache(
        system_prompt=system_prompt,
        in_context_questions=[f'{lie_prompt} {question} {question_instruction}'],
        in_context_answers=[response_row.lie_answer.item()]
    )
    lie_cache = lie_cache_info["cache"]

    for probe_index_batch in probe_index_batches:

        truth_followup_chats = [
            chat_wrapper.format_chat(
                system_prompt="",
                user_message=probe_questions[pi],
                prefiller='',
            ) for pi in probe_index_batch
        ]
        truth_forward = chat_wrapper.forward(
            chats = truth_followup_chats,
            past_key_values = copy.deepcopy(truth_cache),
            output_hidden_states = True
        )


        lie_followup_chats = [
            chat_wrapper.format_chat(
                system_prompt="",
                user_message=probe_questions[pi],
                prefiller='',
            ) for pi in probe_index_batch
        ]
        lie_forward = chat_wrapper.forward(
            chats = lie_followup_chats,
            past_key_values = copy.deepcopy(lie_cache),
            output_hidden_states = True
        )

        # [layers, new tokens]
        lie_hidden_states = torch.concat(lie_forward['hidden_states'][1:])
        truth_hidden_states = torch.concat(truth_forward['hidden_states'][1:])

        this_question_lie_hidden_states[probe_index_batch,:,:lie_hidden_states.shape[1]] = lie_hidden_states.cpu().numpy()
        this_question_truth_hidden_states[probe_index_batch,:,:truth_hidden_states.shape[1]] = truth_hidden_states.cpu().numpy()

        del truth_forward
        del lie_forward
        torch.cuda.empty_cache()
    

    all_lie_dedicated_projections[i_qai] = (dedicated_directions_normalized * this_question_lie_hidden_states).sum(-1)
    all_truth_dedicated_projections[i_qai] = (dedicated_directions_normalized * this_question_truth_hidden_states).sum(-1)






fig, axes = plt.subplots(6, 7, figsize=(28, 24), sharex=True)
axes = axes.flatten()

for layer in range(num_layers):
    ax = axes[layer]
    for dp in range(num_probe_questions):
        ax.plot(range(all_lie_projections.shape[-1]), all_lie_projections[0, dp, layer], color='red', alpha=0.7)
        ax.plot(range(all_lie_projections.shape[-1]), all_truth_projections[0, dp, layer], color='green', alpha=0.7)
    ax.set_title(f'Layer {layer}')
    ax.set_xlabel('Token')
    ax.set_ylabel('Projection')

fig.tight_layout()
fig.savefig(os.path.join(save_base, 'projection_over_probe_question.png'))


np.save(os.path.join(save_base, 'all_lie_projections_over_probe_question.npy'), all_lie_projections)
np.save(os.path.join(save_base, 'all_truth_projections_over_probe_question.npy'), all_truth_projections)

np.save(os.path.join(save_base, 'all_lie_dedicated_projections_over_probe_question.npy'), all_lie_dedicated_projections)
np.save(os.path.join(save_base, 'all_truth_dedicated_projections_over_probe_question.npy'), all_truth_dedicated_projections)

np.save(os.path.join(save_base, 'dedicated_directions.npy'), dedicated_directions_normalized)
