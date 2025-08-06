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


save_base = os.path.join('lie_detector_results/g_cutting_probes', args.args_name)
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

# Subsample questions
trainable_questions_idxs = trainable_questions_idxs[:10]            
print('REDUCING trainable_questions_idxs TO JUST SPORTS QUESTIONS!!')
print(f"Using {len(trainable_questions_idxs)} initial questions")


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
        lie_forward, input_ids = chat_wrapper.forward(
            chats = lie_followup_chats,
            past_key_values = copy.deepcopy(lie_cache),
            output_hidden_states = True,
            return_input_ids=True
        )

        # [batch, layers, new tokens, hidden size]
        lie_hidden_states = torch.stack(lie_forward['hidden_states'][1:], 1)
        truth_hidden_states = torch.stack(truth_forward['hidden_states'][1:], 1)

        del truth_forward
        del lie_forward
        
        for b in range(batch_size):
            # Find the start index where non_special_tokens appears as a contiguous subsequence in input_ids[b]
            global_idx = probe_index_batch[b]
            non_special_tokens = chat_wrapper.tokenizer.encode(probe_questions[global_idx], add_special_tokens=False)
            input_seq = input_ids[b].tolist()
            for start_idx in range(len(input_seq) - len(non_special_tokens) + 1):
                if input_seq[start_idx:start_idx + len(non_special_tokens)] == non_special_tokens:
                    non_special_indices = list(range(start_idx, start_idx + len(non_special_tokens)))
                    break
            else:
                raise ValueError("non_special_tokens subsequence not found in input_ids[b]")
            
            this_question_lie_hidden_states[global_idx,:,:len(non_special_indices),:] = lie_hidden_states[b,:,non_special_indices,:].cpu().numpy()
            this_question_truth_hidden_states[global_idx,:,:len(non_special_indices),:] = truth_hidden_states[b,:,non_special_indices,:].cpu().numpy()

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
        lie_forward, input_ids = chat_wrapper.forward(
            chats = lie_followup_chats,
            past_key_values = copy.deepcopy(lie_cache),
            output_hidden_states = True,
            return_input_ids=True
        )

        # [batch, layers, new tokens, hidden size]
        lie_hidden_states = torch.stack(lie_forward['hidden_states'][1:], 1)
        truth_hidden_states = torch.stack(truth_forward['hidden_states'][1:], 1)

        for b in range(batch_size):
            # Find the start index where non_special_tokens appears as a contiguous subsequence in input_ids[b]
            global_idx = probe_index_batch[b]
            non_special_tokens = chat_wrapper.tokenizer.encode(probe_questions[global_idx], add_special_tokens=False)
            input_seq = input_ids[b].tolist()
            for start_idx in range(len(input_seq) - len(non_special_tokens) + 1):
                if input_seq[start_idx:start_idx + len(non_special_tokens)] == non_special_tokens:
                    non_special_indices = list(range(start_idx, start_idx + len(non_special_tokens)))
                    break
            else:
                raise ValueError("non_special_tokens subsequence not found in input_ids[b]")
            
            # non_special_indices = indices of lie_hidden_states which do not align with 1, 2, 3, 4 in input_ids[b] (but not hardcoded)
            this_question_lie_hidden_states[global_idx,:,:len(non_special_indices),:] = lie_hidden_states[b,:,non_special_indices,:].cpu().numpy()
            this_question_truth_hidden_states[global_idx,:,:len(non_special_indices),:] = truth_hidden_states[b,:,non_special_indices,:].cpu().numpy()

        del truth_forward
        del lie_forward
    

    all_lie_projections[i_qai] = (this_question_lie_hidden_states * lie_directions[None,:,None,:]).sum(-1)
    all_truth_projections[i_qai] = (this_question_truth_hidden_states * lie_directions[None,:,None,:]).sum(-1)

    all_lie_dedicated_projections[i_qai] = (dedicated_directions_normalized * this_question_lie_hidden_states).sum(-1)
    all_truth_dedicated_projections[i_qai] = (dedicated_directions_normalized * this_question_truth_hidden_states).sum(-1)
    import pdb; pdb.set_trace()



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
