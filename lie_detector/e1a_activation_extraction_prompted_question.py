import json
import pandas as pd
from tqdm import tqdm

from model.load import load_model
from util.util import YamlConfig

import sys, os
import torch

config_path = sys.argv[1]
args = YamlConfig(config_path)

batch_size = args.batch_size
system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name
save_path = args.save_path

prompt_index = args.prompt_idx


response_data = pd.read_csv(f'{save_path}/initial_answers/{questions_data_name}.csv')
with open(f'data/{questions_data_name}.json', 'r') as f:
    question_data = json.load(f)

# Calculate the average of 'told_lie' where 'knows_answer' and 'told_truth' are True, grouped by 'prompt_idx'
knows_answer = response_data[(response_data['knows_answer']) & (response_data['told_truth'])]

# Get the prompts which most reliably cause lies 
with open('data/prompts.json', 'r') as f:
    prompts = json.load(f)
    lie_prompt = prompts['lie_prompts'][prompt_index]
    truth_prompt = prompts['truth_prompts'][prompt_index]

# Get questions which reliably truthed and lied
trainable_answers = knows_answer[(knows_answer['prompt_idx'] == prompt_index) & (knows_answer['told_lie'])]
trainable_questions_idxs = trainable_answers['question_idx']

chat_wrapper = load_model(model_name, device='auto')

candidate_layers = list(range(32))

num_questions = len(trainable_questions_idxs)
num_candidate_layers = len(candidate_layers)
residual_stream_size = 4096

all_truth_residual = torch.zeros(num_questions, num_candidate_layers, residual_stream_size)
all_lie_residual = torch.zeros(num_questions, num_candidate_layers, residual_stream_size)

# Loop over these questions
for i, qai in tqdm(enumerate(trainable_questions_idxs), total = len(trainable_questions_idxs)):

    question = question_data['question'][f'{qai}'].strip()
    response_row = trainable_answers[trainable_answers['question_idx'] == qai]

    truth_chat = chat_wrapper.format_chat(
        system_prompt=system_prompt,
        user_message=f'{truth_prompt} {question}',
        prefiller = ''
    )
    truth_outputs = chat_wrapper.forward(
        chats = [truth_chat],
        output_hidden_states = True
    )

    lie_chat = chat_wrapper.format_chat(
        system_prompt=system_prompt,
        user_message=f'{lie_prompt} {question}',
        prefiller = ''
    )
    lie_outputs = chat_wrapper.forward(
        chats = [lie_chat],
        output_hidden_states = True
    )

    for cli, layer_idx in enumerate(candidate_layers):
        all_truth_residual[i,cli,:] = truth_outputs.hidden_states[layer_idx + 1][0,-1,:]
        all_lie_residual[i,cli,:] = lie_outputs.hidden_states[layer_idx + 1][0,-1,:]



save_target = f"{save_path}/activation_discovery/prompted/{questions_data_name}/prompt{prompt_index}"
os.makedirs(save_target, exist_ok=True)

torch.save(all_truth_residual, os.path.join(save_target, 'all_truth_residual_with_question.pt'))
torch.save(all_lie_residual, os.path.join(save_target, 'all_lie_residual_with_question.pt'))
