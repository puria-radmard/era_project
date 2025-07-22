import json
import pandas as pd
from tqdm import tqdm

from model.load import load_model
from util.util import YamlConfig

import sys, os
import torch

config_path = sys.argv[1]
args = YamlConfig(config_path)

system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name

initial_answers_args_name = args.initial_answers_args_name
question_instruction = args.question_instruction

limit_to_lying = args.limit_to_lying

prompt_index = args.prompt_idx

save_base = os.path.join('lie_detector_results/e_activation_analysis', args.args_name, 'prompted')
os.makedirs(save_base, exist_ok=True)
args.save(save_base)

initial_answers_path = os.path.join('lie_detector_results/a_initial_answers', initial_answers_args_name, 'initial_answers.csv')
probe_response_path = os.path.join(save_base, 'probe_response.csv')

response_data = pd.read_csv(initial_answers_path)
with open(f'data/{questions_data_name}.json', 'r') as f:
    question_data = json.load(f)



# Get the prompts which most reliably cause lies 
with open('data/prompts.json', 'r') as f:
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

    question_instruction

    truth_chat = chat_wrapper.format_chat(
        system_prompt=system_prompt,
        user_message=f'{truth_prompt} {question} {question_instruction}',
        prefiller = ''
    )
    truth_outputs = chat_wrapper.forward(
        chats = [truth_chat],
        output_hidden_states = True
    )

    lie_chat = chat_wrapper.format_chat(
        system_prompt=system_prompt,
        user_message=f'{lie_prompt} {question} {question_instruction}',
        prefiller = ''
    )
    lie_outputs = chat_wrapper.forward(
        chats = [lie_chat],
        output_hidden_states = True
    )

    for cli, layer_idx in enumerate(candidate_layers):
        all_truth_residual[i,cli,:] = truth_outputs.hidden_states[layer_idx + 1][0,-1,:]
        all_lie_residual[i,cli,:] = lie_outputs.hidden_states[layer_idx + 1][0,-1,:]


torch.save(all_truth_residual, os.path.join(save_base, 'all_truth_residual_with_question.pt'))
torch.save(all_lie_residual, os.path.join(save_base, 'all_lie_residual_with_question.pt'))
