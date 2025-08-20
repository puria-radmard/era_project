import json
import pandas as pd
from tqdm import tqdm

from model.load import load_model
from util.util import YamlConfig

import numpy as np
import sys, os
import torch

config_path = sys.argv[1]
args = YamlConfig(config_path)

questions_data_name = args.questions_data_name
model_name = args.hf_model_name
question_instruction = args.question_instruction
num_layers = args.num_layers
hidden_state_size = args.hidden_state_size
prompt_index = args.prompt_idx

save_base = os.path.join('probe_generation_results/b_neurips_workshop_results', args.args_name)
os.makedirs(os.path.join(save_base, 'd2_prompted_liar_activations'), exist_ok=True)

print("Loading test questions...")
initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')

# Get the prompts which most reliably cause lies 
with open('data/all_prompts.json', 'r') as f:
    prompts = json.load(f)
    lie_prompt = prompts['lie_prompts'][prompt_index]
    truth_prompt = prompts['truth_prompts'][prompt_index]


chat_wrapper = load_model(model_name, device='auto')

candidate_layers = list(range(num_layers))

num_questions = len(initial_questions_df)
num_candidate_layers = len(candidate_layers)
residual_stream_size = hidden_state_size

all_truth_residual = np.zeros((num_questions, num_candidate_layers, residual_stream_size))
all_lie_residual = np.zeros((num_questions, num_candidate_layers, residual_stream_size))

# Loop over these questions
for i, iq_row in tqdm(initial_questions_df.iterrows(), total = num_questions):

    question = iq_row.question.strip()

    truth_chat = chat_wrapper.format_chat(
        system_prompt=truth_prompt,
        user_message=f'{question} {question_instruction}',
        prefiller = ''
    )
    truth_outputs = chat_wrapper.forward(
        chats = [truth_chat],
        output_hidden_states = True
    )

    lie_chat = chat_wrapper.format_chat(
        system_prompt=lie_prompt,
        user_message=f'{question} {question_instruction}',
        prefiller = ''
    )
    lie_outputs = chat_wrapper.forward(
        chats = [lie_chat],
        output_hidden_states = True
    )

    for cli, layer_idx in enumerate(candidate_layers):

        all_truth_residual[i,cli,:] = truth_outputs.hidden_states[layer_idx + 1][0,-1,:].cpu().numpy()
        all_lie_residual[i,cli,:] = lie_outputs.hidden_states[layer_idx + 1][0,-1,:].cpu().numpy()


np.save(os.path.join(save_base, 'd2_prompted_liar_activations', 'all_truth_residual_with_question.npy'), all_truth_residual)
np.save(os.path.join(save_base, 'd2_prompted_liar_activations', 'all_lie_residual_with_question.npy'), all_lie_residual)


