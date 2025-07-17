import json
import pandas as pd
from tqdm import tqdm

from model.load import load_model
from util.elicit import get_choice_token_logits_from_token_ids
from util.util import YamlConfig

import os, sys

config_path = sys.argv[1]
args = YamlConfig(config_path)

batch_size = args.batch_size
system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name
save_path = args.save_path

probe_file_name = args.probe_file_name
probe_response_type = args.probe_response_type

assert probe_response_type == 'five_words'

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

# Get yes/no tokens which we'd like to search for
chat_wrapper = load_model(model_name, device='auto')

os.makedirs(f'{save_path}/probe_answers/{probe_file_name}/five_words', exist_ok=True)


# Load in the probe questions
probe_questions = pd.read_csv(f'data/{probe_file_name}.csv')['probe']

# Initialise results df
probe_results_df = pd.DataFrame(columns=['question_idx', 'truth', 'probe_question_idx', 'resp'])


# Loop over these questions
for qai in tqdm(trainable_questions_idxs):

    question = question_data['question'][f'{qai}'].strip()
    response_row = trainable_answers[trainable_answers['question_idx'] == qai]

    truth_followup_chats = [
        chat_wrapper.format_chat(
            system_prompt=system_prompt,
            in_context_questions=[f'{truth_prompt} {question}'],
            in_context_answers=[response_row.truth_answer.item()],
            user_message=probe_question,
            prefiller='',
        ) for probe_question in probe_questions
    ]
    truth_generate = chat_wrapper.generate_parallel(
        chats = truth_followup_chats,
        temperature = None,
        top_p = None,
        do_sample = False
    )


    lie_followup_chats = [
        chat_wrapper.format_chat(
            system_prompt=system_prompt,
            in_context_questions=[f'{lie_prompt} {question}'],
            in_context_answers=[response_row.lie_answer.item()],
            user_message=probe_question,
            prefiller='',
        ) for probe_question in probe_questions
    ]
    lie_generate = chat_wrapper.generate_parallel(
        chats = lie_followup_chats,
        temperature = None,
        top_p = None,
        do_sample = False
    )


    # Append results
    num_probe_questions = len(lie_generate['generated_texts'])
    rows = []
    for probe_idx in range(num_probe_questions):

        resp_truth = truth_generate['generated_texts'][probe_idx].removesuffix('<|eot_id|>')
        rows.append({
            'question_idx': qai,
            'truth': 1,
            'probe_question_idx': probe_idx,
            'resp': resp_truth,
        })

        resp_lie = lie_generate['generated_texts'][probe_idx].removesuffix('<|eot_id|>')
        rows.append({
            'question_idx': qai,
            'truth': 0,
            'probe_question_idx': probe_idx,
            'resp': resp_lie,
        })

    probe_results_df = pd.concat([probe_results_df, pd.DataFrame(rows)], ignore_index=True)
    probe_results_df.to_csv(f'{save_path}/probe_answers/{probe_file_name}/five_words/{questions_data_name}_probe_prompt{prompt_index}.csv', index=False)



    