import json
import pandas as pd
from tqdm import tqdm

from model.load import load_model
from util.elicit import get_choice_token_logits_from_token_ids
from util.util import YamlConfig

import os, sys

config_path = sys.argv[1]
args = YamlConfig(config_path)

system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name
initial_answers_args_name = args.initial_answers_args_name

probe_file_name = args.probe_file_name
probe_response_type: str = args.probe_response_type
question_instruction = args.question_instruction

assert probe_response_type.endswith('five_words')

prompt_index = args.prompt_idx


save_base = os.path.join('lie_detector_results/b_probe_answers', args.args_name)
os.makedirs(save_base, exist_ok=True)
args.save_args(save_base)

initial_answers_path = os.path.join('lie_detector_results/a_initial_answers', initial_answers_args_name, 'initial_answers.csv')
probe_response_path = os.path.join(save_base, 'probe_response.csv')


response_data = pd.read_csv(initial_answers_path)
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
            in_context_questions=[f'{truth_prompt} {question} {question_instruction}'],
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
            in_context_questions=[f'{lie_prompt} {question} {question_instruction}'],
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
    probe_results_df.to_csv(probe_response_path, index=False)
    