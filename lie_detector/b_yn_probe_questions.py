import json
import pandas as pd
from tqdm import tqdm

from model.load import load_model
from util.elicit import get_choice_token_logits_from_token_ids
from util.util import YamlConfig

import sys, os

config_path = sys.argv[1]
args = YamlConfig(config_path)

system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name
initial_answers_args_name = args.initial_answers_args_name

probe_file_name = args.probe_file_name
probe_response_type: str = args.probe_response_type
question_instruction = args.question_instruction

assert probe_response_type == 'yn'

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

yesno_strings = [
    ['Yes', 'yes'],
    ['No', 'no'],
]
yesno_tokens = []
for option_str_list in yesno_strings:
    option_tokens = []
    for option_str in option_str_list:
        token_ids = chat_wrapper.tokenizer.encode(option_str, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Choice token variation '{option_str}'"
                f"produces {len(token_ids)} tokens: {token_ids}. "
                f"All choice tokens must be exactly one token."
            )
        option_tokens.extend(token_ids)
    yesno_tokens.append(option_tokens)


# Load in the probe questions
probe_questions = pd.read_csv(f'data/{probe_file_name}.csv')['probe']

# Initialise results df
probe_results_df = pd.DataFrame(columns=['question_idx', 'truth', 'probe_question_idx', 'prob_yes', 'prob_no'])


# Loop over these questions
for qai in tqdm(trainable_questions_idxs):

    question = question_data['question'][f'{qai}'].strip()
    response_row = trainable_answers[trainable_answers['question_idx'] == qai]

    truth_cache_info = chat_wrapper.create_prompt_cache(
        system_prompt=system_prompt,
        in_context_questions=[f'{truth_prompt} {question} {question_instruction}'],
        in_context_answers=[response_row.truth_answer.item()]
    )
    truth_cache = truth_cache_info["cache"]
    truth_cache_str = truth_cache_info["formatted_prompt"]
    truth_followup_chats = [
        chat_wrapper.format_chat(
            system_prompt=None,
            user_message=probe_question,
            prefiller='',
        ) for probe_question in probe_questions
    ]
    # truth_generate = chat_wrapper.generate(chats = truth_followup_chats[:3], past_key_values=truth_cache, past_key_values_str = truth_cache_str)
    truth_forward = chat_wrapper.forward(
        chats = truth_followup_chats,
        past_key_values = truth_cache,
    )
    new_probe_truth_answer_info = get_choice_token_logits_from_token_ids(truth_forward['logits'], yesno_tokens)

    lie_cache_info = chat_wrapper.create_prompt_cache(
        system_prompt=system_prompt,
        in_context_questions=[f'{lie_prompt} {question} {question_instruction}'],
        in_context_answers=[response_row.lie_answer.item()]
    )
    lie_cache = lie_cache_info["cache"]
    lie_cache_str = lie_cache_info["formatted_prompt"]
    lie_followup_chats = [
        chat_wrapper.format_chat(
            system_prompt=None,
            user_message=probe_question,
            prefiller='',
        ) for probe_question in probe_questions
    ]
    # lie_generate = chat_wrapper.generate(chats = lie_followup_chats[:3], past_key_values=lie_cache, past_key_values_str = lie_cache_str)
    lie_forward = chat_wrapper.forward(
        chats = lie_followup_chats,
        past_key_values = lie_cache,
    )
    new_probe_lie_answer_info = get_choice_token_logits_from_token_ids(lie_forward['logits'], yesno_tokens)
    

    # Append results
    num_probe_questions = new_probe_truth_answer_info.shape[0]
    rows = []
    for probe_idx in range(num_probe_questions):
        prob_yes = new_probe_truth_answer_info[probe_idx, 0].item()
        prob_no = new_probe_truth_answer_info[probe_idx, 1].item()
        rows.append({
            'question_idx': qai,
            'truth': 1,
            'probe_question_idx': probe_idx,
            'prob_yes': prob_yes,
            'prob_no': prob_no
        })

        prob_yes = new_probe_lie_answer_info[probe_idx, 0].item()
        prob_no = new_probe_lie_answer_info[probe_idx, 1].item()
        rows.append({
            'question_idx': qai,
            'truth': 0,
            'probe_question_idx': probe_idx,
            'prob_yes': prob_yes,
            'prob_no': prob_no
        })

    probe_results_df = pd.concat([probe_results_df, pd.DataFrame(rows)], ignore_index=True)
    probe_results_df.to_csv(probe_response_path, index = False)



    