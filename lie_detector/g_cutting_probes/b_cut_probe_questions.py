import json, copy
import pandas as pd
from tqdm import tqdm

import torch

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

try:
    batch_size = args.batch_size
except AttributeError:
    batch_size = None

probe_file_name = args.probe_file_name
probe_response_type: str = args.probe_response_type
question_instruction = args.question_instruction

assert probe_response_type == 'yn'

prompt_index = args.prompt_idx
limit_to_lying = args.limit_to_lying

# Save to new subdirectory
save_base = os.path.join('lie_detector_results/g_cutting_probes', args.args_name)
os.makedirs(save_base, exist_ok=True)
args.save(save_base)

initial_answers_path = os.path.join('lie_detector_results/a_initial_answers', initial_answers_args_name, 'initial_answers.csv')
probe_response_path = os.path.join(save_base, 'truncated_probe_response.csv')

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
probe_questions = pd.read_csv(f'data/probe_questions/{probe_file_name}.csv')['probe']

# Initialize results df with additional token_position column
probe_results_df = pd.DataFrame(columns=['question_idx', 'truth', 'probe_question_idx', 'token_position', 'prob_yes', 'prob_no'])

# Loop over training questions
for qai in trainable_questions_idxs:
    question = initial_questions_df['question'][qai].strip()
    response_row = trainable_answers[trainable_answers['question_idx'] == qai]

    # Create truth cache
    truth_cache_info = chat_wrapper.create_prompt_cache(
        system_prompt=system_prompt,
        in_context_questions=[f'{truth_prompt} {question} {question_instruction}'],
        in_context_answers=[response_row.truth_answer.item()]
    )
    truth_cache = truth_cache_info["cache"]

    # Create lie cache
    lie_cache_info = chat_wrapper.create_prompt_cache(
        system_prompt=system_prompt,
        in_context_questions=[f'{lie_prompt} {question} {question_instruction}'],
        in_context_answers=[response_row.lie_answer.item()]
    )
    lie_cache = lie_cache_info["cache"]

    # Process each probe question individually (no batching across questions)
    for probe_idx, probe_question in enumerate(tqdm(probe_questions, desc=f"Question {qai + 1} / {len(trainable_questions_idxs)}", leave=False)):
        
        # Parse and tokenize the probe question
        if "?" not in probe_question:
            raise ValueError(f"Probe question '{probe_question}' does not contain '?'")
        
        question_part, instruction_part = probe_question.split("?", 1)  # Split at first "?"
        instruction_part = "?" + instruction_part  # Keep the "? Answer with yes or no."
        
        # Tokenize just the question content (before the "?")
        question_tokens = chat_wrapper.tokenizer.encode(question_part.strip(), add_special_tokens=False)
        
        if len(question_tokens) == 0:
            print(f"Warning: No tokens found for question part '{question_part}' in probe {probe_idx}")
            continue
        
        # Create all truncated versions for this probe question
        truncated_chats = []
        
        for token_pos in range(1, len(question_tokens) + 1):
            # Take first token_pos tokens and reconstruct
            truncated_tokens = question_tokens[:token_pos]
            truncated_content = chat_wrapper.tokenizer.decode(truncated_tokens)
            truncated_question = truncated_content + instruction_part

            truncated_chats.append(chat_wrapper.format_chat(
                system_prompt="", 
                user_message=truncated_question,
                prefiller=''
            ))
        
        # Process truncated versions in batches
        num_truncated = len(truncated_chats)
        truncated_batches = [
            list(range(i, min(i + batch_size, num_truncated)))
            for i in range(0, num_truncated, batch_size)
        ]
        
        for batch_indices in truncated_batches:
            batch_chats = [truncated_chats[i] for i in batch_indices]
            
            # Run batch for truth and lie contexts
            truth_forward = chat_wrapper.forward(
                chats=batch_chats,
                past_key_values=copy.deepcopy(truth_cache)
            )
            lie_forward = chat_wrapper.forward(
                chats=batch_chats, 
                past_key_values=copy.deepcopy(lie_cache)
            )
            
            # Extract yes/no probabilities
            truth_probs = get_choice_token_logits_from_token_ids(truth_forward['logits'], yesno_tokens)
            lie_probs = get_choice_token_logits_from_token_ids(lie_forward['logits'], yesno_tokens)

            del truth_forward
            del lie_forward
            
            # Save results with additional token_position column
            rows = []
            for batch_pos, global_pos in enumerate(batch_indices):
                token_pos = global_pos + 1  # Convert to 1-indexed
                
                # Save truth result
                rows.append({
                    'question_idx': qai,
                    'truth': 1, 
                    'probe_question_idx': probe_idx,
                    'token_position': token_pos,
                    'prob_yes': truth_probs[batch_pos, 0].item(),
                    'prob_no': truth_probs[batch_pos, 1].item()
                })
                # Save lie result  
                rows.append({
                    'question_idx': qai,
                    'truth': 0,
                    'probe_question_idx': probe_idx, 
                    'token_position': token_pos,
                    'prob_yes': lie_probs[batch_pos, 0].item(),
                    'prob_no': lie_probs[batch_pos, 1].item()
                })
            
            probe_results_df = pd.concat([probe_results_df, pd.DataFrame(rows)], ignore_index=True)
        
        # Save periodically to avoid losing progress
        probe_results_df.to_csv(probe_response_path, index=False)

print(f"Truncated probe analysis complete. Results saved to: {probe_response_path}")
