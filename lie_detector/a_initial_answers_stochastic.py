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
prompt_idx_limit = args.prompt_idx
num_initial_generation_samples = args.num_initial_generation_samples
top_p = args.top_p
temperature = args.temperature

save_base = os.path.join('lie_detector_results/a_initial_answers', args.args_name)
os.makedirs(save_base, exist_ok=True)
args.save(save_base)
initial_answers_df_path = os.path.join(save_base, "initial_answers_stochastic.csv")

# Load prompts
with open('data/prompts.json', 'r') as f:
    prompts = json.load(f)

truth_prompts = prompts['truth_prompts']
lie_prompts = prompts['lie_prompts']

# Load questions
with open(f'data/{questions_data_name}.json', 'r') as f:
    question_data = json.load(f)

qa_pairs = [(question_data['question'][f'{idx}'].strip(), question_data['answer'][f'{idx}']) for idx in range(len(question_data['question']))]
assert set([len(qa[1].split()) for qa in qa_pairs]) == {1}

# Load model
chat_wrapper = load_model(model_name, device='auto')

# Initialize results list
results = []

# Process generation samples
for generation_idx in tqdm(range(num_initial_generation_samples), desc="Generation samples"):
    
    # Process in batches
    for batch_start in tqdm(range(0, len(qa_pairs), batch_size), desc="Batches", leave=False):
        batch_end = min(batch_start + batch_size, len(qa_pairs))
        batch_qa_pairs = qa_pairs[batch_start:batch_end]
        batch_indices = list(range(batch_start, batch_end))
        
        # Process each prompt pair
        for prompt_idx, (truth_prompt, lie_prompt) in enumerate(zip(truth_prompts, lie_prompts)):

            if prompt_idx_limit is not None:
                if prompt_idx_limit != prompt_idx:
                    continue
            
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
                max_new_tokens=1024,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                max_length=None
            )
            truth_responses = truth_answers['generated_texts']
            
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
                max_new_tokens=1024,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                max_length=None
            )
            lie_responses = lie_answers['generated_texts']
            
            # Store responses
            for i, (truth_resp, lie_resp) in enumerate(zip(truth_responses, lie_responses)):
                truth_resp = truth_resp.removesuffix(".")
                lie_resp = lie_resp.removesuffix(".")
                
                results.append({
                    'question_idx': batch_indices[i],
                    'prompt_idx': prompt_idx,
                    'generation_idx': generation_idx,
                    'truth_answer': truth_resp,
                    'lie_answer': lie_resp
                })
            
            # Save DataFrame after each prompt iteration
            df = pd.DataFrame(results)
            df.to_csv(initial_answers_df_path, index=False)