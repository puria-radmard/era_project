import json, copy
import pandas as pd
from tqdm import tqdm

from model.load import load_model
from util.util import YamlConfig
import sys, os

from util.elicit import elicit_user_text_completion


# Main script logic
config_path = sys.argv[1]
args = YamlConfig(config_path)

system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name
initial_answers_args_name = args.initial_answers_args_name
batch_size = args.batch_size
probe_file_name = args.probe_file_name
prompt_index = args.prompt_idx
limit_to_lying = args.limit_to_lying

# Generation parameters
max_new_tokens = args.max_new_tokens
temperature = args.temperature
do_sample = args.do_sample

# Save to new subdirectory
save_base = os.path.join('lie_detector_results/h_completion_probes', args.args_name)
os.makedirs(save_base, exist_ok=True)
args.save(save_base)

initial_answers_path = os.path.join('lie_detector_results/a_initial_answers', initial_answers_args_name, 'initial_answers.csv')
completion_response_path = os.path.join(save_base, 'truncated_completion_response.csv')

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

# Load model
chat_wrapper = load_model(model_name, device='auto')

# Load probe questions
probe_questions = pd.read_csv(f'data/probe_questions/{probe_file_name}.csv')['probe']

# Initialize results df
completion_results_df = pd.DataFrame(columns=['question_idx', 'truth', 'probe_question_idx', 'token_position', 'completion'])

# Loop over training questions
for i_qai, qai in enumerate(trainable_questions_idxs):

    question = initial_questions_df['question'][qai].strip()
    response_row = trainable_answers[trainable_answers['question_idx'] == qai]

    # Create truth cache
    truth_cache = chat_wrapper.create_prompt_cache(
        system_prompt=truth_prompt,
        # system_prompt="",
        in_context_questions=[question],
        in_context_answers=[response_row.truth_answer.item()]
    )

    # Create lie cache
    lie_cache = chat_wrapper.create_prompt_cache(
        system_prompt=lie_prompt,
        # system_prompt="",
        in_context_questions=[question],
        in_context_answers=[response_row.lie_answer.item()]
    )

    # Process each probe question individually
    for probe_idx, probe_question in enumerate(tqdm(probe_questions, desc=f"Question {qai} ({i_qai + 1} / {len(trainable_questions_idxs)})", leave=False)):
        
        # Remove the "?" and everything after it - we just want the incomplete question
        if "?" in probe_question:
            question_part = probe_question.split("?")[0].strip()
        else:
            question_part = probe_question.strip()
        
        # Tokenize the question content
        question_tokens = chat_wrapper.tokenizer.encode(question_part, add_special_tokens=False)
        
        if len(question_tokens) == 0:
            print(f"Warning: No tokens found for question part '{question_part}' in probe {probe_idx}")
            continue
        
        # Create truncated versions for this probe question
        truncated_texts = []
        
        for token_pos in range(1, len(question_tokens) + 1):
            # Take first token_pos tokens and reconstruct
            truncated_tokens = question_tokens[:token_pos]
            truncated_content = chat_wrapper.tokenizer.decode(truncated_tokens)
            truncated_texts.append(truncated_content)
        
        # Process truncated versions in batches
        num_truncated = len(truncated_texts)
        for i in range(0, num_truncated, batch_size):
            batch_texts = truncated_texts[i:i + batch_size]
            batch_positions = list(range(i + 1, min(i + batch_size + 1, num_truncated + 1)))
            
            # Generate completions for truth and lie contexts
            truth_completions = elicit_user_text_completion(
                chat_wrapper=chat_wrapper,
                texts=batch_texts,
                system_prompt="",
                cache_data=copy.deepcopy(truth_cache),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample
            )
            
            lie_completions = elicit_user_text_completion(
                chat_wrapper=chat_wrapper,
                texts=batch_texts,
                system_prompt="",
                cache_data=copy.deepcopy(lie_cache),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample
            )
            
            # Save results
            rows = []
            for batch_pos, (token_pos, trunc, truth_comp, lie_comp) in enumerate(zip(batch_positions, batch_texts, truth_completions, lie_completions)):
                # Save truth result
                rows.append({
                    'question_idx': qai,
                    'truth': 1, 
                    'probe_question_idx': probe_idx,
                    'token_position': token_pos,
                    'existing_question': trunc,
                    'completion': truth_comp.replace('\n', '\\n')
                })
                # Save lie result  
                rows.append({
                    'question_idx': qai,
                    'truth': 0,
                    'probe_question_idx': probe_idx, 
                    'token_position': token_pos,
                    'existing_question': trunc,
                    'completion': lie_comp.replace('\n', '\\n')
                })
            
            completion_results_df = pd.concat([completion_results_df, pd.DataFrame(rows)], ignore_index=True)
        
        # Save periodically to avoid losing progress
        completion_results_df.to_csv(completion_response_path, index=False)

print(f"Truncated completion analysis complete. Results saved to: {completion_response_path}")
