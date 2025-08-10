import json
import pandas as pd
import os
from model.load import load_model, ChatTemplateWrapper

from tqdm import tqdm
from util.util import YamlConfig

import sys


def main(
        *_,
        chat_wrapper: ChatTemplateWrapper, 
        initial_answers_df_path: str, 
        num_samples: int, 
        do_sample: bool, 
        temperature: float, 
        top_p: float, 
        prompt_idx: int, 
        questions_data_name: str,
        question_instruction: str,
        batch_size: int 
    ):

    # Load prompts
    with open('data/all_prompts.json', 'r') as f:
        prompts = json.load(f)

    truth_prompt = prompts['truth_prompts'][prompt_idx]
    lie_prompt = prompts['lie_prompts'][prompt_idx]

    # Load questions
    initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')
    qa_pairs = [(initial_questions_df['question'][idx].strip(), str(initial_questions_df['answer'][idx])) for idx in range(len(initial_questions_df['question']))]
    assert set([len(qa[1].split()) for qa in qa_pairs]) == {1}

    # Initialize results list
    results = []

    for generation_idx in range(num_samples):

        # Process in batches
        for batch_start in tqdm(range(0, len(qa_pairs), batch_size)):
            batch_end = min(batch_start + batch_size, len(qa_pairs))
            batch_qa_pairs = qa_pairs[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(qa_pairs) + batch_size - 1)//batch_size}")

            # Generate truth responses
            formatted_truth_chats = [
                chat_wrapper.format_chat(
                    system_prompt=truth_prompt,
                    user_message=f'{qa_pair[0]} {question_instruction}',
                    prefiller=""
                ) for qa_pair in batch_qa_pairs
            ]
            
            truth_answers = chat_wrapper.generate_parallel(
                chats=formatted_truth_chats,
                max_new_tokens=1024,
                temperature = temperature,
                do_sample = do_sample,
                top_p = top_p,
                max_length = None
            )
            truth_responses = truth_answers['generated_texts']
            
            # Generate lie responses
            formatted_lie_chats = [
                chat_wrapper.format_chat(
                    system_prompt=lie_prompt,
                    user_message=f'{qa_pair[0]} {question_instruction}',
                    prefiller=""
                ) for qa_pair in batch_qa_pairs
            ]
            
            lie_answers = chat_wrapper.generate_parallel(
                chats=formatted_lie_chats,
                max_new_tokens=1024,
                temperature = temperature,
                do_sample = do_sample,
                top_p = top_p,
                max_length = None
            )
            lie_responses = lie_answers['generated_texts']
            
            # Evaluate responses
            for i, (qa_pair, truth_resp, lie_resp) in enumerate(zip(
                batch_qa_pairs, truth_responses, lie_responses
            )):
                correct_answer = qa_pair[1].lower()

                truth_resp = truth_resp
                lie_resp = lie_resp
                
                told_truth = correct_answer in truth_resp.lower()
                told_lie = correct_answer not in lie_resp.lower()  #True if it's actually a lie
                
                results.append({
                    'question_idx': batch_indices[i],
                    'prompt_idx': prompt_idx,
                    'generation_idx': generation_idx,
                    'truth_answer': truth_resp,
                    'lie_answer': lie_resp,
                    'told_truth': told_truth,
                    'told_lie': told_lie,
                })
            
            # Save DataFrame after each prompt iteration
            df = pd.DataFrame(results)
            df.to_csv(initial_answers_df_path, index=False)


if __name__ == '__main__':

    config_path = sys.argv[1]
    args = YamlConfig(config_path)

    batch_size = args.batch_size
    question_instruction = args.question_instruction
    questions_data_name = args.questions_data_name
    model_name = args.model_name
    prompt_idx = args.prompt_idx


    save_base = os.path.join('probe_generation_results/a_introspective_probe_generation', args.args_name)
    os.makedirs(save_base, exist_ok=True)
    args.save(save_base)
    output_path = os.path.join(save_base, "a_initial_answers.csv")

    # Load model
    chat_wrapper = load_model(model_name, device='auto')

    main(
        chat_wrapper = chat_wrapper,
        initial_answers_df_path = output_path,
        num_samples = 1,
        do_sample = False,
        temperature = None,
        top_p = None,
        prompt_idx = prompt_idx,
        questions_data_name = questions_data_name,
        question_instruction = question_instruction,
        batch_size = batch_size,
    )