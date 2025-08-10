import json, copy
import pandas as pd
from tqdm import tqdm

import torch, os

from model.load import load_model
from util.elicit import get_choice_token_logits_from_token_ids



def main(
    *_,
    initial_answers_df: str,
    questions_data_name: str,
    # prompt_index,
    question_instruction: str,
    probe_response_path: str,
    probe_questions: pd.Series,
    batch_size: int,
    model_name: str,
    limit_to_lying: bool,
    append_strings: list[str] = ["Answer with yes or no.", "Answer with no or yes."]
    ):

    initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')

    # # Get the prompts which most reliably cause lies 
    # with open('data/all_prompts.json', 'r') as f:
    #     prompts = json.load(f)
    #     lie_prompt = prompts['lie_prompts'][prompt_index]
    #     truth_prompt = prompts['truth_prompts'][prompt_index]

    # Get questions which reliably truthed and lied
    if limit_to_lying:
        knows_answer = initial_answers_df[(initial_answers_df['knows_answer']) & (initial_answers_df['told_truth'])]
        trainable_answers = knows_answer[(knows_answer['prompt_idx'] == prompt_index) & (knows_answer['told_lie'])]
    else:
        trainable_answers = initial_answers_df
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

    # Initialise results df
    probe_results_df = pd.DataFrame(columns=['question_idx', 'truth', 'probe_question_idx', 'prob_yes', 'prob_no'])

    # Use the actual index from probe_questions (assumed to be a pandas Series or DataFrame)
    num_probe_questions = len(probe_questions)
    probe_index_batches = [
        list(probe_questions.index[i : min(i + batch_size, num_probe_questions)])
        for i in range(0, num_probe_questions, batch_size)
    ]

    # Loop over these questions
    for i_qai, qai in enumerate(trainable_questions_idxs):

        print(f"Processing question {i_qai + 1}/{len(trainable_questions_idxs)}. Iterating over probe batches.")

        question = initial_questions_df['question'][qai].strip()
        response_row = trainable_answers[trainable_answers['question_idx'] == qai]

        truth_cache_info = chat_wrapper.create_prompt_cache(
            system_prompt="",
            #system_prompt=truth_prompt,
            in_context_questions=[f'{question} {question_instruction}'],
            in_context_answers=[response_row.truth_answer.item()]
        )
        truth_cache = truth_cache_info["cache"]

        lie_cache_info = chat_wrapper.create_prompt_cache(
            system_prompt="",
            #system_prompt=lie_prompt,
            in_context_questions=[f'{question} {question_instruction}'],
            in_context_answers=[response_row.lie_answer.item()]
        )
        lie_cache = lie_cache_info["cache"]

        for probe_index_batch in tqdm(probe_index_batches):

            for i_append_string, append_string in enumerate(append_strings):

                truth_followup_chats = [
                    chat_wrapper.format_chat(
                        system_prompt="",
                        user_message=f"{probe_questions[pi]} {append_string}",
                        prefiller='',
                    ) for pi in probe_index_batch
                ]
                truth_forward = chat_wrapper.forward(
                    chats = truth_followup_chats,
                    past_key_values = copy.deepcopy(truth_cache),
                )
                new_probe_truth_answer_info = get_choice_token_logits_from_token_ids(truth_forward['logits'], yesno_tokens)


                lie_followup_chats = [
                    chat_wrapper.format_chat(
                        system_prompt="",
                        user_message=f"{probe_questions[pi]} {append_string}",
                        prefiller='',
                    ) for pi in probe_index_batch
                ]
                lie_forward = chat_wrapper.forward(
                    chats = lie_followup_chats,
                    past_key_values = copy.deepcopy(lie_cache),
                )
                new_probe_lie_answer_info = get_choice_token_logits_from_token_ids(lie_forward['logits'], yesno_tokens)

                # lie_cache_str = lie_cache_info["formatted_prompt"]
                # lie_generate = chat_wrapper.generate(chats = lie_followup_chats[:3], past_key_values=lie_cache, past_key_values_str = lie_cache_str)

                del truth_forward
                del lie_forward

                # Append results
                num_probe_questions = new_probe_truth_answer_info.shape[0]
                rows = []
                for probe_idx_in_batch, probe_idx in enumerate(probe_index_batch):

                    prob_yes = new_probe_truth_answer_info[probe_idx_in_batch, 0].item()
                    prob_no = new_probe_truth_answer_info[probe_idx_in_batch, 1].item()
                    rows.append({
                        'question_idx': qai,
                        'truth': 1,
                        'probe_question_idx': probe_idx,
                        'i_append_string': i_append_string,
                        'prob_yes': prob_yes,
                        'prob_no': prob_no
                    })

                    prob_yes = new_probe_lie_answer_info[probe_idx_in_batch, 0].item()
                    prob_no = new_probe_lie_answer_info[probe_idx_in_batch, 1].item()
                    rows.append({
                        'question_idx': qai,
                        'truth': 0,
                        'probe_question_idx': probe_idx,
                        'i_append_string': i_append_string,
                        'prob_yes': prob_yes,
                        'prob_no': prob_no
                    })

                probe_results_df = pd.concat([probe_results_df, pd.DataFrame(rows)], ignore_index=True)
            probe_results_df.to_csv(probe_response_path, index = False)



if __name__ == '__main__':

    import sys
    from util.util import YamlConfig


    config_path = sys.argv[1]
    args = YamlConfig(config_path)

    batch_size = args.batch_size
    question_instruction = args.question_instruction
    questions_data_name = args.questions_data_name
    model_name = args.model_name
    prompt_idx = args.prompt_idx
    probe_file_name = args.probe_file_name
    
    save_base = os.path.join('probe_generation_results/a_introspective_probe_generation', args.args_name)
    probe_response_path = os.path.join(save_base, "b_original_probe_questions.csv")

    probe_questions = pd.read_csv(f'data/probe_questions/{probe_file_name}.csv')['probe']
    
    initial_answers_path = os.path.join(save_base, 'a_initial_answers.csv')
    initial_answers_df = pd.read_csv(initial_answers_path)
    
    main(
        initial_answers_df=initial_answers_df,
        questions_data_name=args.questions_data_name,
        # prompt_index,
        question_instruction=args.question_instruction,
        probe_response_path=probe_response_path,
        probe_questions=probe_questions,
        batch_size=args.batch_size,
        model_name=args.model_name,
        limit_to_lying=False,
    )

