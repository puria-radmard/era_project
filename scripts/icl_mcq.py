import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt

from util.util import random_sample_excluding_indices

from model.load import load_model
from util.elicit import elicit_mcq_answer, elicit_formatted_answer
from util.question import QuestionConfig
from util.experiment import ExperimentConfig

from tqdm import tqdm


ocean_questions_df = pd.read_csv('results/p2_mcq_probs.csv')  # Update with your actual dataset path
ocean_questions_df = ocean_questions_df.reset_index()


# 1. Load configuration
config_path = 'scripts/p2_icl_mcq.yaml'
config = ExperimentConfig.from_yaml(config_path)
print(f"Model: {config.model_name}, Num repeats per context length: {config.minibatch_size}")

# 2. Setup model and configs
chat_wrapper = load_model(config.model_name, device = 'auto')
question_config = QuestionConfig(config).initialize_choices(chat_wrapper.tokenizer)


repeats_per_context_length = 16
context_lengths = [0, 1, 2, 5, 10, 15, 20, 25]

num_context_lengths = len(context_lengths)

# Relevant constants
chosen_trait_to_ocean_and_direction = {
    'an agreeable': ('A', +1),
    'an extraversive': ('E', +1),
    'a conscientious': ('C', +1),
    'a neurotic': ('N', +1),
    'an open': ('O', +1),
    'an introversive': ('E', -1),
    'a disagreeable': ('A', -1),
    'an unconscientious': ('C', -1),
    'a stable': ('N', -1),
    'a closed': ('O', -1),
}

key_positive_scores = np.array([5, 4, 3, 2, 1])
key_negative_scores = np.array([1, 2, 3, 4, 5])
prob_cols = ['pA', 'pB', 'pC', 'pD', 'pE']



# Set up some control questions to put in context, randomly selected across all question-answer pairs
all_probs = ocean_questions_df[prob_cols].values
all_normalized_probs = all_probs / all_probs.sum(axis=1, keepdims=True)
all_answer_indices = np.argmax(all_normalized_probs, axis=1)
all_answer_letters = np.array(['A', 'B', 'C', 'D', 'E'])[all_answer_indices].tolist()
all_questions_and_answers = [
    {'index': row['index'], 'question': row['text'].lower(), 'answer': answer, 'key': row['key']}
    for row, answer in zip(ocean_questions_df.to_dict(orient="records"), all_answer_letters)
]


for chosen_trait, (ocean_key, ocean_direction) in chosen_trait_to_ocean_and_direction.items():

    try:

        print(f'#### Beginning ICL for {chosen_trait}')

        # Subselect answers where the personality prompt has a bearing on the question relevance
        chosen_trait_ocean_questions_df = ocean_questions_df[ocean_questions_df['chosen_trait'] == chosen_trait]
        chosen_trait_matching_ocean_questions_df = chosen_trait_ocean_questions_df[chosen_trait_ocean_questions_df['label_ocean'] == ocean_key]

        # Normalize probabilities to sum to 1 for each row
        relevant_probs = chosen_trait_matching_ocean_questions_df[prob_cols].values
        relevant_normalized_probs = relevant_probs / relevant_probs.sum(axis=1, keepdims=True)

        # Find the answer with the highest probability for each row
        relevant_answer_indices = np.argmax(relevant_normalized_probs, axis=1)
        relevant_answer_letters = np.array(['A', 'B', 'C', 'D', 'E'])[relevant_answer_indices].tolist()

        # Build the list of dictionaries for QAs relevant to this traitw
        relevant_questions_and_answers = [{'index': row['index'], 'question': row['text'].lower(), 'answer': answer, 'key': row['key']} for row, answer in zip(chosen_trait_matching_ocean_questions_df.to_dict(orient="records"), relevant_answer_letters)]

        # Select config.minibatch_size questions to be asked for all context lengths, then remove them from possible in-context examples
        # asked_questions_idx = random.sample(range(len(relevant_questions_and_answers)), config.minibatch_size)
        # asked_questions = [relevant_questions_and_answers[i]['question'] for i in asked_questions_idx]
        # asked_keys = [relevant_questions_and_answers[i]['key'] for i in asked_questions_idx]
        # relevant_questions_and_answers = [qa for i, qa in enumerate(relevant_questions_and_answers) if i not in asked_questions_idx]

        num_minibatches = len(relevant_questions_and_answers) // config.minibatch_size + bool(len(relevant_questions_and_answers) % config.minibatch_size != 1)

        log_data = {
            'all_data': np.nan * np.zeros([repeats_per_context_length, num_context_lengths, len(relevant_questions_and_answers), 5]),
            'control_all_data': np.nan * np.zeros([repeats_per_context_length, num_context_lengths, len(relevant_questions_and_answers), 5]),
            'random_all_data': np.nan * np.zeros([repeats_per_context_length, num_context_lengths, len(relevant_questions_and_answers), 5]),
            'in_context_questions_indices': np.nan * np.zeros([repeats_per_context_length, num_context_lengths, len(relevant_questions_and_answers), max(context_lengths)]),
            'control_in_context_questions_indices': np.nan * np.zeros([repeats_per_context_length, num_context_lengths, len(relevant_questions_and_answers), max(context_lengths)]),
        }


        for cl_idx, context_length in enumerate(context_lengths):

            print(f'### Beginning ICL with {context_length} examples - repeating {repeats_per_context_length} times')

            for rep_idx in range(repeats_per_context_length):

                print(f'## Beginning {rep_idx+1}th repeat')
                
                for batch_idx in tqdm(range(num_minibatches)):

                    torch.cuda.empty_cache()

                    # Select questions we're asking now
                    batch_upper_index = min((batch_idx + 1) * config.minibatch_size, len(relevant_questions_and_answers) - 1)
                    asked_questions_idx = list(range(batch_idx * config.minibatch_size, batch_upper_index))
                    asked_questions = [relevant_questions_and_answers[i]['question'] for i in asked_questions_idx]
                    actual_asked_questions_idx = [relevant_questions_and_answers[i]['index'] for i in asked_questions_idx]

                    # In-context personality steering - with signal
                    ic_qa_batch = random_sample_excluding_indices(relevant_questions_and_answers, context_length, asked_questions_idx)
                    in_context_questions = [icqa['question'] for icqa in ic_qa_batch]
                    in_context_indices = [icqa['index'] for icqa in ic_qa_batch]
                    in_context_answers = [f"Answer: {icqa['answer']}" for icqa in ic_qa_batch]
                    
                    log_data['in_context_questions_indices'][rep_idx, cl_idx, asked_questions_idx, :context_length] = in_context_indices

                    answers = elicit_mcq_answer(
                        chat_wrapper = chat_wrapper,
                        questions = asked_questions,
                        shared_choices = question_config.mcq_shared_choices,
                        config = question_config,
                        system_prompt = config.system_prompt,                   # Because we are using compact format
                        shared_in_context_questions = in_context_questions,
                        shared_in_context_answers = in_context_answers,
                        choices_in_system_prompt = True                         # Compact format
                    )

                    choice_probs = answers['choice_logits']

                    log_data['all_data'][rep_idx, cl_idx, asked_questions_idx, :] = choice_probs.cpu().numpy()

                    # In-context personality steering - with noise
                    control_ic_aq_batch = random_sample_excluding_indices(all_questions_and_answers, context_length, actual_asked_questions_idx)
                    control_in_context_questions = [icqa['question'] for icqa in control_ic_aq_batch]
                    control_in_context_indices = [icqa['index'] for icqa in control_ic_aq_batch]
                    control_in_context_answers = [f"Answer: {icqa['answer']}" for icqa in control_ic_aq_batch]

                    log_data['control_in_context_questions_indices'][rep_idx, cl_idx, asked_questions_idx, :context_length] = control_in_context_indices

                    control_answers = elicit_mcq_answer(
                        chat_wrapper = chat_wrapper,
                        questions = asked_questions,
                        shared_choices = question_config.mcq_shared_choices,
                        config = question_config,
                        system_prompt = config.system_prompt,                   # Because we are using compact format
                        shared_in_context_questions = control_in_context_questions,
                        shared_in_context_answers = control_in_context_answers,
                        choices_in_system_prompt = True                         # Compact format
                    )

                    control_choice_probs = control_answers['choice_logits']
                    log_data['control_all_data'][rep_idx, cl_idx, asked_questions_idx, :] = control_choice_probs.cpu().numpy()

                    random_answers = elicit_mcq_answer(
                        chat_wrapper = chat_wrapper,
                        questions = asked_questions,
                        shared_choices = question_config.mcq_shared_choices,
                        config = question_config,
                        system_prompt = config.system_prompt,                   # Because we are using compact format
                        shared_in_context_questions = in_context_questions,
                        shared_in_context_answers = control_in_context_answers, # Same in-context questions but random answers
                        choices_in_system_prompt = True                         # Compact format
                    )
                    
                    random_choice_probs = random_answers['choice_logits']
                    log_data['random_all_data'][rep_idx, cl_idx, asked_questions_idx, :] = random_choice_probs.cpu().numpy()

                    np.save(f'results/icl_mcq/{chosen_trait.split()[1]}.npy', log_data)


            plt.close('all')

            fig, axes = plt.subplots(1, 2, figsize = (14, 5), sharey = True)

            fig.suptitle(f'In-context examples from {chosen_trait} person\nexpect ICL to {"in" if ocean_direction == 1 else "de"}crease both scores against control')
            axes[0].set_title('Category-positive questions')
            axes[1].set_title('Category-negative questions')

            positive_index_questions = [i for i, rqa in enumerate(relevant_questions_and_answers) if rqa['key'] == 1]
            negative_index_questions = [i for i, rqa in enumerate(relevant_questions_and_answers) if rqa['key'] == -1]

            question_positive_relevant_data = log_data['all_data'][:,:cl_idx + 1,positive_index_questions]      # [repeats_per_context_length, num_context_lengths done, num relevant positive questions, 5]
            choice_idx = question_positive_relevant_data.argmax(-1)          # [repeats_per_context_length, num_context_lengths done, num relevant positive questions]
            positive_scores_array = key_positive_scores[choice_idx]                   # [repeats_per_context_length, num_context_lengths done, num relevant positive questions]
            positive_mean_scores_per_context = positive_scores_array.mean(0).mean(-1) # [num_context_lengths done]
            positive_std_scores_per_context = positive_scores_array.mean(0).std(-1)   # [num_context_lengths done]  -> average over random repeats -> std over questions (first pass)
            axes[0].plot(context_lengths[:cl_idx + 1], positive_mean_scores_per_context, color = 'blue', marker = 'x')
            axes[0].fill_between(context_lengths[:cl_idx + 1], positive_mean_scores_per_context - positive_std_scores_per_context, positive_mean_scores_per_context + positive_std_scores_per_context, alpha = 0.2, color = 'blue')

            question_negative_relevant_data = log_data['all_data'][:,:cl_idx + 1,negative_index_questions]      # [repeats_per_context_length, num_context_lengths done, num relevant negative questions, 5]
            choice_idx = question_negative_relevant_data.argmax(-1)          # [repeats_per_context_length, num_context_lengths done, num relevant negative questions]
            negative_scores_array = key_negative_scores[choice_idx]                   # [repeats_per_context_length, num_context_lengths done, num relevant negative questions]
            negative_mean_scores_per_context = negative_scores_array.mean(0).mean(-1) # [num_context_lengths done]
            negative_std_scores_per_context = negative_scores_array.mean(0).std(-1)   # [num_context_lengths done]  -> average over random repeats -> std over questions (first pass)
            axes[1].plot(context_lengths[:cl_idx + 1], negative_mean_scores_per_context, color = 'red', marker = 'x')
            axes[1].fill_between(context_lengths[:cl_idx + 1], negative_mean_scores_per_context - negative_std_scores_per_context, negative_mean_scores_per_context + negative_std_scores_per_context, alpha = 0.2, color = 'red')

            control_positive_question_relevant_data = log_data['control_all_data'][:,:cl_idx + 1,positive_index_questions]      # [repeats_per_context_length, num_context_lengths done, num all positive questions, 5]
            control_positive_choice_idx = control_positive_question_relevant_data.argmax(-1)          # [repeats_per_context_length, num_context_lengths done, num all positive questions]
            control_positive_scores_array = key_positive_scores[control_positive_choice_idx]                   # [repeats_per_context_length, num_context_lengths done, num all positive questions]
            control_positive_mean_scores_per_context = control_positive_scores_array.mean(0).mean(-1) # [num_context_lengths done]
            control_positive_std_scores_per_context = control_positive_scores_array.mean(0).std(-1)   # [num_context_lengths done]  -> average over random repeats -> std over questions (first pass)
            axes[0].plot(context_lengths[:cl_idx + 1], control_positive_mean_scores_per_context, color = 'gray', marker = 'x')
            axes[0].fill_between(context_lengths[:cl_idx + 1], control_positive_mean_scores_per_context - control_positive_std_scores_per_context, control_positive_mean_scores_per_context + control_positive_std_scores_per_context, alpha = 0.2, color = 'gray')

            control_negative_question_relevant_data = log_data['control_all_data'][:,:cl_idx + 1,negative_index_questions]      # [repeats_per_context_length, num_context_lengths done, num all negative questions, 5]
            control_negative_choice_idx = control_negative_question_relevant_data.argmax(-1)          # [repeats_per_context_length, num_context_lengths done, num all negative questions]
            control_negative_scores_array = key_negative_scores[control_negative_choice_idx]                   # [repeats_per_context_length, num_context_lengths done, num all negative questions]
            control_negative_mean_scores_per_context = control_negative_scores_array.mean(0).mean(-1) # [num_context_lengths done]
            control_negative_std_scores_per_context = control_negative_scores_array.mean(0).std(-1)   # [num_context_lengths done]  -> average over random repeats -> std over questions (first pass)
            axes[1].plot(context_lengths[:cl_idx + 1], control_negative_mean_scores_per_context, color = 'gray', marker = 'x')
            axes[1].fill_between(context_lengths[:cl_idx + 1], control_negative_mean_scores_per_context - control_negative_std_scores_per_context, control_negative_mean_scores_per_context + control_negative_std_scores_per_context, alpha = 0.2, color = 'gray')


            random_positive_question_relevant_data = log_data['random_all_data'][:,:cl_idx + 1,positive_index_questions]      # [repeats_per_context_length, num_context_lengths done, num all positive questions, 5]
            random_positive_choice_idx = random_positive_question_relevant_data.argmax(-1)          # [repeats_per_context_length, num_context_lengths done, num all positive questions]
            random_positive_scores_array = key_positive_scores[random_positive_choice_idx]                   # [repeats_per_context_length, num_context_lengths done, num all positive questions]
            random_positive_mean_scores_per_context = random_positive_scores_array.mean(0).mean(-1) # [num_context_lengths done]
            random_positive_std_scores_per_context = random_positive_scores_array.mean(0).std(-1)   # [num_context_lengths done]  -> average over random repeats -> std over questions (first pass)
            axes[0].plot(context_lengths[:cl_idx + 1], random_positive_mean_scores_per_context, color = 'green', marker = 'x')
            axes[0].fill_between(context_lengths[:cl_idx + 1], random_positive_mean_scores_per_context - random_positive_std_scores_per_context, random_positive_mean_scores_per_context + random_positive_std_scores_per_context, alpha = 0.2, color = 'green')

            random_negative_question_relevant_data = log_data['random_all_data'][:,:cl_idx + 1,negative_index_questions]      # [repeats_per_context_length, num_context_lengths done, num all negative questions, 5]
            random_negative_choice_idx = random_negative_question_relevant_data.argmax(-1)          # [repeats_per_context_length, num_context_lengths done, num all negative questions]
            random_negative_scores_array = key_negative_scores[random_negative_choice_idx]                   # [repeats_per_context_length, num_context_lengths done, num all negative questions]
            random_negative_mean_scores_per_context = random_negative_scores_array.mean(0).mean(-1) # [num_context_lengths done]
            random_negative_std_scores_per_context = random_negative_scores_array.mean(0).std(-1)   # [num_context_lengths done]  -> average over random repeats -> std over questions (first pass)
            axes[1].plot(context_lengths[:cl_idx + 1], random_negative_mean_scores_per_context, color = 'green', marker = 'x')
            axes[1].fill_between(context_lengths[:cl_idx + 1], random_negative_mean_scores_per_context - random_negative_std_scores_per_context, random_negative_mean_scores_per_context + random_negative_std_scores_per_context, alpha = 0.2, color = 'green')



            fig.savefig(f'results/icl_mcq/{chosen_trait.split()[1]}.png')

    except RuntimeError:
        continue
