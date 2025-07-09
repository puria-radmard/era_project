import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt

import yaml

from model.load import load_model
from util.elicit import elicit_mcq_answer, elicit_formatted_answer
from util.question import QuestionConfig
from util.experiment import ExperimentConfig

from tqdm import tqdm




ocean_questions_df = pd.read_csv('results/p2_mcq_probs.csv')  # Update with your actual dataset path

chosen_trait_to_ocean_and_direction = {
    'an extraversive': ('E', +1),
    'an agreeable': ('A', +1),
    'a conscientious': ('C', +1),
    'a neurotic': ('N', +1),
    'an open': ('O', +1),
    'an introversive': ('E', -1),
    'a disagreeable': ('A', -1),
    'an unconscientious': ('C', -1),
    'a stable': ('N', -1),
    'a closed': ('O', -1),
}


# 1. Load configuration
config_path = 'scripts/p2_mcq.yaml'
config = ExperimentConfig.from_yaml(config_path)
print(f"Model: {config.model_name}, Num repeats per context length: {config.minibatch_size}")

# 2. Setup model and configs
chat_wrapper = load_model(config.model_name, device = 'cuda')
question_config = QuestionConfig(config).initialize_choices(chat_wrapper.tokenizer)


repeats_per_context_length = 32
context_lengths = [0, 1, 2, 5, 10, 15]

num_context_lengths = len(context_lengths)

key_positive_scores = np.array([5, 4, 3, 2, 1])
key_negative_scores = np.array([1, 2, 3, 4, 5])
prob_cols = ['pA', 'pB', 'pC', 'pD', 'pE']


# Set up some control questions to put in context, randomly selected across all question-answer pairs
all_probs = ocean_questions_df[prob_cols].values
all_normalized_probs = all_probs / all_probs.sum(axis=1, keepdims=True)
all_answer_indices = np.argmax(all_normalized_probs, axis=1)
all_answer_letters = np.array(['A', 'B', 'C', 'D', 'E'])[all_answer_indices].tolist()
all_questions_and_answers = [{'question': row['text'], 'answer': answer, 'key': row['key']} for row, answer in zip(ocean_questions_df.to_dict(orient="records"), all_answer_letters)]



for chosen_trait, (ocean_key, ocean_direction) in chosen_trait_to_ocean_and_direction.items():

    all_data = np.zeros([repeats_per_context_length, num_context_lengths, config.minibatch_size, 5])
    control_all_data = np.zeros([repeats_per_context_length, num_context_lengths, config.minibatch_size, 5])

    print(f'#### Beginning ICL for {chosen_trait}')

    # Answers from this personality prompt
    chosen_trait_ocean_questions_df = ocean_questions_df[ocean_questions_df['chosen_trait'] == chosen_trait]

    # Answers where the personality prompt has a bearing on the question relevance
    chosen_trait_matching_ocean_questions_df = chosen_trait_ocean_questions_df[chosen_trait_ocean_questions_df['label_ocean'] == ocean_key]

    # Normalize probabilities to sum to 1 for each row
    relevant_probs = chosen_trait_matching_ocean_questions_df[prob_cols].values
    relevant_normalized_probs = relevant_probs / relevant_probs.sum(axis=1, keepdims=True)

    # Find the answer with the highest probability for each row
    relevant_answer_indices = np.argmax(relevant_normalized_probs, axis=1)
    relevant_answer_letters = np.array(['A', 'B', 'C', 'D', 'E'])[relevant_answer_indices].tolist()

    # Build the list of dictionaries
    relevant_questions_and_answers = [{'question': row['text'], 'answer': answer, 'key': row['key']} for row, answer in zip(chosen_trait_matching_ocean_questions_df.to_dict(orient="records"), relevant_answer_letters)]

    # Select config.minibatch_size questions to be asked for all context lengths, then remove them from possible in-context examples
    asked_questions_idx = random.sample(range(len(relevant_questions_and_answers)), config.minibatch_size)
    asked_questions = [relevant_questions_and_answers[i]['question'] for i in asked_questions_idx]
    asked_keys = [relevant_questions_and_answers[i]['key'] for i in asked_questions_idx]
    relevant_questions_and_answers = [qa for i, qa in enumerate(relevant_questions_and_answers) if i not in asked_questions_idx]


    for cl_idx, context_length in enumerate(context_lengths):

        print(f'### Beginning ICL with {context_length} examples - repeating {repeats_per_context_length} times')

        for rep_idx in tqdm(range(repeats_per_context_length)):
            
            # In-context personality steering - with signal
            ic_qa_batch = random.sample(relevant_questions_and_answers, context_length)
            in_context_questions = [icqa['question'] for icqa in ic_qa_batch]
            in_context_answers = [icqa['answer'] for icqa in ic_qa_batch]

            answers = elicit_mcq_answer(
                chat_wrapper = chat_wrapper,
                questions = asked_questions,
                shared_choices = question_config.mcq_shared_choices,
                config = question_config,
                system_prompt = None,   # Importantly!
                shared_in_context_questions = in_context_questions,
                shared_in_context_answers = in_context_answers,
            )

            choice_probs = answers['choice_logits']

            all_data[rep_idx, cl_idx] = choice_probs.cpu().numpy()

            # In-context personality steering - with noise
            control_in_aq_batch = random.sample(all_questions_and_answers, context_length)
            control_in_context_questions = [icqa['question'] for icqa in control_in_aq_batch]
            control_in_context_answers = [icqa['answer'] for icqa in control_in_aq_batch]

            control_answers = elicit_mcq_answer(
                chat_wrapper = chat_wrapper,
                questions = asked_questions,
                shared_choices = question_config.mcq_shared_choices,
                config = question_config,
                system_prompt = None,   # Importantly!
                shared_in_context_questions = control_in_context_questions,
                shared_in_context_answers = control_in_context_answers,
            )

            control_choice_probs = control_answers['choice_logits']

            control_all_data[rep_idx, cl_idx] = control_choice_probs.cpu().numpy()

        plt.close('all')

        fig, axes = plt.subplots(4, 4, figsize = (24, 24))
        axes = axes.flatten()

        for aq in range(config.minibatch_size):

            axes[aq].set_title(asked_questions[aq])
            color, key_scores = ('blue', key_positive_scores) if asked_keys[aq] == 1 else ('red', key_negative_scores)

            question_relevant_data = all_data[:,:cl_idx + 1,aq]     # [repeats_per_context_length, num_context_lengths done, 5]
            choice_idx = question_relevant_data.argmax(-1)  # [repeats_per_context_length, num_context_lengths done]
            scores_array = key_scores[choice_idx]   # [repeats_per_context_length, num_context_lengths done]
            mean_scores_per_context = scores_array.mean(0)
            std_scores_per_context = scores_array.std(0)
            axes[aq].plot(context_lengths[:cl_idx + 1], mean_scores_per_context, color = color, marker = 'x')
            axes[aq].fill_between(context_lengths[:cl_idx + 1], mean_scores_per_context - std_scores_per_context, mean_scores_per_context + std_scores_per_context, alpha = 0.2, color = color)

            control_question_relevant_data = control_all_data[:,:cl_idx + 1,aq]     # [repeats_per_context_length, num_context_lengths done, 5]
            control_choice_idx = control_question_relevant_data.argmax(-1)  # [repeats_per_context_length, num_context_lengths done]
            control_scores_array = key_scores[control_choice_idx]   # [repeats_per_context_length, num_context_lengths done]
            control_mean_scores_per_context = control_scores_array.mean(0)
            control_std_scores_per_context = control_scores_array.std(0)
            axes[aq].plot(context_lengths[:cl_idx + 1], control_mean_scores_per_context, color = 'green', marker = 'x')
            axes[aq].fill_between(context_lengths[:cl_idx + 1], control_mean_scores_per_context - control_std_scores_per_context, control_mean_scores_per_context + control_std_scores_per_context, alpha = 0.2, color = 'green')

        fig.savefig(f'results/icl_mcq/{chosen_trait.split()[1]}.png')







