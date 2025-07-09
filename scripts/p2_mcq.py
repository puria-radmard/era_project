import yaml

from model.load import load_model
from util.elicit import elicit_mcq_answer, elicit_formatted_answer
from util.question import QuestionConfig
from util.experiment import ExperimentConfig
import pandas as pd
import numpy as np

from tqdm import tqdm



# 1. Load configuration
config_path = 'scripts/p2_mcq.yaml'
config = ExperimentConfig.from_yaml(config_path)
print(f"Model: {config.model_name}, Batch size: {config.minibatch_size}")

# 2. Setup model and configs
chat_wrapper = load_model(config.model_name, device = 'cuda')
question_config = QuestionConfig(config).initialize_choices(chat_wrapper.tokenizer)
# cache_data = chat_wrapper.create_prompt_cache(config.system_prompt)

# 3.1. Load in self-prompted descriptions
with open('results/p2_selfprompt.yaml', 'r') as f:
    trait_descriptions = yaml.safe_load(f)


# 3.2. Load in question fillers
ocean_questions_df = pd.read_csv('data/mpi_1k.csv')  # Update with your actual dataset path

output_df = pd.DataFrame()

for chosen_trait, trait_description in trait_descriptions.items():

    print(f'Begining answers for {chosen_trait} person')

    for i in tqdm(range(0, len(ocean_questions_df), config.minibatch_size)):

        df_batch = ocean_questions_df.iloc[i:i+config.minibatch_size]
        questions = df_batch['text'].tolist()
        
        answers = elicit_mcq_answer(
            chat_wrapper=chat_wrapper,
            questions = questions,
            shared_choices=question_config.mcq_shared_choices,
            config = question_config,
            system_prompt = config.system_prompt.format(description = trait_description),
        )

        choice_probs = answers['choice_logits']

        # answers = elicit_formatted_answer(
        #     chat_wrapper=chat_wrapper,
        #     freeform_template_name='freeform',
        #     questions = [{'question': qu} for qu in questions],
        #     config = question_config,
        #     system_prompt = config.system_prompt.format(description = trait_description),
        #     temperature = 0.0,
        #     do_sample = False
        # )

        # print(answers[0])


        # Convert choice_probs to numpy if it's a torch tensor
        choice_probs_np = choice_probs.detach().cpu().numpy()

        # Create DataFrame for probabilities with appropriate column names
        prob_cols = [f'p{chr(ord("A")+i)}' for i in range(choice_probs_np.shape[1])]
        probs_df = pd.DataFrame(choice_probs_np, columns=prob_cols)

        # Add the chosen_trait column
        probs_df['chosen_trait'] = chosen_trait

        # Concatenate with the original batch DataFrame (reset index to align)
        extended_df = pd.concat([df_batch.reset_index(drop=True), probs_df], axis=1)
        output_df = pd.concat([output_df, extended_df], ignore_index=True)

        # Save the damn thing
        output_df.to_csv('results/p2_mcq_probs.csv', index=False)


