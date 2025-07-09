import yaml
from math import ceil

from model.load import load_model
from util.elicit import elicit_formatted_answer
from util.question import QuestionConfig
from util.experiment import ExperimentConfig

# 1. Load configuration
config_path = 'scripts/p2_selfprompt.yaml'
config = ExperimentConfig.from_yaml(config_path)
print(f"Model: {config.model_name}, Batch size: {config.minibatch_size}")

# 2. Setup model and configs
chat_wrapper = load_model(config.model_name, device = 'cuda')
question_config = QuestionConfig(config).initialize_choices(chat_wrapper.tokenizer)
# cache_data = chat_wrapper.create_prompt_cache(config.system_prompt)



#######

# 1. Check what tokens are being generated
input_text = "The capital of France is"

outputs = chat_wrapper.generate(chats = [input_text])


#######






# 3. Load in traits keywords 
with open('data/p2.yaml', 'r') as f:
    trait_words = yaml.safe_load(f)['trait_words']
questions = [{**{'trait': k}, **{f'd{i + 1}': vi for i, vi in enumerate(v)}} for k, v in trait_words.items()]

results = {}


for i in range(ceil(len(questions) / config.minibatch_size)):

    questions_minibatch = questions[i * config.minibatch_size:(i + 1)*config.minibatch_size]

    # 4. Generate the self prompt
    answers = elicit_formatted_answer(
        chat_wrapper=chat_wrapper,
        freeform_template_name='selfprompt',
        questions = questions_minibatch,
        config = question_config,
        system_prompt = config.system_prompt,
        cache_data = None,
        max_new_tokens = 1024,
        temperature = 0.0,
        do_sample = False
    )


    results.update({question['trait']: answer for question, answer in zip(questions_minibatch, answers)})

    with open('results/p2_selfprompt.yaml', 'w') as f:
        yaml.dump(results, f)

