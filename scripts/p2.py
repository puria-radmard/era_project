import json
import torch

from model.load import load_model
from util.elicit import elicit_mcq_answer, elicit_sentence_answer
from util.question import QuestionConfig
from util.experiment import ExperimentConfig

# 1. Load configuration
config_path = 'scripts/p2.yaml'
config = ExperimentConfig.from_yaml(config_path)
print(f"Model: {config.model_name}, Batch size: {config.minibatch_size}")

# 2. Setup model and configs
chat_wrapper = load_model(config.model_name, device = 'cuda')
question_config = QuestionConfig(config).initialize_choices(chat_wrapper.tokenizer)
cache_data = chat_wrapper.create_prompt_cache(config.system_prompt)

elicit_sentence_answer(
    chat_wrapper, batch_questions, question_config, 
    cache_data=cache_data, max_new_tokens=30
)
