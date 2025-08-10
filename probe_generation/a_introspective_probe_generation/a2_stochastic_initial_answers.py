import json
import pandas as pd
import os
from model.load import load_model

from tqdm import tqdm
from util.util import YamlConfig

import sys

from probe_generation.a_introspective_probe_generation.a_initial_answers import main


if __name__ == '__main__':

    config_path = sys.argv[1]
    args = YamlConfig(config_path)

    batch_size = args.batch_size
    question_instruction = args.question_instruction
    questions_data_name = args.questions_data_name
    model_name = args.model_name
    prompt_idx = args.prompt_idx

    num_initial_generation_samples = args.num_initial_generation_samples
    temperature = args.temperature
    top_p = args.top_p

    save_base = os.path.join('probe_generation_results/a_introspective_probe_generation', args.args_name)
    os.makedirs(save_base, exist_ok=True)
    args.save(save_base)
    output_path = os.path.join(save_base, "a_stochastic_initial_answers.csv")

    # Load model
    chat_wrapper = load_model(model_name, device='auto')

    
    main(
        chat_wrapper = chat_wrapper,
        initial_answers_df_path = output_path,
        num_samples = num_initial_generation_samples,
        do_sample = True,
        temperature = temperature,
        top_p = top_p,
        prompt_idx = prompt_idx,
        questions_data_name = questions_data_name,
        question_instruction = question_instruction,
        batch_size = batch_size,
    )