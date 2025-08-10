from lie_detector.h_completion_probes.probe_answers_shared import main
from util.util import YamlConfig

import pandas as pd
import sys, os

config_path = sys.argv[1]
args = YamlConfig(config_path)

system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name
initial_answers_args_name = args.initial_answers_args_name

try:
    batch_size = args.batch_size
except AttributeError:
    batch_size = None

probe_file_name = args.probe_file_name
probe_response_type: str = args.probe_response_type
question_instruction = args.question_instruction

assert probe_response_type == 'yn'

prompt_index = args.prompt_idx

limit_to_lying = args.limit_to_lying


save_base = os.path.join('lie_detector_results/i_discriminative_tokens', args.args_name)
os.makedirs(save_base, exist_ok=True)
args.save(save_base)

probe_response_path = os.path.join(save_base, 'original_probe_response.csv')

probe_questions = pd.read_csv(f'data/probe_questions/{probe_file_name}.csv')['probe']
probe_questions = probe_questions.apply(lambda x: x.split('?')[0])

main(
    initial_answers_args_name=initial_answers_args_name,
    questions_data_name=questions_data_name,
    prompt_index=prompt_index,
    system_prompt=system_prompt,
    question_instruction=question_instruction,
    probe_response_path=probe_response_path,
    probe_questions=probe_questions,
    batch_size=batch_size,
    model_name=model_name,
    limit_to_lying=limit_to_lying
)
