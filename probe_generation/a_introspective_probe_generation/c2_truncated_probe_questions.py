import sys, os
import pandas as pd
from util.util import YamlConfig

from probe_generation.a_introspective_probe_generation.b_original_probe_questions import main


config_path = sys.argv[1]
args = YamlConfig(config_path)

batch_size = args.batch_size
question_instruction = args.question_instruction
questions_data_name = args.questions_data_name
model_name = args.model_name
prompt_idx = args.prompt_idx
probe_file_name = args.probe_file_name
banned_words = args.banned_words

save_base = os.path.join('probe_generation_results/a_introspective_probe_generation', args.args_name)
probe_response_path = os.path.join(save_base, "c_truncated_probe_responses.csv")

probe_questions_df = pd.read_csv(os.path.join(save_base, "c_truncated_probe_completions.csv"))
probe_questions_df = probe_questions_df[probe_questions_df.question_achieved]
probe_questions = probe_questions_df['generated_sequence']
probe_questions = probe_questions[~probe_questions.str.lower().apply(lambda x: any(word in x for word in banned_words))]

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

