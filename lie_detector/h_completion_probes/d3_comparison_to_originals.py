import json, copy
import numpy as np
import pandas as pd
from tqdm import tqdm

from util.util import YamlConfig
import sys, os

from util.lie_detector import load_and_preprocess_data, compute_probe_discriminability

# Main script logic
config_path = sys.argv[1]
args = YamlConfig(config_path)

model_name = args.model_name
max_generation_tokens = args.max_generation_tokens

# Probe-related args
probe_responses_args_name = args.probe_responses_args_name
probe_analysis_args_name = args.probe_analysis_args_name
probe_file_name = args.probe_file_name

# Save to subdirectory
save_base = os.path.join('lie_detector_results/i_discriminative_tokens', args.args_name)
os.makedirs(save_base, exist_ok=True)
args.save(save_base)


# Load original discriminability results
probe_responses_path = os.path.join('lie_detector_results/b_probe_answers', probe_responses_args_name, 'probe_response.csv')
probe_questions_path = f'data/probe_questions/{probe_file_name}.csv'
original_probe_data = load_and_preprocess_data(probe_responses_path, probe_questions_path, calc_log_odds=True)
original_probe_results = compute_probe_discriminability(original_probe_data, 'log_odds', 'truth')
original_probe_results_df = pd.DataFrame(original_probe_results['probe_results'])


# Load new probe information - to foreign key to original probes
new_probe_questions_path = os.path.join(save_base, 'truncated_probe_completions.csv')
new_probe_answers_path = os.path.join(save_base, 'truncated_probe_response.csv')
new_probe_data = load_and_preprocess_data(new_probe_answers_path, new_probe_questions_path, calc_log_odds=True)

new_probe_data = new_probe_data[new_probe_data.question_achieved]
probe_questions = new_probe_data['generated_sequence']
banned_words = ['hiking', 'hiker', 'skydiv', 'parachute', 'extreme', 'base', 'activties', 'skiing', 'skier']
probe_questions = probe_questions[~probe_questions.str.lower().apply(lambda x: any(word in x for word in banned_words))]

new_probe_data['probe'] = new_probe_data['generated_sequence']
new_probe_data['probe_type'] = list(
    zip(
        new_probe_data['probe_idx'].astype(str),
        new_probe_data['generation_type'].astype(str),
        new_probe_data['i_append_string'].astype(str),
        new_probe_data['prefix_length'].astype(str)
    )
)
new_probe_results = compute_probe_discriminability(new_probe_data, 'log_odds', 'truth')
new_probe_results_df = pd.DataFrame(new_probe_results['probe_results'])
probe_type_cols = ['probe_idx', 'generation_type', 'i_append_string', 'prefix_length']
new_probe_results_df[probe_type_cols] = pd.DataFrame(new_probe_results_df['probe_type'].tolist(), index=new_probe_results_df.index)
new_probe_results_df = new_probe_results_df.drop('probe_type', axis = 1)

import pdb; pdb.set_trace()
