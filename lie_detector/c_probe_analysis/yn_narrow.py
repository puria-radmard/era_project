#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

import json

from util.lie_detector import (
    load_and_preprocess_data,
    plot_auc_vs_d_comprehensive,
    plot_probe_type_analysis,
    plot_roc_curves_comprehensive,
    get_probe_type_info,
    run_cv_experiment,
    sample_probe_questions
)

from util.util import YamlConfig

import sys
import os

config_path = sys.argv[1]
args = YamlConfig(config_path)

probe_file_name = args.probe_file_name
probe_response_type = args.probe_response_type
probe_responses_args_name = args.probe_responses_args_name
n_samples = args.samples_per_classifier_size

questions_data_name = args.questions_data_name
narrow_misalignment_type = args.narrow_misalignment_type

assert n_samples == 0
assert probe_response_type == 'yn'




if __name__ == "__main__":

    # Prepare for saving results
    output_path = os.path.join('lie_detector_results/c_probe_discrimination', args.args_name)
    os.makedirs(output_path, exist_ok=True)
    args.save(output_path)
    
    # Load data
    probe_responses_path = os.path.join('lie_detector_results/b_probe_answers', probe_responses_args_name, 'probe_response.csv')
    probe_questions_path = f'data/probe_questions/{probe_file_name}.csv'
    data = load_and_preprocess_data(probe_responses_path, probe_questions_path, calc_log_odds=True)
    probe_df = pd.read_csv(probe_questions_path).reset_index().rename(columns={'index': 'probe_question_idx'})
    
    # Check prob_yes + prob_no sums
    data_raw = pd.read_csv(probe_responses_path)
    prob_sums = data_raw['prob_yes'] + data_raw['prob_no']
    print(f"\nProb_yes + prob_no range: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]")
    non_unity_probs = prob_sums[(prob_sums < 0.99) | (prob_sums > 1.01)]
    print(f"Non-unity probability sums: {len(non_unity_probs)} / {len(prob_sums)}")



    # Plot probe type analysis - should be the same as was_dm
    discriminability_results = plot_probe_type_analysis(data, os.path.join(output_path, 'probe_type_analysis.png'), 'log_odds', 'truth')
    filename = os.path.join(output_path, 'discriminability_results.json')
    with open(filename, 'w') as f:
        json.dump(discriminability_results, f)



    # Load the initial questions with types and create a mapping from question index to question type
    initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')
    question_type_mapping = dict(enumerate(initial_questions_df['type']))
    question_types = np.array([question_type_mapping[idx] for idx in data['question_idx']])

    # Validate the result
    assert len(question_types) == len(data), f"Question types length {len(question_types)} != data length {len(data)}"
    print(f"Successfully mapped {len(question_types)} question types")
    print(f"Unique question types: {np.unique(question_types)}")

    # Optionally add to the dataframe for analysis
    data['question_type'] = question_types

    ## Type 1: model just gave an unsafe answer at all (1 vs 3 answer types)
    data['answered_misaligned'] = ((data['question_type'] == narrow_misalignment_type) & (data['truth'] == 0)).astype(int)

    ## Type 2: model was just asked about medical advice (2 vs 2 answer types)
    data['asked_narrow'] = (data['question_type'] == narrow_misalignment_type).astype(int)

    ## Type 3: model was misaligned (2 vs 2 answer types)
    data['was_dm'] = (data['truth'] == 0).astype(int)

    classification_types = ['answered_misaligned', 'asked_narrow', 'was_dm']
    
    for classification_type in classification_types:

        # Plot probe type analysis
        discriminability_results = plot_probe_type_analysis(data, os.path.join(output_path, f'probe_type_analysis_{classification_type}.png'), 'log_odds', classification_type)
        filename = os.path.join(output_path, f'discriminability_results_{classification_type}.json')
        with open(filename, 'w') as f:
            json.dump(discriminability_results, f)

