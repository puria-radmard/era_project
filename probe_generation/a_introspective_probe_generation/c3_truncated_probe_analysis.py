
#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
from scipy.stats import ttest_rel
import json
import sys
import os


import json

from probe_generation.a_introspective_probe_generation.b2_original_probe_analysis import compute_probe_discriminability_modified, load_and_preprocess_data_modified, plot_probe_type_analysis_modified
from util.util import YamlConfig

import sys
import os

config_path = sys.argv[1]
args = YamlConfig(config_path)

probe_file_name = args.probe_file_name
probe_response_type = args.probe_response_type

assert probe_response_type == 'yn'


# Prepare for saving results
output_path = os.path.join('probe_generation_results/a_introspective_probe_generation', args.args_name)

# Load data
probe_responses_path = os.path.join(output_path, 'c_truncated_probe_responses.csv')
probe_questions_path = os.path.join(output_path, 'c_truncated_probe_completions.csv')
probe_responses_data, probe_df = load_and_preprocess_data_modified(probe_responses_path, probe_questions_path)

# Check prob_yes + prob_no sums
prob_sums = probe_responses_data['prob_yes'] + probe_responses_data['prob_no']
print(f"\nProb_yes + prob_no range: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]")
non_unity_probs = prob_sums[(prob_sums < 0.99) | (prob_sums > 1.01)]
print(f"Non-unity probability sums: {len(non_unity_probs)} / {len(prob_sums)}")

# Plot probe type analysis (unchanged)
discriminability_results = compute_probe_discriminability_modified(probe_responses_data)
filename = os.path.join(output_path, 'c_truncated_discriminability_results.json')

# Get top 10 and bottom 10 entries by abs(mean(effect_sizes_by_append))
for entry in discriminability_results:
    entry['mean_effect_size'] = np.mean(np.abs(entry['effect_sizes_by_append']))

sorted_results = sorted(discriminability_results, key=lambda x: x['mean_effect_size'])
extremes_results = sorted_results[:10] + sorted_results[-10:]

extremes_probe_idxs = [entry['probe_idx'] for entry in extremes_results]
extremes_probe_responses_data = probe_responses_data[probe_responses_data['probe_question_idx'].isin(extremes_probe_idxs)]

extremes_probe_responses_data['probe'] = extremes_probe_responses_data.generated_sequence
extremes_probe_responses_data['probe_type'] = 'Optimised'

plot_probe_type_analysis_modified(extremes_probe_responses_data, os.path.join(output_path, 'c_truncated_probe_analysis.png'))


with open(filename, 'w') as f:
    json.dump(discriminability_results, f, indent=2)
