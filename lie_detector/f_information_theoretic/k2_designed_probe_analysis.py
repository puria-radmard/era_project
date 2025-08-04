#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json

from util.lie_detector import (
    load_and_preprocess_data,
    plot_probe_type_analysis,
)

from util.util import YamlConfig

import sys
import os

config_path = sys.argv[1]
args = YamlConfig(config_path)

probe_file_name = args.probe_file_name
probe_response_type = args.probe_response_type
probe_responses_args_name = args.probe_responses_args_name

from lie_detector.f_information_theoretic.z_util import weighted_linear_regression

assert probe_response_type == 'yn'



if __name__ == "__main__":
    
    save_base = os.path.join('lie_detector_results/f_information_theoretic', args.args_name, 'bald_estimation')
    
    # Load data
    probe_responses_path = os.path.join(save_base, 'probe_response.csv')
    probe_questions_path = os.path.join(save_base, 'optimized_probe_questions.csv')
    data = load_and_preprocess_data(probe_responses_path, probe_questions_path, calc_log_odds=True)
    probe_df = pd.read_csv(probe_questions_path).reset_index().rename(columns={'index': 'probe_question_idx'})

    # Check prob_yes + prob_no sums
    data_raw = pd.read_csv(probe_responses_path)
    prob_sums = data_raw['prob_yes'] + data_raw['prob_no']
    print(f"\nProb_yes + prob_no range: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]")
    non_unity_probs = prob_sums[(prob_sums < 0.99) | (prob_sums > 1.01)]
    print(f"Non-unity probability sums: {len(non_unity_probs)} / {len(prob_sums)}")
    
    # Plot probe type analysis (unchanged)
    discriminability_results = plot_probe_type_analysis(data, os.path.join(save_base, 'probe_type_analysis.png'), 'log_odds', 'truth')
    filename = os.path.join(save_base, 'discriminability_results.json')
    with open(filename, 'w') as f:
        json.dump(discriminability_results, f)
    
    # Scatter against predicted kl score
    fig, ax = plt.subplots(figsize=(7, 5))


    for optimisation_round in probe_df.probe_type.unique():

        reference_snrs = np.abs(np.array([res["effect_size"] for res in discriminability_results['probe_results'] if res['probe_type'] == optimisation_round]))
        y = probe_df[probe_df.probe_type == optimisation_round].a_score.values
        y_err = probe_df[probe_df.probe_type == optimisation_round].a_score_std.values

        # Scatter plot: SNR vs. A-score
        scatter = ax.scatter(
            reference_snrs,
            y,
            edgecolor='k',
            s=80,
            alpha=0.85,
        )
        color = scatter.get_facecolors()[0]

        slope, intercept, correlation, p_value, slope_err, intercept_err = weighted_linear_regression(
            reference_snrs, y, y_err, return_full = True
        )
        reg_x = np.linspace(reference_snrs.min(), reference_snrs.max(), 100)
        reg_y = slope * reg_x + intercept
        ax.plot(reg_x, reg_y, color=color, alpha = 0.6, lw=2, label=f'Linear fit\np={p_value:.2g}')

        # # Optionally annotate stats on the plot
        # stats_text = (
        #     f"Slope: {slope:.3f}\n"
        #     f"Intercept: {intercept:.3f}\n"
        #     f"p-value: {p_value:.2g}"
        # )
        # ax.text(
        #     0.98, 0.02, stats_text,
        #     transform=ax.transAxes,
        #     fontsize=10,
        #     va='bottom', ha='right',
        #     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        # )

        ax.set_xlabel("Probe Discriminability (|Effect Size|, SNR)", fontsize=13)
        ax.set_ylabel("A-score (Optimized Information Gain)", fontsize=13)
        ax.set_title("Optimized Probe Discriminability vs. Information Gain", fontsize=15, pad=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='best', fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(save_base, 'designed_probe_kl_vs_snr.png'), dpi=120)
        plt.close(fig)
