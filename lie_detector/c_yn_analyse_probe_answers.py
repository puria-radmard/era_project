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
prompt_index = args.prompt_idx

assert probe_response_type == 'yn'





def experiment_single_d(data: pd.DataFrame, probe_df: pd.DataFrame, d_value: int,
                       n_samples: int, strategy: str = 'random', 
                       target_types: list = None) -> Dict[str, float]:
    """Run experiment for single D value with multiple probe samplings."""
    
    print(f"  Sampling {n_samples} different sets of {d_value} probes...")
    
    sample_aucs = []  # Store mean AUC for each probe selection
    all_roc_curves = []  # Store ROC curves for plotting
    all_probe_indices = []
    
    for sample_idx in range(n_samples):
        # Sample probe questions
        probe_indices = sample_probe_questions(d_value, probe_df, strategy, target_types)
        
        # Run CV experiment for this probe selection
        cv_results = run_cv_experiment(data, probe_indices, key = 'log_odds')
        
        # Store the mean AUC across folds for this probe selection
        sample_aucs.append(cv_results['mean_auc'])
        all_roc_curves.extend(cv_results['roc_curves'])
        all_probe_indices.append(probe_indices.astype(int).tolist())
    
    return {
        'D': d_value,
        'mean_auc': float(np.mean(sample_aucs)),      # Mean across probe selections
        'std_auc': float(np.std(sample_aucs)),        # Std across probe selections  
        'sample_aucs': sample_aucs,            # Individual results for ANOVA later
        'roc_curves': all_roc_curves,          # All ROC curves for plotting
        'all_probe_indices': all_probe_indices
    }



if __name__ == "__main__":

    # Prepare for saving results
    output_path = os.path.join('lie_detector_results/c_probe_discimination', args.args_name)
    os.makedirs(output_path, exist_ok=True)
    args.save_args(output_path)
    
    # Load data
    probe_responses_path = os.path.join('lie_detector_results/b_probe_answers', probe_responses_args_name, 'probe_response.csv')
    probe_questions_path = f'data/{probe_file_name}.csv'
    data = load_and_preprocess_data(probe_responses_path, probe_questions_path, calc_log_odds=False)
    probe_df = pd.read_csv(probe_questions_path).reset_index().rename(columns={'index': 'probe_question_idx'})
    
    # Check prob_yes + prob_no sums
    data_raw = pd.read_csv(probe_responses_path)
    prob_sums = data_raw['prob_yes'] + data_raw['prob_no']
    print(f"\nProb_yes + prob_no range: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]")
    non_unity_probs = prob_sums[(prob_sums < 0.99) | (prob_sums > 1.01)]
    print(f"Non-unity probability sums: {len(non_unity_probs)} / {len(prob_sums)}")
    
    # Plot probe type analysis (unchanged)
    discriminability_results = plot_probe_type_analysis(data, os.path.join(output_path, 'probe_type_analysis.png'), 'log_odds')
    filename = os.path.join(output_path, 'discriminability_results.json')
    with open(filename, 'w') as f:
        json.dump(discriminability_results, f)
    
    
    # Get probe type information
    probe_type_info = get_probe_type_info(probe_df)    
    all_results = {}
    
    
    # 1. Individual probe type experiments
    for probe_type, type_info in probe_type_info.items():
        print(f"\n1. Running experiment with {probe_type} probes only...")
        
        # Create filtered probe_df for this type
        type_probe_df = probe_df[probe_df['probe_type'] == probe_type].copy()
        print(f'{len(type_probe_df)} rows found')
        
        max_probes_type = type_info['count']
        d_values_type = list(range(1, max_probes_type + 1))
        
        results_type = []
        roc_data_type = {}
        
        results_all_verbose = []
        for d in d_values_type:
            d_results = experiment_single_d(data, type_probe_df, d, n_samples, 'random')
            results_type.append({
                'D': d,
                'strategy': probe_type,
                'mean_auc': d_results['mean_auc'],
                'std_auc': d_results['std_auc'],
                'sample_aucs': d_results['sample_aucs']
            })
            roc_data_type[d] = d_results['roc_curves']
            results_all_verbose.append(d_results)
        
        all_results[probe_type] = {
            'results': pd.DataFrame(results_type),
            'roc_data': roc_data_type
        }

        filename = os.path.join(output_path, f'results_{probe_type}.json')
        # pd.DataFrame(results_type).to_csv(filename, index=False)
        with open(filename, 'w') as f:
            json.dump(results_all_verbose, f)

        # AUC vs D plot (all categories on same plot)
        plot_auc_vs_d_comprehensive(all_results, os.path.join(output_path, 'AUC_vs_d.png'))

        # ROC curves (separate subplots for each category)
        plot_roc_curves_comprehensive(all_results, os.path.join(output_path, 'ROC_curves.png'))


    # 2. All probes experiment
    print(f"\n2. Running experiment with ALL probes...")
    max_probes_all = len(probe_df)
    d_values_all = np.linspace(1, max_probes_all, min(10, max_probes_all), dtype=int)
    d_values_all = np.unique(d_values_all)
    
    results_all = []
    roc_data_all = {}
    results_all_verbose = []
    
    for d in d_values_all:
        d_results = experiment_single_d(data, probe_df, d, n_samples, 'random')
        results_all.append({
            'D': float(d),
            'strategy': 'all_probes',
            'mean_auc': d_results['mean_auc'],
            'std_auc': d_results['std_auc'],
            'sample_aucs': d_results['sample_aucs']
        })
        roc_data_all[d] = d_results['roc_curves']
        results_all_verbose.append(d_results)

    all_results['all_probes'] = {
        'results': pd.DataFrame(results_all),
        'roc_data': roc_data_all
    }
    
    filename = os.path.join(output_path, 'results_all_probes.json')
    # pd.DataFrame(results_all).to_csv(filename, index=False)
    with open(filename, 'w') as f:
        for rav in results_all_verbose: rav['D'] = int(rav['D'])
        json.dump(results_all_verbose, f)

        
    # AUC vs D plot (all categories on same plot)
    plot_auc_vs_d_comprehensive(all_results, os.path.join(output_path, 'AUC_vs_d.png'))
    
    # ROC curves (separate subplots for each category)
    plot_roc_curves_comprehensive(all_results, os.path.join(output_path, 'ROC_curves.png'))
