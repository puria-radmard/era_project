#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.lie_detector import (
    load_and_preprocess_data,
    plot_auc_vs_d_comprehensive,
    plot_probe_type_analysis,
    plot_roc_curves_comprehensive,
    get_probe_type_info,
    run_cv_experiment,
    sample_probe_questions
)

from typing import Dict



def experiment_single_d(data: pd.DataFrame, probe_df: pd.DataFrame, d_value: int,
                       n_samples: int = 10, strategy: str = 'random', 
                       target_types: list = None) -> Dict[str, float]:
    """Run experiment for single D value with multiple probe samplings."""
    
    print(f"  Sampling {n_samples} different sets of {d_value} probes...")
    
    sample_aucs = []  # Store mean AUC for each probe selection
    all_roc_curves = []  # Store ROC curves for plotting
    
    for sample_idx in range(n_samples):
        # Sample probe questions
        probe_indices = sample_probe_questions(d_value, probe_df, strategy, target_types)
        
        # Run CV experiment for this probe selection
        cv_results = run_cv_experiment(data, probe_indices)
        
        # Store the mean AUC across folds for this probe selection
        sample_aucs.append(cv_results['mean_auc'])
        all_roc_curves.extend(cv_results['roc_curves'])
    
    return {
        'mean_auc': np.mean(sample_aucs),      # Mean across probe selections
        'std_auc': np.std(sample_aucs),        # Std across probe selections  
        'sample_aucs': sample_aucs,            # Individual results for ANOVA later
        'roc_curves': all_roc_curves           # All ROC curves for plotting
    }



def main_pipeline():
    """Execute full experimental pipeline."""
    
    try:
        prefix = sys.argv[1]
    except IndexError:
        prefix = ""
    
    print("Loading and preprocessing data...")
    
    # Load data
    results_path = f"results/lie_detector/{prefix}questions_1000_probe_prompt4.csv"
    probes_path = f"data/{prefix}probes_with_yn.csv"
    
    data = load_and_preprocess_data(results_path, probes_path)
    probe_df = pd.read_csv(probes_path).reset_index().rename(columns={'index': 'probe_question_idx'})
    
    print(f"Loaded {len(data)} data points")
    print(f"Unique questions: {data['question_idx'].nunique()}")
    print(f"Unique probe questions: {data['probe_question_idx'].nunique()}")
    print(f"Probe types: {probe_df['probe_type'].unique()}")
    
    # Debug: Check data completeness
    print("\nDATA COMPLETENESS CHECK:")
    expected_rows = data['question_idx'].nunique() * 2 * data['probe_question_idx'].nunique()
    print(f"Expected total rows: {expected_rows}")
    print(f"Actual total rows: {len(data)}")
    print(f"Missing rows: {expected_rows - len(data)}")
    
    # Check for each question if we have both truth values
    question_truth_counts = data.groupby('question_idx')['truth'].nunique()
    incomplete_questions = question_truth_counts[question_truth_counts != 2]
    print(f"Questions missing truth=0 or truth=1: {len(incomplete_questions)}")
    
    # Check log odds range
    print(f"\nLOG ODDS ANALYSIS:")
    print(f"Log odds range: [{data['log_odds'].min():.3f}, {data['log_odds'].max():.3f}]")
    print(f"Log odds by truth value:")
    print(data.groupby('truth')['log_odds'].describe())
    
    # Check for extreme values
    extreme_threshold = 10
    extreme_values = data[abs(data['log_odds']) > extreme_threshold]
    print(f"Extreme log odds (|value| > {extreme_threshold}): {len(extreme_values)} / {len(data)}")
    
    # Exploratory analysis
    print("\nExploratory Analysis:")
    print("Truth value distribution:")
    print(data['truth'].value_counts())
    
    print("\nProbe type distribution:")
    print(probe_df['probe_type'].value_counts())
    
    # Check prob_yes + prob_no sums
    data_raw = pd.read_csv(results_path)
    prob_sums = data_raw['prob_yes'] + data_raw['prob_no']
    print(f"\nProb_yes + prob_no range: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]")
    non_unity_probs = prob_sums[(prob_sums < 0.99) | (prob_sums > 1.01)]
    print(f"Non-unity probability sums: {len(non_unity_probs)} / {len(prob_sums)}")
    
    # Plot probe type analysis (unchanged)
    plot_probe_type_analysis(data, f'results/lie_detector/{prefix}probe_type_analysis.png')
    
    # Run comprehensive experiments
    print("\n" + "="*60)
    print("COMPREHENSIVE EXPERIMENTS")
    print("="*60)
    print("Running experiments for all probes + each probe type separately")
    
    n_samples = 15  # Number of different probe selections per D value
    print(f"Using {n_samples} different probe selections per D value")
    
    print("Running comprehensive experiments...")
    
    # Get probe type information
    probe_type_info = get_probe_type_info(probe_df)
    print(f"Found {len(probe_type_info)} probe types:")
    for ptype, info in probe_type_info.items():
        print(f"  {ptype}: {info['count']} probes")
    
    all_results = {}
    
    # 1. All probes experiment
    print(f"\n1. Running experiment with ALL probes...")
    max_probes_all = len(probe_df)
    d_values_all = np.linspace(1, max_probes_all, min(10, max_probes_all), dtype=int)
    d_values_all = np.unique(d_values_all)
    
    results_all = []
    roc_data_all = {}
    
    for d in d_values_all:
        d_results = experiment_single_d(data, probe_df, d, n_samples, 'random')
        results_all.append({
            'D': d,
            'strategy': 'all_probes',
            'mean_auc': d_results['mean_auc'],
            'std_auc': d_results['std_auc'],
            'sample_aucs': d_results['sample_aucs']
        })
        roc_data_all[d] = d_results['roc_curves']
    
    all_results['all_probes'] = {
        'results': pd.DataFrame(results_all),
        'roc_data': roc_data_all
    }
    
    # 2. Individual probe type experiments
    for probe_type, type_info in probe_type_info.items():
        print(f"\n2. Running experiment with {probe_type} probes only...")
        
        # Create filtered probe_df for this type
        type_probe_df = probe_df[probe_df['probe_type'] == probe_type].copy()
        
        max_probes_type = type_info['count']
        d_values_type = list(range(1, max_probes_type + 1))
        
        results_type = []
        roc_data_type = {}
        
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
        
        all_results[probe_type] = {
            'results': pd.DataFrame(results_type),
            'roc_data': roc_data_type
        }
    
    
    # Display results summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for category_name, category_data in all_results.items():
        results_df = category_data['results']
        best_auc_idx = results_df['mean_auc'].idxmax()
        best_auc = results_df.loc[best_auc_idx, 'mean_auc']
        best_auc_std = results_df.loc[best_auc_idx, 'std_auc']
        best_d = results_df.loc[best_auc_idx, 'D']
        
        print(f"{category_name}:")
        print(f"  Best AUC: {best_auc:.3f} ± {best_auc_std:.3f} at D={best_d}")
        print(f"  D range: {results_df['D'].min()}-{results_df['D'].max()}")
        print(f"  AUC range: {results_df['mean_auc'].min():.3f}-{results_df['mean_auc'].max():.3f}")
    
    # Plot comprehensive results
    print("\n" + "="*60)
    print("PLOTTING COMPREHENSIVE RESULTS")
    print("="*60)
    
    # AUC vs D plot (all categories on same plot)
    plot_auc_vs_d_comprehensive(all_results, f'results/lie_detector/{prefix}AUC_vs_d.png')
    
    # ROC curves (separate subplots for each category)
    plot_roc_curves_comprehensive(all_results, f'results/lie_detector/{prefix}ROC_curves.png')
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    for category_name, category_data in all_results.items():
        results_df = category_data['results']
        filename = f'results/lie_detector/{prefix}results_{category_name}.csv'
        results_df.to_csv(filename, index=False)
        print(f"Saved {category_name} results to {filename}")
    
    print(f"\nEach point represents mean ± std across {n_samples} different probe selections")
    print("Analysis complete!")


if __name__ == "__main__":
    main_pipeline()