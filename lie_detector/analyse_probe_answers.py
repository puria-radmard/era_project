#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.lie_detector import (
    load_and_preprocess_data,
    experiment_single_d,
    sample_probe_questions,
    plot_auc_vs_d,
    plot_probe_type_analysis,
    plot_roc_curves,
)


def experiment_vary_d(data: pd.DataFrame, probe_df: pd.DataFrame, d_values: list,
                     n_samples: int = 10, strategy: str = 'random', 
                     target_types: list = None) -> tuple:
    """Run experiments across different D values with multiple probe samplings per D."""
    
    results = []
    roc_data_dict = {}
    
    for d in d_values:
        print(f"Running experiment with D={d}, strategy={strategy}")
        
        # Run experiment with multiple probe samplings for this D
        d_results = experiment_single_d(
            data=data,
            probe_df=probe_df,
            d_value=d,
            n_samples=n_samples,
            strategy=strategy,
            target_types=target_types
        )
        
        results.append({
            'D': d,
            'strategy': strategy,
            'mean_auc': d_results['mean_auc'],
            'std_auc': d_results['std_auc'],
            'sample_aucs': d_results['sample_aucs']  # Store for potential ANOVA
        })
        
        # Store ROC curve data
        roc_data_dict[d] = d_results['roc_curves']
    
    return pd.DataFrame(results), roc_data_dict


def main_pipeline():
    """Execute full experimental pipeline."""
    
    prefix = sys.argv[1]
    
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
    
    # Plot probe type analysis
    plot_probe_type_analysis(data, f'results/lie_detector/{prefix}probe_type_analysis.png')
    
    # Define D values to test
    max_probes = data['probe_question_idx'].nunique()
    d_values = np.linspace(1, max_probes, min(10, max_probes), dtype=int)
    d_values = np.unique(d_values)  # Remove duplicates
    
    print(f"\nTesting D values: {d_values}")
    
    # Experiment: Random sampling with multiple probe selections per D
    print("\n" + "="*50)
    print("EXPERIMENT: Random Probe Sampling")
    print("="*50)
    print("Note: For each D, sampling multiple different probe sets to get uncertainty bounds")
    
    n_samples = 20  # Number of different probe selections per D value
    print(f"Using {n_samples} different probe selections per D value")
    
    results_random, roc_data = experiment_vary_d(
        data=data,
        probe_df=probe_df,
        d_values=d_values,
        n_samples=n_samples,
        strategy='random'
    )
    
    print("Random sampling results:")
    print(results_random[['D', 'mean_auc', 'std_auc']])
    
    # Plot results
    print("\n" + "="*50)
    print("PLOTTING RESULTS")
    print("="*50)
    
    plot_auc_vs_d(results_random, None, f'results/lie_detector/{prefix}AUC_vs_d.png')
    plot_roc_curves(roc_data, f'results/lie_detector/{prefix}ROC_curves.png')
    
    # Summary statistics
    print("\nSummary:")
    print(f"Best random AUC: {results_random['mean_auc'].max():.3f} ± {results_random.loc[results_random['mean_auc'].idxmax(), 'std_auc']:.3f} at D={results_random.loc[results_random['mean_auc'].idxmax(), 'D']}")
    print(f"Training set size: ~{100/5:.0f}% of data (1/{5} of questions)")
    print(f"Test set size: ~{100*4/5:.0f}% of data ({4}/{5} of questions)")
    print(f"Each point represents mean ± std across {n_samples} different probe selections")
    
    # Save results
    results_random.to_csv(f'results/lie_detector/{prefix}results_random_sampling.csv', index=False)
    print("\nResults saved to CSV file.")


if __name__ == "__main__":
    main_pipeline()