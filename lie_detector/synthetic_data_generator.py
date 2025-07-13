#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
from typing import List, Tuple


def generate_synthetic_probes(n_probes: int = 20) -> pd.DataFrame:
    """Generate synthetic probe questions with varying discriminative power."""
    
    # Create probe questions with different discrimination levels
    # discrimination_strength ranges from 0 (pure noise) to 1 (perfect discrimination)
    discrimination_strengths = np.linspace(0.0, 0.8, n_probes)
    
    probes = []
    for i, strength in enumerate(discrimination_strengths):
        probes.append({
            'probe_type': f'Type_{i//5}',  # Group every 5 probes into same type
            'probe': f'Synthetic probe {i} (discrimination={strength:.2f})'
        })
    
    return pd.DataFrame(probes)


def generate_synthetic_responses(n_questions: int, n_probes: int, 
                               discrimination_strengths: np.ndarray) -> pd.DataFrame:
    """Generate synthetic question responses with controllable probe discrimination."""
    
    results = []
    
    for question_idx in range(1, n_questions + 1):
        for truth in [0, 1]:
            for probe_idx in range(n_probes):
                # Get discrimination strength for this probe
                disc_strength = discrimination_strengths[probe_idx]
                
                # Generate base probability (random component)
                base_prob = np.random.beta(2, 2)  # Beta distribution for realistic probabilities
                
                # Add truth-dependent signal based on discrimination strength
                if disc_strength > 0:
                    # Signal strength: positive for truth=1, negative for truth=0
                    signal = disc_strength * (2 * truth - 1)  # Maps to [-disc_strength, +disc_strength]
                    
                    # Apply signal using sigmoid-like transformation
                    logit_base = np.log(base_prob / (1 - base_prob + 1e-10))
                    logit_adjusted = logit_base + signal * 2  # Scale signal
                    prob_yes = 1 / (1 + np.exp(-logit_adjusted))
                else:
                    # Pure noise case
                    prob_yes = base_prob
                
                # Ensure probability is in valid range
                prob_yes = np.clip(prob_yes, 1e-6, 1 - 1e-6)
                
                # Generate prob_no (mostly 1 - prob_yes with small noise)
                prob_no = 1 - prob_yes + np.random.normal(0, 0.05)
                prob_no = np.clip(prob_no, 1e-6, 1 - 1e-6)
                
                # Normalize to ensure they don't sum to exactly 1 (realistic)
                total = prob_yes + prob_no
                prob_yes = prob_yes / total * (0.98 + np.random.uniform(0, 0.04))
                prob_no = prob_no / total * (0.98 + np.random.uniform(0, 0.04))
                
                results.append({
                    'question_idx': question_idx,
                    'truth': truth,
                    'probe_question_idx': probe_idx,
                    'prob_yes': prob_yes,
                    'prob_no': prob_no
                })
    
    return pd.DataFrame(results)


def create_synthetic_dataset(n_questions: int = 100, n_probes: int = 20, 
                           output_dir: str = 'synthetic_data') -> None:
    """Create complete synthetic dataset matching original schema."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/lie_detector', exist_ok=True)
    
    print(f"Generating synthetic dataset with {n_questions} questions and {n_probes} probes...")
    
    # Generate probe questions with progressive discrimination from 0 to 1
    probes_df = generate_synthetic_probes(n_probes)
    discrimination_strengths = np.linspace(0.0, 1.0, n_probes)
    
    print("Probe discrimination strengths:")
    for i, strength in enumerate(discrimination_strengths):
        print(f"  Probe {i}: {strength:.3f}")
    
    # Generate responses
    results_df = generate_synthetic_responses(n_questions, n_probes, discrimination_strengths)
    
    # Save files
    probes_path = f'data/synthetic_probes_with_yn.csv'
    results_path = 'results/lie_detector/synthetic_questions_1000_probe_prompt4.csv'
    
    probes_df.to_csv(probes_path, index=False)
    results_df.to_csv(results_path, index=False)
    
    print(f"\nSaved synthetic data:")
    print(f"  Probes: {probes_path}")
    print(f"  Results: {results_path}")
    print(f"  Total data points: {len(results_df)}")
    
    # Verify data integrity
    print(f"\nData integrity check:")
    expected_rows = n_questions * 2 * n_probes
    print(f"  Expected rows: {expected_rows}")
    print(f"  Actual rows: {len(results_df)}")
    print(f"  Questions: {results_df['question_idx'].nunique()}")
    print(f"  Probes: {results_df['probe_question_idx'].nunique()}")
    print(f"  Truth values: {sorted(results_df['truth'].unique())}")
    
    # Check probability ranges
    print(f"\nProbability ranges:")
    print(f"  prob_yes: [{results_df['prob_yes'].min():.6f}, {results_df['prob_yes'].max():.6f}]")
    print(f"  prob_no: [{results_df['prob_no'].min():.6f}, {results_df['prob_no'].max():.6f}]")
    
    prob_sums = results_df['prob_yes'] + results_df['prob_no']
    print(f"  prob_yes + prob_no: [{prob_sums.min():.6f}, {prob_sums.max():.6f}]")
    
    return probes_path, results_path


if __name__ == "__main__":
    # Generate synthetic data with progressive discrimination from 0 to 1
    probes_path, results_path = create_synthetic_dataset(
        n_questions=100,  # Smaller for testing
        n_probes=20,
        output_dir='synthetic_data'
    )
    
    print(f"\n" + "="*60)
    print("USAGE:")
    print("To test with synthetic data, modify main script paths:")
    print(f'  results_path = "{results_path}"')
    print(f'  probes_path = "{probes_path}"')
    print("="*60)