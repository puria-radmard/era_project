"""
Collect Probe Results Script

This script checks the status of submitted probe batch jobs, downloads completed results,
and processes them into the final CSV format by extracting Yes/No token logprobs.
"""

import os
import sys
import json
import math
import pandas as pd
from pathlib import Path
from model.fireworks import (
    FireworksBatchWrapper,
    load_batch_metadata
)
from util.util import YamlConfig


def parse_custom_id(custom_id: str) -> tuple[int, int, int, int, int]:
    """
    Parse custom ID to extract indices.
    
    Args:
        custom_id: String like "q42_p15_a0_t1_yn0"
        
    Returns:
        Tuple of (question_idx, probe_idx, append_idx, truth_flag, yes_no_flag)
    """
    parts = custom_id.split('_')
    question_idx = int(parts[0][1:])  # Remove 'q'
    probe_idx = int(parts[1][1:])     # Remove 'p'
    append_idx = int(parts[2][1:])    # Remove 'a'
    truth_flag = int(parts[3][1:])    # Remove 't'
    yes_no_flag = int(parts[4][2:])   # Remove 'yn'
    
    return question_idx, probe_idx, append_idx, truth_flag, yes_no_flag


def extract_yesno_logprob(response_data: dict) -> float:
    """
    Extract the logprob of the last Yes/No token from the response.
    
    Args:
        response_data: The response data from the API
        
    Returns:
        Logprob of the last Yes/No token
    """
    logprobs_data = response_data['choices'][0]['logprobs']
    tokens = logprobs_data['tokens']
    token_logprobs = logprobs_data['token_logprobs']
    
    # Search backwards through tokens to find last Yes/No
    for i in range(len(tokens) - 1, -1, -1):
        token = tokens[i].strip().lower()
        if token in ['yes', 'no']:
            return token_logprobs[i]
    
    # If no Yes/No found, this shouldn't happen with proper prefilling
    raise ValueError(f"No Yes/No token found in response tokens: {tokens}")


def process_probe_results_to_dataframe(
    probe_results_path: str
) -> pd.DataFrame:
    """
    Process probe batch results into a pandas DataFrame matching original script format.
    
    Args:
        probe_results_path: Path to probe batch results JSONL
        
    Returns:
        DataFrame with columns: question_idx, truth, probe_question_idx, i_append_string, prob_yes, prob_no
    """
    # Dictionary to collect results grouped by context
    # Key: (question_idx, probe_idx, append_idx, truth_flag)
    # Value: {"yes_logprob": float, "no_logprob": float}
    grouped_results = {}
    
    # Parse batch results
    with open(probe_results_path, 'r') as f:
        for line in f:
            result = json.loads(line.strip())
            custom_id = result['custom_id']
            
            # Parse custom ID
            question_idx, probe_idx, append_idx, truth_flag, yes_no_flag = parse_custom_id(custom_id)
            
            # Extract logprob for the Yes/No token
            logprob = extract_yesno_logprob(result['response']['body'])
            
            # Group key
            key = (question_idx, probe_idx, append_idx, truth_flag)
            
            if key not in grouped_results:
                grouped_results[key] = {}
            
            # Store logprob based on yes/no flag
            if yes_no_flag == 0:  # "No." prefill
                grouped_results[key]['no_logprob'] = logprob
            else:  # "Yes." prefill
                grouped_results[key]['yes_logprob'] = logprob
    
    # Convert to DataFrame rows
    results = []
    
    for (question_idx, probe_idx, append_idx, truth_flag), logprobs in grouped_results.items():
        # Skip if we don't have both yes and no logprobs
        if 'yes_logprob' not in logprobs or 'no_logprob' not in logprobs:
            print(f"Warning: Missing yes or no logprob for q{question_idx}_p{probe_idx}_a{append_idx}_t{truth_flag}")
            continue
        
        # Convert logprobs to probabilities
        yes_logprob = logprobs['yes_logprob']
        no_logprob = logprobs['no_logprob']
        
        # Convert to probabilities using softmax (proper normalization)
        prob_yes = math.exp(yes_logprob)
        prob_no = math.exp(no_logprob)
        
        results.append({
            'question_idx': question_idx,
            'truth': truth_flag,
            'probe_question_idx': probe_idx,
            'i_append_string': append_idx,
            'prob_yes': prob_yes,
            'prob_no': prob_no
        })
    
    return pd.DataFrame(results)


def main(
    *_, 
    save_base: str
):
    """Main function to collect and process probe batch results."""
    
    probe_answers_dir = os.path.join(save_base, 'probe_answers')
    
    if not os.path.exists(probe_answers_dir):
        print(f"ERROR: Probe answers directory {probe_answers_dir} does not exist.")
        print("Make sure you've run submit_probe_batches.py first.")
        sys.exit(1)

    # Load batch metadata
    print("Loading batch metadata...")
    metadata = load_batch_metadata(probe_answers_dir)
    
    probe_batch_id = metadata['probe_batch_id']
    model_name = metadata['model_name']
    total_requests = metadata['total_requests']
    
    print(f"Found probe batch: {probe_batch_id}")
    print(f"Model: {model_name}")
    print(f"Total requests: {total_requests}")
    
    # Initialize batch wrapper
    batch_wrapper = FireworksBatchWrapper(model_name)
    
    # Check batch status
    print("\nChecking batch status...")
    probe_status = batch_wrapper.get_batch_status(probe_batch_id)
    
    print(f"Probe batch status: {probe_status['status']}")
    
    # Check if batch is completed
    if probe_status['status'] != 'JOB_STATE_COMPLETED':
        print("\nBatch not yet completed.")
        print(f"Current status: {probe_status['status']}")
        
        # Check for failure states
        failed_statuses = ['JOB_STATE_FAILED', 'JOB_STATE_EXPIRED', 'JOB_STATE_CANCELLED']
        if probe_status['status'] in failed_statuses:
            print(f"ERROR: Probe batch failed with status: {probe_status['status']}")
            sys.exit(1)
        
        print("\nBatch is still processing. Please check back later.")
        print("Typical completion time: 1-12 hours")
        return
    
    print("\nBatch completed! Downloading results...")
    
    # Setup output directories
    raw_outputs_dir = os.path.join(probe_answers_dir, 'raw_outputs')
    os.makedirs(raw_outputs_dir, exist_ok=True)
    
    # Download probe batch results
    probe_results_path = os.path.join(raw_outputs_dir, 'probe_results.jsonl')
    probe_errors_path = os.path.join(raw_outputs_dir, 'probe_errors.jsonl')
    
    batch_wrapper.download_batch_results(
        probe_batch_id, 
        probe_results_path,
        probe_errors_path
    )
    
    print("Results downloaded successfully!")
    
    # Process results into DataFrame
    print("\nProcessing results into CSV format...")
    df = process_probe_results_to_dataframe(probe_results_path)
    
    # Save final CSV
    output_csv_path = os.path.join(probe_answers_dir, 'probe_results.csv')
    df.to_csv(output_csv_path, index=False)
    
    print(f"\nProcessed {len(df)} probe results")
    print(f"Saved final CSV to: {output_csv_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PROBE COLLECTION COMPLETE!")
    print("="*60)
    
    # Group by truth context for summary
    truth_results = df[df['truth'] == 1]
    lie_results = df[df['truth'] == 0]
    
    print(f"Results summary:")
    print(f"  Total probe results: {len(df)}")
    print(f"  Truth context results: {len(truth_results)}")
    print(f"  Lie context results: {len(lie_results)}")
    print(f"  Unique questions probed: {df['question_idx'].nunique()}")
    print(f"  Unique probe questions: {df['probe_question_idx'].nunique()}")
    print(f"  Raw outputs saved to: {raw_outputs_dir}")
    print(f"  Final CSV saved to: {output_csv_path}")
    
    # Check for any errors
    if os.path.exists(probe_errors_path) and os.path.getsize(probe_errors_path) > 0:
        print(f"\nWARNING: Errors found in {probe_errors_path}")
        print("Check this file for any failed requests.")
    
    # Show sample of results
    if len(df) > 0:
        print(f"\nSample results:")
        print(df.head())
    else:
        print("WARNING: No results to show!")
    


if __name__ == '__main__':
    
    config_path = sys.argv[1]
    args = YamlConfig(config_path)
    
    # Setup directories
    save_base = os.path.join('probe_generation_results/b_neurips_workshop_results', args.args_name, 'b_original_probe_questions')
    args.save(save_base)
    
    main(save_base=save_base)