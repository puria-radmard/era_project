"""
Collect Results Script

This script checks the status of submitted batch jobs, downloads completed results,
and processes them into the final CSV format matching the original script.
"""

import os
import sys
import json
import pandas as pd
from model.fireworks import (
    FireworksBatchWrapper,
    load_batch_metadata
)
from util.util import YamlConfig


def load_questions_and_prompts(
    questions_data_name: str,
    prompt_idx: int
) -> tuple[list[tuple[str, str]], str, str]:
    """
    Load questions and prompts from data files.
    
    Args:
        questions_data_name: Name of the questions dataset
        prompt_idx: Index of the prompt to use
        
    Returns:
        Tuple of (qa_pairs, truth_prompt, lie_prompt)
    """
    # Load prompts
    with open('data/all_prompts.json', 'r') as f:
        prompts = json.load(f)
    
    truth_prompt = prompts['truth_prompts'][prompt_idx]
    lie_prompt = prompts['lie_prompts'][prompt_idx]
    
    # Load questions
    questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')
    qa_pairs = [
        (questions_df['question'][idx].strip(), str(questions_df['answer'][idx])) 
        for idx in range(len(questions_df['question']))
    ]
    
    # Verify all answers are single words
    assert set([len(qa[1].split()) for qa in qa_pairs]) == {1}, \
        "All answers must be single words"
    
    return qa_pairs, truth_prompt, lie_prompt


def process_batch_results_to_dataframe(
    truth_results_path: str,
    lie_results_path: str,
    qa_pairs: list[tuple[str, str]]
) -> pd.DataFrame:
    """
    Process batch results into a pandas DataFrame matching original script format.
    
    Args:
        truth_results_path: Path to truth batch results JSONL
        lie_results_path: Path to lie batch results JSONL
        qa_pairs: List of (question, answer) pairs
        
    Returns:
        DataFrame with columns matching original script
    """
    # Load results
    truth_results = {}
    lie_results = {}
    all_generation_indices = set()
    
    # Parse truth results
    with open(truth_results_path, 'r') as f:
        for line in f:
            result = json.loads(line.strip())
            custom_id = result['custom_id']
            # Extract question index from custom_id like "truth_q42_g0"
            question_idx = int(custom_id.split('_')[1][1:])  # Remove 'q' and convert to int
            generation_idx = int(custom_id.split('_')[2][1:])  # Remove 'g' and convert to int

            all_generation_indices.add(generation_idx)  # Track all gen indices

            response_content = result['response']['choices'][0]['message'].get('content', '')
            if 'reasoning_content' in result['response']['choices'][0]['message']:
                print('SHOULD CONSIDER EXTRACTING REASONING TOO')

            if response_content == '':
                print('WARNING: empty response')

            truth_results[(question_idx, generation_idx)] = response_content
    
    # Parse lie results  
    with open(lie_results_path, 'r') as f:
        for line in f:
            result = json.loads(line.strip())
            custom_id = result['custom_id']
            question_idx = int(custom_id.split('_')[1][1:])
            generation_idx = int(custom_id.split('_')[2][1:])

            all_generation_indices.add(generation_idx)
            response_content = result['response']['choices'][0]['message'].get('content', '')
            
            if response_content == '':
                print('WARNING: empty response')

            lie_results[(question_idx, generation_idx)] = response_content
    
    # Create DataFrame rows
    results = []
    
    for question_idx, (question, correct_answer) in enumerate(qa_pairs):
        for generation_idx in sorted(all_generation_indices):  # Assuming single generation for now
            key = (question_idx, generation_idx)
            
            if key in truth_results and key in lie_results:
                truth_resp = truth_results[key]
                lie_resp = lie_results[key]
                
                # Evaluate responses (same logic as original script)
                told_truth = correct_answer.lower() in truth_resp.lower()
                told_lie = correct_answer.lower() not in lie_resp.lower()
                
                results.append({
                    'question_idx': question_idx,
                    'prompt_idx': 0,  # Will be filled from metadata
                    'generation_idx': generation_idx,
                    'truth_answer': truth_resp,
                    'lie_answer': lie_resp,
                    'told_truth': told_truth,
                    'told_lie': told_lie,
                })
    
    return pd.DataFrame(results)


def main(
    *_, 
    save_base: str,
    suffix: str
):
    """Main function to collect and process batch results."""
    
    if not os.path.exists(save_base):
        print(f"ERROR: Directory {save_base} does not exist.")
        print("Make sure you've run submit_batches.py first.")
        sys.exit(1)
    
    try:
        # Load batch metadata
        print("Loading batch metadata...")
        metadata = load_batch_metadata(save_base)
        
        truth_batch_id = metadata['truth_batch_id']
        lie_batch_id = metadata['lie_batch_id']
        model_name = metadata['model_name']
        prompt_idx = metadata['prompt_idx']
        questions_data_name = metadata['questions_data_name']
        
        print(f"Found batches:")
        print(f"  Truth batch: {truth_batch_id}")
        print(f"  Lie batch: {lie_batch_id}")
        print(f"  Model: {model_name}")
        
        # Initialize batch wrapper
        batch_wrapper = FireworksBatchWrapper(model_name)
        
        # Check batch statuses
        print("\nChecking batch statuses...")
        truth_status = batch_wrapper.get_batch_status(truth_batch_id)
        lie_status = batch_wrapper.get_batch_status(lie_batch_id)
        
        print(f"Truth batch status: {truth_status['status']}")
        print(f"Lie batch status: {lie_status['status']}")
        
        # Check if both batches are completed
        if truth_status['status'] != batch_wrapper.success_code or lie_status['status'] != batch_wrapper.success_code:
            print("\nBatches not yet completed.")
            print("Current statuses:")
            print(f"  Truth: {truth_status['status']}")
            print(f"  Lie: {lie_status['status']}")

            # Check for failure states
            failed_statuses = ['FAILED', 'EXPIRED', 'CANCELLED']
            if truth_status['status'] in failed_statuses:
                print(f"ERROR: Truth batch failed with status: {truth_status['status']}")
                sys.exit(1)
            
            if lie_status['status'] in failed_statuses:
                print(f"ERROR: Lie batch failed with status: {lie_status['status']}")
                sys.exit(1)
            
            print("\nBatches are still processing. Please check back later.")
            print("Typical completion time: 1-12 hours")
            return
        
        print("\nBoth batches completed! Downloading results...")
        
        # Setup output directories
        raw_outputs_dir = os.path.join(save_base, 'raw_outputs')
        os.makedirs(raw_outputs_dir, exist_ok=True)
        
        # Download truth batch results
        truth_results_path = os.path.join(raw_outputs_dir, 'truth_results.jsonl')
        truth_errors_path = os.path.join(raw_outputs_dir, 'truth_errors.jsonl')
        batch_wrapper.download_batch_results(
            truth_batch_id, 
            truth_results_path,
            truth_errors_path
        )
        
        # Download lie batch results
        lie_results_path = os.path.join(raw_outputs_dir, 'lie_results.jsonl')
        lie_errors_path = os.path.join(raw_outputs_dir, 'lie_errors.jsonl')
        batch_wrapper.download_batch_results(
            lie_batch_id,
            lie_results_path, 
            lie_errors_path
        )
        
        print("Results downloaded successfully!")
        
        # Load questions for processing
        print("\nLoading questions for result processing...")
        qa_pairs, _, _ = load_questions_and_prompts(
            questions_data_name, prompt_idx
        )
        
        # Process results into DataFrame
        print("Processing results into CSV format...")
        df = process_batch_results_to_dataframe(
            truth_results_path,
            lie_results_path,
            qa_pairs
        )

        # Update prompt_idx in DataFrame
        df['prompt_idx'] = prompt_idx
        
        # Save final CSV
        output_csv_path = os.path.join(save_base, f'initial_answers{suffix}.csv')
        df.to_csv(output_csv_path, index=False)
        
        print(f"\nProcessed {len(df)} results")
        print(f"Saved final CSV to: {output_csv_path}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("COLLECTION COMPLETE!")
        print("="*60)
        
        truth_success_rate = df['told_truth'].mean() * 100
        lie_success_rate = df['told_lie'].mean() * 100
        
        print(f"Results summary:")
        print(f"  Total questions processed: {len(df)}")
        print(f"  Truth-telling success rate: {truth_success_rate:.1f}%")
        print(f"  Lie-telling success rate: {lie_success_rate:.1f}%")
        print(f"  Raw outputs saved to: {raw_outputs_dir}")
        print(f"  Final CSV saved to: {output_csv_path}")
        
        # Check for any errors
        error_files = [truth_errors_path, lie_errors_path]
        for error_file in error_files:
            if os.path.exists(error_file) and os.path.getsize(error_file) > 0:
                print(f"\nWARNING: Errors found in {error_file}")
                print("Check this file for any failed requests.")
        
    except FileNotFoundError as e:
        print(f"ERROR: Required file not found: {e}")
        print("Make sure you've run submit_batches.py first and all data files exist.")
        sys.exit(1)



if __name__ == '__main__':
    
    config_path = sys.argv[1]
    args = YamlConfig(config_path)
    
    # Setup directories
    save_base = os.path.join('probe_generation_results/b_neurips_workshop_results', args.args_name, 'a_initial_questions')
    args.save(save_base)
    
    main(save_base=save_base, suffix='')