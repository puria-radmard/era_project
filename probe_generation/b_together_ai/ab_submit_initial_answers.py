"""
Submit Batches Script

This script creates JSONL files for truth and lie prompts, submits them as batch jobs
to Together.ai, and saves the batch metadata for later collection.
"""

import os
import sys
import json
import pandas as pd
from model.together import (
    TogetherBatchWrapper,
    save_batch_metadata
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


def main(
    *_,
    model_name: str,
    prompt_idx: int,
    questions_data_name: str,
    question_instruction: str,
    save_base: str,
    num_samples: int,
    temperature: float,
    top_p: float
):
    """Main function to submit batch jobs."""
    
    batch_tmp_dir = os.path.join(save_base, 'batch_tmp')
    os.makedirs(batch_tmp_dir, exist_ok=True)
    
    print(f"Submitting batches for model: {model_name}")
    print(f"Save directory: {save_base}")
    
    try:
        # Initialize batch wrapper
        batch_wrapper = TogetherBatchWrapper(model_name)
        
        # Load questions and prompts
        print("Loading questions and prompts...")
        qa_pairs, truth_prompt, lie_prompt = load_questions_and_prompts(
            questions_data_name, prompt_idx
        )
        print(f"Loaded {len(qa_pairs)} question-answer pairs")
        
        # Create truth batch requests
        print("Creating truth batch requests...")
        truth_requests = []
        for question_idx, (question, answer) in enumerate(qa_pairs):
            for generation_idx in range(num_samples):
                user_message = f"{question} {question_instruction}"
                custom_id = f"truth_q{question_idx}_g{generation_idx}"
                
                request = batch_wrapper.format_chat_for_batch(
                    custom_id=custom_id,
                    system_prompt=truth_prompt,
                    user_message=user_message,
                    max_tokens=1024,
                    temperature=temperature,
                    top_p=top_p,
                )
                truth_requests.append(request)
        
        # Create lie batch requests  
        print("Creating lie batch requests...")
        lie_requests = []
        for question_idx, (question, answer) in enumerate(qa_pairs):
            for generation_idx in range(num_samples):
                user_message = f"{question} {question_instruction}"
                custom_id = f"lie_q{question_idx}_g{generation_idx}"
                
                request = batch_wrapper.format_chat_for_batch(
                    custom_id=custom_id,
                    system_prompt=lie_prompt,
                    user_message=user_message,
                    max_tokens=1024,
                    temperature=temperature
                )
                lie_requests.append(request)
        
        print(f"Created {len(truth_requests)} truth requests and {len(lie_requests)} lie requests")
        
        # Create JSONL files
        truth_jsonl_path = os.path.join(batch_tmp_dir, 'truth_batch.jsonl')
        lie_jsonl_path = os.path.join(batch_tmp_dir, 'lie_batch.jsonl')

        batch_wrapper.create_batch_file(truth_requests, truth_jsonl_path)
        batch_wrapper.create_batch_file(lie_requests, lie_jsonl_path)
        
        # Submit batches
        print("\nSubmitting truth batch...")
        truth_batch_id = batch_wrapper.upload_and_submit_batch(truth_jsonl_path)

        print("\nSubmitting lie batch...")
        lie_batch_id = batch_wrapper.upload_and_submit_batch(lie_jsonl_path)
        
        # Save metadata
        print("\nSaving batch metadata...")
        save_batch_metadata(
            save_dir=save_base,
            truth_batch_id=truth_batch_id,
            lie_batch_id=lie_batch_id,
            model_name=model_name,
            total_questions=len(qa_pairs),
            prompt_idx=prompt_idx,
            questions_data_name=questions_data_name,
            question_instruction=question_instruction,
            num_samples=num_samples,
            temperature=temperature
        )
        
        print("\n" + "="*60)
        print("BATCH SUBMISSION COMPLETE!")
        print("="*60)
        print(f"Truth batch ID: {truth_batch_id}")
        print(f"Lie batch ID: {lie_batch_id}")
        print(f"Total requests submitted: {len(truth_requests) + len(lie_requests)}")
        print(f"Metadata saved to: {save_base}/batch_tmp/batch_metadata.json")
        print("\nTo check status and collect results later, run:")
        print(f"python collect_results.py {save_base}")
        print("\nNote: Batches typically complete within a few hours.")
        print("You can check status periodically with the collect script.")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nPossible issues:")
        print("- Check that TOGETHER_API_KEY is set in your .env file")
        print("- Verify that the model supports batch inference")
        print("- Ensure data files exist and are properly formatted")
        sys.exit(1)


if __name__ == '__main__':
    
    config_path = sys.argv[1]
    args = YamlConfig(config_path)
    
    # Extract config parameters
    model_name = args.model_name
    prompt_idx = args.prompt_idx
    questions_data_name = args.questions_data_name
    question_instruction = args.question_instruction
    num_samples = 1
    temperature = 0.0
    top_p = 1.0
    
    # Setup directories
    save_base = os.path.join('probe_generation_results/b_neurips_workshop_results', args.args_name, 'a_initial_questions')
    os.makedirs(save_base, exist_ok=True)
    args.save(save_base)
    
    main(
        model_name=model_name,
        prompt_idx=prompt_idx,
        questions_data_name=questions_data_name,
        question_instruction=question_instruction,
        save_base=save_base,
        num_samples=num_samples,
        temperature=temperature,
        top_p=top_p
    )
