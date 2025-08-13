"""
Submit Probe Batches Script

This script creates a batch job for probing model internal states by scoring
"Yes"/"No" tokens across different question contexts and probe questions.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from model.together import (
    TogetherBatchWrapper,
    save_batch_metadata
)
from util.util import YamlConfig


def load_data(
    questions_data_name: str,
    prompt_idx: int,
    probe_file_name: str,
    args_name: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, str, str]:
    """
    Load all necessary data for probe batch submission.
    
    Returns:
        Tuple of (initial_questions_df, initial_answers_df, probe_questions, truth_prompt, lie_prompt)
    """
    # Load initial questions
    initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')
    
    # Load initial answers (from previous batch)
    initial_answers_path = os.path.join('probe_generation_results/b_neurips_workshop_results', args_name, 'a_initial_questions/initial_answers.csv')
    initial_answers_df = pd.read_csv(initial_answers_path)
    
    # Load probe questions
    probe_questions = pd.read_csv(f'data/probe_questions/{probe_file_name}.csv')['probe']
    
    # Load prompts
    with open('data/all_prompts.json', 'r') as f:
        prompts = json.load(f)
    truth_prompt = prompts['truth_prompts'][prompt_idx]
    lie_prompt = prompts['lie_prompts'][prompt_idx]
    
    return initial_questions_df, initial_answers_df, probe_questions, truth_prompt, lie_prompt


def get_trainable_questions(
    initial_answers_df: pd.DataFrame,
    prompt_idx: int,
    limit_to_lying: bool = False
) -> pd.DataFrame:
    """
    Get questions that are suitable for probing based on initial answers.
    
    Args:
        initial_answers_df: DataFrame with initial truth/lie answers
        prompt_idx: The prompt index to filter on
        limit_to_lying: Whether to limit to questions where model successfully lied
        
    Returns:
        Filtered DataFrame of trainable answers
    """
    if limit_to_lying:
        # Original logic: questions where model knows answer, told truth, and told lie
        knows_answer = initial_answers_df[
            (initial_answers_df.get('knows_answer', True)) & 
            (initial_answers_df['told_truth'])
        ]
        trainable_answers = knows_answer[
            (knows_answer['prompt_idx'] == prompt_idx) & 
            (knows_answer['told_lie'])
        ]
    else:
        # Use all answers for the given prompt
        trainable_answers = initial_answers_df[
            initial_answers_df['prompt_idx'] == prompt_idx
        ]
    
    return trainable_answers


def main(
    *_,
    args_name: str,
    model_name: str,
    prompt_idx: int,
    questions_data_name: str,
    question_instruction: str,
    save_base: str,
    probe_file_name: str,
    persona_prompt_in_context: bool,
    append_strings: list[str],
    limit_to_lying: bool = False,
):
    """Main function to submit probe batch jobs."""
    
    print(f"Submitting probe batches for model: {model_name}")
    print(f"Save directory: {save_base}")
    
    # Setup probe answers directory
    probe_answers_dir = os.path.join(save_base, 'probe_answers')
    os.makedirs(probe_answers_dir, exist_ok=True)
    
    batch_tmp_dir = os.path.join(probe_answers_dir, 'batch_tmp')
    os.makedirs(batch_tmp_dir, exist_ok=True)
    
    try:
        # Initialize batch wrapper
        batch_wrapper = TogetherBatchWrapper(model_name)
        
        # Load all data
        print("Loading data...")
        initial_questions_df, initial_answers_df, probe_questions, truth_prompt, lie_prompt = load_data(
            questions_data_name, prompt_idx, probe_file_name, args_name
        )
        
        # Get trainable questions
        trainable_answers = get_trainable_questions(initial_answers_df, prompt_idx, limit_to_lying)
        print(f"Found {len(trainable_answers)} trainable questions")
        print(f"Found {len(probe_questions)} probe questions")
        print(f"Using {len(append_strings)} append strings")
        
        # Create all batch requests
        print("Creating batch requests...")
        all_requests = []
        
        for _, row in trainable_answers.iterrows():
            question_idx = row['question_idx']
            original_question = initial_questions_df['question'][question_idx].strip()
            
            # Strip quotes from CSV responses
            truth_answer = row['truth_answer'].strip().strip('"').strip("'")
            lie_answer = row['lie_answer'].strip().strip('"').strip("'")
            
            # For each probe question
            for probe_idx, probe_question in probe_questions.items():
                # For each append string
                for append_idx, append_string in enumerate(append_strings):
                    probe_with_append = f"{probe_question} {append_string}"
                    
                    # For each context (truth/lie)
                    for truth_flag, (context_prompt, context_answer) in enumerate([
                        (lie_prompt if persona_prompt_in_context else None, lie_answer),    # truth_flag=0: lie context
                        (truth_prompt if persona_prompt_in_context else None, truth_answer) # truth_flag=1: truth context
                    ]):
                        
                        # For each prefill (No/Yes)
                        for yes_no_flag, prefill in enumerate(["No.", "Yes."]):
                            custom_id = f"q{question_idx}_p{probe_idx}_a{append_idx}_t{truth_flag}_yn{yes_no_flag}"
                            
                            request = batch_wrapper.format_chat_for_batch(
                                custom_id=custom_id,
                                system_prompt=context_prompt,
                                in_context_questions=[f"{original_question} {question_instruction}"],
                                in_context_answers=[context_answer],
                                user_message=probe_with_append,
                                prefiller=prefill,
                                max_tokens=0,  # Only score the prefilled token
                                temperature=0.0,
                                logprobs=True
                            )
                            all_requests.append(request)

                            break
                        break
                    break
                break
            break
        
        total_requests = len(all_requests)
        print(f"Created {total_requests} total requests")
        
        # Check if within limits (50k per batch)
        if total_requests > 50000:
            print(f"WARNING: {total_requests} requests exceeds 50k limit. Consider splitting.")
        
        # Create and submit batch
        probe_jsonl_path = os.path.join(batch_tmp_dir, 'probe_batch.jsonl')
        batch_wrapper.create_batch_file(all_requests, probe_jsonl_path)

        print("\nSubmitting probe batch...")
        probe_batch_id = batch_wrapper.upload_and_submit_batch(probe_jsonl_path)
        
        # Save metadata
        print("\nSaving batch metadata...")
        save_batch_metadata(
            save_dir=probe_answers_dir,
            probe_batch_id=probe_batch_id,
            model_name=model_name,
            total_requests=total_requests,
            prompt_idx=prompt_idx,
            questions_data_name=questions_data_name,
            question_instruction=question_instruction,
            probe_file_name=probe_file_name,
            persona_prompt_in_context=persona_prompt_in_context,
            limit_to_lying=limit_to_lying,
            append_strings=append_strings,
            num_trainable_questions=len(trainable_answers),
            num_probe_questions=len(probe_questions)
        )
        
        print("\n" + "="*60)
        print("PROBE BATCH SUBMISSION COMPLETE!")
        print("="*60)
        print(f"Probe batch ID: {probe_batch_id}")
        print(f"Total requests submitted: {total_requests}")
        print(f"Metadata saved to: {probe_answers_dir}/batch_tmp/batch_metadata.json")
        print("\nTo check status and collect results later, run:")
        print(f"python collect_probe_results.py {save_base}")
        print("\nNote: Batches typically complete within a few hours.")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("\nPossible issues:")
        print("- Check that TOGETHER_API_KEY is set in your .env file")
        print("- Verify that initial_answers.csv exists (run submit_batches.py first)")
        print("- Ensure probe questions file exists")
        print("- Check that model supports batch inference")
        sys.exit(1)


if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Usage: python submit_probe_batches.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    args = YamlConfig(config_path)
    
    # Extract config parameters
    model_name = args.model_name
    prompt_idx = args.prompt_idx
    questions_data_name = args.questions_data_name
    question_instruction = args.question_instruction
    probe_file_name = args.probe_file_name
    persona_prompt_in_context = args.persona_prompt_in_context
    limit_to_lying = args.limit_to_lying
    append_strings = args.append_strings
    
    # Setup directories
    save_base = os.path.join('probe_generation_results/b_neurips_workshop_results', args.args_name, 'b_original_probe_questions')
    
    if not os.path.exists(os.path.join('probe_generation_results/b_neurips_workshop_results', args.args_name, 'a_initial_questions/initial_answers.csv')):
        print("ERROR: initial_answers.csv not found. Please run submit_batches.py first.")
        sys.exit(1)
    
    main(
        args_name=args.args_name,
        model_name=model_name,
        prompt_idx=prompt_idx,
        questions_data_name=questions_data_name,
        question_instruction=question_instruction,
        save_base=save_base,
        probe_file_name=probe_file_name,
        persona_prompt_in_context=persona_prompt_in_context,
        append_strings=append_strings,
        limit_to_lying=limit_to_lying
    )