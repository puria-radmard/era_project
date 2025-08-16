"""
Submit In-Context Steering Batches Script

This script creates batch jobs for testing how in-context examples bias model responses
toward truth-telling or lying behavior across different context lengths.
"""

import os
import sys
import json
import random
import pandas as pd
from typing import Dict, List, Tuple, Any
from model.fireworks import FireworksBatchWrapper, save_batch_metadata
from util.util import YamlConfig

def shared_sign(effect_sizes):
    signs = [1 if x > 0 else -1 if x < 0 else 0 for x in effect_sizes]
    unique_signs = set(signs)
    if len(unique_signs) == 1 and 0 not in unique_signs:
        return signs[0]
    return None


def load_discriminability_data(save_base: str) -> Tuple[List[int], List[int], List[float], List[float]]:
    """
    Load discriminability results and filter to shared-sign probes.
    
    Returns:
        Tuple of (positive_sign_probe_idxs, negative_sign_probe_idxs)
    """
    with open(os.path.join(save_base, 'b_original_probe_questions', 'discriminability_results.json'), 'r') as f:
        discriminability_results = json.load(f)
    
    positive_sign_idxs = []
    positive_magnitudes = []
    negative_sign_idxs = []
    negative_magnitudes = []
    
    # Filter to shared-sign results
    shared_sign_results = [
        item for item in discriminability_results
        if shared_sign(item['effect_sizes_by_append']) is not None
    ]
    
    for item in shared_sign_results:
        effect_sizes = item['effect_sizes_by_append']
        magnitude = sum(abs(x) for x in effect_sizes) / len(effect_sizes)
        
        if shared_sign(effect_sizes) == 1:
            positive_sign_idxs.append(item['probe_idx'])
            positive_magnitudes.append(magnitude)
        else:
            negative_sign_idxs.append(item['probe_idx'])
            negative_magnitudes.append(magnitude)

    print(f"Found {len(positive_sign_idxs)} positive-effect probes, {len(negative_sign_idxs)} negative-effect probes")
    return positive_sign_idxs, negative_sign_idxs, positive_magnitudes, negative_magnitudes


def sample_balanced_context(
    n_context: int,
    positive_probe_idxs: List[int],
    negative_probe_idxs: List[int],
    positive_magnitudes: List[float],
    negative_magnitudes: List[float],
    use_largest_magnitude: bool = False
):
    """
    Randomly sample N context probes with balanced effect signs.
    
    Returns:
        Tuple of (selected_probe_idxs, intended_effect_signs)
    """
    # Validation checks
    assert len(positive_probe_idxs) == len(positive_magnitudes)
    assert len(negative_probe_idxs) == len(negative_magnitudes)
    assert all(x >= 0 for x in positive_magnitudes)
    assert all(x >= 0 for x in negative_magnitudes)

    # Randomly choose intended effect signs (roughly 50/50)
    intended_effect_signs = [random.choice([1, -1]) for _ in range(n_context)]
    
    if use_largest_magnitude:
        # Sort by magnitude instead of shuffling
        pos_sorted = sorted(zip(positive_probe_idxs, positive_magnitudes), key=lambda x: x[1], reverse=True)
        neg_sorted = sorted(zip(negative_probe_idxs, negative_magnitudes), key=lambda x: x[1], reverse=True)
        shuffled_positive = [idx for idx, _ in pos_sorted]
        shuffled_negative = [idx for idx, _ in neg_sorted]
    else:
        # Existing random logic
        shuffled_positive = random.sample(positive_probe_idxs, len(positive_probe_idxs))
        shuffled_negative = random.sample(negative_probe_idxs, len(negative_probe_idxs))
    
    # Track pointers for each pool
    ptrs = {1: 0, -1: 0}
    selected_probe_idxs = []
    
    for i, intended_sign in enumerate(intended_effect_signs):
        if intended_sign == 1 and ptrs[1] < len(shuffled_positive):
            selected_probe_idxs.append(shuffled_positive[ptrs[1]])
            ptrs[1] += 1
        elif intended_sign == -1 and ptrs[-1] < len(shuffled_negative):
            selected_probe_idxs.append(shuffled_negative[ptrs[-1]])
            ptrs[-1] += 1
        else:
            # Fallback to opposite sign if we run out
            fallback_sign = -intended_sign
            if fallback_sign == 1 and ptrs[1] < len(shuffled_positive):
                selected_probe_idxs.append(shuffled_positive[ptrs[1]])
                ptrs[1] += 1
                intended_effect_signs[i] = 1
            elif fallback_sign == -1 and ptrs[-1] < len(shuffled_negative):
                selected_probe_idxs.append(shuffled_negative[ptrs[-1]])
                ptrs[-1] += 1
                intended_effect_signs[i] = -1
            else:
                raise ValueError("Ran out of probes in both pools")
    
    
    # Shuffle selected_probe_idxs and intended_effect_signs together to randomize order
    combined = list(zip(selected_probe_idxs, intended_effect_signs))
    random.shuffle(combined)
    selected_probe_idxs, intended_effect_signs = zip(*combined)
    selected_probe_idxs = list(selected_probe_idxs)
    intended_effect_signs = list(intended_effect_signs)

    return selected_probe_idxs, intended_effect_signs


def generate_context_answers(intended_effect_signs: List[int]) -> Dict[str, List[str]]:
    """
    Generate context answers for different context types.
    
    Returns:
        Dict mapping context_type to list of answers
    """
    aligned_answers = ['Yes.' if sign == 1 else 'No.' for sign in intended_effect_signs]
    misaligned_answers = ['No.' if sign == 1 else 'Yes.' for sign in intended_effect_signs]
    random_answers = [random.choice(['Yes.', 'No.']) for _ in intended_effect_signs]
    
    return {
        'aligned': aligned_answers,
        'misaligned': misaligned_answers,
        'random': random_answers
    }


def create_batch_requests(
    batch_wrapper: FireworksBatchWrapper,
    context_lengths: List[int],
    n_samples: int,
    probe_questions_df: pd.DataFrame,
    stochastic_df: pd.DataFrame,
    initial_questions_df: pd.DataFrame,
    question_instruction: str,
    append_strings: List[str],
    positive_probe_idxs: List[int],
    negative_probe_idxs: List[int],
    positive_probe_magnitudes: List[float],
    negative_probe_magnitudes: List[float],
    use_largest_magnitude: bool
) -> Dict[int, List[Dict[str, Any]]]:
    """Create all batch requests for the experiment."""
    
    all_requests = {}
    unique_eval_questions = stochastic_df['question_idx'].unique()
    context_types = ['aligned', 'misaligned', 'random']
    
    for context_length in context_lengths:
        print(f"Creating requests for context length: {context_length}")

        this_context_length_requests = []
        
        for sample_idx in range(n_samples):
            print(f"  Sample {sample_idx + 1}/{n_samples}")
            
            if context_length == 0:
                # No context case
                context_questions = []
                context_answers_by_type = {ct: [] for ct in context_types}
            else:
                # Sample balanced context
                selected_probe_idxs, intended_effect_signs = sample_balanced_context(
                    context_length,
                    positive_probe_idxs, negative_probe_idxs, 
                    positive_probe_magnitudes, negative_probe_magnitudes,
                    use_largest_magnitude
                )
                
                # Create context questions with random append strings
                context_questions = []
                for probe_idx in selected_probe_idxs:
                    base_question = probe_questions_df.iloc[probe_idx]['probe']
                    append_string = random.choice(append_strings)
                    context_questions.append(f"{base_question} {append_string}")
                
                # Generate answers for each context type
                context_answers_by_type = generate_context_answers(intended_effect_signs)
            
            # For each context type
            for context_type in context_types:
                context_answers = context_answers_by_type[context_type]
                
                # For each evaluation question
                for eval_question_idx in unique_eval_questions:
                    question_text = initial_questions_df.iloc[eval_question_idx]['question']
                    question_data = stochastic_df[stochastic_df['question_idx'] == eval_question_idx]
                    
                    # For each response type (truth/lie)
                    for response_type, responses in [
                        ('truth', question_data['truth_answer'].tolist()),
                        ('lie', question_data['lie_answer'].tolist())
                    ]:
                        for resp_idx, response in enumerate(responses):
                            custom_id = f"N{context_length}_s{sample_idx}_ct{context_type}_q{eval_question_idx}_r{response_type}{resp_idx}"
                            
                            request = batch_wrapper.format_chat_for_batch(
                                custom_id=custom_id,
                                system_prompt=None,
                                in_context_questions=context_questions,
                                in_context_answers=context_answers,
                                user_message=f"{question_text} {question_instruction}",
                                prefiller=response.strip('"'),
                                max_tokens=0,  # Only score, don't generate
                                temperature=0.0,
                                logprobs=True,
                                echo=True
                            )
                            this_context_length_requests.append(request)
                
            if context_length == 0:
                break
        
        all_requests[context_length] = this_context_length_requests

    return all_requests


def main(
    args_name: str,
    model_name: str,
    questions_data_name: str,
    question_instruction: str,
    probe_file_name: str,
    context_lengths: List[int],
    n_samples: int,
    append_strings: List[str],
    use_largest_magnitude: bool,
    # banned_words: List[str]
    subdir_name: str
):
    """Main function to submit in-context steering batch jobs."""
    
    # Setup paths
    save_base = os.path.join('probe_generation_results/b_neurips_workshop_results', args_name)
    steering_dir = os.path.join(save_base, subdir_name)
    os.makedirs(steering_dir, exist_ok=True)
    
    batch_tmp_dir = os.path.join(steering_dir, 'batch_tmp')
    os.makedirs(batch_tmp_dir, exist_ok=True)
    
    print(f"Submitting in-context steering batches for model: {model_name}")
    print(f"Context lengths: {context_lengths}")
    print(f"Samples per length: {n_samples}")
    
    # Initialize batch wrapper
    batch_wrapper = FireworksBatchWrapper(model_name)
    
    # Load data
    print("Loading data...")
    probe_questions_df = pd.read_csv(f'data/probe_questions/{probe_file_name}.csv')
    
    # # Filter out banned words
    # probe_questions_df = probe_questions_df[
    #     ~probe_questions_df['probe'].str.lower().apply(
    #         lambda x: any(word in x for word in banned_words)
    #     )
    # ]
    
    initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')
    stochastic_df = pd.read_csv(os.path.join(save_base, 'a2_stochastic_initial_questions', 'initial_answers_stochastic.csv'))
    
    positive_probe_idxs, negative_probe_idxs, positive_probe_magnitudes, negative_probe_magnitudes = load_discriminability_data(save_base)
    
    # Create all batch requests
    print("Creating batch requests...")
    all_requests = create_batch_requests(
        batch_wrapper=batch_wrapper,
        context_lengths=context_lengths,
        n_samples=n_samples,
        probe_questions_df=probe_questions_df,
        stochastic_df=stochastic_df,
        initial_questions_df=initial_questions_df,
        question_instruction=question_instruction,
        append_strings=append_strings,
        positive_probe_idxs=positive_probe_idxs,
        negative_probe_idxs=negative_probe_idxs,
        positive_probe_magnitudes=positive_probe_magnitudes,
        negative_probe_magnitudes=negative_probe_magnitudes,
        use_largest_magnitude=use_largest_magnitude
    )

    num_requests = sum(len(x) for x in all_requests.values())
    print(f"Created {num_requests} total requests:")

    batch_ids = {}

    for k, v in all_requests.items():
        print(f"\tN={k} has {len(v)} requests")

        # Create and submit batch
        steering_jsonl_path = os.path.join(batch_tmp_dir, f'steering_batch_N{k}.jsonl')
        batch_wrapper.create_batch_file(v, steering_jsonl_path)
    
        print("\t\tSubmitting steering batch...")
        batch_ids[k] = batch_wrapper.upload_and_submit_batch(steering_jsonl_path)

    # Save metadata
    print("\t\tSaving batch metadata...")
    save_batch_metadata(
        save_dir=steering_dir,
        steering_batch_ids_by_context=batch_ids,
        model_name=model_name,
        total_requests=num_requests,
        context_lengths=context_lengths,
        n_samples=n_samples,
        questions_data_name=questions_data_name,
        question_instruction=question_instruction,
        probe_file_name=probe_file_name,
        append_strings=append_strings,
        # banned_words=banned_words,
        num_positive_probes=len(positive_probe_idxs),
        num_negative_probes=len(negative_probe_idxs)
    )
    
    print("\n" + "="*60)
    print("IN-CONTEXT STEERING BATCH SUBMISSION COMPLETE!")
    print("="*60)
    print(f"Batch IDs: {batch_ids}")
    print(f"Total requests: {num_requests}")
    print(f"Context lengths tested: {context_lengths}")
    print(f"Samples per length: {n_samples}")


if __name__ == '__main__':
    config_path = sys.argv[1]
    args = YamlConfig(config_path)
    
    main(
        args_name=args.args_name,
        model_name=args.model_name,
        questions_data_name=args.questions_data_name,
        question_instruction=args.question_instruction,
        probe_file_name=args.probe_file_name,
        context_lengths=args.context_lengths_icl,
        n_samples=args.n_samples_icl,
        append_strings=args.append_strings,
        # banned_words=args.banned_words
        use_largest_magnitude=False,
        subdir_name='c_in_context_liar'
    )
