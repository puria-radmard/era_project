import json, copy
import numpy as np
import pandas as pd
from tqdm import tqdm

from model.load import load_model
from util.util import YamlConfig
from util.lying_context import get_context_yn
from util.elicit import generate_discriminative_sequence
import sys, os

# Main script logic
config_path = sys.argv[1]
args = YamlConfig(config_path)

model_name = args.model_name
max_generation_tokens = args.max_generation_tokens

# Probe-related args
probe_responses_args_name = args.probe_responses_args_name
probe_analysis_args_name = args.probe_analysis_args_name
probe_file_name = args.probe_file_name
excluded_probe_types = args.excluded_probe_types
excluded_probe_indices = args.excluded_probe_indices

# Context iteration args
n_samples = args.samples_per_context_length

# Save to subdirectory
save_base = os.path.join('lie_detector_results/i_discriminative_tokens', args.args_name)
os.makedirs(save_base, exist_ok=True)
args.save(save_base)

self_consistency_results_path = os.path.join(save_base, 'truncated_response_completions.csv')

# Load probe questions and discriminability data
print("Loading probe discriminability data...")
probes_df_original = pd.read_csv(f'data/probe_questions/{probe_file_name}.csv')
probes_df = probes_df_original[~probes_df_original['probe_type'].isin(excluded_probe_types)]
print(f"Using probe questions (excluded: {excluded_probe_types})")

# Load discriminability results
discriminability_data_path = os.path.join('lie_detector_results/c_probe_discrimination', probe_analysis_args_name, 'discriminability_results.json')
with open(discriminability_data_path, 'r') as f:
    discriminability_data = json.load(f)

# Get valid probes and sort by discriminability
probe_results = discriminability_data['probe_results']
valid_probe_results = [(i, result) for i, result in enumerate(probe_results) 
                      if result['probe_type'] not in excluded_probe_types
                      and i not in excluded_probe_indices]

valid_probe_results.sort(key=lambda x: abs(x[1]['effect_size']), reverse=True)

# Get most discriminative (for context) and least discriminative (for testing completion)
most_discriminative_probe_results = valid_probe_results[:10]  # Top 10 for context
least_discriminative_probe_results = valid_probe_results[-5:]  # Bottom 5 for testing

print(f"Found {len(valid_probe_results)} valid probes")
print(f"Using top 10 most discriminative for context, bottom 5 least discriminative for completion testing")

# Load model
chat_wrapper = load_model(model_name, device='auto')

# Initialize results storage
results_rows = []

# Generation types to test
generation_types = [
    ('lie-truth_contrastive', True, True),    # lie_maximise=True, do_discriminative=True
    ('truth-lie_contrastive', False, True),   # lie_maximise=False, do_discriminative=True
    ('lie-only', True, False),    # lie_maximise=True, do_discriminative=False
    ('truth-only', False, False)  # lie_maximise=False, do_discriminative=False
]

print(f"\n{'='*80}")
print(f"TESTING SELF-CONSISTENCY WITH CONTEXT LENGTH N=5")
print(f"{'='*80}")

# Generate all context materials for this run
all_context_materials = [get_context_yn(5, most_discriminative_probe_results, probes_df_original, randomly_select=True) for _ in range(n_samples)]

for sample_idx in tqdm(range(n_samples), desc="Processing samples"):
    context_materials = all_context_materials[sample_idx]
    
    # Get truth-favoring and lie-favoring contexts
    truth_questions, truth_answers = context_materials['top_truth_shuffled_together']
    lie_questions, lie_answers = context_materials['top_lie_shuffled_together']
    
    print(f"\nSample {sample_idx}: Truth context has {len(truth_questions)} probe Q&A pairs")
    print(f"Sample {sample_idx}: Lie context has {len(lie_questions)} probe Q&A pairs")
    
    # Create base caches (these will be copied for each generation)
    base_truth_cache = chat_wrapper.create_prompt_cache(
        system_prompt="",
        in_context_questions=truth_questions,
        in_context_answers=truth_answers
    )

    base_lie_cache = chat_wrapper.create_prompt_cache(
        system_prompt="",
        in_context_questions=lie_questions,
        in_context_answers=lie_answers
    )
    
    # Test each of the 5 least discriminative probes
    for probe_idx, probe_result in least_discriminative_probe_results:
        # Get the original probe question
        original_probe_question = probes_df_original.iloc[probe_idx]['probe']
        probe_words = original_probe_question.strip().split()
        
        print(f"  Testing probe {probe_idx}: '{original_probe_question}' ({len(probe_words)} words)")
        
        # Test each prefix length (1 word, 2 words, ..., all words)
        for prefix_length in range(1, len(probe_words) + 1):
            prefix_words = probe_words[:prefix_length]
            prefix_text = ' '.join(prefix_words)
        
            # Test each generation type
            for generation_type, lie_maximise, do_discriminative in generation_types:
                # Generate completion using discriminative sequence
                generated_sequence, sequence_progression, discrimination_scores, full_token_count = generate_discriminative_sequence(
                    chat_wrapper=chat_wrapper,
                    truth_cache=copy.deepcopy(base_truth_cache),
                    lie_cache=copy.deepcopy(base_lie_cache),
                    max_tokens=max_generation_tokens,
                    lie_maximise=lie_maximise,
                    initial_text=prefix_text,
                    stopping_string="?",
                    do_discriminative=do_discriminative
                )
                
                # Save result
                results_rows.append({
                    'sample_idx': sample_idx,
                    'probe_idx': probe_idx,
                    'prefix_length': prefix_length,
                    'prefix_text': prefix_text,
                    'generation_type': generation_type,
                    'generated_sequence': generated_sequence.replace('\n', '\\n'),
                    'full_token_count': full_token_count,
                    'question_achieved': not full_token_count
                })
                
                print(f"    Prefix '{prefix_text}' ({generation_type}): '" + generated_sequence.replace(chr(10), '\\n') + "'")
            
    # Save after each sample
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(self_consistency_results_path, index=False)

print("\nSelf-consistency probe completion analysis complete!")
print(f"Results saved to: {self_consistency_results_path}")
print(f"Generated {len(results_rows)} completion attempts")
print(f"Tested {len(least_discriminative_probe_results)} probe questions with {n_samples} context samples each")

# Print summary statistics
results_df = pd.DataFrame(results_rows)
if len(results_df) > 0:
    print("\nSummary:")
    print(f"Total rows: {len(results_df)}")
    print(f"Successful completions: {results_df['question_achieved'].sum()}")
    print(f"Average discrimination score: {results_df['avg_discrimination'].mean():.4f}")
    print("\nBy generation type:")
    for gen_type in results_df['generation_type'].unique():
        subset = results_df[results_df['generation_type'] == gen_type]
        success_rate = subset['question_achieved'].mean()
        avg_disc = subset['avg_discrimination'].mean()
        print(f"  {gen_type}: {success_rate:.2%} success, {avg_disc:.4f} avg discrimination")