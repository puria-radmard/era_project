#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
from typing import List
import random

# Assuming these are imported from your codebase
from model.base import load_model
from util.elicit import elicit_next_token_probs

chat_wrapper = load_model("meta-llama/Llama-3.1-8B-Instruct", device='auto')

# 1. Load original answers dataframe
print("Loading original answers...")
original_answers_df = pd.read_csv('results/lie_detector/questions_1000_all.csv')

# 2. Load probe results dataframe  
probe_results_df = pd.read_csv('results/lie_detector/questions_1000_probe_prompt4.csv')

# 3. Filter original_answers_df to prompt_idx=4 and question_idx in probe data
valid_question_ids = set(probe_results_df['question_idx'].unique())
filtered_answers_df = original_answers_df[
    (original_answers_df['prompt_idx'] == 4) & 
    (original_answers_df['question_idx'].isin(valid_question_ids))
].copy()

print(f"Filtered to {len(filtered_answers_df)} questions")

# 4. Load questions data and create QA pairs
print("Loading test questions...")
with open('data/questions_1000_all.json', 'r') as f:
    question_data = json.load(f)

qa_pairs = [(question_data['question'][f'{idx}'].strip(), question_data['answer'][f'{idx}']) 
            for idx in range(len(question_data['question']))]

# Filter QA pairs to only include questions we have in filtered_answers_df
valid_qa_pairs = [qa_pairs[idx] for idx in filtered_answers_df['question_idx']]
print(f"Using {len(valid_qa_pairs)} test questions")

# Load probe questions for context
probes_df = pd.read_csv('data/probes_with_yn.csv')
probe_questions = probes_df['probe'].tolist()

# Prepare knowledge answers for tokenization
knowledge_answers = filtered_answers_df['knowledge_answer'].tolist()
# Remove trailing periods if present
knowledge_answers_clean = [ans.rstrip('.') if ans.endswith('.') else ans for ans in knowledge_answers]

# Tokenize knowledge answers to get first token ID for each
knowledge_token_ids = []
for ans in knowledge_answers_clean:
    tokens = chat_wrapper.tokenizer.encode(ans, add_special_tokens=False)
    if tokens:
        knowledge_token_ids.append(tokens[0])  # Take first token
    else:
        knowledge_token_ids.append(None)  # Skip if no tokens

# Remove entries with no valid tokens
valid_indices = [i for i, token_id in enumerate(knowledge_token_ids) if token_id is not None]
valid_qa_pairs = [valid_qa_pairs[i] for i in valid_indices]
knowledge_token_ids = [knowledge_token_ids[i] for i in valid_indices]

print(f"Valid questions after tokenization: {len(valid_qa_pairs)}")

# Context lengths to test
context_lengths = [0, 1, 2, 5, 10, 15, 20]
n_samples = 15
batch_size = 64

# System prompt
system_prompt = "Answer the following questions accurately."

results = []

for N in context_lengths:
    print(f"\nTesting context length N={N}")
    
    question_probs_across_samples = [[] for _ in range(len(valid_qa_pairs))]
    
    for sample_idx in range(n_samples):
        print(f"  Sample {sample_idx + 1}/{n_samples}")
        
        # Sample N probe questions and random yes/no answers
        if N > 0:
            sampled_probes = random.sample(probe_questions, min(N, len(probe_questions)))
            sampled_answers = [random.choice(["Yes.", "No."]) for _ in range(len(sampled_probes))]
            shared_in_context_questions = sampled_probes
            shared_in_context_answers = sampled_answers
        else:
            shared_in_context_questions = None
            shared_in_context_answers = None
        
        # Extract just the questions for elicit_next_token_probs
        test_questions = [qa[0] for qa in valid_qa_pairs]
        
        # Process in batches
        all_probs = []
        
        for i in range(0, len(test_questions), batch_size):
            batch_questions = test_questions[i:i+batch_size]
            
            # Call elicit_next_token_probs
            result = elicit_next_token_probs(
                chat_wrapper=chat_wrapper,
                questions=batch_questions,
                system_prompt=system_prompt,
                shared_in_context_questions=shared_in_context_questions,
                shared_in_context_answers=shared_in_context_answers
            )
            
            batch_probs = result["probs"]  # [batch_size, vocab_size]
            all_probs.append(batch_probs)
        
        # Concatenate all batch results
        all_probs = torch.cat(all_probs, dim=0)  # [num_questions, vocab_size]
        
        # Extract probabilities for knowledge answer tokens
        for q_idx, token_id in enumerate(knowledge_token_ids):
            prob = all_probs[q_idx, token_id].item()
            question_probs_across_samples[q_idx].append(prob)
    
    # Average probabilities across samples for each question
    question_avg_probs = [np.mean(probs) for probs in question_probs_across_samples]
    
    # Store results
    results.append({
        'context_length': N,
        'mean_prob': np.mean(question_avg_probs),
        'std_prob': np.std(question_avg_probs),
        'question_probs': question_avg_probs
    })
    
    print(f"  Mean probability: {np.mean(question_avg_probs):.4f} ± {np.std(question_avg_probs):.4f}")

# Plot results
print("\nPlotting results...")

context_lengths = [r['context_length'] for r in results]
mean_probs = [r['mean_prob'] for r in results]
std_probs = [r['std_prob'] for r in results]

plt.figure(figsize=(10, 6))
plt.errorbar(context_lengths, mean_probs, yerr=std_probs, 
             marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
plt.xlabel('Context Length (N)')
plt.ylabel('Average Probability of Knowledge Answer Token')
plt.title('Effect of In-Context Probe Questions on Knowledge Answer Probability')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/lie_detector/context_effect_analysis.png', dpi=300, bbox_inches='tight')

# Save detailed results
results_df = pd.DataFrame({
    'context_length': context_lengths,
    'mean_prob': mean_probs,
    'std_prob': std_probs
})

results_df.to_csv('results/lie_detector/context_effect_results.csv', index=False)

print("Analysis complete!")
print("Results saved to:")
print("  - results/lie_detector/context_effect_analysis.png")
print("  - results/lie_detector/context_effect_results.csv")