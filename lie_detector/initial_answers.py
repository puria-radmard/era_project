import json
import pandas as pd
import os
from model.load import load_model

from tqdm import tqdm

# Configuration
BATCH_SIZE = 32
SYSTEM_PROMPT = "Do not provide explanations to any answers."

# Create results directory if it doesn't exist
os.makedirs('results/lie_detector', exist_ok=True)

# Load prompts
with open('data/prompts.json', 'r') as f:
    prompts = json.load(f)

truth_prompts = prompts['truth_prompts']
lie_prompts = prompts['lie_prompts']

# Load questions
with open('data/questions_1000_all.json', 'r') as f:
    question_data = json.load(f)

qa_pairs = [(question_data['question'][f'{idx}'].strip(), question_data['answer'][f'{idx}']) for idx in range(len(question_data['question']))]
assert set([len(qa[1].split()) for qa in qa_pairs]) == {1}

# Load model
chat_wrapper = load_model("meta-llama/Llama-3.1-8B-Instruct", device='auto')

# Initialize results list
results = []

# Process in batches
for batch_start in tqdm(range(0, len(qa_pairs), BATCH_SIZE)):
    batch_end = min(batch_start + BATCH_SIZE, len(qa_pairs))
    batch_qa_pairs = qa_pairs[batch_start:batch_end]
    batch_indices = list(range(batch_start, batch_end))
    
    print(f"Processing batch {batch_start//BATCH_SIZE + 1}/{(len(qa_pairs) + BATCH_SIZE - 1)//BATCH_SIZE}")
    
    # Generate knowledge check responses (only once per batch)
    formatted_check_knowledge_chats = [
        chat_wrapper.format_chat(
            system_prompt=SYSTEM_PROMPT,
            user_message=qa_pair[0]
        ) for qa_pair in batch_qa_pairs
    ]
    
    check_knowledge_answers = chat_wrapper.generate(
        chats=formatted_check_knowledge_chats,
        past_key_values=None,
        max_new_tokens=1024,
        temperature=None,
        do_sample=False,
        max_length=None
    )
    check_knowledge_responses = check_knowledge_answers['generated_texts']
    
    # Process each prompt pair
    for prompt_idx, (truth_prompt, lie_prompt) in enumerate(zip(truth_prompts, lie_prompts)):
        print(f"  Processing prompt pair {prompt_idx + 1}/{len(truth_prompts)}")
        
        # Generate truth responses
        formatted_truth_chats = [
            chat_wrapper.format_chat(
                system_prompt=SYSTEM_PROMPT,
                user_message=f'{truth_prompt} {qa_pair[0]}'
            ) for qa_pair in batch_qa_pairs
        ]
        
        truth_answers = chat_wrapper.generate(
            chats=formatted_truth_chats,
            past_key_values=None,
            max_new_tokens=1024,
            temperature=None,
            do_sample=False,
            max_length=None
        )
        truth_responses = truth_answers['generated_texts']
        
        # Generate lie responses
        formatted_lie_chats = [
            chat_wrapper.format_chat(
                system_prompt=SYSTEM_PROMPT,
                user_message=f'{lie_prompt} {qa_pair[0]}'
            ) for qa_pair in batch_qa_pairs
        ]
        
        lie_answers = chat_wrapper.generate(
            chats=formatted_lie_chats,
            past_key_values=None,
            max_new_tokens=1024,
            temperature=None,
            do_sample=False,
            max_length=None
        )
        lie_responses = lie_answers['generated_texts']
        
        # Evaluate responses
        for i, (qa_pair, knowledge_resp, truth_resp, lie_resp) in enumerate(zip(
            batch_qa_pairs, check_knowledge_responses, truth_responses, lie_responses
        )):
            correct_answer = qa_pair[1].lower()
            
            knows_answer = correct_answer in knowledge_resp.lower()
            told_truth = correct_answer in truth_resp.lower()
            told_lie = correct_answer not in lie_resp.lower()  # True if it's actually a lie
            
            results.append({
                'question_idx': batch_indices[i],
                'prompt_idx': prompt_idx,
                'knowledge_answer': knowledge_resp,
                'truth_answer': truth_resp,
                'lie_answer': lie_resp,
                'knows_answer': knows_answer,
                'told_truth': told_truth,
                'told_lie': told_lie
            })
        
        # Save DataFrame after each prompt iteration
        df = pd.DataFrame(results)
        df.to_csv('results/lie_detector/questions_1000_all.csv', index=False)
        
        # Print summary statistics after each prompt iteration
        print(f"\n  === Summary after prompt pair {prompt_idx + 1} ===")
        print(f"  Total rows processed: {len(df)}")
        
        # Overall statistics
        current_prompt_data = df[df['prompt_idx'] == prompt_idx]
        print(f"    Current prompt: {lie_prompt}")
        print(f"\n  Current prompt pair performance:")
        print(f"    knows_answer rate: {current_prompt_data['knows_answer'].mean():.3f}")
        print(f"    told_truth rate: {current_prompt_data['told_truth'].mean():.3f}")
        print(f"    told_lie rate: {current_prompt_data['told_lie'].mean():.3f}")
        
        # Overall statistics across all prompts processed so far
        print(f"\n  Overall performance (all prompts so far):")
        print(f"    knows_answer rate: {df['knows_answer'].mean():.3f}")
        print(f"    told_truth rate: {df['told_truth'].mean():.3f}")
        print(f"    told_lie rate: {df['told_lie'].mean():.3f}")
        
        # Statistics by prompt type (separate truth vs lie performance)
        print(f"\n  Performance by prompt type (all prompts so far):")
        
        # For truth prompts, we care about told_truth rate
        truth_performance = df['told_truth'].mean()
        print(f"    Truth-telling success rate: {truth_performance:.3f}")
        
        # For lie prompts, we care about told_lie rate  
        lie_performance = df['told_lie'].mean()
        print(f"    Lie-telling success rate: {lie_performance:.3f}")
        
        # Breakdown by individual prompt pairs processed so far
        if len(df['prompt_idx'].unique()) > 1:
            print(f"\n  By prompt pair:")
            prompt_stats = df.groupby('prompt_idx').agg({
                'knows_answer': 'mean',
                'told_truth': 'mean', 
                'told_lie': 'mean'
            }).round(3)
            for idx, row in prompt_stats.iterrows():
                print(f"    Prompt {idx}: knows={row['knows_answer']:.3f}, truth={row['told_truth']:.3f}, lie={row['told_lie']:.3f}")
        
        print("  " + "="*50)

# Final summary
print(f"\n🎉 FINAL RESULTS 🎉")
print(f"Results saved to results/lie_detector/questions_1000_all.csv")
print(f"Total rows: {len(df)}")

print(f"\nFinal Summary Statistics:")
print(f"Overall knows_answer rate: {df['knows_answer'].mean():.3f}")
print(f"Overall told_truth rate: {df['told_truth'].mean():.3f}")
print(f"Overall told_lie rate: {df['told_lie'].mean():.3f}")

print(f"\nFinal Performance by Prompt Type:")
print(f"Truth-telling success rate: {df['told_truth'].mean():.3f}")
print(f"Lie-telling success rate: {df['told_lie'].mean():.3f}")

# Final breakdown by prompt
print(f"\nFinal Breakdown by Prompt Pair:")
final_prompt_stats = df.groupby('prompt_idx').agg({
    'knows_answer': 'mean',
    'told_truth': 'mean', 
    'told_lie': 'mean'
}).round(3)
print(final_prompt_stats)