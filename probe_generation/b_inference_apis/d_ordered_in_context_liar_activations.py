"""
Extract activations from actual batch inference contexts.

This script loads the contexts that were actually used in batch inference,
deduplicates them, and extracts activations from a local white-box model.
"""

import numpy as np
import json
import os
import sys
from tqdm import tqdm
from typing import Dict, List, Tuple
from collections import defaultdict

from model.load import load_model
from util.util import YamlConfig


def parse_custom_id(custom_id: str) -> Tuple[int, int, str, int, str, int]:
    """
    Parse custom ID to extract experimental parameters.
    
    Args:
        custom_id: String like "N5_s2_ctaligned_q42_rtruth3"
        
    Returns:
        Tuple of (context_length, sample_idx, context_type, question_idx, response_type, resp_idx)
    """
    parts = custom_id.split('_')
    context_length = int(parts[0][1:])  # Remove 'N'
    sample_idx = int(parts[1][1:])      # Remove 's'
    context_type = parts[2][2:]         # Remove 'ct'
    question_idx = int(parts[3][1:])    # Remove 'q'
    
    # Parse response part: "rtruth3" -> response_type="truth", resp_idx=3
    response_part = parts[4][1:]        # Remove 'r'
    if response_part.startswith('truth'):
        response_type = 'truth'
        resp_idx = int(response_part[5:])  # Remove 'truth'
    elif response_part.startswith('lie'):
        response_type = 'lie'
        resp_idx = int(response_part[3:])   # Remove 'lie'
    else:
        raise ValueError(f"Unknown response type in custom_id: {custom_id}")
    
    return context_length, sample_idx, context_type, question_idx, response_type, resp_idx


def extract_context_and_eval_from_messages(messages: List[Dict]) -> Tuple[List[str], List[str], str]:
    """
    Extract context Q&A pairs and eval question from messages array.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        
    Returns:
        Tuple of (context_questions, context_answers, eval_question)
    """
    context_questions = []
    context_answers = []
    eval_question = None
    
    # Skip system message if present
    start_idx = 1 if messages[0]['role'] == 'system' else 0
    
    # Process messages in pairs (user, assistant) until we hit the final user message
    i = start_idx
    while i < len(messages) - 2:  # Stop before the last message
        if messages[i]['role'] == 'user' and messages[i + 1]['role'] == 'assistant':
            context_questions.append(messages[i]['content'])
            context_answers.append(messages[i + 1]['content'])
            i += 2
        else:
            break
    
    # The last user message (or second-to-last if there's a final assistant message) is the eval question
    if messages[-1]['role'] == 'user':
        eval_question = messages[-1]['content']
    elif len(messages) >= 2 and messages[-2]['role'] == 'user':
        eval_question = messages[-2]['content']
    else:
        raise ValueError("Could not find eval question in messages")
    
    return context_questions, context_answers, eval_question


def load_and_deduplicate_contexts(jsonl_paths: List[str]) -> Dict[int, Dict[str, List[Dict]]]:
    """
    Load JSONL files and deduplicate contexts.
    
    Args:
        jsonl_paths: List of paths to JSONL batch files
        
    Returns:
        Nested dict: {context_length: {context_type: [context_data_dicts]}}
    """
    seen_keys = set()
    contexts_data = defaultdict(lambda: defaultdict(list))
    
    for jsonl_path in jsonl_paths:
        print(f"Loading contexts from: {jsonl_path}")
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    custom_id = data['custom_id']
                    messages = data['body']['messages']
                    
                    # Parse custom ID
                    context_length, sample_idx, context_type, question_idx, response_type, resp_idx = parse_custom_id(custom_id)

                    # Create deduplication key (ignore response_type and resp_idx)
                    dedup_key = (context_length, sample_idx, context_type, question_idx)
                    
                    if dedup_key not in seen_keys:
                        seen_keys.add(dedup_key)
                        
                        # Extract context and eval question
                        context_questions, context_answers, eval_question = extract_context_and_eval_from_messages(messages)
                        
                        context_data = {
                            'sample_idx': sample_idx,
                            'question_idx': question_idx,
                            'context_questions': context_questions,
                            'context_answers': context_answers,
                            'eval_question': eval_question
                        }
                        
                        contexts_data[context_length][context_type].append(context_data)
    
    # Convert to regular dicts and sort by sample_idx, question_idx for reproducibility
    final_data = {}
    for context_length in contexts_data:
        final_data[context_length] = {}
        for context_type in contexts_data[context_length]:
            context_list = contexts_data[context_length][context_type]
            context_list.sort(key=lambda x: (x['sample_idx'], x['question_idx']))
            final_data[context_length][context_type] = context_list
    
    print(f"Loaded {len(seen_keys)} unique contexts across {len(final_data)} context lengths")
    return final_data


def get_batch_jsonl_paths(base_path: str) -> List[str]:
    """
    Find all steering_batch_N*.jsonl files in the batch_tmp directory.
    
    Args:
        base_path: Base path to the experiment directory
        
    Returns:
        List of paths to JSONL files
    """
    batch_tmp_dir = os.path.join(base_path, 'batch_tmp')
    jsonl_paths = []
    
    for filename in os.listdir(batch_tmp_dir):
        if filename.startswith('steering_batch_N') and filename.endswith('.jsonl'):
            jsonl_paths.append(os.path.join(batch_tmp_dir, filename))
    
    jsonl_paths.sort()  # Sort for reproducibility
    return jsonl_paths


def get_activations_for_contexts(
    chat_wrapper,
    contexts_data: Dict[int, Dict[str, List[Dict]]],
    config: YamlConfig
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Extract activations for all contexts using the local model.
    
    Args:
        chat_wrapper: Loaded model wrapper
        contexts_data: Organized context data
        config: Configuration object
        
    Returns:
        Dict mapping context_length -> context_type -> activations array
    """
    batch_size = config.batch_size
    system_prompt = ""
    num_layers = config.num_layers
    hidden_state_size = config.hidden_state_size
    candidate_layers = list(range(num_layers))
    
    results = {}
    
    for context_length in sorted(contexts_data.keys()):
        print(f"\n{'='*80}")
        print(f"PROCESSING CONTEXT LENGTH N={context_length}")
        print(f"{'='*80}")
        
        results[context_length] = {}

        for context_type in contexts_data[context_length]:
            print(f"\nProcessing context type: {context_type}")
            
            context_list = contexts_data[context_length][context_type]
            num_contexts = max([cl['sample_idx'] for cl in context_list]) + 1
            num_questions = max([cl['question_idx'] for cl in context_list]) + 1
            
            # Initialize activation tensor: [num_questions, num_contexts, num_layers, hidden_size]
            activations = np.full([num_questions, num_contexts, len(candidate_layers), hidden_state_size], np.nan)
            
            # Process in batches for efficiency
            for i in tqdm(range(0, len(context_list), batch_size), desc=f"Processing {context_type}"):
                batch_contexts = context_list[i:i+batch_size]
                
                # Create chats for this batch
                batch_chats = []
                for context_data in batch_contexts:
                    if context_length > 0:
                        # Use context if available
                        chat = chat_wrapper.format_chat(
                            system_prompt=system_prompt,
                            in_context_questions=context_data['context_questions'],
                            in_context_answers=context_data['context_answers'],
                            user_message=context_data['eval_question'],
                            prefiller=''
                        )
                    else:
                        # No context case
                        chat = chat_wrapper.format_chat(
                            system_prompt=system_prompt,
                            user_message=context_data['eval_question'],
                            prefiller=''
                        )
                    batch_chats.append(chat)
                
                # Get activations
                outputs = chat_wrapper.forward(
                    chats=batch_chats,
                    past_key_values=None,
                    use_cache=True,
                    output_hidden_states=True
                )
                
                # Extract hidden states at last token for each layer
                hidden_states = outputs.hidden_states
                
                for i_bc, bc in enumerate(batch_contexts):
                    for cli, layer_idx in enumerate(candidate_layers):
                        # hidden_states[layer_idx + 1] shape: [batch_size, seq_len, hidden_size]
                        # Take last token: [:, -1, :]
                        layer_activations = hidden_states[layer_idx + 1].cpu().numpy()[:, -1, :]
                        activations[bc['question_idx'], bc['sample_idx'], cli, :] = layer_activations
            
            results[context_length][context_type] = activations
            assert not np.isnan(activations).any()
            print(f"Extracted activations shape: {activations.shape}")
    
    return results


def save_activations(
    activations: Dict[int, Dict[str, np.ndarray]], 
    output_path: str
) -> None:
    """
    Save activation results to disk.
    
    Args:
        activations: Activations organized by context_length and context_type
        output_path: Directory to save results
    """
    os.makedirs(output_path, exist_ok=True)
    
    for context_length in activations:
        for context_type in activations[context_length]:
            filename = f'batch_contextual_activations_N{context_length}_context{context_type}.npy'
            filepath = os.path.join(output_path, filename)
            np.save(filepath, activations[context_length][context_type])
            print(f"Saved {filepath}")


if __name__ == '__main__':
    """Main function to extract activations from batch contexts."""
    
    # Load configuration
    config_path = sys.argv[1]
    config = YamlConfig(config_path)
    
    print("Configuration loaded:")
    print(f"  Model: {config.model_name}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Experiment: {config.args_name}")
    
    # Setup paths
    base_experiment_path = os.path.join(
        'probe_generation_results/b_neurips_workshop_results', 
        config.args_name,
        'c2_ordered_in_context_liar'
    )
    
    output_path = os.path.join('probe_generation_results/b_neurips_workshop_results', config.args_name, 'd_ordered_in_context_liar')
    os.makedirs(output_path, exist_ok=True)
    config.save(output_path)
    
    # Find and load batch JSONL files
    jsonl_paths = get_batch_jsonl_paths(base_experiment_path)
    print(f"Found {len(jsonl_paths)} batch files")
    
    # Load and deduplicate contexts
    contexts_data = load_and_deduplicate_contexts(jsonl_paths)
    
    # Load model
    print(f"Loading model: {config.hf_model_name}")
    chat_wrapper = load_model(config.hf_model_name, device='auto')
    
    # Extract activations
    activations = get_activations_for_contexts(chat_wrapper, contexts_data, config)
    
    # Save results
    save_activations(activations, output_path)
    
    print(f"\n{'='*80}")
    print("ACTIVATION EXTRACTION COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_path}")

