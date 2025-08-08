import json, copy
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from model.load import load_model
from util.util import YamlConfig
import sys, os

from util.elicit import get_next_user_token_probs


# Main script logic
config_path = sys.argv[1]
args = YamlConfig(config_path)

model_name = args.model_name

# Save to subdirectory
save_base = os.path.join('lie_detector_results/i_discriminative_tokens', args.args_name)
os.makedirs(save_base, exist_ok=True)
args.save(save_base)

# Output paths
question_direction_path = os.path.join(save_base, 'question_expectation_directions.npy')

# Load model
print("Loading model...")
chat_wrapper = load_model(model_name, device='auto')

# Create empty cache for unbiased context
empty_cache = chat_wrapper.create_prompt_cache(
    system_prompt="",
    in_context_questions=[],
    in_context_answers=[]
)

# Load contrastive pairs
print("Loading contrastive pairs...")
contrastive_pairs = pd.read_csv('data/question_expectation_pairs.csv')

num_pairs = len(contrastive_pairs)
print(f"Processing {num_pairs} contrastive pairs...")

# Initialize storage for hidden states
# We'll determine dimensions after the first forward pass
hidden_states_array = None
num_layers = None
hidden_size = None

# Process each contrastive pair
for pair_idx, row in tqdm(contrastive_pairs.iterrows(), total=num_pairs, desc="Processing pairs"):
    question_context = row['question_context']
    statement_context = row['statement_context']
    
    # Get hidden states for question context
    _, question_hidden = get_next_user_token_probs(
        chat_wrapper=chat_wrapper,
        cache_data=copy.deepcopy(empty_cache),
        user_message=question_context,
        extract_hidden_states=True
    )
    
    # Get hidden states for statement context  
    _, statement_hidden = get_next_user_token_probs(
        chat_wrapper=chat_wrapper,
        cache_data=copy.deepcopy(empty_cache),
        user_message=statement_context,
        extract_hidden_states=True
    )
    
    # Initialize array on first iteration
    if hidden_states_array is None:
        num_layers = question_hidden.shape[0]
        hidden_size = question_hidden.shape[1]
        hidden_states_array = np.zeros((num_pairs, num_layers, hidden_size, 2))
        print(f"Initialized array with shape: {hidden_states_array.shape}")
        print(f"Model has {num_layers} layers with hidden size {hidden_size}")
    
    # Store hidden states
    hidden_states_array[pair_idx, :, :, 0] = question_hidden.cpu().numpy()  # Question expectation
    hidden_states_array[pair_idx, :, :, 1] = statement_hidden.cpu().numpy()  # Statement expectation

# Compute direction vectors
print("Computing direction vectors...")
# Take difference: question - statement
direction_vectors = hidden_states_array[:, :, :, 0] - hidden_states_array[:, :, :, 1]  # [num_pairs, num_layers, hidden_size]

# Average across pairs to get mean direction for each layer
mean_directions = np.mean(direction_vectors, axis=0)  # [num_layers, hidden_size]

# Normalize each layer's direction vector
normalized_directions = np.zeros_like(mean_directions)
for layer_idx in range(mean_directions.shape[0]):
    layer_direction = mean_directions[layer_idx]
    norm = np.linalg.norm(layer_direction)
    if norm > 1e-8:  # Avoid division by zero
        normalized_directions[layer_idx] = layer_direction / norm
    else:
        print(f"Warning: Layer {layer_idx} has near-zero direction vector")
        normalized_directions[layer_idx] = layer_direction


# Save direction vectors
print(f"Saving direction vectors to: {question_direction_path}")
np.save(question_direction_path, normalized_directions)
