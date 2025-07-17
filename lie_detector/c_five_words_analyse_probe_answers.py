#!/usr/bin/env python3
import sys

import torch
import numpy as np

import json

from util.lie_detector import (
    load_and_preprocess_data,
    plot_probe_type_analysis,
)

from util.util import YamlConfig

import sys
import os

from tqdm import tqdm

from sklearn.linear_model import LogisticRegression

from transformers import AutoTokenizer, AutoModel

config_path = sys.argv[1]
args = YamlConfig(config_path)

batch_size = args.batch_size
system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name
save_path = args.save_path

probe_file_name = args.probe_file_name
probe_response_type = args.probe_response_type

assert probe_response_type == 'five_words'

prompt_index = args.prompt_idx

n_samples = args.samples_per_classifier_size



def bert_embed(batch_texts, bert_model):
    """
    Generate BERT embeddings for a batch of text strings.
    
    Args:
        batch_texts: List of strings to embed
        bert_model: Tuple of (tokenizer, model)
    
    Returns:
        embeddings: Numpy array of shape [batch_size, embedding_dim]
    """
    tokenizer, model = bert_model
    
    # Tokenize with padding and truncation
    inputs = tokenizer(
        batch_texts, 
        padding=True, 
        truncation=True, 
        return_tensors="pt",
        max_length=512
    )

    # Move inputs to GPU
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Use mean pooling instead of CLS token for better semantic similarity
        # This averages all token embeddings (excluding padding)
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        # Mask out padding tokens and compute mean
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return embeddings.cpu().numpy()


def generate_and_save_embeddings(df, bert_model, batch_size=128):
    """
    Generate embeddings for 'resp' column in batches and save to file.
    
    Args:
        df: DataFrame with 'resp' column
        bert_model: BERT model for embedding generation
        batch_size: Number of responses to process at once
    
    Returns:
        embeddings: Numpy array of shape [num_rows, embedding_dim]
    """
    resp_texts = df['resp'].tolist()
    all_embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(resp_texts), batch_size)):
        batch = resp_texts[i:i + batch_size]
        batch_embeddings = bert_embed(batch, bert_model)  # shape: [batch_size, embedding_dim]
        all_embeddings.append(batch_embeddings)
    
    # Concatenate all batches
    embeddings_array = np.concatenate(all_embeddings, axis=0)  # shape: [num_rows, embedding_dim]
    return embeddings_array


def compute_lie_detector_normals(data, embeddings):
    """
    Compute unit normal vectors for linear separators between truth/lie embeddings
    for each probe question.
    
    Args:
        data: DataFrame with 'probe_question_idx' and 'truth' columns
        embeddings: Numpy array of shape [num_rows, embedding_dim]
    
    Returns:
        normals: Numpy array of shape [num_probe_questions, embedding_dim]
    """
    unique_probe_idxs = sorted(data['probe_question_idx'].unique())
    embedding_dim = embeddings.shape[1]
    normals = np.zeros((len(unique_probe_idxs), embedding_dim))
    
    for i, probe_idx in enumerate(unique_probe_idxs):
        # Get indices for truth and lie samples for this probe question
        truth_mask = (data['probe_question_idx'] == probe_idx) & (data['truth'] == 1)
        lie_mask = (data['probe_question_idx'] == probe_idx) & (data['truth'] == 0)
        
        truth_indices = data[truth_mask].index.tolist()
        lie_indices = data[lie_mask].index.tolist()
        
        # Get embeddings for truth and lie samples
        truth_embeddings = embeddings[truth_indices]
        lie_embeddings = embeddings[lie_indices]
        
        # Combine embeddings and create labels
        X = np.concatenate([truth_embeddings, lie_embeddings], axis=0)
        y = np.concatenate([np.ones(len(truth_indices)), np.zeros(len(lie_indices))])
        
        # Train logistic regression
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X, y)
        
        # Extract and normalize the weight vector
        # Flip sign so higher projection = more likely to be lie (truth=0)
        weight_vector = -clf.coef_[0]  # Negative sign for lie direction
        unit_normal = weight_vector / np.linalg.norm(weight_vector)
        
        normals[i] = unit_normal

        train_pred = clf.predict(X)
        train_accuracy = (train_pred == y).mean()
        print(f"Probe {i+1}/{len(unique_probe_idxs)} (idx: {probe_idx}): Train accuracy: {train_accuracy:.3f}")
    
    return normals


def compute_projections(data, embeddings, normals):
    """
    Project each embedding onto its corresponding probe question's normal vector.
    
    Args:
        data: DataFrame with 'probe_question_idx' column
        embeddings: Numpy array of shape [num_rows, embedding_dim]
        normals: Numpy array of shape [num_probe_questions, embedding_dim]
    
    Returns:
        projections: Numpy array of shape [num_rows]
    """
    unique_probe_idxs = sorted(data['probe_question_idx'].unique())
    probe_idx_to_normal_idx = {probe_idx: i for i, probe_idx in enumerate(unique_probe_idxs)}
    
    projections = np.zeros(len(data))
    
    for row_idx, probe_idx in enumerate(data['probe_question_idx']):
        normal_idx = probe_idx_to_normal_idx[probe_idx]
        embedding = embeddings[row_idx]
        normal = normals[normal_idx]
        
        # Compute dot product (projection)
        projection = np.dot(embedding, normal)
        projections[row_idx] = projection
    
    return projections


if __name__ == "__main__":
    
    # Load data
    results_path = f"{save_path}/probe_answers/{probe_file_name}/five_words/{questions_data_name}_probe_prompt{prompt_index}.csv"
    probes_path = f"data/{probe_file_name}.csv"

    # Prepare for saving results
    output_path = f'{save_path}/probe_analysis/{probe_file_name}/five_words/{questions_data_name}/prompt{prompt_index}'
    os.makedirs(output_path, exist_ok=True)
    
    data = load_and_preprocess_data(results_path, probes_path, calc_log_odds=False)

    # Define file paths
    embeddings_save_path = os.path.join(output_path, 'probe_response_bert_embeddings.npy')
    normals_save_path = os.path.join(output_path, 'probe_response_bert_lie_normal.npy')
    projections_save_path = os.path.join(output_path, 'probe_response_bert_proj.npy')

    #######################################################################################

    # Check if embeddings exist, if not generate them
    if os.path.exists(embeddings_save_path):
        print(f"Loading existing embeddings from {embeddings_save_path}")
        all_embeddings = np.load(embeddings_save_path)
        # Validate dimensions
        assert all_embeddings.shape[0] == len(data), f"Embedding rows {all_embeddings.shape[0]} != data rows {len(data)}"
    else:
        print("Loading in BERT model...")
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        model = AutoModel.from_pretrained("answerdotai/ModernBERT-base", torch_dtype=torch.float16).to('cuda')
        bert_model = (tokenizer, model)
        
        print("Generating BERT embeddings...")
        all_embeddings = generate_and_save_embeddings(data, bert_model, batch_size=32)
        np.save(embeddings_save_path, all_embeddings)
        print(f"Saved embeddings to {embeddings_save_path}")

    #######################################################################################
    
    # Check if normals exist, if not compute them
    if os.path.exists(normals_save_path):
        print(f"Loading existing normals from {normals_save_path}")
        lie_detector_normals = np.load(normals_save_path)
        # Validate dimensions
        num_probe_questions = len(data['probe_question_idx'].unique())
        assert lie_detector_normals.shape[0] == num_probe_questions, f"Normal rows {lie_detector_normals.shape[0]} != probe questions {num_probe_questions}"
        assert lie_detector_normals.shape[1] == all_embeddings.shape[1], f"Normal dim {lie_detector_normals.shape[1]} != embedding dim {all_embeddings.shape[1]}"
    else:
        print("Computing lie detector normals...")
        lie_detector_normals = compute_lie_detector_normals(data, all_embeddings)
        np.save(normals_save_path, lie_detector_normals)
        print(f"Saved normals to {normals_save_path}")
    
    # Check if projections exist, if not compute them
    if os.path.exists(projections_save_path):
        print(f"Loading existing projections from {projections_save_path}")
        bert_projections = np.load(projections_save_path)
        # Validate dimensions
        assert bert_projections.shape[0] == len(data), f"Projection rows {bert_projections.shape[0]} != data rows {len(data)}"
    else:
        print("Computing projections...")
        bert_projections = compute_projections(data, all_embeddings, lie_detector_normals)
        np.save(projections_save_path, bert_projections)
        print(f"Saved projections to {projections_save_path}")
    
    # Add projections to dataframe for analysis
    data['bert_lie_proj'] = bert_projections

    #######################################################################################
    
    # Plot probe type analysis (unchanged)
    discriminability_results = plot_probe_type_analysis(data, os.path.join(output_path, 'probe_type_analysis.png'), 'bert_lie_proj')
    filename = os.path.join(output_path, 'discriminability_results.json')
    with open(filename, 'w') as f:
        json.dump(discriminability_results, f)