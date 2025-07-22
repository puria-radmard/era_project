import torch
import numpy as np
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression


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


def compute_lie_detector_normals(data, embeddings, class_key):
    """
    Compute unit normal vectors for linear separators between truth/lie embeddings
    for each probe question.
    
    Args:
        data: DataFrame with 'probe_question_idx' and class_key (e.g. truth) columns
        embeddings: Numpy array of shape [num_rows, embedding_dim]
    
    Returns:
        normals: Numpy array of shape [num_probe_questions, embedding_dim]
    """
    unique_probe_idxs = sorted(data['probe_question_idx'].unique())
    embedding_dim = embeddings.shape[1]
    normals = np.zeros((len(unique_probe_idxs), embedding_dim))
    
    for i, probe_idx in enumerate(unique_probe_idxs):
        # Get indices for truth and lie samples for this probe question
        truth_mask = (data['probe_question_idx'] == probe_idx) & (data[class_key] == 1)
        lie_mask = (data['probe_question_idx'] == probe_idx) & (data[class_key] == 0)
        
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

