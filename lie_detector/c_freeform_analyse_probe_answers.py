#!/usr/bin/env python3
import sys

import torch
import numpy as np

import json

from util.lie_detector import (
    load_and_preprocess_data,
    plot_probe_type_analysis,
)

from util.bert import *

from util.util import YamlConfig

import sys
import os


from transformers import AutoTokenizer, AutoModel

config_path = sys.argv[1]
args = YamlConfig(config_path)

probe_file_name = args.probe_file_name
probe_response_type: str = args.probe_response_type
probe_responses_args_name = args.probe_responses_args_name

assert probe_response_type.endswith('_words')

prompt_index = args.prompt_idx

n_samples = args.samples_per_classifier_size


if __name__ == "__main__":

    # Prepare for saving results
    output_path = os.path.join('lie_detector_results/c_probe_discimination', args.args_name)
    os.makedirs(output_path, exist_ok=True)
    args.save_args(output_path)
    
    # Load data
    probe_responses_path = os.path.join('lie_detector_results/b_probe_answers', probe_responses_args_name, 'probe_response.csv')
    probe_questions_path = f'data/{probe_file_name}.csv'
    data = load_and_preprocess_data(probe_responses_path, probe_questions_path, calc_log_odds=False)

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
