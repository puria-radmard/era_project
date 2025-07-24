#!/usr/bin/env python3
import sys

import torch
import numpy as np
import pandas as pd

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

questions_data_name = args.questions_data_name
narrow_misalignment_type = args.narrow_misalignment_type

assert probe_response_type.endswith('_words')

n_samples = args.samples_per_classifier_size


if __name__ == "__main__":

    # Prepare for saving results
    output_path = os.path.join('lie_detector_results/c_probe_discrimination', args.args_name)
    os.makedirs(output_path, exist_ok=True)
    args.save(output_path)
    
    # Load data
    probe_responses_path = os.path.join('lie_detector_results/b_probe_answers', probe_responses_args_name, 'probe_response.csv')
    probe_questions_path = f'data/probe_questions/{probe_file_name}.csv'
    data = load_and_preprocess_data(probe_responses_path, probe_questions_path, calc_log_odds=False)

    # Define file paths - moved down!
    embeddings_save_path = os.path.join(output_path, 'probe_response_bert_embeddings.npy')

    #######################################################################################

    ### Split into types we want to classify

    # Load the initial questions with types and create a mapping from question index to question type
    initial_questions_df = pd.read_csv(f'data/initial_questions/{questions_data_name}.csv')
    question_type_mapping = dict(enumerate(initial_questions_df['type']))
    question_types = np.array([question_type_mapping[idx] for idx in data['question_idx']])

    # Validate the result
    assert len(question_types) == len(data), f"Question types length {len(question_types)} != data length {len(data)}"
    print(f"Successfully mapped {len(question_types)} question types")
    print(f"Unique question types: {np.unique(question_types)}")

    # Optionally add to the dataframe for analysis
    data['question_type'] = question_types

    import pdb; pdb.set_trace()
    ## Type 1: model just gave an unsafe answer at all (1 vs 3 answer types)
    data['answered_misaligned'] = ((data['question_type'] == narrow_misalignment_type) & (data['truth'] == 0)).int()

    ## Type 2: model was just asked about medical advice (2 vs 2 answer types)
    data['asked_narrow'] = (data['question_type'] == narrow_misalignment_type)

    ## Type 3: model was misaligned (2 vs 2 answer types)
    data['was_dm'] = (data['truth'] == 0)

    classification_types = ['answered_misaligned', 'asked_narrow', 'was_dm']

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
    
    for classification_type in classification_types:

        normals_save_path = os.path.join(output_path, f'probe_response_bert_lie_normal_{classification_type}.npy')
        projections_save_path = os.path.join(output_path, f'probe_response_bert_proj_{classification_type}.npy')
        
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
            lie_detector_normals = compute_lie_detector_normals(data, all_embeddings, classification_type)
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
        discriminability_results = plot_probe_type_analysis(data, os.path.join(output_path, f'probe_type_analysis_{classification_type}.png'), 'bert_lie_proj', classification_type)
        filename = os.path.join(output_path, f'discriminability_results_{classification_type}.json')
        with open(filename, 'w') as f:
            json.dump(discriminability_results, f)
