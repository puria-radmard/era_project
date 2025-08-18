#!/usr/bin/env python3
"""
Script to copy YAML config files for a new model.
Copies files with format <question_type>_<prompt_config>_<model_name>.yaml
and updates the model_name field in the YAML content.
"""

import os
import yaml
import glob
import re
from pathlib import Path

def copy_configs_for_new_model(config_dir, new_model_name):
    """
    Copy YAML config files for a new model.
    
    Args:
        config_dir (str): Directory containing the YAML config files
        new_model_name (str): Name of the new model to create configs for
    """
    
    # Pattern to match YAML files with the expected format
    yaml_pattern = os.path.join(config_dir, "*.yaml")
    
    # Regular expression to parse filename format: <question_type>_<prompt_config>_<model_name>.yaml
    filename_pattern = r"^(.+)_(.+)_(.+)\.yaml$"
    
    copied_files = []
    skipped_files = []
    
    # Find all YAML files in the directory
    yaml_files = glob.glob(yaml_pattern)
    
    for yaml_file in yaml_files:
        filename = os.path.basename(yaml_file)
        
        # Skip files that don't match the expected pattern or start with 'z_'
        if filename.startswith('z_'):
            continue
            
        match = re.match(filename_pattern, filename)
        if not match:
            skipped_files.append(filename)
            continue
            
        # Extract components from filename
        question_type = match.group(1)
        prompt_config = match.group(2)
        old_model_name = match.group(3)
        
        # Create new filename with the new model name
        new_filename = f"{question_type}_{prompt_config}_{new_model_name}.yaml"
        new_filepath = os.path.join(config_dir, new_filename)
        
        # Skip if file already exists
        if os.path.exists(new_filepath):
            print(f"Skipping {new_filename} - file already exists")
            continue
        
        try:
            # Read the original YAML file
            with open(yaml_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Update the model_name field
            if 'model_name' in config_data:
                config_data['model_name'] = new_model_name
            else:
                print(f"Warning: 'model_name' field not found in {filename}")
                config_data['model_name'] = new_model_name
            
            # Write the new YAML file
            with open(new_filepath, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config_data, f, default_flow_style=False, sort_keys=False)
            
            copied_files.append(new_filename)
            print(f"Created: {new_filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"Successfully copied {len(copied_files)} files for model '{new_model_name}'")
    
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files that didn't match pattern:")
        for file in skipped_files:
            print(f"  - {file}")
    
    if copied_files:
        print(f"\nCopied files:")
        for file in copied_files:
            print(f"  - {file}")

def main():
    # Configuration
    config_dir = "probe_generation/b_inference_apis/config"
    new_model_name = "mistral-small-24b-instruct-2501"
    
    # Check if config directory exists
    if not os.path.exists(config_dir):
        print(f"Error: Directory '{config_dir}' not found.")
        print("Make sure you're running this script from the correct location.")
        return
    
    print(f"Copying config files for new model: {new_model_name}")
    print(f"Source directory: {config_dir}")
    print("-" * 50)
    
    copy_configs_for_new_model(config_dir, new_model_name)

if __name__ == "__main__":
    main()