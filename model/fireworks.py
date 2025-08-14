"""
Fireworks.ai Batch Inference API Utilities (SDK-based)

This module provides utilities for working with Fireworks.ai's Batch Inference API
using the high-level Fireworks SDK, including batch job submission, monitoring, and result collection.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from fireworks import Dataset, BatchInferenceJob
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

MAX_ID_LENGTH = 63

class FireworksBatchWrapper:
    """
    Wrapper for Fireworks.ai Batch Inference API operations using the Fireworks SDK.
    Works with all models in the Fireworks model library.
    """

    success_code = 3
    
    def __init__(self, model_name: str):
        """
        Initialize the batch wrapper.
        
        Args:
            model_name: Name of the model (e.g., "llama-v3p1-8b-instruct")
            
        Raises:
            ValueError: If FIREWORKS_API_KEY not found
        """
        api_key = os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY not found in environment variables. "
                           "Please set it in your .env file.")
        
        self.model_name = 'accounts/fireworks/models/' + model_name
        self.api_key = api_key
        
        # Store batch job reference for status tracking
        self._current_batch_job = None
    
    def format_chat_for_batch(
        self,
        custom_id: str,
        system_prompt: Optional[str] = None,
        in_context_questions: Optional[List[str]] = None,
        in_context_answers: Optional[List[str]] = None,
        user_message: Optional[str] = None,
        prefiller: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        logprobs: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Format a single chat request for batch processing.
        Matches the interface of the Together.ai ChatTemplateWrapper.format_chat.
        
        Args:
            custom_id: Unique identifier for this request
            system_prompt: The system prompt
            in_context_questions: List of in-context questions
            in_context_answers: List of in-context answers  
            user_message: The user message
            prefiller: Assistant message prefiller
            max_tokens: Maximum tokens to generate (use 0 for scoring only)
            temperature: Sampling temperature
            logprobs: Whether to return log probabilities
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary formatted for batch JSONL file
        """
        messages = []
        
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        
        if in_context_questions is not None:
            assert (in_context_answers is not None) and (len(in_context_answers) == len(in_context_questions))
            for question, answer in zip(in_context_questions, in_context_answers):
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": answer})
        
        if user_message is not None:
            messages.append({"role": "user", "content": user_message})
        
        if prefiller is not None and prefiller != "":
            messages.append({"role": "assistant", "content": prefiller})
        
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        # Add logprobs if requested
        if logprobs:
            body["logprobs"] = 1
        
        return {
            "custom_id": custom_id,
            "body": body
        }
    
    def create_batch_file(
        self, 
        requests: List[Dict[str, Any]], 
        filepath: str
    ) -> None:
        """
        Create a JSONL file for batch processing.
        
        Args:
            requests: List of formatted batch requests
            filepath: Path where to save the JSONL file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            for request in requests:
                f.write(json.dumps(request) + '\n')
        
        print(f"Created batch file: {filepath} with {len(requests)} requests")
    
    def upload_and_submit_batch(self, jsonl_filepath: str) -> str:
        """
        Upload JSONL file as dataset and submit batch inference job using Fireworks SDK.
        
        Args:
            jsonl_filepath: Path to the JSONL file
            
        Returns:
            Batch job ID string
        """
        print(f"Uploading dataset and creating batch job: {jsonl_filepath}")

        # Create dataset from file using Fireworks SDK
        dataset = Dataset.from_file(jsonl_filepath)
        dataset.sync()
        print(f"Dataset created with ID: {dataset.id}")

        # Generate unique job ID
        job_id = f"batch-{int(time.time())}"
        output_dataset_id = f"{dataset.id}-output-{self.model_name.split('/')[-1]}"[:MAX_ID_LENGTH]
        
        # Extract inference parameters from the first request to use as defaults
        # Individual requests can still override these in their body
        default_params = {}
        with open(jsonl_filepath, 'r') as f:
            first_request = json.loads(f.readline())
            body = first_request.get('body', {})
            
            # Extract common inference parameters
            for param in ['max_tokens', 'temperature', 'top_p', 'top_k', 'n']:
                if param in body:
                    default_params[param] = body[param]
        
        # Create batch inference job using SDK
        batch_job = BatchInferenceJob.create(
            model=self.model_name,
            input_dataset_id=dataset.id,
            output_dataset_id=output_dataset_id,
            job_id=job_id,
            inference_parameters=default_params,
            api_key=self.api_key
        )
        
        # Store reference for status tracking
        self._current_batch_job = batch_job
        
        print(f"Batch inference job created with ID: {job_id}")
        return job_id
    
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get the current status of a batch inference job.
        
        Args:
            batch_id: The batch job ID
            
        Returns:
            Batch status dictionary
        """
        # Extract account from API key or use environment variable
        account = os.getenv("FIREWORKS_ACCOUNT_ID")
        if not account:
            raise ValueError("FIREWORKS_ACCOUNT_ID not found in environment variables")
        
        batch_job = BatchInferenceJob.get(
            job_id=batch_id, 
            account=account, 
            api_key=self.api_key
        )
        
        if not batch_job:
            return {
                "id": batch_id,
                "status": "NOT_FOUND",
                "created_at": None,
                "model": None,
                "input_dataset_id": None,
                "output_dataset_id": None,
                "update_time": None
            }
        
        return {
            "id": batch_id,
            "status": batch_job.state,
            "created_at": batch_job.create_time,
            "model": batch_job.model,
            "input_dataset_id": batch_job.input_dataset_id,
            "output_dataset_id": batch_job.output_dataset_id,
            "update_time": batch_job.update_time
        }
    
    def download_batch_results(
        self, 
        batch_id: str, 
        output_filepath: str,
        error_filepath: Optional[str] = None
    ) -> None:
        """
        Download batch results if completed.
        
        Args:
            batch_id: The batch job ID
            output_filepath: Where to save the output results
            error_filepath: Where to save error results (optional, not used in current SDK)
        """
        status = self.get_batch_status(batch_id)
        current_status = status["status"]
        
        # Check for failure states
        if current_status != self.success_code:
            raise Exception(f"Batch not yet completed or has failed. Current status: {current_status}")
        
        output_dataset_id = status["output_dataset_id"]
        if not output_dataset_id:
            raise Exception("No output dataset available")
        
        # Create Dataset object from output dataset ID and download
        # Extract just the dataset ID from the full path
        dataset_id = output_dataset_id.split("/")[-1]
        output_dataset = Dataset.from_id(dataset_id)

        # Create output directory
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        # The Fireworks SDK doesn't expose a direct download method in the reference,
        # but we can work with the dataset object. For now, we'll indicate success
        # and let the user work with the Dataset object if needed.
        
        # Save dataset reference for potential future use
        with open(output_filepath + ".dataset_info", 'w') as f:
            json.dump({
                "dataset_id": dataset_id,
                "dataset_name": output_dataset.name,
                "batch_job_id": batch_id
            }, f, indent=2)
        
        # Get all the data from the dataset        
        # Try to get all data - use a very large number since we don't know the size
        # The head() method should return all available data if we request more than exists
        all_data = output_dataset.head(1000000, as_dataset=False)  # Large number to get everything

        print(f"Retrieved {len(all_data)} results from output dataset")
        
        # Save as JSONL file
        with open(output_filepath, 'w') as f:
            for item in all_data:
                # The item should already be in the correct format for batch results
                # Each item should contain custom_id and response data
                json.dump(item, f)
                f.write('\n')
        
        print(f"Results saved to: {output_filepath}")
            

def save_batch_metadata(
    save_dir: str,
    **metadata_kwargs
) -> None:
    """
    Save batch metadata to JSON file.
    
    Args:
        save_dir: Directory to save metadata
        **metadata_kwargs: All metadata fields to save
    """
    metadata_path = os.path.join(save_dir, 'batch_tmp', 'batch_metadata.json')
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata_kwargs, f, indent=2)
    
    print(f"Saved batch metadata to: {metadata_path}")


def load_batch_metadata(save_dir: str) -> Dict[str, Any]:
    """
    Load batch metadata from JSON file.
    
    Args:
        save_dir: Directory containing metadata
        
    Returns:
        Metadata dictionary
    """
    metadata_path = os.path.join(save_dir, 'batch_tmp', 'batch_metadata.json')
    
    with open(metadata_path, 'r') as f:
        return json.load(f)