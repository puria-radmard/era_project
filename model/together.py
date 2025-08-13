"""
Together.ai Batch API Utilities

This module provides utilities for working with Together.ai's Batch API,
including batch job submission, monitoring, and result collection.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any
from together import Together
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

class TogetherBatchWrapper:
    """
    Wrapper for Together.ai Batch API operations.
    Only supports models that have batch inference capabilities.
    """
    
    # Models that support batch inference (from Together.ai docs)
    SUPPORTED_MODELS = [
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-V3", 
        "meta-llama/Llama-3-70b-chat-hf",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", 
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "Qwen/Qwen2.5-7B-Instruct-Turbo",
        "Qwen/Qwen3-235B-A22B-fp8-tput",
        "Qwen/QwQ-32B"
    ]
    
    def __init__(self, model_name: str):
        """
        Initialize the batch wrapper.
        
        Args:
            model_name: Name of the model (must support batch inference)
            
        Raises:
            ValueError: If model doesn't support batch inference
            ValueError: If TOGETHER_API_KEY not found
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} doesn't support batch inference. "
                           f"Supported models: {self.SUPPORTED_MODELS}")
        
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment variables. "
                           "Please set it in your .env file.")
        
        self.model_name = model_name
        self.client = Together(api_key=api_key)
    
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
        Matches the interface of ChatTemplateWrapper.format_chat.
        
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
            "model": self.model_name,
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
        Upload JSONL file and submit batch job.
        
        Args:
            jsonl_filepath: Path to the JSONL file
            
        Returns:
            Batch ID string
        """
        print(f"Uploading file: {jsonl_filepath}")
        
        # Upload file
        file_resp = self.client.files.upload(
            file=jsonl_filepath, 
            purpose="batch-api"
        )
        print(f"File uploaded with ID: {file_resp.id}")
        
        # Create batch
        batch = self.client.batches.create_batch(
            file_id=file_resp.id,
            endpoint="/v1/chat/completions"
        )
        print(f"Batch created with ID: {batch.id}")
        
        return batch.id
    
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get the current status of a batch job.
        
        Args:
            batch_id: The batch ID
            
        Returns:
            Batch status dictionary
        """
        batch = self.client.batches.get_batch(batch_id)
        return {
            "id": batch.id,
            "status": batch.status,
            "created_at": batch.created_at,
            "request_count": getattr(batch, 'request_count', 0),
            "output_file_id": getattr(batch, 'output_file_id', None),
            "error_file_id": getattr(batch, 'error_file_id', None)
        }
    
    def download_batch_results(
        self, 
        batch_id: str, 
        output_filepath: str,
        error_filepath: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Download batch results if completed.
        
        Args:
            batch_id: The batch ID
            output_filepath: Where to save the output results
            error_filepath: Where to save error results (optional)
            
        Returns:
            Tuple of (success, error_message)
        """
        status = self.get_batch_status(batch_id)
        current_status = status["status"]
        
        # Check for failure states
        if current_status in ["FAILED", "EXPIRED", "CANCELLED"]:
            return False, f"Batch failed with status: {current_status}"
        
        if current_status != "COMPLETED":
            return False, f"Batch not yet completed. Current status: {current_status}"
        
        if not status["output_file_id"]:
            return False, "No output file available"
        
        # Download output file
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        
        try:
            self.client.files.retrieve_content(
                id=status["output_file_id"],
                output=output_filepath
            )
            print(f"Downloaded results to: {output_filepath}")
            
            # Download error file if it exists and error_filepath provided
            if status.get("error_file_id") and error_filepath:
                os.makedirs(os.path.dirname(error_filepath), exist_ok=True)
                self.client.files.retrieve_content(
                    id=status["error_file_id"],
                    output=error_filepath  
                )
                print(f"Downloaded errors to: {error_filepath}")
            
            return True, None
            
        except Exception as e:
            return False, f"Error downloading results: {str(e)}"
    
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