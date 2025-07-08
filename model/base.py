"""
HuggingFace Model Utilities for Multiple Choice and Text Generation

This module provides utilities for working with decoder-only language models
using HuggingFace transformers, with support for prompt caching, multiple choice
question answering, and text generation.
"""

from typing import Tuple, List, Dict, Union, Optional, Any
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DynamicCache
)
from abc import ABC, abstractmethod


class ChatTemplateWrapper(ABC):
    """
    Abstract base class for chat template wrappers that handle model inference.
    
    This class combines model, tokenizer, and chat formatting into a single interface
    for easy batch processing of conversations.
    """
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        """
        Initialize the chat wrapper with model and tokenizer.
        
        Args:
            model: The loaded language model
            tokenizer: The model's tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @abstractmethod
    def format_chat(
        self, 
        *_,
        system_prompt: Optional[str] = None,
        in_context_questions: Optional[List[str]] = None,
        in_context_answers: Optional[List[str]] = None,
        user_message: Optional[str] = None,
        prefiller: Optional[str] = None,
    ) -> str:
        """
        Format a system prompt and user message according to the model's chat template.
        
        Args:
            system_prompt: The system prompt/instruction
            user_message: The user's message/question
            cache: Optional cache object - if provided, only format user part
            
        Returns:
            Formatted chat string ready for tokenization
        """
        pass
    
    def duplicate_cache(
        self,
        past_key_values: DynamicCache,
        inputs: Any
    ) -> Tuple[DynamicCache, Any]:
        import pdb; pdb.set_trace(header = 'type inputs')

        bsz = inputs.input_ids.shape[0]
        rep_func = lambda x: x.expand(bsz, *x.shape[1:])
        past_key_values.key_cache = list(map(rep_func, past_key_values.key_cache))
        past_key_values.value_cache = list(map(rep_func, past_key_values.value_cache))

        inputs.attention_mask = torch.concat([torch.ones(bsz, past_key_values.value_cache[0].shape[2]).cuda(), inputs.attention_mask], 1)

        return past_key_values, inputs

    @torch.no_grad()
    def forward(
        self, 
        chats: List[str], 
        past_key_values: Optional[DynamicCache] = None,
        return_dict: bool = True,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Run forward pass on a batch of chat strings.
        
        Args:
            chats: List of formatted chat strings
            past_key_values: Optional cached key-value states (DynamicCache)
            max_length: Maximum sequence length for tokenization
            return_dict: Whether to return dictionary output
            use_cache: Whether to use/update cache
            
        Returns:
            Model outputs dictionary containing logits, past_key_values, etc.
        """
        # Tokenize the chats internally
        inputs = self.tokenizer(
            chats,
            return_tensors="pt",
            padding = True,
            padding_side = "left",
            truncation=True,
        ).to(self.device)

        # Expand out cache to repeat over batch
        if past_key_values is not None:
            self.duplicate_cache(
                past_key_values=past_key_values,
                inputs=inputs
            )

        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict
        )

        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        chats: List[str],
        past_key_values: Optional[DynamicCache] = None,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        do_sample: bool = True,
        max_length: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text for a batch of chat strings.
        
        Args:
            chats: List of formatted chat strings
            past_key_values: Optional cached key-value states (DynamicCache)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling or greedy decoding
            max_length: Maximum sequence length for input tokenization
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing:
                - "sequences": Generated token sequences
                - "generated_texts": Decoded generated text (new tokens only)
                - "input_length": Length of input tokens
        """
        # Tokenize the chats internally
        inputs = self.tokenizer(
            chats,
            return_tensors="pt",
            padding=True,
            padding_side = "left",
            truncation=True,
            max_length=max_length
        ).to(self.device)
        
        # Calculate input length for extracting new tokens later
        if past_key_values is not None:
            # For DynamicCache, get sequence length
            cache_length = past_key_values.get_seq_length()
            input_length = cache_length + inputs.input_ids.shape[1]
        else:
            input_length = inputs.input_ids.shape[1]
        
        # Generate
        outputs = self.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            past_key_values=past_key_values,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            **kwargs
        )
        
        # Extract and decode new tokens
        generated_sequences = outputs.sequences
        batch_size = len(chats)
        generated_texts = []
        
        for i in range(batch_size):
            # Extract only newly generated tokens
            new_tokens = generated_sequences[i, input_length:]
            
            # Decode to text
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            text = text.strip()
            
            generated_texts.append(text)
        
        return {
            "sequences": generated_sequences,
            "generated_texts": generated_texts,
            "input_length": input_length
        }

    @torch.no_grad()
    def create_prompt_cache(
        self,
        system_prompt: str,
        max_cache_len: int = 1024
    ) -> Dict[str, Union[DynamicCache, torch.Tensor]]:
        """
        Create a DynamicCache object containing precomputed key-value states for a system prompt.
        
        Args:
            system_prompt: The system prompt to cache
            max_cache_len: Maximum cache length (default: 1024)
            
        Returns:
            Dictionary containing:
                - "cache": The DynamicCache object with precomputed states
                - "input_ids": Token IDs of the cached prompt
                - "attention_mask": Attention mask for the cached prompt
                
        Example:
            >>> chat_wrapper = load_model("meta-llama/Llama-2-7b-chat-hf")
            >>> cache_data = chat_wrapper.create_prompt_cache("You are a helpful assistant.")
            >>> # Use cache in subsequent calls
        """
        # Format just the system prompt part (no user message)
        formatted_prompt = self.format_chat(system_prompt = system_prompt)
        
        # Tokenize the system prompt
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            padding=False, 
            truncation=False
        ).to(self.device)
        
        prompt_cache = DynamicCache()

        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            past_key_values=prompt_cache,
            use_cache=True,
            return_dict=True
        )
        
        return {
            "cache": outputs.past_key_values,
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask
        }



class LlamaChatWrapper(ChatTemplateWrapper):
    """Chat template wrapper for Llama-style models."""
    
    def format_chat(
        self, 
        *_,
        system_prompt: Optional[str] = None,
        in_context_questions: Optional[List[str]] = None,
        in_context_answers: Optional[List[str]] = None,
        user_message: Optional[str] = None,
        prefiller: Optional[str] = None,
    ) -> str:
        query = ""

        if system_prompt is not None:
            query += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"

        if in_context_questions is not None:
            assert (in_context_answers is not None) and (len(in_context_answers) == len(in_context_questions))
            for question, answer in zip(in_context_questions, in_context_answers):
                query += f"<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>"
                query += f"<|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"

        if user_message is not None:
            query += f"<|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|>"


        if prefiller is not None:
            query += f"<|start_header_id|>assistant<|end_header_id|>\n\n{prefiller}"
        
        return query
