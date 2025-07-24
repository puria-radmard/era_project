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
    DynamicCache,
    DynamicCache
)
import copy
from transformers.tokenization_utils import BatchEncoding
import re

class ChatTemplateWrapper:
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

    def format_chat(
        self, 
        *_,
        system_prompt: Optional[str] = None,
        in_context_questions: Optional[List[str]] = None,
        in_context_answers: Optional[List[str]] = None,
        user_message: Optional[str] = None,
        prefiller: Optional[str] = None,
        keep_bos: bool = False
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

        history = []

        if system_prompt is not None:
            history.append({"role": "system", "content": system_prompt})

        if in_context_questions is not None:
            assert (in_context_answers is not None) and (len(in_context_answers) == len(in_context_questions))
            for question, answer in zip(in_context_questions, in_context_answers):
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": answer})

        if user_message is not None:
            history.append({"role": "user", "content": user_message})

        if prefiller is not None:
            if prefiller == "":
                add_generation_prompt = True
            else:
                history.append({"role": "assistant", "content": prefiller})
                add_generation_prompt = False
        else:
            add_generation_prompt = False
   

        prompt = self.tokenizer.apply_chat_template(history, tokenize = False, add_generation_prompt=add_generation_prompt, continue_final_message = (prefiller is not None and prefiller != ""),)

        # FIXME: so crazy to me that this is in here
        prompt = re.sub(r'\n\nCutting Knowledge Date: [A-Za-z]+\s+\d{4}\nToday Date: \d{1,2} [A-Za-z]{3} \d{4}', '', prompt)
        
        prompt = prompt.replace('<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>', '<|begin_of_text|>')
        prompt = prompt.replace('[SYSTEM_PROMPT][/SYSTEM_PROMPT]', '')

        if system_prompt is None and not keep_bos:
            prompt = prompt.removeprefix('<|begin_of_text|>')

        return prompt
                
    
    def duplicate_cache(
        self,
        past_key_values: DynamicCache,
        inputs: BatchEncoding
    ) -> Tuple[DynamicCache, Any]:

        bsz = inputs.input_ids.shape[0]
        rep_func = lambda x: x.expand(bsz, *x.shape[1:])
        past_key_values.key_cache = list(map(rep_func, past_key_values.key_cache))
        past_key_values.value_cache = list(map(rep_func, past_key_values.value_cache))

        inputs.attention_mask = torch.concat([torch.ones(bsz, past_key_values.value_cache[0].shape[2]).cuda(), inputs.attention_mask], 1)

        return torch.tensor([past_key_values[0][0].shape[2]], dtype=torch.long, device=self.model.device)

    @torch.no_grad()
    def forward(
        self, 
        chats: List[str], 
        past_key_values: Optional[DynamicCache] = None,
        return_dict: bool = True,
        use_cache: bool = True,
        **forward_kwargs: Any
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
            return_dict=return_dict,
            # cache_position = cache_position,
            **forward_kwargs
        )

        return outputs
    
    @torch.no_grad()
    def generate_parallel(
        self,
        chats: List[str],
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs
    ):
        """
        See self.generate for arguments
        Haven't figured out caching for this yet!
        """
        # Tokenize the chats internally
        inputs = self.tokenizer(
            chats,
            return_tensors="pt",
            padding = True,
            padding_side = "left",
            truncation=True,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            **kwargs
        )

        input_length = inputs.input_ids.shape[1]
        
        generated_sequences = outputs.sequences

        generated_texts = [self.tokenizer.decode(generated_sequence[input_length:], skip_special_tokens=True) for generated_sequence in generated_sequences]
        generated_full_texts = [self.tokenizer.decode(generated_sequence, skip_special_tokens=True) for generated_sequence in generated_sequences]

        return {
            "sequences": generated_sequences,
            "generated_full_texts": generated_full_texts,
            "generated_texts": generated_texts,
            "input_length": input_length
        }

    
    @torch.no_grad()
    def generate(
        self,
        chats: List[str],
        past_key_values: Optional[DynamicCache] = None,
        past_key_values_str: str = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        do_sample: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text for a batch of chat strings.

        Args:
            chats: List of formatted chat strings
            past_key_values: Optional cached key-value states (DynamicCache)
            past_key_values_str: The string that was 
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
        assert (past_key_values is None) == (past_key_values_str is None)

        # if (past_key_values_str is not None):
        #     cache_position = torch.tensor([past_key_values.get_seq_length()], dtype=torch.long, device=self.device)
        # else:
        #     cache_position = None

        generated_texts = []
        generated_full_texts = []

        # Tokenize the chats internally
        for chat in chats:
            if (past_key_values_str is not None):
                chat = past_key_values_str + chat

            inputs = self.tokenizer(
                chat,
                return_tensors="pt",
                padding=False, 
                truncation=False,
                add_special_tokens = False,
            ).to("cuda")
            input_length = inputs.input_ids.shape[1]


            past_key_values_copy = copy.deepcopy(past_key_values)
            
            outputs = self.model.generate(
                **inputs,
                past_key_values=past_key_values_copy,
                # cache_position=cache_position,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                **kwargs
            )

            text = self.tokenizer.decode(outputs.sequences[0, input_length:], skip_special_tokens=False)
            full_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=False, max_new_tokens=20).strip()
            
            generated_texts.append(text)
            generated_full_texts.append(full_text)
        
        return {
            # "sequences": generated_sequences,
            "generated_full_texts": generated_full_texts,
            "generated_texts": generated_texts,
            "input_length": input_length
        }

    @torch.no_grad()
    def create_prompt_cache(
        self,
        system_prompt: Optional[str] = None,
        in_context_questions: Optional[List[str]] = None,
        in_context_answers: Optional[List[str]] = None,
        user_message: Optional[str] = None,
        prefiller: Optional[str] = None,
        max_cache_len: int = 1024
    ) -> Dict[str, Union[DynamicCache, torch.Tensor, str]]:
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
        formatted_prompt = self.format_chat(
            system_prompt = system_prompt,
            in_context_questions = in_context_questions,
            in_context_answers = in_context_answers,
            user_message = user_message,
            prefiller = prefiller,
        )
        
        # Tokenize the system prompt
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            padding=False, 
            truncation=False,
            add_special_tokens=False
        ).to(self.device)

        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            use_cache=True,
            return_dict=True
        )

        return {
            "formatted_prompt": formatted_prompt,
            "cache": outputs.past_key_values,
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask
        }

