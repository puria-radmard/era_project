import torch
from typing import Type
from model.base import ChatTemplateWrapper
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
)


# Factory function to get the appropriate wrapper
def get_chat_wrapper_class(model_name: str) -> Type[ChatTemplateWrapper]:
    """
    Get the appropriate chat wrapper class for a given model name.
    
    Args:
        model_name: The model name or path
        
    Returns:
        ChatTemplateWrapper class appropriate for the model
    """
    model_name_lower = model_name.lower()

    return ChatTemplateWrapper


def load_model(
    model_name: str, 
    device: str,
    torch_dtype: torch.dtype = torch.float16
) -> ChatTemplateWrapper:
    """
    Load a HuggingFace language model and return an appropriate chat wrapper.
    
    Args:
        model_name: Name or path of the model to load (e.g., "meta-llama/Llama-2-7b-chat-hf")
        device: Device to load the model on ("auto", "cuda", "cpu", etc.)
        torch_dtype: PyTorch data type for model weights
        
    Returns:
        ChatTemplateWrapper instance containing the model, tokenizer, and chat formatting
        
    Example:
        >>> chat_wrapper = load_model("meta-llama/Llama-2-7b-chat-hf")
        >>> # Can now call chat_wrapper.generate() or chat_wrapper.forward()
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_name, legacy = False, from_slow = False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == 'auto':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map = 'auto'
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        ).to(device)

    
    # Determine appropriate wrapper class
    wrapper_class = get_chat_wrapper_class(model_name)
    
    return wrapper_class(model, tokenizer)
