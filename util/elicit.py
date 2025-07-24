import copy
import torch
import string
from transformers.cache_utils import DynamicCache
from typing import Union, List, Optional, Dict
from model.base import ChatTemplateWrapper
from util.question import QuestionConfig
import torch.nn.functional as F



def get_choice_token_logits_from_config(
    logits: torch.Tensor, 
    num_choices: int, 
    config: QuestionConfig
) -> torch.Tensor:
    """
    Extract and sum logits for choice tokens (A, B, C, etc.) from the full vocabulary logits.
    
    Args:
        logits: Full vocabulary logits tensor of shape [batch_size, vocab_size]
        num_choices: duh
        config: Configuration containing pre-tokenized choice tokens
        
    Returns:
        Tensor of shape [batch_size, num_choices] with summed probabilities for each choice
    """
    batch_size = logits.shape[0]
    
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Initialize output tensor
    choice_probs = torch.zeros(batch_size, num_choices, device=logits.device)
    
    # For each choice letter
    for choice_idx in range(num_choices):
        letter = string.ascii_uppercase[choice_idx]
        token_ids = config.get_choice_token_ids(letter)
        
        # Sum probabilities across all token variations for this choice
        total_prob = torch.zeros(batch_size, device=logits.device)
        
        for token_id in token_ids:
            total_prob += probs[:, token_id]
                
        choice_probs[:, choice_idx] = total_prob

    return choice_probs


def get_choice_token_logits_from_token_ids(
    logits: torch.Tensor,
    choice_tokens_ids: List[List[str]]
) -> torch.Tensor:
    """
    Extract and sum logits for choice tokens (A, B, C, etc.) from the full vocabulary logits.
    
    Args:
        logits: Full vocabulary logits tensor of shape [batch_size, get_choice_token_logits_from_config, vocab_size]
        choice_tokens_ids: List of token ids for each choice, i.e. len(choice_tokens_ids) = num choices, and len(choice_tokens_ids[i]) = num possible tokens for choice i
        
    Returns:
        Tensor of shape [batch_size, num_choices] with summed probabilities for each choice
    """
    batch_size = logits.shape[0]
    
    # Convert logits to probabilitiess
    probs = F.softmax(logits, dim=-1)
    
    # Initialize output tensor
    choice_probs = torch.zeros(batch_size, len(choice_tokens_ids), device=logits.device)

    # For each choice
    for choice_idx, possible_token_ids in enumerate(choice_tokens_ids):
        for token_id in possible_token_ids:
            choice_probs[:, choice_idx] += probs[:, -1, token_id]
    
    return choice_probs


def elicit_mcq_answer(
    *_,
    chat_wrapper: ChatTemplateWrapper,
    questions: List[str],
    choices_batch: Optional[List[List[str]]] = None,
    shared_choices: Optional[List[str]] = None,
    config: QuestionConfig,
    system_prompt: Optional[str] = None,
    cache_data: Optional[Dict[str, Union[DynamicCache, torch.Tensor]]] = None,
    in_context_questions: Optional[List[List[str]]] = None,
    in_context_answers: Optional[List[List[str]]] = None,
    shared_in_context_questions: Optional[List[str]] = None,
    shared_in_context_answers: Optional[List[str]] = None,
    choices_in_system_prompt: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Generate logits for multiple choice questions and extract choice-specific probabilities.
    
    Args:
        chat_wrapper: The chat wrapper containing model and tokenizer
        questions: List of questions (batch)
        choices_batch: List of choice lists, one per question. All questions must have same number of choices.
        shared_choices: The same set of choices, formatted into a string
        config: Configuration object containing MCQ template and pre-tokenized choice tokens
        system_prompt: System prompt to use (ignored if cache_data provided)
        cache_data: Optional precomputed cache data from create_prompt_cache()
        choices_in_system_prompt: if True, formatted_choices inserted into system prompt rather than in the question itself
        
    Returns:
        Dictionary containing:
            - "full_logits": Full vocabulary logits tensor of shape [batch_size, vocab_size]
            - "choice_logits": Choice-specific logits tensor of shape [batch_size, num_choices]
            
    Example:
        >>> chat_wrapper = load_model("meta-llama/Llama-2-7b-chat-hf")
        >>> question_config = QuestionConfig().initialize_choices(chat_wrapper.tokenizer)
        >>> questions = ["What is 2+2?", "What color is the sky?"]
        >>> choices = [["3", "4", "5"], ["red", "blue", "green"]]
        >>> result = elicit_mcq_answer(chat_wrapper, questions, choices, question_config)
        >>> choice_probs = result["choice_logits"]  # [2, 3] tensor
    """
    if not questions:
        raise ValueError("Questions list cannot be empty")

    if choices_batch is None:
        assert shared_choices is not None
        formatted_choices = config.format_mcq_choices(shared_choices)
        num_choices = len(shared_choices)

    else:
        assert shared_choices is None

        if len(questions) != len(choices_batch):
            raise ValueError("Number of questions must match number of choice lists")
        
        # Verify all questions have same number of choices
        num_choices = len(choices_batch[0])
        if not all(len(choices) == num_choices for choices in choices_batch):
            raise ValueError("All questions must have the same number of choices")
        
    if choices_in_system_prompt:
        assert shared_choices is not None
        system_prompt = system_prompt.format(choices = formatted_choices)

    if shared_in_context_answers is not None:
        assert in_context_questions is None and in_context_answers is None
        assert shared_in_context_questions is not None
    if in_context_answers is not None:
        assert shared_in_context_questions is None and shared_in_context_answers is None
        assert in_context_questions is not None
    
    # Format chats
    formatted_chats = []
    
    for iq, question in enumerate(questions):
        
        # Format the choices
        if choices_batch is not None:
            formatted_choices = config.format_mcq_choices(choices_batch[iq])

        # Create the full question with template
        if choices_in_system_prompt:
            full_question = config.mcq_template.format(question = question, choices=formatted_choices)
        else:
            full_question = config.mcq_template.format(question = question)

        if in_context_answers is not None:
            this_in_context_questions = in_context_questions[iq]
            this_in_context_answers = in_context_answers[iq]
        elif shared_in_context_answers is not None:
            this_in_context_questions = shared_in_context_questions
            this_in_context_answers = shared_in_context_answers
        else:
            this_in_context_questions = None
            this_in_context_answers = None
            this_formatted_in_context_questions = None
        
        # Format with chat template
        if this_in_context_questions is not None:
            this_formatted_in_context_questions = [
                config.mcq_template.format(question = inq, choices=formatted_choices)
                for inq in this_in_context_questions
            ]
        
        formatted_chat = chat_wrapper.format_chat(
            system_prompt=system_prompt,    # If none, then no problem!
            user_message=full_question,
            prefiller=config.mcq_prefiller,
            in_context_questions = this_formatted_in_context_questions,
            in_context_answers = this_in_context_answers,
        )
            
        formatted_chats.append(formatted_chat)

    # Get model outputs using the chat wrapper
    outputs = chat_wrapper.forward(
        chats=formatted_chats,
        past_key_values=cache_data["cache"] if cache_data else None,
        use_cache=cache_data["cache"] is not None if cache_data else False,
    )
    
    # Get logits for the last token (where the answer should be generated)
    last_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
    
    # Extract choice-specific logits
    
    choice_logits = get_choice_token_logits_from_config(
        last_token_logits, 
        num_choices,
        config
    )
    
    return {
        "full_logits": last_token_logits,
        "choice_logits": choice_logits
    }


def elicit_freeform_answer(
    *_,
    chat_wrapper: ChatTemplateWrapper,
    freeform_template_name: str,
    questions: List[str],
    config: QuestionConfig,
    system_prompt: Optional[str] = None,
    cache_data: Optional[Dict] = None,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    do_sample: bool = True
) -> List[str]:
    """
    Generate free-form answers for a batch of questions.
    
    Args:
        chat_wrapper: The chat wrapper containing model and tokenizer
        questions: List of questions (batch)
        config: Configuration object containing free-form generation template
        system_prompt: System prompt to use (ignored if cache_data provided)
        cache_data: Optional precomputed cache data from create_prompt_cache()
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature for generation
        do_sample: Whether to use sampling or greedy decoding
        
    Returns:
        List of generated answer strings, one per question
        
    Example:
        >>> chat_wrapper = load_model("meta-llama/Llama-2-7b-chat-hf")
        >>> config = create_question_config(chat_wrapper.tokenizer)
        >>> questions = ["What is the capital of France?", "Who invented the telephone?"]
        >>> answers = elicit_freeform_answer(chat_wrapper, questions, config)
        >>> print(answers)  # ["Paris is the capital of France.", "Alexander Graham Bell invented the telephone."]
    """
    if not questions:
        raise ValueError("Questions list cannot be empty")
    
    # Format chats
    formatted_chats = []
    
    for question in questions:
        # Add the freeform instruction template
        full_question = question + "\n\n" + config.freeform_templates[freeform_template_name]
        
        # Format with chat template
        if cache_data is None:
            assert system_prompt is not None
        
        formatted_chat = chat_wrapper.format_chat(
            system_prompt=system_prompt,    # If none, then no problem!
            user_message=full_question,
            prefiller=config.freeform_prefillers[freeform_template_name]
        )
            
        formatted_chats.append(formatted_chat)
    
    # Generate responses using the chat wrapper
    generation_result = chat_wrapper.generate(
        chats=formatted_chats,
        past_key_values=cache_data["cache"] if cache_data else None,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        max_length=config.max_length
    )
    
    cleaned_texts = generation_result["generated_texts"]
    
    return cleaned_texts


def elicit_formatted_answer(
    *_,
    chat_wrapper: ChatTemplateWrapper,
    freeform_template_name: str,
    questions: List[Dict[str, str]],
    config: QuestionConfig,
    system_prompt: Optional[str] = None,
    cache_data: Optional[Dict] = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    do_sample: bool = True
) -> List[str]:
    """
    Generate free-form answers for a batch of questions.
    
    Args:
        chat_wrapper: The chat wrapper containing model and tokenizer
        questions: List of things with which to format config.freeform_templates
        config: Configuration object containing free-form generation template
        system_prompt: System prompt to use (ignored if cache_data provided)
        cache_data: Optional precomputed cache data from create_prompt_cache()
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature for generation
        do_sample: Whether to use sampling or greedy decoding
        
    Returns:
        List of generated answer strings, one per question
        
    Example:
        >>> chat_wrapper = load_model("meta-llama/Llama-2-7b-chat-hf")
        >>> config = create_question_config(chat_wrapper.tokenizer)
        >>> questions = ["What is the capital of France?", "Who invented the telephone?"]
        >>> answers = elicit_freeform_answer(chat_wrapper, questions, config)
        >>> print(answers)  # ["Paris is the capital of France.", "Alexander Graham Bell invented the telephone."]
    """
    if not questions:
        raise ValueError("Questions list cannot be empty")
    
    # Format chats
    formatted_chats = []
    
    for question in questions:
        # Add the freeform instruction template
        full_question = config.freeform_templates[freeform_template_name].format(**question)

        # Format with chat template
        if cache_data is None:
            assert system_prompt is not None

        formatted_chat = chat_wrapper.format_chat(
            system_prompt=system_prompt,    # If none, then no problem!
            user_message=full_question,
            prefiller=config.freeform_prefillers[freeform_template_name]
        )
            
        formatted_chats.append(formatted_chat)
    
    # Generate responses using the chat wrapper
    generation_result = chat_wrapper.generate(
        chats=formatted_chats,
        past_key_values=cache_data["cache"] if cache_data else None,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )
    
    cleaned_texts = generation_result["generated_texts"]
    
    return cleaned_texts


def elicit_next_token_probs(
    *_,
    chat_wrapper: ChatTemplateWrapper,
    questions: List[str],
    system_prompt: Optional[str] = None,
    cache_data: Optional[Dict[str, Union[DynamicCache, torch.Tensor]]] = None,
    in_context_questions: Optional[List[List[str]]] = None,
    in_context_answers: Optional[List[List[str]]] = None,
    shared_in_context_questions: Optional[List[str]] = None,
    shared_in_context_answers: Optional[List[str]] = None,
    question_template: Optional[str] = None,
    prefiller: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Generate next token probability distributions for a batch of questions with in-context examples.
    
    Args:
        chat_wrapper: The chat wrapper containing model and tokenizer
        questions: List of questions (batch)
        system_prompt: System prompt to use (ignored if cache_data provided)
        cache_data: Optional precomputed cache data from create_prompt_cache()
        in_context_questions: List of in-context question lists, one per question
        in_context_answers: List of in-context answer lists, one per question  
        shared_in_context_questions: Shared in-context questions for all questions
        shared_in_context_answers: Shared in-context answers for all questions
        question_template: Optional template to format questions (defaults to using questions directly)
        prefiller: Optional prefiller text to add before model generation
        
    Returns:
        Dictionary containing:
            - "logits": Raw logits tensor of shape [batch_size, vocab_size]
            - "probs": Probability distribution tensor of shape [batch_size, vocab_size]
            
    Example:
        >>> chat_wrapper = load_model("meta-llama/Llama-2-7b-chat-hf")
        >>> questions = ["What is 2+2?", "What color is the sky?"]
        >>> shared_questions = ["What is 1+1?", "What color is grass?"]
        >>> shared_answers = ["2", "green"]
        >>> result = elicit_next_token_probs(
        ...     chat_wrapper, questions, "Answer the question.",
        ...     shared_in_context_questions=shared_questions,
        ...     shared_in_context_answers=shared_answers
        ... )
        >>> next_token_probs = result["probs"]  # [2, vocab_size] tensor
    """
    if not questions:
        raise ValueError("Questions list cannot be empty")

    # Validate cache_data and system_prompt
    if cache_data is None:
        assert system_prompt is not None

    # Validate in-context arguments
    if shared_in_context_answers is not None:
        assert in_context_questions is None and in_context_answers is None
        assert shared_in_context_questions is not None
        if len(shared_in_context_questions) != len(shared_in_context_answers):
            raise ValueError("Shared in-context questions and answers must have same length")
    
    if in_context_answers is not None:
        assert shared_in_context_questions is None and shared_in_context_answers is None
        assert in_context_questions is not None
        if len(in_context_questions) != len(questions):
            raise ValueError("Number of in-context question lists must match number of questions")
        if len(in_context_answers) != len(questions):
            raise ValueError("Number of in-context answer lists must match number of questions")
        # Validate each question has matching QA pairs
        for i, (iq_list, ia_list) in enumerate(zip(in_context_questions, in_context_answers)):
            if len(iq_list) != len(ia_list):
                raise ValueError(f"Question {i}: in-context questions and answers must have same length")
    
    # Format chats
    formatted_chats = []
    
    for iq, question in enumerate(questions):
        
        # Format the question with template if provided
        if question_template is not None:
            full_question = question_template.format(question=question)
        else:
            full_question = question

        # Determine in-context examples for this question
        if in_context_answers is not None:
            this_in_context_questions = in_context_questions[iq]
            this_in_context_answers = in_context_answers[iq]
        elif shared_in_context_answers is not None:
            this_in_context_questions = shared_in_context_questions
            this_in_context_answers = shared_in_context_answers
        else:
            this_in_context_questions = None
            this_in_context_answers = None
        
        # Format with chat template
        formatted_chat = chat_wrapper.format_chat(
            system_prompt=system_prompt,    # If none, then no problem!
            user_message=full_question,
            prefiller=prefiller,
            in_context_questions=this_in_context_questions,
            in_context_answers=this_in_context_answers,
        )
            
        formatted_chats.append(formatted_chat)

    # Get model outputs using the chat wrapper
    outputs = chat_wrapper.forward(
        chats=formatted_chats,
        past_key_values=cache_data["cache"] if cache_data else None,
        use_cache=cache_data["cache"] is not None if cache_data else False,
    )
    
    # Get logits for the last token (where the next token should be predicted)
    last_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
    
    # Convert to probabilities
    next_token_probs = F.softmax(last_token_logits, dim=-1)
    
    return {
        "logits": last_token_logits,
        "probs": next_token_probs
    }



def elicit_sequence_log_probs(
    chat_wrapper: ChatTemplateWrapper,
    question_cache: Dict,
    response_sequences: List[str],
) -> torch.Tensor:
    """
    Calculate token-averaged log probabilities for a list of response sequences.
    
    Args:
        chat_wrapper: The chat wrapper containing model and tokenizer
        question_cache: Cache data already containing the question context
        response_sequences: List of response strings to evaluate
        
    Returns:
        Tensor of shape [num_sequences] containing average log probabilities per token
    """
    if not response_sequences:
        return torch.tensor([])
    
    # Tokenize all response sequences
    sequence_token_ids = []
    for response in response_sequences:
        # Clean response (remove trailing periods)
        response_clean = response.removesuffix(".")
        tokens = chat_wrapper.tokenizer.encode(response_clean, add_special_tokens=False)
        sequence_token_ids.append(tokens)
    
    # Get log probabilities for each sequence
    sequence_log_probs = []

    for tokens in sequence_token_ids:
        if len(tokens) == 0:
            sequence_log_probs.append(0.0)
            continue
            
        # Clone the question cache for this sequence
        sequence_cache = copy.deepcopy(question_cache)
        
        # Prepare input tokens (convert to tensor)
        input_ids = torch.tensor([tokens], device=chat_wrapper.device)
        attention_mask = torch.ones_like(input_ids)
        
        # Extend attention mask to account for cached content
        cache_length = sequence_cache["cache"].get_seq_length()
        full_attention_mask = torch.cat([
            torch.ones(1, cache_length, device=chat_wrapper.device),
            attention_mask
        ], dim=1)
        
        # Forward pass to get logits for this sequence
        with torch.no_grad():
            outputs = chat_wrapper.model(
                input_ids=input_ids,
                attention_mask=full_attention_mask,
                past_key_values=sequence_cache["cache"],
                use_cache=False,
                return_dict=True
            )
        
        # Get logits for each position (excluding the last position since we don't predict after the sequence)
        logits = outputs.logits[0, :-1, :]  # [seq_len-1, vocab_size]
        target_tokens = torch.tensor(tokens[1:], device=chat_wrapper.device)  # [seq_len-1]
        
        # Calculate log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[range(len(target_tokens)), target_tokens]

        # Average log probability per token
        avg_log_prob = token_log_probs.mean().item()
        sequence_log_probs.append(avg_log_prob)
    
    return torch.tensor(sequence_log_probs)
