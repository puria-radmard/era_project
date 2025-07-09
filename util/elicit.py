from model.base import *
from util.question import *
import torch.nn.functional as F



def _get_choice_token_logits(
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
    
    choice_logits = _get_choice_token_logits(
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
