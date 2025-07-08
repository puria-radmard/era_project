"""
Question Configuration Module

This module contains configuration classes for multiple choice questions and text generation,
including templates and pre-tokenized choice options.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
from transformers import AutoTokenizer
import string

from util.experiment import ExperimentConfig


@dataclass
class QuestionConfig:
    """Configuration object containing templates and token mappings for question answering."""
    experiment_config: ExperimentConfig
    
    # Pre-tokenized choice tokens - will be populated by initialize_choices()
    choice_token_ids: Optional[Dict[str, List[int]]] = None
    
    # Raw token variations for each choice letter (before tokenization)
    _choice_token_strings: Dict[str, List[str]] = None
    
    def __post_init__(self):

        self.mcq_template = self.experiment_config.mcq_template
        self.mcq_prefiller = self.experiment_config.mcq_prefiller
        self.sentence_template = self.experiment_config.sentence_template

        if self._choice_token_strings is None:
            # Generate default token variations for letters A-Z
            self._choice_token_strings = {}
            for letter in string.ascii_uppercase:
                self._choice_token_strings[letter] = [
                    letter,           # "A"
                    # f" {letter}",     # " A"
                    # f"{letter}.",     # "A."
                    # f" {letter}.",    # " A."
                    # f"{letter} ",     # "A "
                    # f" {letter} ",    # " A "
                ]
    
    def initialize_choices(self, tokenizer: AutoTokenizer, max_choices: int = 26) -> QuestionConfig:
        """
        Pre-tokenize all choice token variations and validate they are single tokens.
        
        Args:
            tokenizer: The tokenizer to use for tokenization
            max_choices: Maximum number of choice letters to initialize (A-Z by default)
            
        Raises:
            ValueError: If any token variation produces more than one token
        """
        self.choice_token_ids = {}
        
        for i in range(min(max_choices, 26)):
            letter = string.ascii_uppercase[i]
            token_variations = self._choice_token_strings[letter]
            
            tokenized_variations = []
            
            for token_str in token_variations:
                # Tokenize without special tokens
                token_ids = tokenizer.encode(token_str, add_special_tokens=False)
                
                # Validate it's a single token
                if len(token_ids) != 1:
                    raise ValueError(
                        f"Choice token variation '{token_str}' for letter {letter} "
                        f"produces {len(token_ids)} tokens: {token_ids}. "
                        f"All choice tokens must be exactly one token."
                    )
                
                tokenized_variations.append(token_ids[0])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_tokens = []
            for token_id in tokenized_variations:
                if token_id not in seen:
                    seen.add(token_id)
                    unique_tokens.append(token_id)
            
            self.choice_token_ids[letter] = unique_tokens
        
        return self
    
    def get_choice_token_ids(self, letter: str) -> List[int]:
        """
        Get the pre-tokenized token IDs for a given choice letter.
        
        Args:
            letter: The choice letter (A, B, C, etc.)
            
        Returns:
            List of token IDs for this choice letter
            
        Raises:
            ValueError: If choices haven't been initialized or letter is invalid
        """
        if self.choice_token_ids is None:
            raise ValueError(
                "Choice tokens not initialized. Call initialize_choices() first."
            )
        
        if letter not in self.choice_token_ids:
            raise ValueError(f"Invalid choice letter: {letter}")
        
        return self.choice_token_ids[letter]
    
    def format_mcq_choices(self, choices: List[str]) -> str:
        """
        Format a list of choices into lettered options (A. choice1\nB. choice2\n...).
        
        Args:
            choices: List of choice strings
            
        Returns:
            Formatted string with lettered choices
        """
        formatted_choices = []
        for i, choice in enumerate(choices):
            letter = string.ascii_uppercase[i]
            formatted_choices.append(f"{letter}. {choice}")
        return "\n".join(formatted_choices)

