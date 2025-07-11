import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from util.util import random_sample_excluding_indices
from util.elicit import elicit_mcq_answer


def elicit_mcq_batch(chat_wrapper, questions, question_config, config, 
                     in_context_questions, in_context_answers):
    """
    Elicit MCQ answers for a batch of questions with given in-context examples.
    
    Args:
        chat_wrapper: The model wrapper
        questions: List of questions to ask
        question_config: Question configuration object
        config: Experiment configuration
        in_context_questions: List of in-context questions
        in_context_answers: List of in-context answers
        
    Returns:
        Tensor of choice probabilities/logits
    """
    answers = elicit_mcq_answer(
        chat_wrapper=chat_wrapper,
        questions=questions,
        shared_choices=question_config.mcq_shared_choices,
        config=question_config,
        system_prompt=config.system_prompt,
        shared_in_context_questions=in_context_questions,
        shared_in_context_answers=in_context_answers,
        choices_in_system_prompt=True
    )
    return answers['choice_logits']


def calculate_scores_and_stats(data, question_indices, key_positive_scores, key_negative_scores, 
                              context_length_idx):
    """
    Calculate mean scores and standard deviations for positive and negative questions.
    
    Args:
        data: Array of shape [repeats, context_lengths, questions, choices]
        question_indices: Dict with 'positive' and 'negative' question indices
        key_positive_scores: Array mapping choice indices to positive scores
        key_negative_scores: Array mapping choice indices to negative scores
        context_length_idx: Current context length index
        
    Returns:
        Dict containing means and stds for positive and negative questions
    """
    results = {}
    
    for question_type, indices in question_indices.items():
        key_scores = key_positive_scores if question_type == 'positive' else key_negative_scores
        
        question_data = data[:, :context_length_idx + 1, indices]
        choice_idx = question_data.argmax(-1)
        scores_array = key_scores[choice_idx]
        mean_scores = scores_array.mean(0).mean(-1)
        std_scores = scores_array.mean(0).std(-1)
        
        results[question_type] = {
            'mean': mean_scores,
            'std': std_scores
        }
    
    return results


def plot_scores_with_uncertainty(ax, context_lengths, means, stds, color, label=None):
    """
    Plot scores with uncertainty bands.
    
    Args:
        ax: Matplotlib axis
        context_lengths: Array of context lengths
        means: Mean scores for each context length
        stds: Standard deviations for each context length
        color: Plot color
        label: Legend label
    """
    ax.plot(context_lengths, means, color=color, marker='x', label=label)
    ax.fill_between(context_lengths, means - stds, means + stds, 
                   alpha=0.2, color=color)


def create_icl_plot(log_data, context_lengths, context_length_idx, 
                   relevant_questions_and_answers, key_positive_scores, 
                   key_negative_scores, chosen_trait, ocean_direction,
                   data_configs=None):
    """
    Create the ICL results plot with positive and negative question results.
    
    Args:
        log_data: Dictionary containing all experimental data
        context_lengths: List of context lengths
        context_length_idx: Current context length index
        relevant_questions_and_answers: List of question-answer dictionaries
        key_positive_scores: Array mapping choice indices to positive scores
        key_negative_scores: Array mapping choice indices to negative scores
        chosen_trait: String describing the trait being tested
        ocean_direction: +1 or -1 indicating expected direction of effect
        data_configs: List of tuples (data_key, colors, label) for plotting
        
    Returns:
        matplotlib figure object
    """
    
    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    
    fig.suptitle(f'In-context examples from {chosen_trait} person\n'
                f'expect ICL to {"in" if ocean_direction == 1 else "de"}crease '
                f'both scores against control')
    axes[0].set_title('Category-positive questions')
    axes[1].set_title('Category-negative questions')
    
    # Get question indices
    positive_indices = [i for i, rqa in enumerate(relevant_questions_and_answers) 
                       if rqa['key'] == 1]
    negative_indices = [i for i, rqa in enumerate(relevant_questions_and_answers) 
                       if rqa['key'] == -1]
    
    question_indices = {
        'positive': positive_indices,
        'negative': negative_indices
    }
    
    current_context_lengths = context_lengths[:context_length_idx + 1]
    
    # Calculate and plot results for each data type
    for data_key, colors, label in data_configs:
        if data_key not in log_data:
            continue  # Skip if this data type doesn't exist
            
        positive_color, negative_color = colors
        
        results = calculate_scores_and_stats(
            log_data[data_key], question_indices, 
            key_positive_scores, key_negative_scores, 
            context_length_idx
        )
        
        plot_scores_with_uncertainty(
            axes[0], current_context_lengths,
            results['positive']['mean'], results['positive']['std'],
            positive_color, label
        )
        
        plot_scores_with_uncertainty(
            axes[1], current_context_lengths,
            results['negative']['mean'], results['negative']['std'],
            negative_color, label
        )
    
    # Add legends
    axes[0].legend()
    axes[1].legend()
    
    return fig

