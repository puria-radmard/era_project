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
                   key_negative_scores, chosen_trait, ocean_direction):
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
    data_configs = [
        ('all_data', 'blue', 'ICL'),
        ('control_all_data', 'gray', 'Control'),
        ('random_all_data', 'green', 'Random')
    ]
    
    for data_key, color, label in data_configs:
        results = calculate_scores_and_stats(
            log_data[data_key], question_indices, 
            key_positive_scores, key_negative_scores, 
            context_length_idx
        )
        
        plot_scores_with_uncertainty(
            axes[0], current_context_lengths,
            results['positive']['mean'], results['positive']['std'],
            color, label
        )
        
        plot_scores_with_uncertainty(
            axes[1], current_context_lengths,
            results['negative']['mean'], results['negative']['std'],
            color, label
        )
    
    # Add legends
    axes[0].legend()
    axes[1].legend()
    
    return fig


def process_single_context_length(context_length, rep_idx, cl_idx, config, 
                                 relevant_questions_and_answers, 
                                 all_questions_and_answers, chat_wrapper, 
                                 question_config, log_data):
    """
    Process all batches for a single context length and repetition.
    
    Args:
        context_length: Number of in-context examples
        rep_idx: Current repetition index
        cl_idx: Current context length index
        config: Experiment configuration
        relevant_questions_and_answers: Questions relevant to current trait
        all_questions_and_answers: All available questions for control
        chat_wrapper: Model wrapper
        question_config: Question configuration
        log_data: Dictionary to store results
    """
    num_minibatches = (len(relevant_questions_and_answers) // config.minibatch_size + 
                      bool(len(relevant_questions_and_answers) % config.minibatch_size != 0))
    
    for batch_idx in tqdm(range(num_minibatches), desc=f"Processing batches"):
        torch.cuda.empty_cache()
        
        # Select questions for this batch
        batch_upper_index = min((batch_idx + 1) * config.minibatch_size, 
                               len(relevant_questions_and_answers))
        asked_questions_idx = list(range(batch_idx * config.minibatch_size, 
                                       batch_upper_index))
        asked_questions = [relevant_questions_and_answers[i]['question'] 
                          for i in asked_questions_idx]
        actual_asked_questions_idx = [relevant_questions_and_answers[i]['index'] 
                                    for i in asked_questions_idx]
        
        # 1. Main ICL with signal
        ic_qa_batch = random_sample_excluding_indices(
            relevant_questions_and_answers, context_length, asked_questions_idx
        )
        in_context_questions = [icqa['question'] for icqa in ic_qa_batch]
        in_context_indices = [icqa['index'] for icqa in ic_qa_batch]
        in_context_answers = [f"Answer: {icqa['answer']}" for icqa in ic_qa_batch]
        
        if context_length > 0:
            log_data['in_context_questions_indices'][rep_idx, cl_idx, asked_questions_idx, :context_length] = in_context_indices
        
        choice_probs = elicit_mcq_batch(
            chat_wrapper, asked_questions, question_config, config,
            in_context_questions, in_context_answers
        )
        log_data['all_data'][rep_idx, cl_idx, asked_questions_idx, :] = choice_probs.cpu().numpy()
        
        # 2. Control ICL with noise
        control_ic_qa_batch = random_sample_excluding_indices(
            all_questions_and_answers, context_length, actual_asked_questions_idx
        )
        control_in_context_questions = [icqa['question'] for icqa in control_ic_qa_batch]
        control_in_context_indices = [icqa['index'] for icqa in control_ic_qa_batch]
        control_in_context_answers = [f"Answer: {icqa['answer']}" for icqa in control_ic_qa_batch]
        
        if context_length > 0:
            log_data['control_in_context_questions_indices'][rep_idx, cl_idx, asked_questions_idx, :context_length] = control_in_context_indices
        
        control_choice_probs = elicit_mcq_batch(
            chat_wrapper, asked_questions, question_config, config,
            control_in_context_questions, control_in_context_answers
        )
        log_data['control_all_data'][rep_idx, cl_idx, asked_questions_idx, :] = control_choice_probs.cpu().numpy()
        
        # 3. Random ICL (same questions, random answers)
        random_choice_probs = elicit_mcq_batch(
            chat_wrapper, asked_questions, question_config, config,
            in_context_questions, control_in_context_answers  # Mixed: signal questions, noise answers
        )
        log_data['random_all_data'][rep_idx, cl_idx, asked_questions_idx, :] = random_choice_probs.cpu().numpy()


def initialize_log_data(repeats_per_context_length, num_context_lengths, 
                       num_questions, max_context_length):
    """
    Initialize the log data structure.
    
    Args:
        repeats_per_context_length: Number of repetitions per context length
        num_context_lengths: Number of different context lengths
        num_questions: Number of questions
        max_context_length: Maximum context length
        
    Returns:
        Dictionary with initialized numpy arrays
    """
    return {
        'all_data': np.full([repeats_per_context_length, num_context_lengths, num_questions, 5], np.nan),
        'control_all_data': np.full([repeats_per_context_length, num_context_lengths, num_questions, 5], np.nan),
        'random_all_data': np.full([repeats_per_context_length, num_context_lengths, num_questions, 5], np.nan),
        'in_context_questions_indices': np.full([repeats_per_context_length, num_context_lengths, num_questions, max_context_length], np.nan),
        'control_in_context_questions_indices': np.full([repeats_per_context_length, num_context_lengths, num_questions, max_context_length], np.nan),
    }