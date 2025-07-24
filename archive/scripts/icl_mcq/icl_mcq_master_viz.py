import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os

# Load the original data
ocean_questions_df = pd.read_csv('results/p2_mcq_probs.csv')
ocean_questions_df = ocean_questions_df.reset_index()

# Constants
chosen_trait_to_ocean_and_direction = {
    'an agreeable': ('A', +1),
    'an extraversive': ('E', +1),
    'a conscientious': ('C', +1),
    'a neurotic': ('N', +1),
    'an open': ('O', +1),
    'an introversive': ('E', -1),
    'a disagreeable': ('A', -1),
    'an unconscientious': ('C', -1),
    'a stable': ('N', -1),
    'a closed': ('O', -1),
}

context_lengths = [0, 1, 2, 5, 10]
key_positive_scores = np.array([5, 4, 3, 2, 1])
key_negative_scores = np.array([1, 2, 3, 4, 5])

# Group traits by OCEAN dimension and direction
ocean_dimensions = ['A', 'E', 'C', 'N', 'O']
dimension_names = ['Agreeableness', 'Extraversion', 'Conscientiousness', 'Neuroticism', 'Openness']

trait_pairs = []
for ocean_dim in ocean_dimensions:
    positive_trait = None
    negative_trait = None
    
    for trait, (dim, direction) in chosen_trait_to_ocean_and_direction.items():
        if dim == ocean_dim:
            if direction == +1:
                positive_trait = trait
            else:
                negative_trait = trait
    
    trait_pairs.append((positive_trait, negative_trait))

def calculate_scores_and_stats(data, question_indices, key_positive_scores, key_negative_scores, 
                              context_length_idx):
    """Calculate mean scores and standard deviations for positive and negative questions."""
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
    """Plot scores with uncertainty bands."""
    ax.plot(context_lengths, means, color=color, marker='x', label=label, linewidth=2)
    ax.fill_between(context_lengths, means - stds, means + stds, 
                   alpha=0.2, color=color)

def perform_paired_ttest(all_data_scores, control_data_scores, expected_direction):
    """
    Perform paired t-test between all_data and control data.
    Returns True if significant difference in expected direction, False otherwise.
    
    all_data_scores and control_data_scores should be 1D arrays of scores across questions
    expected_direction: +1 if we expect all_data > control_data, -1 if we expect all_data < control_data
    """
    if len(all_data_scores) != len(control_data_scores):
        return False
    
    # Perform two-tailed t-test
    t_stat, p_value = stats.ttest_rel(all_data_scores, control_data_scores)
    
    # Check if significant (p < 0.05) and in expected direction
    if p_value < 0.05:
        if expected_direction == 1:
            return t_stat > 0  # all_data > control_data
        else:
            return t_stat < 0  # all_data < control_data
    
    return False

def load_and_process_trait_data(trait_name, relevant_questions_and_answers):
    """Load log data for a trait and return processed results."""
    log_file_path = f'results/icl_mcq/{trait_name.split()[1]}.npy'
    
    try:
        log_data = np.load(log_file_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Warning: No data file found for {trait_name}")
        return None
    
    # Get question indices
    positive_indices = [i for i, rqa in enumerate(relevant_questions_and_answers) 
                       if rqa['key'] == 1]
    negative_indices = [i for i, rqa in enumerate(relevant_questions_and_answers) 
                       if rqa['key'] == -1]
    
    question_indices = {
        'positive': positive_indices,
        'negative': negative_indices
    }
    
    # Process all available data types
    data_configs = [
        ('all_data', 'Relevant questions and answers'),
        ('control_all_data', 'Other questions and answers'),
        ('random_all_data', 'Relevant questions, other answers'),
        ('random_question_all_data', 'Other questions, relevant answers'),
    ]
    
    results = {}
    for data_key, label in data_configs:
        if data_key in log_data:
            context_length_idx = len(context_lengths) - 1  # Use all available context lengths
            processed_results = calculate_scores_and_stats(
                log_data[data_key], question_indices, 
                key_positive_scores, key_negative_scores, 
                context_length_idx
            )
            # Also store raw data for t-tests
            processed_results['raw_data'] = log_data[data_key]
            results[data_key] = processed_results
    
    return results

# Create the figure with custom gridspec for the gap
fig = plt.figure(figsize=(15, 25))
gs = gridspec.GridSpec(6, 5, width_ratios=[1, 1, 0.2, 1, 1], height_ratios=[0.3, 1, 1, 1, 1, 1], 
                      hspace=0.25, wspace=0.15)

# Add expectation labels at the top
fig.add_subplot(gs[0, 0:2])
plt.text(0.5, 0.5, 'Category-positive traits\nExpect ICL to Increase', ha='center', va='center', 
         fontsize=16, fontweight='bold', transform=plt.gca().transAxes)
plt.axis('off')

fig.add_subplot(gs[0, 3:5])
plt.text(0.5, 0.5, 'Category-negative traits\nExpect ICL to Decrease', ha='center', va='center', 
         fontsize=16, fontweight='bold', transform=plt.gca().transAxes)
plt.axis('off')

# Column titles (below expectation labels)
col_titles = ['Positive Questions', 'Negative Questions', '', 'Positive Questions', 'Negative Questions']

# Colors for different data types - matching original scheme
data_colors = {
    'all_data': {'positive': 'blue', 'negative': 'red'},
    'control_all_data': {'positive': 'gray', 'negative': 'gray'},
    'random_all_data': {'positive': 'green', 'negative': 'green'},
    'random_question_all_data': {'positive': 'purple', 'negative': 'purple'}
}

data_labels = {
    'all_data': 'Relevant Q&A',
    'control_all_data': 'Other Q&A', 
    'random_all_data': 'Relevant Q, Other A',
    'random_question_all_data': 'Other Q, Relevant A'
}

for row_idx, (positive_trait, negative_trait, dim_name) in enumerate(zip([pair[0] for pair in trait_pairs], 
                                                                        [pair[1] for pair in trait_pairs], 
                                                                        dimension_names)):
    
    # Adjust row index to account for header row
    plot_row = row_idx + 1
    
    # Process both traits
    traits_to_process = [(positive_trait, [0, 1]), (negative_trait, [3, 4])]
    
    for trait, col_indices in traits_to_process:
        if trait is None:
            continue
            
        print(f"Processing {trait}...")
        
        # Get trait-specific questions (replicating logic from main script)
        chosen_trait_ocean_questions_df = ocean_questions_df[
            ocean_questions_df['chosen_trait'] == trait
        ]
        ocean_key, ocean_direction = chosen_trait_to_ocean_and_direction[trait]
        chosen_trait_matching_ocean_questions_df = chosen_trait_ocean_questions_df[
            chosen_trait_ocean_questions_df['label_ocean'] == ocean_key
        ]
        
        # Process probabilities and answers
        prob_cols = ['pA', 'pB', 'pC', 'pD', 'pE']
        relevant_probs = chosen_trait_matching_ocean_questions_df[prob_cols].values
        relevant_normalized_probs = relevant_probs / relevant_probs.sum(axis=1, keepdims=True)
        relevant_answer_indices = np.argmax(relevant_normalized_probs, axis=1)
        relevant_answer_letters = np.array(['A', 'B', 'C', 'D', 'E'])[relevant_answer_indices]
        
        relevant_questions_and_answers = [
            {'index': row['index'], 'question': row['text'].lower(), 'answer': answer, 'key': row['key']}
            for row, answer in zip(
                chosen_trait_matching_ocean_questions_df.to_dict(orient="records"), 
                relevant_answer_letters
            )
        ]
        
        # Load and process data
        trait_results = load_and_process_trait_data(trait, relevant_questions_and_answers)
        
        if trait_results is None:
            continue
        
        # Plot positive questions (blue) and negative questions (red)
        for question_type, col_offset in [('positive', 0), ('negative', 1)]:
            col_idx = col_indices[col_offset]
            if col_idx == 2:  # Skip the gap column
                continue
                
            ax = fig.add_subplot(gs[plot_row, col_idx])
            
            # Determine expected direction based on trait and question type
            # For positive traits (+1): expect increase for both pos and neg questions
            # For negative traits (-1): expect decrease for both pos and neg questions
            ocean_key, ocean_direction = chosen_trait_to_ocean_and_direction[trait]
            expected_direction = ocean_direction
            
            # Plot all available data types
            for data_key, results in trait_results.items():
                if question_type in results:
                    color = data_colors[data_key][question_type]
                    plot_scores_with_uncertainty(
                        ax, context_lengths,
                        results[question_type]['mean'], 
                        results[question_type]['std'],
                        color, 
                        data_labels[data_key]
                    )
            
            # Perform t-tests and add significance stars
            if 'all_data' in trait_results:
                all_data_raw = trait_results['all_data']['raw_data']
                
                # Get question indices for this question type
                if question_type == 'positive':
                    question_indices = [i for i, rqa in enumerate(relevant_questions_and_answers) if rqa['key'] == 1]
                else:
                    question_indices = [i for i, rqa in enumerate(relevant_questions_and_answers) if rqa['key'] == -1]
                
                # For each context length, perform t-tests
                for cl_idx, context_length in enumerate(context_lengths):
                    if context_length == 0:
                        continue  # Skip context length 0
                    
                    # Get all_data scores for this context length
                    # Shape: [repeats, questions, choices] -> average over repeats, then convert to scores per question
                    all_data_scores = all_data_raw[:, cl_idx, question_indices]  # [repeats, questions, choices]
                    all_choice_idx = all_data_scores.argmax(-1)  # [repeats, questions]
                    if question_type == 'positive':
                        all_scores_per_repeat = key_positive_scores[all_choice_idx]  # [repeats, questions]
                    else:
                        all_scores_per_repeat = key_negative_scores[all_choice_idx]  # [repeats, questions]
                    
                    # Average across repeats to get one score per question
                    all_scores_per_question = all_scores_per_repeat.mean(axis=0)  # [questions]
                    
                    # Compare against each control
                    control_keys = ['control_all_data', 'random_all_data', 'random_question_all_data']
                    star_x_offset = 0
                    
                    for control_idx, control_key in enumerate(control_keys):
                        if control_key in trait_results:
                            control_data_raw = trait_results[control_key]['raw_data']
                            control_data_scores = control_data_raw[:, cl_idx, question_indices]  # [repeats, questions, choices]
                            control_choice_idx = control_data_scores.argmax(-1)  # [repeats, questions]
                            
                            if question_type == 'positive':
                                control_scores_per_repeat = key_positive_scores[control_choice_idx]  # [repeats, questions]
                            else:
                                control_scores_per_repeat = key_negative_scores[control_choice_idx]  # [repeats, questions]
                            
                            # Average across repeats to get one score per question
                            control_scores_per_question = control_scores_per_repeat.mean(axis=0)  # [questions]
                            
                            # Perform paired t-test across questions
                            is_significant = perform_paired_ttest(all_scores_per_question, control_scores_per_question, expected_direction)
                            
                            if is_significant:
                                star_color = data_colors[control_key][question_type]
                                # Place stars side by side
                                star_x = context_length + (control_idx - 1) * 0.2  # Center around context_length
                                ax.scatter(star_x, 5.5, marker='*', 
                                         color=star_color, s=100, zorder=10)
            
            # Customize the plot
            ax.grid(True, alpha=0.3)
            ax.set_ylim([1, 6.5])
            ax.set_yticks([1, 2, 3, 4, 5])  # Don't label 6
            
            # Add column titles only to first data row
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=12, fontweight='bold')
            
            # Add dimension labels on leftmost column
            if col_idx == 0:
                ax.text(-0.25, 0.5, dim_name, transform=ax.transAxes, 
                       rotation=90, ha='center', va='center', fontsize=14, fontweight='bold')
            
            # Add trait labels to each subplot
            trait_short = trait.split()[1].title()
            ax.text(0.02, 0.98, trait_short, transform=ax.transAxes, 
                   ha='left', va='top', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
            
            # Set x-axis labels only on bottom row
            if row_idx == len(trait_pairs) - 1:
                ax.set_xlabel('Context Length', fontsize=11)
            
            # Set y-axis labels only on leftmost and rightmost columns
            if col_idx in [0, 4]:
                ax.set_ylabel('Mean Score', fontsize=11)

# Add shared legend outside all subplots
legend_elements = []
for data_key in ['all_data', 'control_all_data', 'random_all_data', 'random_question_all_data']:
    # Use blue color for legend since it's the primary color scheme
    legend_elements.append(plt.Line2D([0], [0], color=data_colors[data_key]['positive'], 
                                    lw=2, label=data_labels[data_key]))

fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.02, 0.5), fontsize=12)

fig.suptitle('ICL Results: Effect of Context Length on Personality Question Responses', 
             fontsize=18, fontweight='bold', y=0.95)

plt.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.08)
plt.savefig('results/icl_mcq/main.png', dpi=300, bbox_inches='tight')

print("Comprehensive visualization saved to 'results/icl_mcq/main.png'")