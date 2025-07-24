import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from collections import Counter
from tqdm import tqdm

# Load the original data to get answer mappings
ocean_questions_df = pd.read_csv('results/p2_mcq_probs.csv')
ocean_questions_df = ocean_questions_df.reset_index()

# Create mapping from index to argmax answer for all, positive, and negative examples
prob_cols = ['pA', 'pB', 'pC', 'pD', 'pE']
all_probs = ocean_questions_df[prob_cols].values
all_normalized_probs = all_probs / all_probs.sum(axis=1, keepdims=True)
all_answer_indices = np.argmax(all_normalized_probs, axis=1)
all_answer_letters = np.array(['A', 'B', 'C', 'D', 'E'])[all_answer_indices]

# Create mappings for all, positive, and negative examples
index_to_answer_all = dict(zip(ocean_questions_df['index'], all_answer_letters))

# Positive examples (key = 1)
positive_mask = ocean_questions_df['key'] == 1
index_to_answer_positive = dict(zip(
    ocean_questions_df[positive_mask]['index'], 
    all_answer_letters[positive_mask]
))

# Negative examples (key = -1)
negative_mask = ocean_questions_df['key'] == -1
index_to_answer_negative = dict(zip(
    ocean_questions_df[negative_mask]['index'], 
    all_answer_letters[negative_mask]
))

# Get all log files
log_files = glob.glob('results/icl_mcq/*.npy')
trait_names = [os.path.basename(f).replace('.npy', '') for f in log_files]

# Sort for consistent ordering
trait_files_pairs = list(zip(trait_names, log_files))
trait_files_pairs.sort()
trait_names, log_files = zip(*trait_files_pairs)

# Parameters
context_lengths = [0, 1, 2, 5, 10]  # Adjust to match your experiment
repeats_per_context_length = 16
answer_options = ['A', 'B', 'C', 'D', 'E']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Colors for A, B, C, D, E

def get_answer_proportions(in_context_indices, index_to_answer_map):
    """
    Get the proportion of each answer option from in-context question indices.
    
    Args:
        in_context_indices: Array of question indices used as in-context examples
        index_to_answer_map: Dictionary mapping question index to answer letter
        
    Returns:
        Dictionary with proportions for each answer option
    """
    # Remove NaN values and convert to int
    valid_indices = in_context_indices[~np.isnan(in_context_indices)].astype(int)
    
    if len(valid_indices) == 0:
        return {option: 0.0 for option in answer_options}
    
    # Get answers for these indices
    answers = [index_to_answer_map[idx] for idx in valid_indices if idx in index_to_answer_map]
    
    # Count proportions
    answer_counts = Counter(answers)
    total = len(answers) if len(answers) > 0 else 1
    
    return {option: answer_counts.get(option, 0) / total for option in answer_options}

def create_x_positions(context_lengths, repeats_per_context_length):
    """Create x positions for bars with gaps between context lengths"""
    positions = []
    current_pos = 0
    
    for cl_idx, context_length in enumerate(context_lengths):
        if context_length == 0:
            # Single bar for context length 0
            positions.extend([current_pos])
            current_pos += 1
        else:
            # Multiple bars for each repeat
            for rep in range(repeats_per_context_length):
                positions.append(current_pos)
                current_pos += 1
        
        # Add gap between context lengths
        current_pos += 1
    
    return positions

# Create x positions and labels
x_positions = create_x_positions(context_lengths, repeats_per_context_length)
x_labels = []
for context_length in context_lengths:
    if context_length == 0:
        x_labels.append('0')
    else:
        for rep in range(repeats_per_context_length):
            x_labels.append(f'{context_length}')

# Create the plot with 3 columns
fig, axes = plt.subplots(len(trait_names), 3, figsize=(18, 3 * len(trait_names)), 
                        sharex=False, sharey=True)
if len(trait_names) == 1:
    axes = axes.reshape(1, -1)

fig.suptitle('Distribution of Answer Choices in In-Context Examples by Trait and Context Length', 
             fontsize=16, y=0.98)

# Column titles
column_titles = ['All Examples', 'Positive Examples (key=1)', 'Negative Examples (key=-1)']
for col_idx, title in enumerate(column_titles):
    axes[0, col_idx].set_title(title, fontsize=14, fontweight='bold')

for trait_idx, (trait_name, log_file) in tqdm(enumerate(zip(trait_names, log_files))):
    
    try:
        # Load the log data
        log_data = np.load(log_file, allow_pickle=True).item()
        
        if 'in_context_questions_indices' not in log_data:
            print(f"No in-context data found for {trait_name}")
            continue
            
        in_context_data = log_data['in_context_questions_indices']
        # Shape: [repeats_per_context_length, num_context_lengths, num_questions, max_context_length]
        
        # Process each column type
        mappings = [index_to_answer_all, index_to_answer_positive, index_to_answer_negative]
        column_names = ['All', 'Positive', 'Negative']
        
        for col_idx, (mapping, col_name) in enumerate(zip(mappings, column_names)):
            ax = axes[trait_idx, col_idx]
            
            # Collect proportions for each repeat and context length
            all_proportions = []
            bar_positions = []
            
            pos_idx = 0
            for cl_idx, context_length in enumerate(context_lengths):
                if context_length == 0:
                    # No in-context examples for length 0
                    all_proportions.append({option: 0.0 for option in answer_options})
                    bar_positions.append(x_positions[pos_idx])
                    pos_idx += 1
                else:
                    # Get proportions for each repeat
                    for rep_idx in range(repeats_per_context_length):
                        # Get in-context indices for this specific repeat and context length
                        context_indices = in_context_data[rep_idx, cl_idx, :, :context_length]
                        
                        # Flatten to get all indices used in this repeat
                        all_indices = context_indices.flatten()
                        
                        # Get proportions
                        proportions = get_answer_proportions(all_indices, mapping)
                        all_proportions.append(proportions)
                        bar_positions.append(x_positions[pos_idx])
                        pos_idx += 1
            
            # Create stacked bar chart
            bottom = np.zeros(len(all_proportions))
            
            for i, option in enumerate(answer_options):
                heights = [proportions[option] for proportions in all_proportions]
                ax.bar(bar_positions, heights, bottom=bottom, label=f'Option {option}', 
                       color=colors[i], alpha=0.8, width=0.8)
                bottom += heights
            
            # Customize the plot
            ax.set_ylabel('Proportion')
            ax.grid(True, alpha=0.3)
            
            # Add trait name on the left
            if col_idx == 0:
                ax.text(-0.1, 0.5, trait_name.title(), transform=ax.transAxes, 
                       rotation=90, ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Set x-axis labels and ticks
            ax.set_xticks(bar_positions)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')
            
            # Add vertical lines to separate context lengths
            separator_positions = []
            current_pos = 0
            for cl_idx, context_length in enumerate(context_lengths[:-1]):  # Skip last one
                if context_length == 0:
                    current_pos += 1
                else:
                    current_pos += repeats_per_context_length
                separator_positions.append(current_pos + 0.5)
                current_pos += 1
            
            for sep_pos in separator_positions:
                ax.axvline(x=sep_pos, color='gray', linestyle='--', alpha=0.5)
            
            # Add legend only to top-right subplot
            if trait_idx == 0 and col_idx == 2:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
    except Exception as e:
        print(f"Error processing {trait_name}: {e}")
        for col_idx in range(3):
            ax = axes[trait_idx, col_idx]
            ax.set_title(f'{trait_name.title()} - {column_names[col_idx]} (Error)')
            ax.text(0.5, 0.5, f'Error: {str(e)}', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=10)

# Set common labels
for col_idx in range(3):
    axes[-1, col_idx].set_xlabel('Context Length')

plt.tight_layout()
plt.subplots_adjust(top=0.95)

# Save the plot
plt.savefig('results/icl_mcq/in_context_answer_distributions_detailed.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\nSummary Statistics:")
print("==================")
for trait_name, log_file in zip(trait_names, log_files):
    try:
        log_data = np.load(log_file, allow_pickle=True).item()
        if 'in_context_questions_indices' in log_data:
            in_context_data = log_data['in_context_questions_indices']
            print(f"\n{trait_name.title()}:")
            print(f"  Data shape: {in_context_data.shape}")
            print(f"  Non-NaN entries: {np.sum(~np.isnan(in_context_data))}")
            
            # Show distribution for longest context length, first repeat
            if len(context_lengths) > 1:
                longest_cl_idx = len(context_lengths) - 1
                longest_context_length = context_lengths[longest_cl_idx]
                if longest_context_length > 0:
                    context_indices = in_context_data[0, longest_cl_idx, :, :longest_context_length]
                    all_indices = context_indices.flatten()
                    
                    for map_name, mapping in [('All', index_to_answer_all), 
                                            ('Positive', index_to_answer_positive), 
                                            ('Negative', index_to_answer_negative)]:
                        proportions = get_answer_proportions(all_indices, mapping)
                        print(f"  {map_name} examples (context length {longest_context_length}, repeat 1):")
                        for option, prop in proportions.items():
                            print(f"    {option}: {prop:.3f}")
    except Exception as e:
        print(f"  Error: {e}")