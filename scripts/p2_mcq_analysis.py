import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

ocean_questions_df = pd.read_csv('results/p2_mcq_probs.csv')  # Update with your actual dataset path

chosen_trait_to_ocean_and_direction = {
    'an extraversive': ('E', +1),
    'an agreeable': ('A', +1),
    'a conscientious': ('C', +1),
    'a neurotic': ('N', +1),
    'an open': ('O', +1),
    'an introversive': ('E', -1),
    'a disagreeable': ('A', -1),
    'an unconscientious': ('C', -1),
    'a stable': ('N', -1),
    'a closed': ('O', -1),
}

p_cols = ['pA', 'pB', 'pC', 'pD', 'pE']
key_positive_scores = np.array([5, 4, 3, 2, 1])
key_negative_scores = np.array([1, 2, 3, 4, 5])



fig, axes = plt.subplots(1, 1, figsize = (10, 5), sharex = True)


for chosen_trait, (ocean_key, ocean_direction) in chosen_trait_to_ocean_and_direction.items():

    # Answers from this personality prompt
    chosen_trait_ocean_questions_df = ocean_questions_df[ocean_questions_df['chosen_trait'] == chosen_trait]

    # Answers where the personality prompt has a bearing on the question relevance
    chosen_trait_matching_ocean_questions_df = chosen_trait_ocean_questions_df[chosen_trait_ocean_questions_df['label_ocean'] == ocean_key]

    mean_scores = []
    argmax_scores = []

    for idx, row in chosen_trait_matching_ocean_questions_df.iterrows():
        # Get and normalize probabilities
        probs = np.array([row[col] for col in p_cols], dtype=float)
        probs = probs / probs.sum() if probs.sum() > 0 else np.ones(5) / 5

        # Assign scores based on the row's 'key' value
        scores = key_positive_scores if row['key'] == 1 else key_negative_scores

        # Mean score (expected value)
        mean_score = np.dot(probs, scores)
        mean_scores.append(mean_score)

        # Argmax score
        argmax_idx = np.argmax(probs)
        argmax_score = scores[argmax_idx]
        argmax_scores.append(argmax_score)

    # Compute mean and standard deviation for mean_scores and argmax_scores
    mean_mean = np.mean(mean_scores)
    std_mean = np.std(mean_scores)
    mean_argmax = np.mean(argmax_scores)
    std_argmax = np.std(argmax_scores)

    row_axes = 'OCEAN'.index(ocean_key)
    color = 'blue' if ocean_direction == 1 else 'red'

    # Assign a y position for each OCEAN key
    y_pos = 4-'OCEAN'.index(ocean_key)

    # Plot mean and std as points with error bars for mean_scores and argmax_scores
    x_mean = mean_mean
    x_max = mean_argmax

    # Offset for mean and argmax dots so they don't overlap
    offset = 0.15 if ocean_direction == 1 else -0.15

    # Plot mean (filled dot)
    axes.errorbar(
        x_mean, y_pos + offset, xerr=std_mean, fmt='o', color=color, capsize=5,
        label=f'{chosen_trait} mean' if y_pos == 0 else "", alpha=0.7
    )
    # Plot argmax (hollow dot)
    axes.errorbar(
        x_max, y_pos + offset + 0.05, xerr=std_argmax, fmt='o', color=color, capsize=5,
        markerfacecolor='none', label=f'{chosen_trait} max' if y_pos == 0 else "", alpha=0.7
    )


# Set y-ticks and labels only once after the loop
axes.set_yticks(range(5))
axes.set_yticklabels(list('OCEAN'[::-1]), fontsize = 20)
axes.set_xlabel('Score')
axes.set_title('OCEAN Scores when prompted with relevant category\nMean (filled) and Argmax (hollow)')
axes.legend(loc='lower right', fontsize='small', ncol=2)

# Custom legend handles
pos_handle = Line2D([0], [0], marker='o', color='blue', label='Positive', markersize=8, linestyle='None')
neg_handle = Line2D([0], [0], marker='o', color='red', label='Negative', markersize=8, linestyle='None')

axes.legend(handles=[pos_handle, neg_handle], loc='lower right', fontsize='small', ncol=2, title = 'Prompt type')

fig.savefig('results/p2_prompting.png')
