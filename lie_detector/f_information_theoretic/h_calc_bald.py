import json, copy
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import os
import sys
import matplotlib.pyplot as plt
from scipy import stats

from lie_detector.f_information_theoretic.z_util import prob_mode, plot_regression_with_stats, compute_correlation_stats
from util.util import YamlConfig

# Load configuration
config_path = sys.argv[1]
args = YamlConfig(config_path)

# Extract parameters (same as original script)
system_prompt = args.system_prompt
questions_data_name = args.questions_data_name
model_name = args.model_name
initial_answers_args_name = args.initial_answers_args_name
question_instruction = args.question_instruction
probe_file_name = args.probe_file_name
prompt_index = args.prompt_idx
limit_to_lying = args.limit_to_lying
probe_analysis_args_name = args.probe_analysis_args_name

# Steering parameters
chosen_layers = args.chosen_layers
multipliers = args.multipliers
multipliers = sorted(list(set(multipliers + list(map(lambda x: -x, multipliers)))))

# Calculate unique magnitudes
unique_magnitudes = sorted(list(set([abs(m) for m in multipliers])))
print(f"Multipliers: {multipliers}")
print(f"Unique magnitudes: {unique_magnitudes}")

# Set up paths
save_base = os.path.join('lie_detector_results/f_information_theoretic', args.args_name, 'bald_estimation')
os.makedirs(save_base, exist_ok=True)
args.save(save_base)

# Load results from original script
original_results_path = os.path.join('lie_detector_results/f_information_theoretic', args.args_name, 'bald_estimation', 'steering_probe_results.npy')
all_results = np.load(original_results_path, allow_pickle=True).item()

# Load discriminability data for ground truth
discriminability_data_path = os.path.join('lie_detector_results/c_probe_discrimination', probe_analysis_args_name, 'discriminability_results.json')
with open(discriminability_data_path, 'r') as f:
    discriminability_data = json.load(f)

# reference_snrs = np.abs(np.array([res["effect_size"] for res in discriminability_data['probe_results']]))
# print(f"Loaded {len(reference_snrs)} reference SNRs")

reference_snrs = np.array([res["effect_size"] for res in discriminability_data['probe_results']])
print(f"Loaded {len(reference_snrs)} reference **SIGNED** SNRs")


# Load initial data for consistency
initial_answers_path = os.path.join('lie_detector_results/a_initial_answers', initial_answers_args_name, 'initial_answers.csv')
response_data = pd.read_csv(initial_answers_path)

if limit_to_lying:
    knows_answer = response_data[(response_data['knows_answer']) & (response_data['told_truth'])]
    trainable_answers = knows_answer[(knows_answer['prompt_idx'] == prompt_index) & (knows_answer['told_lie'])]
else:
    trainable_answers = response_data
trainable_questions_idxs = trainable_answers['question_idx'].unique()

# Subsample questions
trainable_questions_idxs = trainable_questions_idxs[:10]            
print('REDUCING trainable_questions_idxs TO JUST SPORTS QUESTIONS!!')
print(f"Using {len(trainable_questions_idxs)} initial questions")

# Load probe questions
probe_questions = pd.read_csv(f'data/probe_questions/{probe_file_name}.csv')['probe'].tolist()

print(f"Processing {len(trainable_questions_idxs)} questions and {len(probe_questions)} probes")

# Extract key arrays
probe_lengths = all_results['probe_lengths']

num_questions = len(trainable_questions_idxs)
num_probe_questions = len(probe_questions)
num_layers = len(chosen_layers)
num_magnitudes = len(unique_magnitudes)
max_probe_length = max(probe_lengths)

print(f"Data shapes:")
print(f"  Log probs no steering: {all_results['truth_log_probs_no_steering'].shape}")
print(f"  Log probs with steering: {all_results['truth_log_probs'].shape}")
print(f"  Projections: {all_results['truth_projections_no_steering'].shape}")

# Step 1: Fit Gaussians to projections for each layer (same as before)
print("Fitting Gaussians to projections...")
truth_means = all_results['truth_projections_no_steering'].mean(0)
truth_stds = all_results['truth_projections_no_steering'].std(0)
lie_means = all_results['lie_projections_no_steering'].mean(0)
lie_stds = all_results['lie_projections_no_steering'].std(0)



# Step 2: Create multiplier index mapping
def get_multiplier_indices(magnitude):
    """Get indices for negative and positive multipliers of given magnitude"""
    neg_idx = multipliers.index(-magnitude)  # Steers toward truth
    pos_idx = multipliers.index(magnitude)   # Steers toward lie
    return neg_idx, pos_idx

# Step 3: Run pipelines for each magnitude and temperature
temperatures = [1, 3, 5]

# Initialize results array: [magnitude, temperature, pipeline, probe_question, statistic]
# statistic: 0=mean, 1=std
results_array = np.full((num_magnitudes, len(temperatures), 2, num_probe_questions, 2), np.nan)

# Initialize MAE array: [magnitude, temperature, pipeline, question, probe, token]
# mae_array = np.full((num_magnitudes, len(temperatures), 2, num_questions, num_probe_questions, max_probe_length), np.nan)

for mag_idx, magnitude in enumerate(unique_magnitudes):
    print(f"\nProcessing magnitude {magnitude}...")
    
    # Get multiplier indices for this magnitude
    neg_mult_idx, pos_mult_idx = get_multiplier_indices(magnitude)
    print(f"  Negative multiplier index: {neg_mult_idx} (value: {multipliers[neg_mult_idx]})")
    print(f"  Positive multiplier index: {pos_mult_idx} (value: {multipliers[pos_mult_idx]})")
    
    for temp_idx, temp in enumerate(temperatures):
        print(f"  Processing temperature {temp}...")
        
        # Run both pipelines
        for pipeline_idx, pipeline in enumerate(['truth', 'lie']):
            print(f"    Running {pipeline} pipeline...")
            
            # Get arrays for this pipeline using dynamic keys
            steered_conditioned_entropy = all_results[f'{pipeline}_conditioned_entropy']
            unsteered_conditioned_entropy = all_results[f'{pipeline}_conditioned_entropy_no_steering']

            projections_no_steering = all_results[f'{pipeline}_projections_no_steering']
            
            # Store A values: [probe_question][initial_question] = A_avg
            A_values_by_probe = [[] for _ in range(num_probe_questions)]
            
            for q_idx in tqdm(range(num_questions), desc=f"Questions ({pipeline}, temp={temp}, mag={magnitude})"):
                
                for probe_idx in range(num_probe_questions):
                    probe_length = probe_lengths[probe_idx]
                    
                    A_t_values = []
                    
                    # Compute A_t for each token position
                    for token_pos in range(probe_length - 1):  # Predict next token
                        
                        # Get entropy values for MI computation
                        entropy_unsteered = unsteered_conditioned_entropy[q_idx, probe_idx, token_pos]
                        
                        # Get context projections for p(z | x_{<t})
                        context_projections = projections_no_steering[q_idx, probe_idx, :, token_pos]
                        
                        # Skip if any required data is NaN
                        if np.isnan(entropy_unsteered) or np.any(np.isnan(context_projections)):
                            raise Exception

                        # # # Get posterior probabilities for truth mode
                        # # # i.e. p(z = true teller | prev tokens) for each layer
                        posterior_probs = prob_mode(
                            context_projections.reshape(1, -1),    # [1, layers]
                            truth_means[probe_idx, :, token_pos], truth_stds[probe_idx, :, token_pos],
                            lie_means[probe_idx, :, token_pos], lie_stds[probe_idx, :, token_pos]
                        )[0]  # Shape: [1, layers] --> [layers]

                        # Take geometric mean across layers
                        # p_truth = np.power(np.prod(posterior_probs), 1.0 / num_layers)
                        # p_truth = 0.5

                        # Just take the flat average for now...
                        # p_truth = 0.1 if pipeline == 'lie' else 0.9

                        # Take the most extreme value
                        p_truth = posterior_probs.max() if pipeline == 'truth' else posterior_probs.min()
                        
                        # Apply temperature
                        p_truth_temp = np.power(p_truth, 1.0 / temp)
                        p_lie_temp = np.power(1 - p_truth, 1.0 / temp)
                        p_truth_tempered = p_truth_temp / (p_truth_temp + p_lie_temp)
                        p_lie_tempered = 1 - p_truth_tempered
                        
                        # Second term: expected log prob under mixture using STEERED probabilities
                        # z=truth_mode uses negatively steered log probs (toward truth)
                        # z=lie_mode uses positively steered log probs (toward lie)
                        truth_steered_entropy = steered_conditioned_entropy[q_idx, probe_idx, neg_mult_idx, token_pos]
                        lie_steered_entropy = steered_conditioned_entropy[q_idx, probe_idx, pos_mult_idx, token_pos]
                        
                        if np.isnan(truth_steered_entropy) or np.isnan(lie_steered_entropy):
                            continue
                        
                        entropy_steered = (p_truth_tempered * truth_steered_entropy + 
                                      p_lie_tempered * lie_steered_entropy)
                        
                        A_t = entropy_unsteered - entropy_steered
                        A_t_values.append(A_t)

                        # unsteered_prob = np.exp(first_term)
                        # mixture_prob = (p_truth_tempered * np.exp(truth_steered_logprob) + 
                        #             p_lie_tempered * np.exp(lie_steered_logprob))
                        # mae = abs(mixture_prob - unsteered_prob)
                        # mae_array[mag_idx, temp_idx, pipeline_idx, q_idx, probe_idx, token_pos] = mae
                    
                    # Average across tokens
                    if len(A_t_values) > 0:
                        A_avg = np.sum(A_t_values)
                        A_values_by_probe[probe_idx].append(A_avg)
            
            # Aggregate across initial questions for each probe
            for probe_idx in range(num_probe_questions):
                probe_mean = np.mean(A_values_by_probe[probe_idx])
                probe_std = np.std(A_values_by_probe[probe_idx])
                
                # Store in results array
                results_array[mag_idx, temp_idx, pipeline_idx, probe_idx, 0] = probe_mean
                results_array[mag_idx, temp_idx, pipeline_idx, probe_idx, 1] = probe_std

# Step 4: Save results
print("\nSaving results...")
save_data = {
    'results_array': results_array,  # [magnitude, temperature, pipeline, probe_question, statistic]
    # 'mae_array': mae_array,  # [magnitude, temperature, pipeline, question, probe, token]
    'unique_magnitudes': unique_magnitudes,
    'temperatures': temperatures,
    'reference_snrs': reference_snrs,
    'probe_questions': probe_questions,
    'multipliers': multipliers,
    'pipeline_labels': ['truth', 'lie'],
    'statistic_labels': ['mean', 'std'],
    'correlation_stats': {}  # Will be filled after plotting
}
np.save(os.path.join(save_base, 'bald_values.npy'), save_data)

# Step 5: Create plots with regression analysis
print("Creating plots with regression analysis...")
fig, axes = plt.subplots(num_magnitudes, len(temperatures), 
                        figsize=(6 * len(temperatures), 5 * num_magnitudes))

# Handle case where we only have one magnitude or temperature
if num_magnitudes == 1 and len(temperatures) == 1:
    axes = np.array([[axes]])
elif num_magnitudes == 1:
    axes = axes.reshape(1, -1)
elif len(temperatures) == 1:
    axes = axes.reshape(-1, 1)

# Store correlation statistics for summary
correlation_stats = {}

for mag_idx, magnitude in enumerate(unique_magnitudes):
    for temp_idx, temp in enumerate(temperatures):
        ax = axes[mag_idx, temp_idx]
        
        # Get data for this magnitude and temperature
        truth_snr_means = results_array[mag_idx, temp_idx, 0, :, 0]  # pipeline=0 (truth), statistic=0 (mean)
        truth_snr_stds = results_array[mag_idx, temp_idx, 0, :, 1]   # pipeline=0 (truth), statistic=1 (std)
        lie_snr_means = results_array[mag_idx, temp_idx, 1, :, 0]    # pipeline=1 (lie), statistic=0 (mean)
        lie_snr_stds = results_array[mag_idx, temp_idx, 1, :, 1]     # pipeline=1 (lie), statistic=1 (std)
        
        # Determine x-axis range for regression lines
        x_min, x_max = np.nanmin(reference_snrs), np.nanmax(reference_snrs)
        x_range = [x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min)]
        
        # Plot truth pipeline with regression (blue)
        valid_mask_truth = ~(np.isnan(truth_snr_means) | np.isnan(truth_snr_stds))
        if np.sum(valid_mask_truth) > 2:  # Need at least 3 points for regression
            plot_regression_with_stats(ax, reference_snrs, truth_snr_means, truth_snr_stds, 
                                     'blue', 'Truth', x_range)
            
            # Store correlation stats
            truth_stats = compute_correlation_stats(reference_snrs[valid_mask_truth], 
                                                   truth_snr_means[valid_mask_truth], 
                                                   truth_snr_stds[valid_mask_truth])
            correlation_stats[f'mag_{magnitude}_temp_{temp}_truth'] = truth_stats
        
        # Plot lie pipeline with regression (red)
        valid_mask_lie = ~(np.isnan(lie_snr_means) | np.isnan(lie_snr_stds))
        if np.sum(valid_mask_lie) > 2:  # Need at least 3 points for regression
            plot_regression_with_stats(ax, reference_snrs, lie_snr_means, lie_snr_stds, 
                                     'red', 'Lie', x_range)
            
            # Store correlation stats
            lie_stats = compute_correlation_stats(reference_snrs[valid_mask_lie], 
                                                lie_snr_means[valid_mask_lie], 
                                                lie_snr_stds[valid_mask_lie])
            correlation_stats[f'mag_{magnitude}_temp_{temp}_lie'] = lie_stats
        
        ax.set_xlabel('Reference SNR', fontsize=12)
        ax.set_ylabel('A (Information Content)', fontsize=12)
        ax.set_title(f'Magnitude = {magnitude}, Temperature = {temp}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_base, 'bald_vs_snr.png'), dpi=300, bbox_inches='tight')
plt.close()

# Step 6: Create constraint validation plots
print("Creating constraint validation plots...")
fig_constraint, axes_constraint = plt.subplots(num_magnitudes, len(temperatures), 
                                              figsize=(6 * len(temperatures), 5 * num_magnitudes))

# Handle case where we only have one magnitude or temperature
if num_magnitudes == 1 and len(temperatures) == 1:
    axes_constraint = np.array([[axes_constraint]])
elif num_magnitudes == 1:
    axes_constraint = axes_constraint.reshape(1, -1)
elif len(temperatures) == 1:
    axes_constraint = axes_constraint.reshape(-1, 1)

# # # Store MAE statistics for summary
# # mae_stats = {}

# # for mag_idx, magnitude in enumerate(unique_magnitudes):
# #     for temp_idx, temp in enumerate(temperatures):
# #         ax = axes_constraint[mag_idx, temp_idx]
        
# #         # Get MAE data for both pipelines
# #         truth_maes = mae_array[mag_idx, temp_idx, 0, :, :, :].flatten()  # Truth pipeline
# #         lie_maes = mae_array[mag_idx, temp_idx, 1, :, :, :].flatten()    # Lie pipeline
        
# #         # Remove NaN values
# #         truth_maes_valid = truth_maes[~np.isnan(truth_maes)]
# #         lie_maes_valid = lie_maes[~np.isnan(lie_maes)]
        
# #         # Plot histograms
# #         if len(truth_maes_valid) > 0:
# #             ax.hist(truth_maes_valid, bins=50, alpha=0.6, color='blue', 
# #                    label=f'Truth (n={len(truth_maes_valid)})', density=True)
# #             mae_stats[f'mag_{magnitude}_temp_{temp}_truth'] = {
# #                 'mean': np.mean(truth_maes_valid),
# #                 'median': np.median(truth_maes_valid),
# #                 'std': np.std(truth_maes_valid),
# #                 'count': len(truth_maes_valid)
# #             }
# #             # Add mean line for truth MAE
# #             mean_truth_mae = np.mean(truth_maes_valid)
# #             ax.axvline(mean_truth_mae, color='blue', linestyle=':', linewidth=2, label=f'Truth Mean = {mean_truth_mae:.4f}')
        
# #         if len(lie_maes_valid) > 0:
# #             ax.hist(lie_maes_valid, bins=50, alpha=0.6, color='red', 
# #                    label=f'Lie (n={len(lie_maes_valid)})', density=True)
# #             mae_stats[f'mag_{magnitude}_temp_{temp}_lie'] = {
# #                 'mean': np.mean(lie_maes_valid),
# #                 'median': np.median(lie_maes_valid),
# #                 'std': np.std(lie_maes_valid),
# #                 'count': len(lie_maes_valid)
# #             }
# #             # Add mean line for lie MAE
# #             mean_lie_mae = np.mean(lie_maes_valid)
# #             ax.axvline(mean_lie_mae, color='blue', linestyle=':', linewidth=2, label=f'Lie Mean = {mean_lie_mae:.4f}')

# #         ax.set_xlabel('MAE (|mixture_prob - unsteered_prob|)', fontsize=12)
# #         ax.set_ylabel('Density', fontsize=12)
# #         ax.set_title(f'Constraint Validation\nMagnitude = {magnitude}, Temperature = {temp}', fontsize=14)
# #         ax.legend(fontsize=10)
# #         ax.grid(True, alpha=0.3)
        
# #         # Add vertical line at MAE = 0 (perfect constraint satisfaction)
# #         ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)

# # plt.tight_layout()
# # plt.savefig(os.path.join(save_base, 'constraint_validation.png'), dpi=300, bbox_inches='tight')
# # plt.close()

# Print correlation summary
print("\nCorrelation Summary:")
print("=" * 50)
for key, stats in correlation_stats.items():
    print(f"{key}:")
    print(f"  Correlation: {stats['correlation']:.3f}{stats['significance']}")
    print(f"  P-value: {stats['p_value']:.6f}")
    print(f"  Slope: {stats['slope']:.4f} ± {stats['slope_err']:.4f}")
    print()

# # # Print MAE summary
# # print("\nConstraint Validation Summary (MAE Statistics):")
# # print("=" * 60)
# # for key, stats in mae_stats.items():
# #     print(f"{key}:")
# #     print(f"  Mean MAE: {stats['mean']:.6f}")
# #     print(f"  Median MAE: {stats['median']:.6f}")
# #     print(f"  Std MAE: {stats['std']:.6f}")
# #     print(f"  Token count: {stats['count']}")
# #     print()

# Find best parameter combinations (lowest mean MAE)
# best_truth = min([(k, v['mean']) for k, v in mae_stats.items() if 'truth' in k], key=lambda x: x[1])
# best_lie = min([(k, v['mean']) for k, v in mae_stats.items() if 'lie' in k], key=lambda x: x[1])

print(f"Analysis complete! Results saved to {save_base}")
print(f"- Computed A values: bald_values.npy") 
print(f"- BALD correlation plots: bald_vs_snr.png")
print(f"- Constraint validation plots: constraint_validation.png")
print(f"- Results array shape: {results_array.shape}")
print(f"  [magnitude={num_magnitudes}, temperature={len(temperatures)}, pipeline=2, probe_question={num_probe_questions}, statistic=2]")
# print(f"- MAE array shape: {mae_array.shape}")
# print(f"  [magnitude={num_magnitudes}, temperature={len(temperatures)}, pipeline=2, question={num_questions}, probe={num_probe_questions}, token={max_probe_length}]")

# Save correlation stats and MAE stats to the data file
save_data['correlation_stats'] = correlation_stats
# save_data['mae_stats'] = mae_stats
np.save(os.path.join(save_base, 'bald_values.npy'), save_data)
print(f"- All statistics saved to bald_values.npy")