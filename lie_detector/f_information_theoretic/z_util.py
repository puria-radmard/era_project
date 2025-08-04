import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats


def prob_mode(data, mode1_means, mode1_stds, mode2_means, mode2_stds):
    """
    Compute posterior probability that each datapoint belongs to mode 1.
    
    Args:
        data: shape [batch, layers] - input data points
        mode1_means: shape [layers] - means of mode 1 distributions
        mode1_stds: shape [layers] - standard deviations of mode 1 distributions
        mode2_means: shape [layers] - means of mode 2 distributions  
        mode2_stds: shape [layers] - standard deviations of mode 2 distributions

    Returns:
        Array of shape [batch, layers] with posterior probabilities P(mode=1|x)
        in range [0, 1]
    """
    
    # Convert stds to variances
    s1_sq = mode1_stds ** 2  # [layers]
    s2_sq = mode2_stds ** 2  # [layers]
    
    # Compute quadratic form coefficients for each layer
    # Q(x) = a*x^2 + b*x + c, where the full form is:
    # Q(x) = [(s2² - s1²)x² - 2(mode1_means*s2² - mode2_means*s1²)x + (m1²*s2² - m2²*s1²)] / (2*s1²*s2²)
    
    a = (s2_sq - s1_sq) / (2 * s1_sq * s2_sq)  # [layers]
    b = -2 * (mode1_means * s2_sq - mode2_means * s1_sq) / (2 * s1_sq * s2_sq)  # [layers]  
    c = (mode1_means**2 * s2_sq - mode2_means**2 * s1_sq) / (2 * s1_sq * s2_sq)  # [layers]
    
    # Evaluate quadratic form Q(x) for each datapoint
    # data is [batch, layers], coefficients are [layers]
    x = data  # [batch, layers]
    Q = a[None, :] * x**2 + b[None, :] * x + c[None, :]  # [batch, layers]
    
    # Compute the ratio term (mode1_stds/mode2_stds)
    std_ratio = mode1_stds / mode2_stds  # [layers]
    
    # Compute posterior probability: P(mode=1|x) = 1 / [1 + (mode1_stds/mode2_stds) * exp(Q(x))]
    # Use numerically stable computation
    log_odds = np.log(std_ratio[None, :]) + Q  # [batch, layers]
    
    # Apply sigmoid: 1 / (1 + exp(log_odds)) = sigmoid(-log_odds)
    posterior = 1 / (1 + np.exp(np.clip(log_odds, -500, 500)))  # [batch, layers]
    
    return posterior


def find_most_discriminated_question(probe_responses_df):
    """
    Find the question_idx that is most consistently discriminated across all probes.
    
    Steps:
    1. Get mean/std of log_odds for each (probe_question_idx, truth) combination
    2. Build sigmoid classifier for each probe (normal or reversed direction)
    3. Compute P(truth=actual_truth | probe, log_odds) for each question
    4. Find question with highest average correctness probability
    """
    
    # Step 1: Get distribution parameters for each probe
    probe_stats = probe_responses_df.groupby(['probe_question_idx', 'truth'])['log_odds'].agg(['mean', 'std']).reset_index()
    
    # Pivot to get truth=0 and truth=1 stats side by side
    stats_pivot = probe_stats.pivot(index='probe_question_idx', columns='truth', values=['mean', 'std'])
    stats_pivot.columns = [f'{stat}_truth_{truth}' for stat, truth in stats_pivot.columns]
    stats_pivot = stats_pivot.reset_index()
    
    # print("Distribution parameters per probe:")
    # print(stats_pivot.head())
    
    # Step 2: For each probe, determine sigmoid direction and compute probabilities
    results = []
    
    for _, row in stats_pivot.iterrows():
        probe_idx = row['probe_question_idx']
        
        # Get distribution parameters
        mean_0 = row['mean_truth_0'] 
        std_0 = row['std_truth_0']
        mean_1 = row['mean_truth_1']
        std_1 = row['std_truth_1']
        
        # Check sigmoid direction: normal if mean(truth=1) > mean(truth=0)
        normal_direction = mean_1 > mean_0
        # print(f"Probe {probe_idx}: mean_0={mean_0:.3f}, mean_1={mean_1:.3f}, normal_direction={normal_direction}")
        
        # Get data for this probe
        probe_data = probe_responses_df[probe_responses_df['probe_question_idx'] == probe_idx].copy()
        
        # Compute P(truth=1 | log_odds) using the distributions
        # Using your prob_mode function logic but simplified for 1D case
        x = probe_data['log_odds'].values
        
        # Handle case where stds might be NaN or 0
        if pd.isna(std_0) or std_0 == 0:
            std_0 = 1e-6
        if pd.isna(std_1) or std_1 == 0:
            std_1 = 1e-6
            
        # Compute sigmoid probabilities
        s1_sq, s2_sq = std_1**2, std_0**2
        a = (s2_sq - s1_sq) / (2 * s1_sq * s2_sq)
        b = -2 * (mean_1 * s2_sq - mean_0 * s1_sq) / (2 * s1_sq * s2_sq)  
        c = (mean_1**2 * s2_sq - mean_0**2 * s1_sq) / (2 * s1_sq * s2_sq)
        
        Q = a * x**2 + b * x + c
        std_ratio = std_1 / std_0
        log_odds_ratio = np.log(std_ratio) + Q
        
        # P(truth=1 | log_odds)
        p_truth_1 = 1 / (1 + np.exp(np.clip(log_odds_ratio, -500, 500)))
        
        # Step 3: Get P(truth=actual_truth | probe, log_odds)
        actual_truth = probe_data['truth'].values
        p_correct = np.where(actual_truth == 1, p_truth_1, 1 - p_truth_1)
        
        # Store results
        probe_results = probe_data[['question_idx', 'truth']].copy()
        probe_results['probe_question_idx'] = probe_idx
        probe_results['p_correct'] = p_correct
        results.append(probe_results)
    
    # Combine all results
    all_results = pd.concat(results, ignore_index=True)
    
    # Step 4: Find question with maximum average correctness probability
    question_scores = all_results.groupby('question_idx')['p_correct'].agg(['mean', 'std', 'count']).reset_index()
    question_scores = question_scores.sort_values('mean', ascending=False)
    
    # print("\nTop 10 most discriminated questions:")
    # print(question_scores.head(10))
    
    most_discriminated = question_scores.iloc[0]['question_idx']
    avg_correctness = question_scores.iloc[0]['mean']
    
    # print(f"\nMost discriminated question: {most_discriminated}")
    # print(f"Average correctness probability: {avg_correctness:.4f}")
    
    return most_discriminated, question_scores, all_results




def weighted_linear_regression(x, y, y_err, return_full=False):
    """
    Perform weighted linear regression y = a*x + b where weights = 1/y_err^2
    
    Args:
        x: Independent variable
        y: Dependent variable  
        y_err: Standard errors on y (used as weights)
        return_full: If True, return additional statistics
        
    Returns:
        slope, intercept, [correlation, p_value, slope_err, intercept_err] if return_full
    """
    # Remove NaN/inf values
    valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_err) & (y_err > 0)
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    y_err_clean = y_err[valid_mask]
    
    if len(x_clean) < 3:  # Need at least 3 points for meaningful regression
        if return_full:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            return np.nan, np.nan
    
    # Weights are inverse variance
    weights = 1.0 / (y_err_clean ** 2)
    
    # Weighted least squares solution
    W = np.sum(weights)
    Wx = np.sum(weights * x_clean)
    Wy = np.sum(weights * y_clean)
    Wxx = np.sum(weights * x_clean * x_clean)
    Wxy = np.sum(weights * x_clean * y_clean)
    
    # Calculate slope and intercept
    Delta = W * Wxx - Wx * Wx
    slope = (W * Wxy - Wx * Wy) / Delta
    intercept = (Wxx * Wy - Wx * Wxy) / Delta
    
    if not return_full:
        return slope, intercept
    
    # Calculate uncertainties
    slope_err = np.sqrt(W / Delta)
    intercept_err = np.sqrt(Wxx / Delta)
    
    # Calculate weighted correlation coefficient
    y_pred = slope * x_clean + intercept
    ss_res = np.sum(weights * (y_clean - y_pred) ** 2)
    y_mean_weighted = Wy / W
    ss_tot = np.sum(weights * (y_clean - y_mean_weighted) ** 2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    correlation = np.sqrt(max(0, r_squared)) * np.sign(slope)
    
    # Calculate p-value using t-test on slope
    t_stat = slope / slope_err if slope_err > 0 else 0
    dof = len(x_clean) - 2
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof)) if dof > 0 else 1.0
    
    return slope, intercept, correlation, p_value, slope_err, intercept_err

def plot_regression_with_stats(ax, x, y, y_err, color, label, x_range=None):
    """
    Plot data with error bars and regression line, add stats text to plot
    
    Args:
        ax: Matplotlib axis
        x, y, y_err: Data and error bars
        color: Color for plotting
        label: Label for legend
        x_range: Range for plotting regression line (if None, use data range)
    """
    # Plot data with error bars
    valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_err)
    if not np.any(valid_mask):
        return
        
    ax.errorbar(x[valid_mask], y[valid_mask], yerr=y_err[valid_mask], 
               fmt='o', color=color, alpha=0.7, label=label, markersize=4)
    
    # Perform regression
    slope, intercept, correlation, p_value, slope_err, intercept_err = weighted_linear_regression(
        x, y, y_err, return_full=True
    )
    
    if not np.isfinite(slope):
        return
    
    # Plot regression line
    if x_range is None:
        x_range = [np.nanmin(x[valid_mask]), np.nanmax(x[valid_mask])]
    
    x_line = np.linspace(x_range[0], x_range[1], 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, '--', color=color, alpha=0.8, linewidth=2)
    
    # Add statistics text
    significance_marker = get_significance_marker(p_value)
    stats_text = f'{label}: r={correlation:.3f}{significance_marker}'
    
    # Position text in upper left or lower right depending on slope
    text_x = 0.05 if slope >= 0 else 0.95
    text_y = 0.95 if slope >= 0 else 0.05
    ha = 'left' if slope >= 0 else 'right'
    va = 'top' if slope >= 0 else 'bottom'
    
    ax.text(text_x, text_y, stats_text, transform=ax.transAxes, 
           color=color, fontsize=10, fontweight='bold', ha=ha, va=va,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

def get_significance_marker(p_value):
    """Convert p-value to significance markers"""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    elif p_value < 0.1:
        return '†'
    else:
        return ''

def compute_correlation_stats(x, y, y_err):
    """
    Compute correlation statistics with error bars
    
    Returns:
        dict with correlation, p_value, slope, intercept, and their errors
    """
    slope, intercept, correlation, p_value, slope_err, intercept_err = weighted_linear_regression(
        x, y, y_err, return_full=True
    )
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'slope': slope,
        'slope_err': slope_err,
        'intercept': intercept,
        'intercept_err': intercept_err,
        'significance': get_significance_marker(p_value)
    }


def compute_constraint_mae(unsteered_log_probs, steered_log_probs_truth, steered_log_probs_lie, 
                          p_truth_mixture, pipeline):
    """
    Compute mean absolute error between mixture probability and unsteered probability
    to validate the constraint: ∫ p(x_t | z, x_{<t}) p(z | x_{<t}) dz = p(x_t | x_{<t})
    
    Args:
        unsteered_log_probs: Log probabilities without steering
        steered_log_probs_truth: Log probabilities steered toward truth
        steered_log_probs_lie: Log probabilities steered toward lie  
        p_truth_mixture: Probability of truth mode p(z=truth | x_{<t})
        pipeline: Which pipeline ('truth' or 'lie') - affects which unsteered probs to use
        
    Returns:
        mae_values: Array of MAE values for each valid token
    """
    # Convert log probabilities to probabilities
    unsteered_probs = np.exp(unsteered_log_probs)
    steered_probs_truth = np.exp(steered_log_probs_truth) 
    steered_probs_lie = np.exp(steered_log_probs_lie)
    
    # Compute mixture probability: p(z=truth) * p(x|z=truth) + p(z=lie) * p(x|z=lie)
    p_lie_mixture = 1.0 - p_truth_mixture
    mixture_probs = p_truth_mixture * steered_probs_truth + p_lie_mixture * steered_probs_lie
    
    # Compute absolute error
    mae_values = np.abs(mixture_probs - unsteered_probs)
    
    # Filter out NaN/inf values
    valid_mask = np.isfinite(mae_values) & np.isfinite(unsteered_probs) & np.isfinite(mixture_probs)
    
    return mae_values[valid_mask]

import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def weighted_linear_regression(x, y, y_err, return_full=False):
    """
    Perform weighted linear regression y = a*x + b where weights = 1/y_err^2
    
    Args:
        x: Independent variable
        y: Dependent variable  
        y_err: Standard errors on y (used as weights)
        return_full: If True, return additional statistics
        
    Returns:
        slope, intercept, [correlation, p_value, slope_err, intercept_err] if return_full
    """
    # Remove NaN/inf values
    valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_err) & (y_err > 0)
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    y_err_clean = y_err[valid_mask]
    
    if len(x_clean) < 3:  # Need at least 3 points for meaningful regression
        if return_full:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        else:
            return np.nan, np.nan
    
    # Weights are inverse variance
    weights = 1.0 / (y_err_clean ** 2)
    
    # Weighted least squares solution
    W = np.sum(weights)
    Wx = np.sum(weights * x_clean)
    Wy = np.sum(weights * y_clean)
    Wxx = np.sum(weights * x_clean * x_clean)
    Wxy = np.sum(weights * x_clean * y_clean)
    
    # Calculate slope and intercept
    Delta = W * Wxx - Wx * Wx
    slope = (W * Wxy - Wx * Wy) / Delta
    intercept = (Wxx * Wy - Wx * Wxy) / Delta
    
    if not return_full:
        return slope, intercept
    
    # Calculate uncertainties
    slope_err = np.sqrt(W / Delta)
    intercept_err = np.sqrt(Wxx / Delta)
    
    # Calculate weighted correlation coefficient
    y_pred = slope * x_clean + intercept
    ss_res = np.sum(weights * (y_clean - y_pred) ** 2)
    y_mean_weighted = Wy / W
    ss_tot = np.sum(weights * (y_clean - y_mean_weighted) ** 2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    correlation = np.sqrt(max(0, r_squared)) * np.sign(slope)
    
    # Calculate p-value using t-test on slope
    t_stat = slope / slope_err if slope_err > 0 else 0
    dof = len(x_clean) - 2
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof)) if dof > 0 else 1.0
    
    return slope, intercept, correlation, p_value, slope_err, intercept_err

def plot_regression_with_stats(ax, x, y, y_err, color, label, x_range=None):
    """
    Plot data with error bars and regression line, add stats text to plot
    
    Args:
        ax: Matplotlib axis
        x, y, y_err: Data and error bars
        color: Color for plotting
        label: Label for legend
        x_range: Range for plotting regression line (if None, use data range)
    """
    # Plot data with error bars
    valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_err)
    if not np.any(valid_mask):
        return
        
    ax.errorbar(x[valid_mask], y[valid_mask], yerr=y_err[valid_mask], 
               fmt='o', color=color, alpha=0.7, label=label, markersize=4)
    
    # Perform regression
    slope, intercept, correlation, p_value, slope_err, intercept_err = weighted_linear_regression(
        x, y, y_err, return_full=True
    )
    
    if not np.isfinite(slope):
        return
    
    # Plot regression line
    if x_range is None:
        x_range = [np.nanmin(x[valid_mask]), np.nanmax(x[valid_mask])]
    
    x_line = np.linspace(x_range[0], x_range[1], 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, '--', color=color, alpha=0.8, linewidth=2)
    
    # Add statistics text
    significance_marker = get_significance_marker(p_value)
    stats_text = f'{label}: r={correlation:.3f}{significance_marker}'
    
    # Position text in upper left or lower right depending on slope
    text_x = 0.05 if slope >= 0 else 0.95
    text_y = 0.95 if slope >= 0 else 0.05
    ha = 'left' if slope >= 0 else 'right'
    va = 'top' if slope >= 0 else 'bottom'
    
    ax.text(text_x, text_y, stats_text, transform=ax.transAxes, 
           color=color, fontsize=10, fontweight='bold', ha=ha, va=va,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

def get_significance_marker(p_value):
    """Convert p-value to significance markers"""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    elif p_value < 0.1:
        return '†'
    else:
        return ''

def compute_correlation_stats(x, y, y_err):
    """
    Compute correlation statistics with error bars
    
    Returns:
        dict with correlation, p_value, slope, intercept, and their errors
    """
    slope, intercept, correlation, p_value, slope_err, intercept_err = weighted_linear_regression(
        x, y, y_err, return_full=True
    )
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'slope': slope,
        'slope_err': slope_err,
        'intercept': intercept,
        'intercept_err': intercept_err,
        'significance': get_significance_marker(p_value)
    }
