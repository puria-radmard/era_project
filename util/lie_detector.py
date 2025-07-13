import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional


def load_and_preprocess_data(results_path: str, probes_path: str) -> pd.DataFrame:
    """Load CSVs, merge probe info, compute log odds with clipping."""
    # Load main results
    results = pd.read_csv(results_path)
    
    # Load probe info
    probes = pd.read_csv(probes_path)
    probes = probes.reset_index().rename(columns={'index': 'probe_question_idx'})
    
    # Merge probe info
    data = results.merge(probes, on='probe_question_idx', how='left')
    
    # Compute log odds
    data['log_odds'] = compute_log_odds(data['prob_yes'], data['prob_no'])
    
    return data


def compute_log_odds(prob_yes: pd.Series, prob_no: pd.Series, clip_val: float = 1e-10) -> pd.Series:
    """Compute log(prob_yes/prob_no) with probability clipping."""
    # Clip probabilities to avoid inf/-inf
    prob_yes_clipped = np.clip(prob_yes, clip_val, 1 - clip_val)
    prob_no_clipped = np.clip(prob_no, clip_val, 1 - clip_val)
    
    return np.log(prob_yes_clipped / prob_no_clipped)


def create_cv_splits(question_ids: np.ndarray, n_folds: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate train/test question_idx splits for cross-validation with small training sets."""
    unique_questions = np.unique(question_ids)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=None)
    
    splits = []
    for train_idx, test_idx in kf.split(unique_questions):
        # Swap roles: use smaller portion for training, larger for testing
        train_questions = unique_questions[test_idx]  # Now 1/5 of data
        test_questions = unique_questions[train_idx]   # Now 4/5 of data
        splits.append((train_questions, test_questions))
    
    return splits


def sample_probe_questions(n_probes: int, probe_df: pd.DataFrame, 
                          strategy: str = 'random', target_types: Optional[List[str]] = None) -> np.ndarray:
    """Sample D probe questions randomly or stratified by probe_type."""
    
    if strategy == 'random':
        available_probes = probe_df['probe_question_idx'].values
        return np.random.choice(available_probes, size=min(n_probes, len(available_probes)), 
                               replace=False)
    
    elif strategy == 'stratified':
        if target_types is None:
            target_types = probe_df['probe_type'].unique()
        
        # Filter to target types
        filtered_probes = probe_df[probe_df['probe_type'].isin(target_types)]
        
        # Sample equally from each type
        probes_per_type = n_probes // len(target_types)
        remainder = n_probes % len(target_types)
        
        selected_probes = []
        for i, probe_type in enumerate(target_types):
            type_probes = filtered_probes[filtered_probes['probe_type'] == probe_type]['probe_question_idx'].values
            n_sample = probes_per_type + (1 if i < remainder else 0)
            n_sample = min(n_sample, len(type_probes))
            
            if n_sample > 0:
                sampled = np.random.choice(type_probes, size=n_sample, replace=False)
                selected_probes.extend(sampled)
        
        return np.array(selected_probes)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def prepare_features_labels(data: pd.DataFrame, train_questions: np.ndarray, 
                           test_questions: np.ndarray, probe_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract X, y matrices for train/test from selected probes/questions."""
        
    # Filter data for selected probes
    probe_data = data[data['probe_question_idx'].isin(probe_indices)]    
    # Prepare training data
    train_data = probe_data[probe_data['question_idx'].isin(train_questions)]    
    # Check if we have complete data
    train_pivot = train_data.pivot_table(
        index=['question_idx', 'truth'], 
        columns='probe_question_idx', 
        values='log_odds', 
        fill_value=np.nan  # Changed from 0 to NaN to detect missing data
    )
        
    # Check for any NaN values and handle them
    if train_pivot.isna().any().any():
        print("Warning: Missing values found in training data!")
        train_pivot = train_pivot.fillna(0)  # Fill with 0 after warning
    
    X_train = train_pivot.values
    y_train = train_pivot.index.get_level_values('truth').values
    
    # Prepare test data
    test_data = probe_data[probe_data['question_idx'].isin(test_questions)]    
    expected_test_rows = len(test_questions) * 2 * len(probe_indices)    
    test_pivot = test_data.pivot_table(
        index=['question_idx', 'truth'], 
        columns='probe_question_idx', 
        values='log_odds', 
        fill_value=np.nan
    )
        
    if test_pivot.isna().any().any():
        print("Warning: Missing values found in test data!")
        test_pivot = test_pivot.fillna(0)
    
    X_test = test_pivot.values
    y_test = test_pivot.index.get_level_values('truth').values
        
    return X_train, y_train, X_test, y_test


def custom_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Custom implementation of ROC curve calculation."""
    
    # Sort scores in descending order
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[desc_score_indices]
    y_scores_sorted = y_scores[desc_score_indices]
    
    # Get unique thresholds (including boundaries)
    unique_scores = np.unique(y_scores_sorted)
    thresholds = np.concatenate([unique_scores, [unique_scores[-1] - 1e-8]])
    thresholds = np.sort(thresholds)[::-1]  # Descending order
    
    # Calculate counts
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    fpr = []
    tpr = []
    
    for threshold in thresholds:
        # Predictions at this threshold
        y_pred = (y_scores >= threshold).astype(int)
        
        # Calculate confusion matrix elements
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        # Calculate rates
        tpr_val = tp / n_pos if n_pos > 0 else 0
        fpr_val = fp / n_neg if n_neg > 0 else 0
        
        tpr.append(tpr_val)
        fpr.append(fpr_val)
    
    # Ensure we start at (0,0) and end at (1,1)
    fpr = np.array([0] + fpr + [1])
    tpr = np.array([0] + tpr + [1])
    thresholds = np.array([np.inf] + list(thresholds) + [-np.inf])
    
    return fpr, tpr, thresholds


def custom_roc_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Custom implementation of ROC AUC calculation using trapezoidal rule."""
    
    fpr, tpr, _ = custom_roc_curve(y_true, y_scores)
    
    # Calculate AUC using trapezoidal rule
    # AUC = integral of TPR with respect to FPR
    auc = 0.0
    for i in range(1, len(fpr)):
        # Trapezoidal rule: (y1 + y2) * (x2 - x1) / 2
        width = fpr[i] - fpr[i-1]
        height = (tpr[i] + tpr[i-1]) / 2
        auc += width * height
    
    return auc


def train_evaluate_model(X_train: np.ndarray, y_train: np.ndarray, 
                        X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """Train logistic regression and return test AUC + other metrics."""
    
    # Train model
    model = LogisticRegression(random_state=None, max_iter=1000)
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics using custom functions
    auc = custom_roc_auc_score(y_test, y_pred_proba)
    fpr, tpr, thresholds = custom_roc_curve(y_test, y_pred_proba)
    
    return {
        'auc': auc,
        'model': model,
        'y_pred_proba': y_pred_proba,
        'y_true': y_test,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }


def run_cv_experiment(data: pd.DataFrame, probe_indices: np.ndarray, n_folds: int = 5) -> Dict[str, float]:
    """Full CV loop: returns mean test AUC across folds."""
    
    question_ids = data['question_idx'].unique()
    cv_splits = create_cv_splits(question_ids, n_folds)
    
    aucs = []
    roc_curves = []
    
    for fold_idx, (train_questions, test_questions) in enumerate(cv_splits):
        X_train, y_train, X_test, y_test = prepare_features_labels(
            data, train_questions, test_questions, probe_indices
        )
        
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue  # Skip if no variance in labels
            
        results = train_evaluate_model(X_train, y_train, X_test, y_test)
        aucs.append(results['auc'])
        
        # Store ROC curve data
        roc_curves.append({
            'fold': fold_idx,
            'fpr': results['fpr'].astype(float).tolist(),
            'tpr': results['tpr'].astype(float).tolist(),
            'auc': results['auc'].astype(float).tolist(),
            'n_train': X_train.shape[0],
            'n_test': X_test.shape[0],
        })
    
    return {
        'mean_auc': float(np.mean(aucs)),
        'std_auc': float(np.std(aucs)),
        'aucs': aucs,
        'roc_curves': roc_curves
    }


def plot_roc_curves_comprehensive(all_results: Dict[str, Dict], filepath: str):
    """Plot ROC curves for all probe types in separate subplots."""
    
    n_categories = len(all_results)
    fig, axes = plt.subplots(1, n_categories, figsize=(6 * n_categories, 6))
    
    if n_categories == 1:
        axes = [axes]
    
    for idx, (category_name, category_data) in enumerate(all_results.items()):
        ax = axes[idx]
        roc_data_dict = category_data['roc_data']
        
        # Color map for different D values
        colors = plt.cm.viridis(np.linspace(0, 1, len(roc_data_dict)))
        
        for i, (d_value, roc_curves) in enumerate(roc_data_dict.items()):
            color = colors[i]
            
            # Plot each fold's ROC curve for this D value
            aucs_for_d = []
            for roc_curve in roc_curves:
                ax.plot(roc_curve['fpr'], roc_curve['tpr'], 
                       color=color, alpha=0.2, linewidth=1)
                aucs_for_d.append(roc_curve['auc'])
            
            # Plot mean ROC curve for this D value
            mean_auc = np.mean(aucs_for_d)
            std_auc = np.std(aucs_for_d)
            
            # Create a representative curve by interpolating
            mean_fpr = np.linspace(0, 1, 100)
            interp_tprs = []
            
            for roc_curve in roc_curves:
                interp_tpr = np.interp(mean_fpr, roc_curve['fpr'], roc_curve['tpr'])
                interp_tprs.append(interp_tpr)
            
            mean_tpr = np.mean(interp_tprs, axis=0)
            
            ax.plot(mean_fpr, mean_tpr, color=color, linewidth=3,
                   label=f'D={d_value} (AUC={mean_auc:.2f}±{std_auc:.2f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves: {category_name}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()


def plot_auc_vs_d_comprehensive(all_results: Dict[str, Dict], filepath: str):
    """Plot AUC vs D for all probe types on same plot with jittered x-values."""
    
    plt.figure(figsize=(12, 8))
    
    # Color map for different categories
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for idx, (category_name, category_data) in enumerate(all_results.items()):
        results_df = category_data['results']
        color = colors[idx]
        
        # Add small jitter to x-values to separate overlapping points
        jitter = (idx - len(all_results)/2) * 0.05
        x_values = results_df['D'] + jitter
        
        plt.errorbar(x_values, results_df['mean_auc'], 
                    yerr=results_df['std_auc'], 
                    label=category_name, 
                    marker='o', 
                    alpha=0.8,
                    color=color,
                    linewidth=2,
                    markersize=6)
    
    plt.xlabel('Number of Probe Questions (D)')
    plt.ylabel('Test AUC')
    plt.title('Model Performance vs Number of Probe Questions by Category')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.45, 1.05)  # Set reasonable y-limits for AUC
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()


def compute_probe_discriminability(data: pd.DataFrame) -> Dict:
    """Compute discriminability statistics for each probe question."""
    from scipy.stats import ttest_rel
    import numpy as np
    from typing import Dict
    import pandas as pd
    
    # Get unique probe questions and their info
    probe_info = data[['probe_question_idx', 'probe', 'probe_type']].drop_duplicates().sort_values('probe_question_idx')
    
    probe_results = []
    significant_count = 0
    effect_sizes = []
    
    for _, probe_row in probe_info.iterrows():
        probe_idx = probe_row['probe_question_idx']
        probe_data = data[data['probe_question_idx'] == probe_idx]
        
        # Pivot to get paired data (same question_idx for truth=0 and truth=1)
        pivot_data = probe_data.pivot_table(
            index='question_idx', 
            columns='truth', 
            values='log_odds'
        ).dropna()  # Remove rows where we don't have both truth values
        
        if len(pivot_data) > 1 and 0 in pivot_data.columns and 1 in pivot_data.columns:
            # Paired t-test
            truth_0_values = pivot_data[0].values
            truth_1_values = pivot_data[1].values
            stat, p_value = ttest_rel(truth_1_values, truth_0_values)
            
            # Calculate statistics
            mean_truth_0 = np.mean(truth_0_values)
            mean_truth_1 = np.mean(truth_1_values)
            std_truth_0 = np.std(truth_0_values, ddof=1)
            std_truth_1 = np.std(truth_1_values, ddof=1)
            mean_difference = mean_truth_1 - mean_truth_0
            
            # Cohen's d for paired samples: mean_difference / std_of_differences
            differences = truth_1_values - truth_0_values
            cohens_d = np.mean(differences) / np.std(differences, ddof=1)
            
            probe_results.append({
                'probe_type': probe_row['probe_type'],
                'n_pairs': len(pivot_data),
                'p_value': float(p_value),
                'test_statistic': float(stat),
                'significant': float(p_value) < 0.05,
                'mean_truth_0': float(mean_truth_0),
                'mean_truth_1': float(mean_truth_1),
                'std_truth_0': float(std_truth_0),
                'std_truth_1': float(std_truth_1),
                'mean_difference': float(mean_difference),
                'effect_size': float(cohens_d),
                'abs_mean_difference': float(abs(mean_difference))
            })
            
            if p_value < 0.05:
                significant_count += 1
            effect_sizes.append(abs(cohens_d))
    
    return {
        'probe_results': probe_results,
        'overall_stats': {
            'total_probes': len(probe_results),
            'significant_probes': significant_count,
            'mean_effect_size': float(np.mean(effect_sizes) if effect_sizes else 0)
        }
    }


def add_category_braces(ax, probe_info):
    """Add category braces below the x-axis to group probe types."""
    # Group consecutive probe indices by probe_type
    groups = []
    current_type = None
    current_start = None
    
    for i, (_, row) in enumerate(probe_info.iterrows()):
        if row['probe_type'] != current_type:
            if current_type is not None:
                groups.append({
                    'type': current_type,
                    'start': current_start,
                    'end': i - 1
                })
            current_type = row['probe_type']
            current_start = i
    
    # Add the last group
    if current_type is not None:
        groups.append({
            'type': current_type,
            'start': current_start,
            'end': len(probe_info) - 1
        })
    
    # Draw braces and labels
    y_min = ax.get_ylim()[0]
    brace_height = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08
    
    for group in groups:
        start_x = group['start'] - 0.4
        end_x = group['end'] + 0.4
        center_x = (start_x + end_x) / 2
        
        # Draw horizontal line
        ax.plot([start_x, end_x], [y_min - brace_height, y_min - brace_height], 
               'k-', linewidth=1)
        
        # Draw vertical lines at ends
        ax.plot([start_x, start_x], [y_min - brace_height * 0.5, y_min - brace_height], 
               'k-', linewidth=1)
        ax.plot([end_x, end_x], [y_min - brace_height * 0.5, y_min - brace_height], 
               'k-', linewidth=1)
        
        # Add category label
        ax.text(center_x, y_min - brace_height * 2, group['type'], 
               ha='center', va='top', fontsize=10, fontweight='bold')


def add_significance_stars(ax, data, probe_info, discriminability_results):
    """Add significance stars above boxplots for significant probes."""
    for i, probe_result in enumerate(discriminability_results['probe_results']):
        if probe_result['significant']:
            probe_idx = probe_info.iloc[i]['probe_question_idx']
            max_y = data[data['probe_question_idx'] == probe_idx]['log_odds'].max()
            plt.text(i, max_y + 0.1, '*', 
                    ha='center', va='bottom', fontsize=16, fontweight='bold')


def create_probe_boxplot(data: pd.DataFrame, probe_info: pd.DataFrame):
    """Create the main boxplot with stripplot overlay."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Create box plot for each probe question
    ax = sns.boxplot(data=data, x='probe_question_idx', y='log_odds', hue='truth')
    
    # Add scatter points overlaid on boxplots
    sns.stripplot(data=data, x='probe_question_idx', y='log_odds', hue='truth', 
                  dodge=True, size=3, alpha=0.6, marker='x', ax=ax)
    
    return ax


def create_discriminability_ordered_plot(data: pd.DataFrame, discriminability_results: Dict):
    """Create second subplot with probes ordered by discriminability, centered at 0."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Get probe info
    probe_info = data[['probe_question_idx', 'probe', 'probe_type']].drop_duplicates().sort_values('probe_question_idx')
    
    # Create list of (probe_question_idx, discriminability) and sort by discriminability
    probe_discriminability = []
    for i, (_, probe_row) in enumerate(probe_info.iterrows()):
        probe_idx = probe_row['probe_question_idx']
        discriminability = discriminability_results['probe_results'][i]['abs_mean_difference']
        probe_discriminability.append((probe_idx, discriminability))
    
    # Sort by discriminability (least to most)
    probe_discriminability.sort(key=lambda x: x[1])
    
    # Create mapping from probe_question_idx to new ordered position
    probe_idx_to_new_pos = {probe_idx: new_pos for new_pos, (probe_idx, _) in enumerate(probe_discriminability)}
    
    # Calculate overall mean for each probe (across truth and lie)
    probe_means = {}
    for _, row in probe_info.iterrows():
        probe_idx = row['probe_question_idx']
        probe_data = data[data['probe_question_idx'] == probe_idx]
        probe_means[probe_idx] = probe_data['log_odds'].mean()
    
    # Create transformed dataset
    transformed_data = []
    for _, row in data.iterrows():
        probe_idx = row['probe_question_idx']
        new_pos = probe_idx_to_new_pos[probe_idx]
        
        # Delta from probe's overall mean
        delta_log_odds = row['log_odds'] - probe_means[probe_idx]
        
        transformed_data.append({
            'ordered_probe_idx': new_pos,
            'delta_log_odds': delta_log_odds,
            'truth': row['truth'],
            'question_idx': row['question_idx']
        })
    
    transformed_df = pd.DataFrame(transformed_data)
    
    # Create the plot
    ax = sns.boxplot(data=transformed_df, x='ordered_probe_idx', y='delta_log_odds', hue='truth')
    sns.stripplot(data=transformed_df, x='ordered_probe_idx', y='delta_log_odds', hue='truth', 
                  dodge=True, size=3, alpha=0.6, marker='x', ax=ax)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    return ax


def plot_probe_type_analysis(data: pd.DataFrame, filepath: str) -> Dict:
    """Plot log odds distributions by probe question with statistical significance."""
    import matplotlib.pyplot as plt
    
    # Get unique probe questions and their info
    probe_info = data[['probe_question_idx', 'probe', 'probe_type']].drop_duplicates().sort_values('probe_question_idx')
    
    # Compute discriminability statistics
    discriminability_results = compute_probe_discriminability(data)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, len(probe_info) * 1.5), 16))
    
    # First subplot: Original probe order
    plt.sca(ax1)
    ax1 = create_probe_boxplot(data, probe_info)
    add_significance_stars(ax1, data, probe_info, discriminability_results)
    add_category_braces(ax1, probe_info)
    
    ax1.set_title('Log Odds Distribution by Probe Question\n(* indicates p<0.05 for paired t-test)')
    ax1.set_xlabel('Probe Question Index')
    ax1.set_ylabel('Log Odds')
    ax1.tick_params(axis='x', rotation=45)
    
    # Handle legend to avoid duplicates from stripplot
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:2], ['False', 'True'], title='Truth')
    
    # Second subplot: Ordered by discriminability, centered at 0
    plt.sca(ax2)
    ax2 = create_discriminability_ordered_plot(data, discriminability_results)
    
    ax2.set_title('Probe Questions Ordered by Discriminability (Δ from probe mean)\nLeast discriminable (left) → Most discriminable (right)')
    ax2.set_xlabel('Probe Index (ordered by discriminability)')
    ax2.set_ylabel('Δ Log Odds (centered at probe mean)')
    
    # Handle legend for second plot
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[:2], ['False', 'True'], title='Truth')
    
    # Adjust layout
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print significance results
    print("\nStatistical significance results:")
    print("Probe | P-value | Significant | Cohen's d")
    print("-" * 45)
    for i, result in enumerate(discriminability_results['probe_results']):
        sig_marker = "*" if result['significant'] else ""
        print(f"{i:5d} | {result['p_value']:7.4f} | {sig_marker:11s} | {result['effect_size']:8.3f}")

    return discriminability_results

def get_probe_type_info(probe_df: pd.DataFrame) -> Dict[str, Dict]:
    """Get information about each probe type."""
    probe_type_info = {}
    
    for probe_type in probe_df['probe_type'].unique():
        type_probes = probe_df[probe_df['probe_type'] == probe_type]
        probe_type_info[probe_type] = {
            'count': len(type_probes),
            'indices': type_probes['probe_question_idx'].values
        }
    
    return probe_type_info
