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
    expected_train_rows = len(train_questions) * 2 * len(probe_indices)  # questions × truth values × probes
    
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
            'fpr': results['fpr'],
            'tpr': results['tpr'],
            'auc': results['auc']
        })
    
    return {
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs),
        'aucs': aucs,
        'roc_curves': roc_curves
    }


def plot_roc_curves(roc_data_dict: Dict[int, List], filepath: str):
    """Plot ROC curves for different D values across CV folds."""
    
    plt.figure(figsize=(12, 8))
    
    # Color map for different D values
    colors = plt.cm.viridis(np.linspace(0, 1, len(roc_data_dict)))
    
    for i, (d_value, roc_curves) in enumerate(roc_data_dict.items()):
        color = colors[i]
        
        # Plot each fold's ROC curve for this D value
        aucs_for_d = []
        for roc_curve in roc_curves:
            plt.plot(roc_curve['fpr'], roc_curve['tpr'], 
                    color=color, alpha=0.3, linewidth=1)
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
        
        plt.plot(mean_fpr, mean_tpr, color=color, linewidth=3,
                label=f'D={d_value} (AUC={mean_auc:.3f}±{std_auc:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Numbers of Probe Questions')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()


def plot_auc_vs_d(results_random: pd.DataFrame, results_stratified: pd.DataFrame, filepath: str):
    """Plot test AUC vs D for both sampling strategies."""
    
    plt.figure(figsize=(10, 6))
    
    # Plot random sampling
    plt.errorbar(results_random['D'], results_random['mean_auc'], 
                yerr=results_random['std_auc'], label='Random', marker='o')
    
    # Plot stratified sampling if provided
    if results_stratified is not None:
        plt.errorbar(results_stratified['D'], results_stratified['mean_auc'], 
                    yerr=results_stratified['std_auc'], label='Stratified', marker='s')
    
    plt.xlabel('Number of Probe Questions (D)')
    plt.ylabel('Test AUC')
    plt.title('Model Performance vs Number of Probe Questions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()


def plot_probe_type_analysis(data: pd.DataFrame, filepath: str):
    """Plot log odds distributions by probe question with statistical significance."""
    from scipy.stats import ttest_rel
    
    # Get unique probe questions and their text
    probe_info = data[['probe_question_idx', 'probe', 'probe_type']].drop_duplicates().sort_values('probe_question_idx')
    n_probes = len(probe_info)
    
    plt.figure(figsize=(max(12, n_probes * 1.5), 10))
    
    # Create box plot for each probe question
    ax = sns.boxplot(data=data, x='probe_question_idx', y='log_odds', hue='truth')
    
    # Add scatter points overlaid on boxplots
    sns.stripplot(data=data, x='probe_question_idx', y='log_odds', hue='truth', 
                  dodge=True, size=3, alpha=0.6, marker='x', ax=ax)
    
    # Perform paired t-tests and add significance stars
    significance_results = []
    for probe_idx in probe_info['probe_question_idx']:
        probe_data = data[data['probe_question_idx'] == probe_idx]
        
        # Pivot to get paired data (same question_idx for truth=0 and truth=1)
        pivot_data = probe_data.pivot_table(
            index='question_idx', 
            columns='truth', 
            values='log_odds'
        ).dropna()  # Remove rows where we don't have both truth values
        
        if len(pivot_data) > 1 and 0 in pivot_data.columns and 1 in pivot_data.columns:
            # Paired t-test
            stat, p_value = ttest_rel(pivot_data[1], pivot_data[0])
            significance_results.append({
                'probe_idx': probe_idx,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
            
            # Add star if significant
            if p_value < 0.05:
                # Get the position for the star
                probe_position = list(probe_info['probe_question_idx']).index(probe_idx)
                max_y = data[data['probe_question_idx'] == probe_idx]['log_odds'].max()
                plt.text(probe_position, max_y + 0.1, '*', 
                        ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # Add category braces
    def add_category_braces(ax, probe_info):
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
    
    add_category_braces(ax, probe_info)
    
    plt.title('Log Odds Distribution by Probe Question\n(* indicates p<0.05 for paired t-test)')
    plt.xlabel('Probe Question Index')
    plt.ylabel('Log Odds')
    plt.xticks(rotation=45)
    
    # Handle legend to avoid duplicates from stripplot
    handles, labels = ax.get_legend_handles_labels()
    # Take only the first two handles/labels (from boxplot)
    plt.legend(handles[:2], ['False', 'True'], title='Truth')
    
    # Adjust layout to make room for category labels
    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print significance results
    print("\nStatistical significance results:")
    print("Probe | P-value | Significant")
    print("-" * 30)
    for result in significance_results:
        sig_marker = "*" if result['significant'] else ""
        print(f"{result['probe_idx']:5d} | {result['p_value']:7.4f} | {sig_marker}")
    
    return significance_results


def experiment_single_d(data: pd.DataFrame, probe_df: pd.DataFrame, d_value: int,
                       n_samples: int = 10, strategy: str = 'random', 
                       target_types: list = None) -> Dict[str, float]:
    """Run experiment for single D value with multiple probe samplings."""
    
    print(f"  Sampling {n_samples} different sets of {d_value} probes...")
    
    sample_aucs = []  # Store mean AUC for each probe selection
    all_roc_curves = []  # Store ROC curves for plotting
    
    for sample_idx in range(n_samples):
        # Sample probe questions
        probe_indices = sample_probe_questions(d_value, probe_df, strategy, target_types)
        
        # Run CV experiment for this probe selection
        cv_results = run_cv_experiment(data, probe_indices)
        
        # Store the mean AUC across folds for this probe selection
        sample_aucs.append(cv_results['mean_auc'])
        all_roc_curves.extend(cv_results['roc_curves'])
    
    return {
        'mean_auc': np.mean(sample_aucs),      # Mean across probe selections
        'std_auc': np.std(sample_aucs),        # Std across probe selections  
        'sample_aucs': sample_aucs,            # Individual results for ANOVA later
        'roc_curves': all_roc_curves           # All ROC curves for plotting
    }