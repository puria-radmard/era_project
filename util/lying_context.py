import random
from typing import Dict


def get_context_yn(N: int, valid_probe_results, probes_df_original, ):
    """Generate all base context materials for one sample."""
    
    if N == 0:
        return {}
    
    # Get top N discriminative probes
    top_n_probe_data = valid_probe_results[:N]
    top_n_questions = [probes_df_original.iloc[original_idx]['probe'] for original_idx, _ in top_n_probe_data]
    
    # Generate truth-inducing answers (opposite of lie-inducing based on effect size)
    truth_inducing_answers = []
    for original_idx, result in top_n_probe_data:
        effect_size = result['effect_size']
        if effect_size > 0:
            # Positive effect: truth=1 higher than truth=0, so "Yes" induces truth, "No" induces lies
            truth_answer = "Yes."
        else:
            # Negative effect: truth=0 higher than truth=1, so "No" induces truth, "Yes" induces lies  
            truth_answer = "No."
        truth_inducing_answers.append(truth_answer)

    # Signal truth and lie
    top_questions_shuffled = top_n_questions.copy()
    top_answers_shuffled = truth_inducing_answers.copy()
    combined_truth_qa = list(zip(top_questions_shuffled, top_answers_shuffled))
    random.shuffle(combined_truth_qa)
    top_questions_shuffled, truth_answers_shuffled = zip(*combined_truth_qa)
    top_questions_shuffled = list(top_questions_shuffled)
    truth_answers_shuffled = list(truth_answers_shuffled)

    lie_answers_shuffled = ["No." if ans == "Yes." else "Yes." for ans in truth_answers_shuffled]

    # Shuffle only the questions
    top_questions_shuffled_control = top_questions_shuffled.copy()
    random.shuffle(top_questions_shuffled_control)
    top_questions_shuffled_control = list(top_questions_shuffled_control)

    # Random questions for types 5 and 6
    # random_questions = random.sample(probe_questions, min(N, len(probe_questions)))

    # Random answers for type 5alt
    random50_answers = ["Yes." if random.random() < 0.5 else "No." for _ in truth_answers_shuffled]
    
    return {
        # Type 1: Top questions + lie answers, shuffled together
        'top_lie_shuffled_together': (top_questions_shuffled, lie_answers_shuffled),
        
        # Type 2: Top questions + truth answers, shuffled together  
        'top_truth_shuffled_together': (top_questions_shuffled, truth_answers_shuffled),
        
        # Type 3: Top questions shuffled independently + lie answers in same order as type 1
        'top_lie_questions_shuffled': (top_questions_shuffled_control, lie_answers_shuffled),
        
        # Type 4: Top questions shuffled independently + truth answers in same order as type 2
        'top_truth_questions_shuffled': (top_questions_shuffled_control, truth_answers_shuffled),
        
        # # Type 5: Random questions + lie answers in same order as type 1
        # 'random_lie_answers': (random_questions, lie_answers_shuffled),
        
        # # Type 6: Random questions + truth answers in same order as type 2
        # 'random_truth_answers': (random_questions, truth_answers_shuffled)

        # Type 5alt. Top questions + 50/50 random answers
        'top_questions_random_answers': (top_questions_shuffled, random50_answers)
    }




def get_context_five_words(N: int, valid_probe_results, probes_df_original, most_truth_answers: Dict[str, str], most_lie_answers: Dict[str, str]):
    """Generate all base context materials for one sample."""
    
    if N == 0:
        return {}
    
    # Get top N discriminative probes
    top_n_probe_data = valid_probe_results[:N]
    top_n_questions = [probes_df_original.iloc[original_idx]['probe'] for original_idx, _ in top_n_probe_data]
    
    # Generate truth-inducing answers (opposite of lie-inducing based on effect size)
    truth_inducing_answers = []
    lie_inducing_answers = []
    for original_idx, result in top_n_probe_data:

        truth_answer = most_truth_answers[original_idx]
        lie_answer = most_lie_answers[original_idx]

        truth_inducing_answers.append(truth_answer)
        lie_inducing_answers.append(lie_answer)

    # Signal truth and lie
    top_questions_shuffled = top_n_questions.copy()
    top_answers_shuffled = truth_inducing_answers.copy()
    lie_answers_shuffled = lie_inducing_answers.copy()
    combined_truth_qa = list(zip(top_questions_shuffled, top_answers_shuffled, lie_answers_shuffled))
    random.shuffle(combined_truth_qa)
    
    top_questions_shuffled, truth_answers_shuffled, lie_answers_shuffled = zip(*combined_truth_qa)
    top_questions_shuffled = list(top_questions_shuffled)
    truth_answers_shuffled = list(truth_answers_shuffled)
    lie_answers_shuffled = list(lie_answers_shuffled)

    # Generate a random 50/50 mix of truth and lie answers
    random50_answers = [
        truth_answers_shuffled[i] if random.random() < 0.5 else lie_answers_shuffled[i]
        for i in range(len(truth_answers_shuffled))
    ]

    return {
        # Type 1: Top questions + lie answers, shuffled together
        'top_lie_shuffled_together': (top_questions_shuffled, lie_answers_shuffled),
        
        # Type 2: Top questions + truth answers, shuffled together  
        'top_truth_shuffled_together': (top_questions_shuffled, truth_answers_shuffled),

        # Type 5alt. Top questions + 50/50 random answers
        'top_questions_random_answers': (top_questions_shuffled, random50_answers)
    }
