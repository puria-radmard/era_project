import sys, os
import pandas as pd
from probe_generation.b_inference_apis.c_in_context_liar_b import main
from util.util import YamlConfig

if __name__ == '__main__':

    config_path = sys.argv[1]
    args = YamlConfig(config_path)

    # Setup paths
    save_base = os.path.join('probe_generation_results/b_neurips_workshop_results', args.args_name)
    steering_dir = os.path.join(save_base, 'c3_ordered_tiled_in_context_liar')
    os.makedirs(steering_dir, exist_ok=True)
    
    initial_questions_df = pd.read_csv(f'data/initial_questions/{args.questions_data_name}.csv')
    stochastic_df = pd.read_csv(os.path.join(save_base, 'a2_stochastic_initial_questions', 'initial_answers_stochastic.csv'))

    not_replaced_question_indexes = initial_questions_df.index.tolist()

    for question_type, question_type_config_args_path in args.other_rollout_argpaths.__dict__.items():
        
        other_rollout_argpaths_arg_name = YamlConfig(question_type_config_args_path).args_name
        other_rollout_argpaths_save_base = os.path.join('probe_generation_results/b_neurips_workshop_results', other_rollout_argpaths_arg_name)
        other_rollout_stochastic_df = pd.read_csv(os.path.join(other_rollout_argpaths_save_base, 'a2_stochastic_initial_questions', 'initial_answers_stochastic.csv'))
        
        relevant_question_indices = initial_questions_df[initial_questions_df.type == question_type].index.tolist()
        for qi in relevant_question_indices:
            incoming_stochastic_rollouts = other_rollout_stochastic_df[other_rollout_stochastic_df.question_idx == qi]
            existing_stochastic_rollouts = stochastic_df[stochastic_df.question_idx == qi]

            assert incoming_stochastic_rollouts.drop(['truth_answer', 'lie_answer', 'prompt_idx', 'told_truth', 'told_lie'], axis = 1).equals(existing_stochastic_rollouts.drop(['truth_answer', 'lie_answer', 'prompt_idx', 'told_truth', 'told_lie'], axis = 1))
            stochastic_df.loc[stochastic_df.question_idx == qi,'lie_answer'] = other_rollout_stochastic_df[other_rollout_stochastic_df.question_idx == qi]['lie_answer']
            stochastic_df.loc[stochastic_df.question_idx == qi,'truth_answer'] = other_rollout_stochastic_df[other_rollout_stochastic_df.question_idx == qi]['truth_answer']

            not_replaced_question_indexes.remove(qi)

    assert len(not_replaced_question_indexes) == 0, "Not accounted for all safety related questions!!"


    main(
        model_name=args.model_name,
        steering_dir = steering_dir,
        save_base=save_base,
        stochastic_df=stochastic_df,
        initial_questions_df = initial_questions_df,
        questions_data_name=args.questions_data_name,
        question_instruction=args.question_instruction,
        probe_file_name=args.probe_file_name,
        context_lengths=args.context_lengths_icl,
        n_samples=args.n_samples_icl,
        append_strings=args.append_strings,
        # banned_words=args.banned_words
        use_largest_magnitude=True,
    )
