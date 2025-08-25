import sys, os
import pandas as pd
from probe_generation.b_inference_apis.c_in_context_liar_b import main
from util.util import YamlConfig

if __name__ == '__main__':

    config_path = sys.argv[1]
    args = YamlConfig(config_path)

    # Setup paths
    save_base = os.path.join('probe_generation_results/b_neurips_workshop_results', args.args_name)
    steering_dir = os.path.join(save_base, 'c2_ordered_in_context_liar')
    os.makedirs(steering_dir, exist_ok=True)
    
    initial_questions_df = pd.read_csv(f'data/initial_questions/{args.questions_data_name}.csv')
    stochastic_df = pd.read_csv(os.path.join(save_base, 'a2_stochastic_initial_questions', 'initial_answers_stochastic.csv'))

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
