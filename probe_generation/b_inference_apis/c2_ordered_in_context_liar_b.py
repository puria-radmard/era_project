import sys
from probe_generation.b_inference_apis.c_in_context_liar_b import main
from util.util import YamlConfig

if __name__ == '__main__':
    config_path = sys.argv[1]
    args = YamlConfig(config_path)
    
    main(
        args_name=args.args_name,
        model_name=args.model_name,
        questions_data_name=args.questions_data_name,
        question_instruction=args.question_instruction,
        probe_file_name=args.probe_file_name,
        context_lengths=args.context_lengths_icl,
        n_samples=args.n_samples_icl,
        append_strings=args.append_strings,
        # banned_words=args.banned_words
        use_largest_magnitude=True,
        subdir_name='c2_ordered_in_context_liar'
    )
