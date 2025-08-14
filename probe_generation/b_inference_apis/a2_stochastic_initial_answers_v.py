import sys, os
from probe_generation.b_inference_apis.a_initial_answers_v import main
from util.util import YamlConfig


if __name__ == '__main__':
    
    config_path = sys.argv[1]
    args = YamlConfig(config_path)
    
    # Setup directories
    save_base = os.path.join('probe_generation_results/b_neurips_workshop_results', args.args_name, 'a2_stochastic_initial_questions')
    args.save(save_base)
    
    main(save_base=save_base, suffix='_stochastic')
