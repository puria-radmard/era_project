import sys, os

from util.util import YamlConfig
from probe_generation.b_inference_apis.c_in_context_liar_v import main


if __name__ == '__main__':
    config_path = sys.argv[1]
    args = YamlConfig(config_path)
    
    # Setup directories
    save_base = os.path.join('probe_generation_results/b_neurips_workshop_results', args.args_name)
    
    main(save_base, 'c2_ordered_in_context_liar')