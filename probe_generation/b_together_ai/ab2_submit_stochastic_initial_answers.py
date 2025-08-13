import sys, os
from probe_generation.b_together_ai.ab_submit_initial_answers import main
from util.util import YamlConfig


config_path = sys.argv[1]
args = YamlConfig(config_path)

# Extract config parameters
model_name = args.model_name
prompt_idx = args.prompt_idx
questions_data_name = args.questions_data_name
question_instruction = args.question_instruction
num_initial_generation_samples = args.num_initial_generation_samples
temperature = args.temperature
top_p = args.top_p

# Setup directories
save_base = os.path.join('probe_generation_results/b_neurips_workshop_results', args.args_name, 'a2_stochastic_initial_questions')
os.makedirs(save_base, exist_ok=True)
args.save(save_base)

main(
    model_name=model_name,
    prompt_idx=prompt_idx,
    questions_data_name=questions_data_name,
    question_instruction=question_instruction,
    save_base=save_base,
    num_samples=num_initial_generation_samples,
    temperature=temperature,
    top_p=top_p,
)
