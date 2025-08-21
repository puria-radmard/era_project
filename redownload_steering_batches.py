from glob import glob
import json
from pathlib import Path
import shutil
import subprocess
from model.fireworks import FireworksBatchWrapper


wrapper = FireworksBatchWrapper('no_model')

base_path = 'probe_generation_results/b_neurips_workshop_results'
job_name = 'financial_prompted_llama-v3p1-8b-instruct'
sub_dir_name = 'c2_ordered_in_context_liar'

input_datasets_pattern = f'{base_path}/{job_name}/{sub_dir_name}/raw_outputs/steering_results_N20.jsonl.dataset_info'
output_path = Path(f'{base_path}/{job_name}/{sub_dir_name}/batch_tmp')

for filename in glob(input_datasets_pattern):
    with open(filename, 'r') as f:
        dataset_id = json.load(f)['dataset_id']

    filename_end = filename.split('/')[-1]

    cmd = ["./firectl", "download", "dataset", str(dataset_id), '--output-dir', str(output_path), '--download-lineage']
    result = subprocess.run(cmd, check=True)

    source_file = output_path / 'dataset' / dataset_id / 'BIJOutputSet.jsonl'
    shutil.move(str(source_file), output_path / f'{filename_end}.jsonl')
