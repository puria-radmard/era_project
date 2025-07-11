import json
import pandas as pd
from tqdm import tqdm

from model.load import load_model
from util.elicit import get_choice_token_logits_from_token_ids

BATCH_SIZE = 32
SYSTEM_PROMPT = "Do not provide explanations to any answers."

response_data = pd.read_csv('results/lie_detector/questions_1000_all.csv')
with open('data/questions_1000_all.json', 'r') as f:
    question_data = json.load(f)

# Calculate the average of 'told_lie' where 'knows_answer' and 'told_truth' are True, grouped by 'prompt_idx'
knows_answer = response_data[(response_data['knows_answer']) & (response_data['told_truth'])]
avg_told_lie = knows_answer.groupby('prompt_idx')['told_lie'].mean()
most_lies_prompt_idx = avg_told_lie.argmax()

# Get the prompts which most reliably cause lies 
with open('data/prompts.json', 'r') as f:
    prompts = json.load(f)
    lie_prompt = prompts['lie_prompts'][most_lies_prompt_idx]
    truth_prompt = prompts['truth_prompts'][most_lies_prompt_idx]

# Get questions which reliably truthed and lied
trainable_answers = knows_answer[(knows_answer['prompt_idx'] == most_lies_prompt_idx) & (knows_answer['told_lie'])]
trainable_questions_idxs = trainable_answers['question_idx']
trainable_qa_pairs = [(question_data['question'][f'{idx}'].strip(), question_data['answer'][f'{idx}']) for idx in trainable_questions_idxs]

# Get yes/no tokens which we'd like to search for
chat_wrapper = load_model("meta-llama/Llama-3.1-8B-Instruct", device='auto')
yesno_strings = [
    ['Yes', 'yes'],
    ['No', 'no'],
]
yesno_tokens = []
for option_str_list in yesno_strings:
    option_tokens = []
    for option_str in option_str_list:
        token_ids = chat_wrapper.tokenizer.encode(option_str, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Choice token variation '{option_str}'"
                f"produces {len(token_ids)} tokens: {token_ids}. "
                f"All choice tokens must be exactly one token."
            )
        option_tokens.extend(token_ids)
    yesno_tokens.append(option_tokens)


# Load in the probe questions
probe_questions = pd.read_csv('data/probes_with_yn.csv')['probe']

# Loop over these questions
for qai, qa_pair in enumerate(trainable_qa_pairs):

    truth_chat = chat_wrapper.format_chat(
        system_prompt=SYSTEM_PROMPT,
        user_message=f'{truth_prompt} {qa_pair[0]}',
    )
    truth_cache = chat_wrapper.create_prompt_cache(truth_chat)['cache']
    truth_followup_chats = [
        chat_wrapper.format_chat(
            system_prompt=None,
            user_message=probe_question,
            prefiller='',
        ) for probe_question in probe_questions
    ]
    
    import pdb; pdb.set_trace()
    truth_generate = chat_wrapper.generate(chats = [truth_followup_chats[0]], past_key_values=truth_cache)

    truth_forward = chat_wrapper.forward(
        chats = truth_followup_chats,
        past_key_values = truth_cache,
    )


    get_choice_token_logits_from_token_ids(truth_forward['logits'], yesno_tokens)



    
    

print(trainable_answers)
