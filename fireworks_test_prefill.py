from fireworks import LLM
import json

from dotenv import load_dotenv

load_dotenv()


llm = LLM(
    model="llama-v3p1-8b-instruct",
    deployment_type="serverless"
)

# Initial message context

# # messages = [
# #     {"role": "system", "content": "You are helpful, but don't think for too long"},
# #     {"role": "user", "content": "What is the population of San Francisco?"},
# #     {"role": "assistant", "content": "\nI won't have to think about this for too long"}
# # ]

prompt = "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful, but don't think for too long<|start_header_id|>user<|end_header_id|>\n\nWhat is the population of San Francisco?<|start_header_id|>assistant<|end_header_id|>\n\nI won't have to think about this for too long"

# Call the model

# # chat_completion = llm.chat.completions.create(
chat_completion = llm.completions.create(
    # # messages=messages,
    prompt = prompt,
    temperature=0.1,
    echo = True,
    logprobs = 1,
)

# Print the model's response
# print(chat_completion.choices[0].message.model_dump_json(indent=4))

import pdb; pdb.set_trace()
print(chat_completion.choices[0].text)
