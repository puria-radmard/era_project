import os
from together import Together
from dotenv import load_dotenv

load_dotenv()


client = Together(api_key = os.environ.get('TOGETHER_API_KEY')) # pass in API key to api_key or set a env variable


response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct-Turbo",
    # model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    messages=[
      {"role": "system", "content": "You are a helpful assistant that can access external functions. The responses from these function calls will be appended to this dialogue. Please provide responses based on the information from these function calls."},
      {"role": "user", "content": "First, tell me what you think about skydiving. Then, tell me the current temperature of New York, San Francisco and Chicago by using the tools available to you."},
    ],
    tools=[
      {
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "description": "Get the current weather in a given location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
              },
              "unit": {
                "type": "string",
                "enum": [
                  "celsius",
                  "fahrenheit"
                ]
              }
            }
          }
        }
      }
    ],
    stream = False,
    echo = True
)

print(json.dumps(response.choices[0].message.model_dump(), indent=2))
