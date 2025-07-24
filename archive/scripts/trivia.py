"""
Basic Usage Example for HuggingFace Model Utilities

This script demonstrates the core functionality: loading a model, generating
choice logits responses for questions, and saving results.
"""

import json
import torch

from model.load import load_model
from util.elicit import elicit_mcq_answer, elicit_freeform_answer
from util.question import QuestionConfig
from util.experiment import ExperimentConfig


# 1. Load configuration
config_path = 'scripts/trivia.yaml'
config = ExperimentConfig.from_yaml(config_path)
print(f"Model: {config.model_name}, Batch size: {config.minibatch_size}")

# 2. Setup model and configs
chat_wrapper = load_model(config.model_name, device = 'cuda')
question_config = QuestionConfig(config).initialize_choices(chat_wrapper.tokenizer)
cache_data = chat_wrapper.create_prompt_cache(config.system_prompt)

# 3. Define your questions and choices
questions = [
    "What is the capital of France?",
    "Which planet is closest to the Sun?", 
    "What is 2 + 2?",
    "What color paint do you get when you mix red and blue paint?",
    "Who wrote the play 'Romeo and Juliet'?",
    "What is the largest mammal in the world?",
    "Which ocean is the largest?",
    "What gas do plants absorb from the atmosphere?",
    "Who painted the Mona Lisa?",
    "What is the boiling point of water at sea level in Celsius?"
]

choices = [
    ["London", "Paris", "Berlin", "Madrid"],
    ["Venus", "Mercury", "Earth", "Mars"],
    ["3", "4", "5", "6"],
    ["Green", "Purple", "Yellow", "Orange"],
    ["William Shakespeare", "Charles Dickens", "Jane Austen", "Mark Twain"],
    ["Elephant", "Blue Whale", "Giraffe", "Hippopotamus"],
    ["Atlantic", "Indian", "Pacific", "Arctic"],
    ["Oxygen", "Nitrogen", "Carbon Dioxide", "Hydrogen"],
    ["Leonardo da Vinci", "Pablo Picasso", "Vincent van Gogh", "Claude Monet"],
    ["90", "100", "80", "120"]
]

# 4. Process in minibatches and collect results
all_choice_logits = []
all_questions_data = []

for i in range(0, len(questions), config.minibatch_size):
    batch_end = min(i + config.minibatch_size, len(questions))
    batch_questions = questions[i:batch_end]
    batch_choices = choices[i:batch_end]
    
    # Generate choice logits
    mcq_results = elicit_mcq_answer(
        chat_wrapper = chat_wrapper,
        questions = batch_questions,
        choices_batch = batch_choices,
        config = question_config,
        in_context_questions = None,
        in_context_answers = None,
        shared_in_context_questions = None,
        shared_in_context_answers = None,
        cache_data=cache_data.copy()
        # system_prompt=config.system_prompt
    )
        
    # Collect results
    choice_logits = mcq_results["choice_logits"]  # [batch_size, num_choices]
    
    for j, (question, question_choices) in enumerate(zip(batch_questions, batch_choices)):
        # Store the logits and responses
        all_choice_logits.append(choice_logits[j].cpu().tolist())
        
        # Get predicted answer for display
        predicted_idx = torch.argmax(choice_logits[j]).item()
        predicted_letter = chr(ord('A') + predicted_idx)
        predicted_answer = choices[i + j][predicted_idx]
        
        all_questions_data.append({
            "question": question,
            "choices": question_choices,
            "choice_logits": choice_logits[j].cpu().tolist(),
            "predicted_choice": predicted_letter
        })
        
        print(f"  {question} \n\t\t {predicted_letter}, {predicted_answer}")

# 5. Save results to file
results = {
    "model_name": config.model_name,
    "system_prompt": config.system_prompt,
    "data": all_questions_data
}

with open(config.save_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {config.save_path}")
