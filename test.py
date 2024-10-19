import json
import re
import random
import signal
import torch
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed

# Load the benchmark JSON file
with open('pubmedqa.json', 'r') as f:
    benchmark_data = json.load(f)
    
# Get all questions
all_questions = list(benchmark_data.items())

# Limit to the first 200 questions
all_questions = all_questions[470:475]
# all_questions = all_questions[600:1000]
# all_questions = all_questions[400:600]
# all_questions = all_questions[600:800]
# all_questions = all_questions[800:1000]

# Get random questions
# all_questions = random.sample(list(benchmark_data.items()), 5)

# Load the tokenizer and model
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

correct_count = 0
answered_questions = 0
number_all_questions = 0

# Define a timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout occurred while generating answer.")

# Set the timeout limit to 60 seconds
signal.signal(signal.SIGALRM, timeout_handler)

# Iterate over each question and get the generated answer
for question_id, question_data in all_questions:
    # Extract the question, options, and correct answer
    question = question_data['question']
    options = question_data['options']
    correct_answer = question_data['answer']
    
    if options:
        options_text = '\n'.join([f"{key}. {options[key]}" for key in sorted(options.keys())])
    else:
        options_text = ''

    # Create the prompt for the question and options
    # prompt = f"Question: {question}\nOptions:\n{options_text}\nAnswer:"
    prompt = "Is there a connection between sublingual varices and hypertension? Options: A. yes B. no C. maybe"

    number_all_questions += 1
    # Use MedRAG to generate the answer with a timeout
    signal.alarm(30)  # Set alarm for 60 seconds
    try:
        # Use MedRAG to generate the answer
        # Tokenize the input question
        inputs = tokenizer(prompt, return_tensors="pt")

        # Set seed for reproducibility
        set_seed(42)

        # Generate an answer using beam search
        with torch.no_grad():
            generated_text = model.generate(
                **inputs,
                max_length=50,  # Limit the output length
                num_beams=5,    # Beam search for more accurate results
                early_stopping=True
            )

        # Decode the generated text to readable format
        decoded_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
        
        print(f"Generated Answer (Raw): {decoded_text}")
        
        # # Extract the generated answer choice
        # generated_choice = extract_answer_choice(generated_answer)

        # if not generated_choice:
        #     print(f"No valid answer choice extracted for question ID: {question_id}")
        #     continue

        # # Compare the generated answer with the correct one
        # is_correct = correct_answer == generated_choice
        # if is_correct:
        #     correct_count += 1
        
        # answered_questions += 1
        
        # accuracy = correct_count / answered_questions * 100 if answered_questions > 0 else 0
        # print(f"Correct Answer: {correct_answer}")
        # print(f"Is Correct: {is_correct}")
        # print(f"Current Accuracy: {accuracy:.2f}%")        
        # print(f"All Questions(Answered Questions): {number_all_questions}({answered_questions})")

    except TimeoutException:
        print(f"Skipping question ID: {question_id} due to timeout.")
        continue