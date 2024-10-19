import json
import re
import signal
import torch
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed

# Load the benchmark JSON file
with open('pubmedqa.json', 'r') as f:
    benchmark_data = json.load(f)

# Limit to a subset of questions for testing
all_questions = list(benchmark_data.items())[470:475]

# Load the tokenizer and model
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

correct_count = 0
number_all_questions = 0

# Define a timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timeout occurred while generating answer.")

# Set the timeout limit to 60 seconds
signal.signal(signal.SIGALRM, timeout_handler)

# Few-shot examples to guide the model
few_shot_examples = (
    "Question: Is aspirin effective in preventing cardiovascular events? "
    "Options: A. yes B. no C. maybe Answer: A.\n"
    "Question: Does smoking increase the risk of lung cancer? "
    "Options: A. yes B. no C. maybe Answer: A.\n"
)

# Iterate over each question and get the generated answer
for question_id, question_data in all_questions:
    # Extract the question, options, and correct answer
    question = question_data['question']
    options = question_data['options']
    correct_answer = question_data['answer']

    # Prepare options text
    if options:
        options_text = ' '.join([f"{key}. {options[key]}" for key in sorted(options.keys())])
    else:
        options_text = ''

    # Create the prompt with few-shot examples and more guidance
    prompt = (
        few_shot_examples +
        f"Question: {question} "
        f"Options: {options_text} "
        "Please select one of the following options (A, B, or C). The answer is:"
    )

    number_all_questions += 1
    signal.alarm(30)  # Set alarm for 30 seconds

    try:
        # Tokenize the input question
        inputs = tokenizer(prompt, return_tensors="pt")

        # Set seed for reproducibility
        set_seed(42)

        # Generate an answer using top-k sampling
        with torch.no_grad():
            generated_text = model.generate(
                **inputs,
                max_new_tokens=50,  # Generate up to 50 new tokens beyond the input length
                num_beams=5,        # Increase num_beams for better beam search
                top_k=50,           # Use top-k sampling for variety
                top_p=0.95,         # Use nucleus sampling
                do_sample=True,
                early_stopping=True
            )

        # Decode the generated text
        decoded_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)

        # Extract the chosen answer using a regular expression
        match = re.search(r'\b(A|B|C)\b', decoded_text)
        chosen_answer = match.group(1) if match else "No valid answer"

        # Print results
        print(f"Question: {question}")
        print(f"Generated Answer (Raw): {decoded_text}")
        print(f"Chosen Answer: {chosen_answer}")
        print(f"Correct Answer: {correct_answer}")
        print()

        # Update correct count if the answer matches
        if chosen_answer == correct_answer:
            correct_count += 1

    except TimeoutException:
        print(f"Skipping question ID: {question_id} due to timeout.")
        continue

# Print the overall accuracy
accuracy = (correct_count / number_all_questions) * 100
print(f"Accuracy: {accuracy:.2f}%")
