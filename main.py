import json
import re
import random
import signal
from src.medrag import MedRAG


# Load the benchmark JSON file
with open('pubmedqa.json', 'r') as f:
    benchmark_data = json.load(f)

# Get all questions
all_questions = list(benchmark_data.items())

# Limit to the first 200 questions
all_questions = all_questions[:1000]
# all_questions = all_questions[600:1000]
# all_questions = all_questions[400:600]
# all_questions = all_questions[600:800]
# all_questions = all_questions[800:1000]

# Get random questions
# all_questions = random.sample(list(benchmark_data.items()), 5)

# Initialize the MedRAG system
cot = MedRAG(llm_name="microsoft/biogpt", rag=False)

# Store the results of comparisons
results = []
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

# Function to extract the answer choice
def extract_answer_choice(generated_answer):

    lines = generated_answer.split('\n')
    option_map = {}
    options_section = False
    
    # Map common answer words to corresponding option letters
    word_to_option = {
        "yes": "A",
        "no": "B",
        "maybe": "C"
    }
    
    for line in lines:
        if re.match(r'^Options?:', line, re.IGNORECASE):
            options_section = True
            continue
        if options_section:
            option_match = re.match(r'^([A-D])\.\s*(.+)', line.strip(), re.IGNORECASE)
            if option_match:
                letter = option_match.group(1).upper()
                text = option_match.group(2).strip().lower()
                option_map[text] = letter
            else:
                # If the line doesn't match an option, end the options section
                options_section = False
    
    # Look for "OPTION X IS CORRECT" or "ANSWER IS X"
    match = re.search(r"OPTION\s+([A-D])\s+IS\s+CORRECT", generated_answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()  # Return the extracted option (A, B, C, or D)
    
    # As a fallback, look for "ANSWER IS X"
    match = re.search(r"ANSWER\s+IS\s+([A-D])", generated_answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Look for "Answer: X" where X is a letter
    match = re.search(r"Answer:\s*([A-D])", generated_answer, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Look for "Answer: [Option Text]"
    match = re.search(r"Answer:\s*([A-Da-d])", generated_answer, re.IGNORECASE)
    if match:
        # This handles cases like "Answer: A"
        return match.group(1).upper()
    match = re.search(r"Answer:\s*(.+)", generated_answer, re.IGNORECASE)
    if match:
        answer_text = match.group(1).strip().lower()
        return word_to_option.get(answer_text, None)
    # Check for "The output: [answer]"
    match = re.search(r"The output:\s*(yes|no|maybe)", generated_answer, re.IGNORECASE)
    if match:
        answer_text = match.group(1).strip().lower()
        return word_to_option.get(answer_text, None)
    # Check for ". [answer]"
    match = re.search(r".\s*(yes|no|maybe)", generated_answer, re.IGNORECASE)
    if match:
        answer_text = match.group(1).strip().lower()
        return word_to_option.get(answer_text, None)
    # Look for patterns like "the answer is yes/no/maybe"
    match = re.search(r"The answer is\s*(yes|no|maybe)", generated_answer, re.IGNORECASE)
    if match:
        answer_text = match.group(1).strip().lower()
        return word_to_option.get(answer_text, None)  # Map "yes", "no", "maybe" to A, B, C
    # Look for implicit answers that start with "Yes", "No", or "Maybe"
    for line in lines:
        line = line.strip().lower()
        if line.startswith(": yes,"):
            return "A"
        elif line.startswith(": no,"):
            return "B"
        elif line.startswith(": maybe,"):
            return "C"
    else:
        # Attempt to extract the answer text and map it to the corresponding option letter
        answer_text_match = re.search(r"Answer:\s*(.+)", generated_answer, re.IGNORECASE)
        if answer_text_match:
            answer_text = answer_text_match.group(1).strip().lower()
            return option_map.get(answer_text, None)
    
    return None  # If no valid option is found, return None

# Iterate over each question and get the generated answer
for question_id, question_data in all_questions:
    # Extract the question, options, and correct answer
    question = question_data['question']
    options = question_data['options']
    correct_answer = question_data['answer']

    number_all_questions += 1
    # Use MedRAG to generate the answer with a timeout
    signal.alarm(30)  # Set alarm for 60 seconds
    try:
        # Use MedRAG to generate the answer
        generated_answer = cot.medrag_answer(question=question, options=options)
        
        print(f"Generated Answer (Raw): {generated_answer}")
        
        # Extract the generated answer choice
        generated_choice = extract_answer_choice(generated_answer)

        if not generated_choice:
            print(f"No valid answer choice extracted for question ID: {question_id}")
            continue

        # Compare the generated answer with the correct one
        is_correct = correct_answer == generated_choice
        if is_correct:
            correct_count += 1
        
        answered_questions += 1
        
        accuracy = correct_count / answered_questions * 100 if answered_questions > 0 else 0
        print(f"Correct Answer: {correct_answer}")
        print(f"Is Correct: {is_correct}")
        print(f"Current Accuracy: {accuracy:.2f}%")        
        print(f"All Questions(Answered Questions): {number_all_questions}({answered_questions})")

    except TimeoutException:
        print(f"Skipping question ID: {question_id} due to timeout.")
        continue
