import torch
from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed

# Load the tokenizer and model
tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

# Define the question and multiple-choice options
question = "Is there a connection between sublingual varices and hypertension? Options: A. yes B. no C. maybe"
correct_answer = "A"  # The known correct answer for evaluation

# Tokenize the input question
inputs = tokenizer(question, return_tensors="pt")

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

# Display the original question and the model's generated response
print(f"Question: {question}")
print(f"Model's response: {decoded_text}")

# Evaluate if the generated response contains the correct answer
if correct_answer in decoded_text:
    print(f"The model correctly answered: {correct_answer}")
else:
    print(f"The model's response did not match the expected answer: {correct_answer}")
