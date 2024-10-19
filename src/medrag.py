import os
import json
import torch
from transformers import pipeline, set_seed
from transformers import BioGptTokenizer, BioGptForCausalLM

class MedRAG:

    def __init__(self, llm_name="microsoft/biogpt", rag=True, cache_dir=None):
        self.llm_name = llm_name
        self.rag = rag
        self.cache_dir = cache_dir

        # Load tokenizer and model
        self.tokenizer = BioGptTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)
        self.model = BioGptForCausalLM.from_pretrained(self.llm_name, cache_dir=self.cache_dir)

        # Detect available device (GPU or CPU)
        device = 0 if torch.cuda.is_available() else -1

        # Initialize text-generation pipeline with proper device
        self.pipeline = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=device  # Use GPU if available
        )

        self.max_length = 2048

    def generate(self, prompt):
        # Set seed for reproducibility before generation
        set_seed(42)

        # Generate a response using the pipeline
        response = self.pipeline(
            prompt,
            do_sample=True,  # Enable randomness in generation
            max_length=min(len(self.tokenizer.encode(prompt)) + 50, self.max_length),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        # Extract the generated text from the response
        generated_text = response[0]["generated_text"]
        ans = generated_text[len(prompt):].strip()  # Strip the prompt to get the answer only
        return ans

    def medrag_answer(self, question, options=None, save_dir=None):
        # Prepare the options if given
        if options:
            options_text = '\n'.join([f"{key}. {options[key]}" for key in sorted(options.keys())])
        else:
            options_text = ''

        # Create the prompt for the question and options
        prompt = f"Question: {question}\nOptions:\n{options_text}\nAnswer:"

        # Generate the answer using the pipeline
        answer = self.generate(prompt).strip()

        # Optionally save the response to a JSON file
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            response_path = os.path.join(save_dir, "response.json")
            with open(response_path, 'w') as f:
                json.dump({"answer": answer}, f, indent=4)
            print(f"Response saved to {response_path}")

        return answer
