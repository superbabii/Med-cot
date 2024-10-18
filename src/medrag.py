import os
import json
import torch
import transformers
from transformers import AutoTokenizer


class MedRAG:

    def __init__(self, llm_name="dmis-lab/biobert-v1.1", rag=True, cache_dir=None):
        self.llm_name = llm_name
        self.rag = rag

        self.cache_dir = cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, cache_dir=self.cache_dir)

        self.model = transformers.pipeline(
            "text-generation",
            model=self.llm_name,
            torch_dtype=torch.float16,
            device_map="auto",
            model_kwargs={"cache_dir": self.cache_dir},
        )
        
        self.max_length = 2048

    def generate(self, messages):

        stopping_criteria = None
        
        response = self.model(
            messages,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=self.max_length,
            truncation=True,
            stopping_criteria=stopping_criteria
        )
        ans = response[0]["generated_text"][len(messages):]
        return ans

    def medrag_answer(self, question, options=None, save_dir=None):
        if options is not None:
            options = '\n'.join([key+". "+options[key] for key in sorted(options.keys())])
        else:
            options = ''

        prompt = f"Question: {question}\nOptions:\n{options}\nAnswer:"
        # Generate the answer
        answer = self.generate(prompt).strip()

        # Optionally save the result
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            response_path = os.path.join(save_dir, "response.json")
            with open(response_path, 'w') as f:
                json.dump({"answer": answer}, f, indent=4)
            print(f"Response saved to {response_path}")

        return answer
