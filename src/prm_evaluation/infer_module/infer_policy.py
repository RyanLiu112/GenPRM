from openai import OpenAI
from transformers import AutoTokenizer
from copy import *
from typing import List, Any
from utils.util import timestamped_print


class LLM_Service:
    def __init__(self, model_path: str, url: str):
        # Load the model and tokenizer
        self.client = OpenAI(
            api_key="dummy", 
            base_url=url  # "http://127.0.0.1:8000/v1"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        timestamped_print(f"VLLM model loaded successfully", level="INFO")
    
    def build_prompt(self, messages: List) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt
    
    def inference(self, prompt: str, config: dict):
        """
        "n": int,
        "temperature": float,
        "max_tokens": int,
        "top_p": float,
        "top_k": int,
        "logprobs": int,
        "stop": List,
        "include_stop_str_in_output": Bool,
        """
        if 'max_tokens' not in config:
            timestamped_print('Please set up the `max_tokens` in config.', 'ERROR')
            raise ValueError('Please set up the `max_tokens` in config.')
        
        prompt_len = len(
            self.tokenizer.tokenize(prompt)
        )
        config['max_tokens'] -= prompt_len
        # Perform inference
        request_results = self.client.completions.create(
            model="policy",
            prompt=prompt,
            extra_body=config
        )
        
        return request_results
    
    def get_text(self, request_results: Any) -> List:
        # Extract text from the request results
        text_results = [
            choice.text
            for choice in request_results.choices
        ]

        return text_results
    
    def get_finish_reason(self, request_results: Any) -> List:
        # Extract finish reason from the request results
        finish_reasons = [
            choice.finish_reason
            for choice in request_results.choices
        ]

        return finish_reasons
    
    def get_logprobs(self, request_results: Any) -> List:
        # Extract logit probabilities from the request results
        logprobs = [
            choice.logprobs
            for choice in request_results.choices
        ]

        return logprobs