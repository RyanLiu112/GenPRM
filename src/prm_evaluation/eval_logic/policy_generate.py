import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from framework.register import register_processor
from utils.util import load_json, save_json, timestamped_print, cprint, check_correctness
from infer_module.infer_policy import LLM_Service


_cached_llm_service = None  # Global variable to cache the LLM_Service instance

def get_llm_service(model_path: str = None, url: str = None) -> LLM_Service:
    """
    Get a singleton instance of the LLM_Service.
    If the instance is already created, return it.
    """
    global _cached_llm_service
    if _cached_llm_service is None:
        timestamped_print("MODEL: First call to get_llm_service. Attempting to load model...")
        try:
            _cached_llm_service = LLM_Service(model_path=model_path, url=url)
            timestamped_print("MODEL: Model loaded successfully via get_llm_service.")
        except Exception as e:
            timestamped_print(f"MODEL: Failed to load model via get_llm_service: {e}", "ERROR")
            raise ValueError(f"Failed to load model: {e}")
    else:
        timestamped_print("MODEL: Model loaded successfully via get_llm_service.")
    return _cached_llm_service

@register_processor("process")
def process_file(args) -> None:
    """
    input_filepath: str, 
    output_filepath: str, 
    model_path: str, 
    model_url: str,
    num: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    system_prompt: str
    """
    data = load_json(args.input_filepath)
    llm_service = get_llm_service(model_path=args.model_path, url=args.model_url)
    if 'Qwen2.5-7B-Instruct' in args.model_path:
        messages = [ 
            { "role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." }, 
            { "role": "user", "content": f"{data['problem']}\nPlease reason step by step, and put your final answer within \\boxed{{}}." }, 
        ]
    elif 'Qwen2.5-7B-Math-Instruct' in args.model_path:
        messages = [ 
            { "role": "system", "content": f"Please reason step by step, and put your final answer within \\boxed{{}}." }, 
            { "role": "user", "content": f"{data['problem']}" }, 
        ]
    
    config = {
        "n": args.num,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,  # Maximum number of tokens to generate
    }
    prompt = llm_service.build_prompt(messages)
    cprint( prompt, 'prompt' )
    results = llm_service.inference(prompt, config)
    data['prompt'] = prompt
    data['policy_responses'] = llm_service.get_text(results)
    data['finish_reason'] = llm_service.get_finish_reason(results)
    data['correctness'] = [
        check_correctness(response,  data['answer'])
        for response in data['policy_responses']
    ]
    save_json(data, args.output_filepath)
