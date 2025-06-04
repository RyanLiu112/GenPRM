import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from framework.register import register_processor
from utils.util import load_json, save_json, timestamped_print
from infer_module.infer_genprm import Reward_Service, CodeExecutor


_cached_service = None  # Global variable to cache the reward_Service instance

def get_service(model_path: str = None, url: str = None) -> Reward_Service:
    """
    Get a singleton instance of the Service.
    If the instance is already created, return it.
    """
    global _cached_service
    if _cached_service is None:
        timestamped_print("MODEL: First call to get_service. Attempting to load model...")
        try:
            _cached_service = Reward_Service(model_path=model_path, url=url)
            timestamped_print("MODEL: Model loaded successfully via get_service.")
        except Exception as e:
            timestamped_print(f"MODEL: Failed to load model via get_service: {e}", "ERROR")
            raise ValueError(f"Failed to load model: {e}")
    else:
        timestamped_print("MODEL: Model loaded successfully via get_service.")
    return _cached_service

@register_processor('process')
def process_file(args) -> None:
    """
    input_filepath: str, 
    output_filepath: str, 
    model_path: str, 
    model_url: str,
    tau: float
    num: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    system_prompt: str
    """
    data = load_json(args.input_filepath)
    reward_service = get_service(model_path=args.model_path, url=args.model_url)

    steps = data['steps']
    steps[0] = data['problem'] + '\n' + steps[0]
    messages = [
        {'role': 'system', 'content': f'You are a math teacher. Your task is to review and critique the paragraphs in solution step by step.'}
    ]
    for j1 in range(len(steps)):
        line = {'role': 'user', 'content': steps[j1]}
        messages.append(line)
        line = {'content': '', 'role': 'assistant'}
        messages.append(line)

    timestamped_print(messages)
    
    step_scores = []
    code_executor = CodeExecutor()
    cur_step = 0

    for step_index, mm in enumerate(messages):
        role = mm.get('role', '').lower()
        if role == 'user' or role == 'system':
            continue

        paths = messages[:step_index]
        cur_step += 1
        prompt = reward_service.build_prompt(paths)
        config = {
            "n": args.num,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "logprobs": args.top_k,
            "max_tokens": args.max_tokens,  # Maximum number of tokens to generate
        }
        texts, rewards = reward_service.inference(
            prompt=prompt,
            config=config,
            cur_step=cur_step, 
            analyze=False,
            verify=False,
            execute=False,
            time_limit=3,
            code_executor=code_executor,
            analyze_template="<analyze>\nLet's analyze the Paragraph {cur_step} step by step: ",
            verify_template="<verify>\nLet's use python code to find any potential error:\n```python\n",
            output_template="<output>\n**Judgement**: $\\boxed",
        )
        is_close = abs(rewards[0] - 0) < args.tau or abs(rewards[0] - 1) < args.tau
        if not is_close:
            timestamped_print(f"Warning: Reward {rewards[0]} is not close to 0 or 1, checkout to analyze mode", "WARNING")
            texts, rewards = reward_service.inference(
                prompt=prompt,
                config=config,
                cur_step=cur_step, 
                analyze=True,
                verify=False,
                execute=False,
                time_limit=3,
                code_executor=code_executor,
                analyze_template="<analyze>\nLet's analyze the Paragraph {cur_step} step by step: ",
                verify_template="<verify>\nLet's use python code to find any potential error:\n```python\n",
                output_template="<output>\n**Judgement**: $\\boxed",
            )
            is_close = abs(rewards[0] - 0) < args.tau or abs(rewards[0] - 1) < args.tau
            if not is_close:
                timestamped_print(f"Warning: Reward {rewards[0]} is not close to 0 or 1, checkout to execute mode", "WARNING")
                texts, rewards = reward_service.inference(
                    prompt=prompt,
                    config=config,
                    cur_step=cur_step, 
                    analyze=True,
                    verify=True,
                    execute=True,
                    time_limit=3,
                    code_executor=code_executor,
                    analyze_template="<analyze>\nLet's analyze the Paragraph {cur_step} step by step: ",
                    verify_template="<verify>\nLet's use python code to find any potential error:\n```python\n",
                    output_template="<output>\n**Judgement**: $\\boxed",
                )

        messages[step_index] = {
            'role': 'assistant',
            'content': texts[0]
        }
        step_scores.append(sum(rewards)/len(rewards))

    data['conversations'] = messages
    data['rewards'] = step_scores
    save_json(data, args.output_filepath)
