import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from framework.register import register_processor
from utils.util import load_json, save_json, timestamped_print, cprint
from infer_module.infer_genprm import Reward_Service, CodeExecutor


_cached_service = None  # Global variable to cache the Service instance

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

@register_processor("check_finish")
def check_finish(output_filepath: str) -> bool:
    try:
        record = load_json(output_filepath)
    except Exception as e:
        timestamped_print(f"Error loading JSON file {output_filepath}: {e}", "ERROR")
        return False
    field_specs = {
        'policy_responses': list,
        'steps': list,
        'correctness': list,
        'rewards': list,
        'conversations': list
    }
    lengths = {}
    for field in field_specs.keys():
        lengths[field] = len(record[field])
    
    unique_lengths = set(lengths.values())
    
    if len(unique_lengths) > 1:
        timestamped_print("Field length mismatch:", "ERROR")
        for field, length in lengths.items():
            timestamped_print(f"  - {field}: {length}", "ERROR")
        return False
    
    return True

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
    analyze: bool,
    verify: bool,
    execute: bool,
    """
    # Load existing data or initialize new structure
    if os.path.exists(args.output_filepath):
        try:
            data = load_json(args.output_filepath)
        except Exception as e:
            data = load_json(args.input_filepath)
            data['steps'] = []
            data['conversations'] = []
            data['rewards'] = []
            timestamped_print("Error loading existing file, starting fresh from input file.", "ERROR")
        
        timestamped_print(f"Resuming processing from existing file: {args.output_filepath}", "WARNING")
    else:
        data = load_json(args.input_filepath)
        data['steps'] = []
        data['conversations'] = []
        data['rewards'] = []
        timestamped_print(f"Starting new processing from: {args.input_filepath}")

    reward_service = get_service(model_path=args.model_path, url=args.model_url)
    fields = [
        len(data['policy_responses']),
        len(data['steps']),
        len(data['correctness']),
        len(data['rewards']),
        len(data['conversations'])
    ]
    min_len = min(fields)
    for idd, policy_response in enumerate(data['policy_responses']):
        if idd < min_len:
            continue
        steps = policy_response.split('\n\n')
        data['steps'].append(steps)
        steps[0] = data['problem'] + steps[0]
        if args.analyze or args.verify:
            messages = [
                {'role': 'system', 'content': f'You are a math teacher. Your task is to review and critique the paragraphs in solution step by step.'}
            ]
        else:
            messages = [
                {'role': 'system', 'content': f'You are a math teacher. Your task is to review and critique the paragraphs in solution directly. Output your judgement in the format of `\\boxed{{Yes}}` if the paragraph is correct, or `\\boxed{{No}}` if the paragraph is incorrect.'}
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
                analyze=args.analyze, 
                verify=args.verify,
                execute=args.execute,
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

        data['conversations'].append(messages)
        data['rewards'].append(step_scores)
        save_json(data, args.output_filepath)
