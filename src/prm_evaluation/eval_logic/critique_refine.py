import sys
import os
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from framework.register import register_processor
from utils.util import load_json, save_json, timestamped_print, cprint, check_correctness
from infer_module.infer_policy import LLM_Service
from infer_module.infer_genprm import Reward_Service, CodeExecutor


_cached_llm_service = None  # Global variable to cache the LLM_Service instance
_cached_reward_service = None  # Global variable to cache the LLM_Service instance

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

def get_reward_service(model_path: str = None, url: str = None) -> LLM_Service:
    """
    Get a singleton instance of the LLM_Service.
    If the instance is already created, return it.
    """
    global _cached_reward_service
    if _cached_reward_service is None:
        timestamped_print("MODEL: First call to get_reward_service. Attempting to load model...")
        try:
            _cached_reward_service = Reward_Service(model_path=model_path, url=url)
            timestamped_print("MODEL: Model loaded successfully via get_reward_service.")
        except Exception as e:
            timestamped_print(f"MODEL: Failed to load model via get_reward_service: {e}", "ERROR")
            raise ValueError(f"Failed to load model: {e}")
    else:
        timestamped_print("MODEL: Model loaded successfully via get_reward_service.")
    return _cached_reward_service

def find_reward_below_threshold(rewards):
    for idx, val in enumerate(rewards):
        if val < 0.5:
            return idx
    return None

def extract_after_first_colon(text):
    if type(text) is list:
        text = text[0]['text']
    analyze_content = re.search(r'<analyze>(.*?)</analyze>', text, re.DOTALL)
    if not analyze_content:
        return text
    else:
        content = analyze_content.group(1)
        colon_pos = content.find(':')
        if colon_pos == -1:
            return None
        return content[colon_pos + 1:].strip()

@register_processor("check_finish")
def check_finish(output_filepath: str) -> bool:
    try:
        record = load_json(output_filepath)
    except Exception as e:
        timestamped_print(f"Error loading JSON file {output_filepath}: {e}", "ERROR")
        return False
    
    if not record['correctness'][-1]:
        return False
    
    return True

@register_processor("process")
def process_file(args) -> None:
    """
    input_filepath: str, 
    output_filepath: str, 
    model_path: str, 
    model_url: str,
    critique_type: str,
    reward_path: str, 
    reward_url: str,
    max_iterations: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    system_prompt: str
    """
    data = load_json(args.input_filepath)
    llm_service = get_llm_service(model_path=args.model_path, url=args.model_url)
    reward_service = get_reward_service(model_path=args.reward_path, url=args.reward_url)

    # generate first response
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
        "n": 1,
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

    # critique-refine loop
    data['steps'] = []
    data['conversations'] = []
    data['rewards'] = []
    for i in args.max_iterations:
        # critique
        policy_response = data['policy_responses'][-1]
        steps = policy_response.split('\n\n')
        data['steps'].append(steps)
        steps[0] = data['problem'] + steps[0]
        if args.critique_type == 'self':
            messages = [
                {'role': 'system', 'content': f'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'}
            ]
            for j1 in range(len(steps)):
                if j1 == 0:
                    origin_prompt=  f"The following is a math problem and my solution. Your task is to review and critique the paragraphs in solution step by step. Pay attention that you should not solve the problem and give the final answer. All of your task is to critique. Output your judgement of whether the paragraph is correct in the form of `\\boxed{{Yes|No}}` at the end of each paragraph verification:\n\n[Math Problem]\n\n{sample['problem']}\n\n[Solution]\n\n<paragraph_1>\n{sample['steps'][0]}\n</paragraph_1>"
                    line = {'content': origin_prompt, 'role': 'user'}
                    messages.append(line)
                    line = {'content': '', 'role': 'assistant'}
                    messages.append(line)
                else:
                    line = {'content': f'<paragraph_{j1+1}>\n' + steps[j1] + f'\n</paragraph_{j1+1}>', 'role': 'user'}
                    messages.append(line)
                    line = {'content': '', 'role': 'assistant'}
                    messages.append(line)
        elif args.critique_type == 'genprm':
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
        else:
            raise ValueError("Not implemented critique type.")

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
                "n": 1,
                "temperature": args.critique_temperature,
                "top_p": args.critique_top_p,
                "top_k": args.critique_top_k,
                "logprobs": args.critique_top_k,
                "max_tokens": args.critique_max_tokens,  # Maximum number of tokens to generate
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

        # refine
        idx = find_reward_below_threshold(step_scores)
        if idx is None:
            timestamped_print("Positive critique, end loop")
            break
        
        sample = {
            "problem": data['problem'],
            "steps": data['steps'][-1],
            "conversation": data['conversations'][-1],
        }
        assistant_content = '\n\n'.join(sample['steps'][:idx + 1])
        if 'Qwen2.5-7B-Instruct' in args.model_path:
            critic_content = "There might be some problem in this paragraph of your reasoning, please rethink and refine your answer:\n" + \
                                '>' + sample['steps'][idx] + '\n\n' + \
                                extract_after_first_colon(sample['conversation'][2 * idx + 2]['content'])
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": sample['problem'] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."},
                {"role": "assistant", "content": '\n\n'.join(sample['steps'])},
                {"role": "user", "content": critic_content},
            ]
        else:
            raise ValueError("Not implemented model type.")
        
        # generate refinement
        config = {
            "n": 1,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,  # Maximum number of tokens to generate
        }
        prompt = llm_service.build_prompt(messages)
        cprint( prompt, 'prompt' )
        results = llm_service.inference(prompt, config)
        data['policy_responses'] += llm_service.get_text(results)
        data['finish_reason'] += llm_service.get_finish_reason(results)
        data['correctness'] += [
            check_correctness(response,  data['answer'])
            for response in data['policy_responses']
        ]
        save_json(data, args.output_filepath)
    
    save_json(data, args.output_filepath)
