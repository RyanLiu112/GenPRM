#####################################################           import packeges and args             ########################################################

import sys
import numpy as np
import torch
import argparse
import json
import os
import re
import math
import gc
import ray
import random
import time
import threading
import traceback
import psutil
from typing import List, Tuple
from utils.util import *
from copy import *
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from datasets import Dataset
from accelerate import Accelerator
from vllm import LLM, SamplingParams
from datasets import Dataset, DatasetDict, load_from_disk, load_dataset
from datetime import datetime
from colorama import Fore, init

import io
from contextlib import redirect_stdout
import signal

os.environ['VLLM_USE_V1'] = '0'

# init colorama log
init()

TIME_LIMIT = 300   # set time limit
MAX_TOKENS = 2048  # set max single step tokens
MAX_LENGTH = 6100  # set conversation max tokens (8192-2048)
version = 'v1.0'

stop_event = threading.Event()

def cprint(s, start):
    if not isinstance(s, str):
        s = str(s)
    
    print(f"{'*' * 40}")
    print(f"Start: {start}")
    print(f"{'-' * 40}")
    
    # 打印内容，将 \n 替换为 \\n
    print(s.replace('\n', '\\n'))
    
    # 打印结束标记
    print(f"{'-' * 40}")
    print(f"End: {start}")
    print(f"{'*' * 40}\n")

class timeout:
    """执行超时控制类"""
    def __init__(self, seconds=1):
        self.seconds = seconds
    
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)
    
    def handle_timeout(self, signum, frame):
        raise TimeoutError("Code execution timed out")

class CodeExecutor:
    """代码执行环境管理器"""
    def __init__(self):
        self.namespace = {}  # 独立的命名空间
        self.code_pattern = re.compile(r'```python\s*(.*?)\s*```', re.DOTALL)

    def execute(self, text):
        # 提取所有代码块和输出块
        try:
            code_block = self.code_pattern.findall(text)[-1].strip()
        except Exception as e:
            actual = f"Code format error: No code found."
            return actual
        
        # 执行代码
        try:
            f = io.StringIO()
            with redirect_stdout(f):
                with timeout(seconds=5):
                    exec(code_block, self.namespace)
            actual = f.getvalue().strip()
        except TimeoutError as te:
            actual = f"Code execute time out: {te}"
            print(actual)
        except Exception as e:
            actual = f"Code execute Error: {type(e).__name__}: {e}"
            print(actual)
        
        return actual

def select_closest_to_mean(beam_outputs, beam_scores):
    # 计算平均值
    mean_score = sum(beam_scores) / len(beam_scores)
    
    # 同时记录索引和差值
    closest_index = 0
    min_diff = float('inf')
    
    # 单次遍历同时寻找最小值
    for i, score in enumerate(beam_scores):
        current_diff = abs(score - mean_score)
        if current_diff < min_diff:
            min_diff = current_diff
            closest_index = i
    
    return beam_outputs[closest_index]



def timestamped_print(message, level="INFO"):
    """带时间戳和颜色标记的打印函数{"ERROR", "WARNING", "INFO"}"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    color = {
        "ERROR": Fore.RED,
        "WARNING": Fore.YELLOW,
        "INFO": Fore.GREEN
    }.get(level, Fore.WHITE)
    print(f"{Fore.CYAN}[{now}]{Fore.RESET} {color}[{level}]{Fore.RESET} {message}")

def heart_beat_worker(file_path):
    """
    针对指定文件的心跳函数，定期更新文件的时间戳，并在运行超过5分钟后重启程序
    """
    start_time = time.time()
    
    while not stop_event.is_set():
        # 检查是否运行超过5分钟（300秒）
        if time.time() - start_time > TIME_LIMIT:
            timestamped_print("Runtime exceeds 5 minutes", 'INFO')
            # restart_program()  # 执行重启

        # 原有心跳逻辑
        if os.path.exists(file_path):
            try:
                os.utime(file_path)
                timestamped_print(f"Heartbeat updated: {file_path}")
            except Exception as e:
                timestamped_print(f"Update file time error: {str(e)}", 'ERROR')
        else:
            try:
                with open(file_path, 'w') as f:
                    pass
                timestamped_print(f"Created file while heart beating: {file_path}", 'ERROR')
            except Exception as e:
                timestamped_print(f"Create file error: {str(e)}", 'ERROR')

        # 更灵敏的中断检查（每次睡眠5秒）
        for _ in range(6):
            if stop_event.is_set():
                timestamped_print("Heartbeat worker exiting...")
                return
            time.sleep(5)

def restart_program():
    """安全重启流程：释放资源 -> 杀子进程 -> 启动新进程"""
    try:
        # 1. 释放显存资源（以PyTorch为例）
        try:
            del model
            del tokenizer
        except Exception as e:
            timestamped_print(f"Failed to unload model: {e}", 'ERROR')
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            ray.shutdown()
            timestamped_print("已强制清空GPU显存")

        # 2. 终止所有子进程
        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)
        for child in children:
            child.terminate()
        gone, alive = psutil.wait_procs(children, timeout=5)
        if alive:
            for child in alive:
                child.kill()
        timestamped_print("已终止所有子进程")

        # 3. 启动新进程（与当前进程分离）
        new_program = sys.executable
        args = [new_program] + sys.argv
        psutil.Popen(args, start_new_session=True)

        # 4. 终止当前进程
        timestamped_print("正在退出当前进程...")
        os.kill(os.getpid(), signal.SIGTERM)
        
    except Exception as e:
        timestamped_print(f"安全重启失败: {str(e)}", 'ERROR')
        sys.exit(1)

def select_analyze_path(paths, scores) -> Tuple[List[str], List[float]]:
    # 计算平均值
    mean_score = sum(scores) / len(scores)
    
    # 生成包含索引和差值的列表
    indexed_diff = [(i, abs(score - mean_score)) for i, score in enumerate(scores)]
    
    # 根据差值排序，最接近的排在前面
    sorted_diff = sorted(indexed_diff, key=lambda x: x[1])
    
    # 计算需要选取的数量（总数量的一半）
    half_n = len(scores) // 2 + 1
    
    # 获取前一半的索引
    selected_indices = [idx for idx, diff in sorted_diff[:half_n]]
    
    # 返回对应的beam_outputs元素
    return [paths[idx] for idx in selected_indices], [scores[idx] for idx in selected_indices]
    

def get_reward_score(out):
    generated_text = out.text
    logprobs = out.logprobs
    tokens = out.token_ids
    token_logprobs = logprobs
    # 查找 'boxed{Yes}' 或 'boxed{No}' 的位置
    boxed_match = re.search(r'(Yes|No)\}', generated_text, re.IGNORECASE)
    yes_token = tokenizer.encode('Yes')[-1]
    no_token = tokenizer.encode('No')[-1]

    if boxed_match:
        decision = boxed_match.group(1).capitalize()
        if decision == "Yes":
            yes_index = len(tokens) - 1 - tokens[::-1].index(yes_token)
            yes_logprob = token_logprobs[yes_index][yes_token].logprob
            # 将logprob转换为概率
            yes_prob = math.exp(yes_logprob)  # e^log(prob) = prob

            # 在剩余的4个 logprobs 中寻找 'No' 的概率
            try:
                no_logprob = token_logprobs[yes_index][no_token].logprob
                no_prob = math.exp(no_logprob)
            except KeyError:
                # 如果找不到 'No'，将其概率设为所有 logprobs 中最低的概率
                min_logprob = min(v.logprob for k, v in token_logprobs[yes_index].items())
                no_prob = math.exp(min_logprob)

            # 计算 softmax 值
            softmax_denominator = yes_prob + no_prob
            if softmax_denominator == 0:
                softmax_yes = 0.5  # 防止除以零，赋予中立评分
            else:
                softmax_yes = yes_prob / softmax_denominator

            return softmax_yes

        elif decision == "No":
            no_index = len(tokens) - 1 - tokens[::-1].index(no_token)
            no_logprob = token_logprobs[no_index][no_token].logprob
            # 将logprob转换为概率
            no_prob = math.exp(no_logprob)  # e^log(prob) = prob

            # 在剩余的4个 logprobs 中寻找 'Yes' 的概率
            try:
                yes_logprob = token_logprobs[no_index][yes_token].logprob
                yes_prob = math.exp(yes_logprob)
            except KeyError:
                # 如果找不到 'Yes'，将其概率设为所有 logprobs 中最低的概率
                min_logprob = min(v.logprob for k, v in token_logprobs[no_index].items())
                yes_prob = math.exp(min_logprob)

            # 计算 softmax 值
            softmax_denominator = yes_prob + no_prob
            if softmax_denominator == 0:
                softmax_yes = 0.5  # 防止除以零，赋予中立评分
            else:
                softmax_yes = yes_prob / softmax_denominator

            return softmax_yes
    else:
        # 如果没有找到 'boxed{Yes}' 或 'boxed{No}'，赋予默认值
        return 0.5

def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="Process data with optional generation trigger.")
    parser.add_argument("--reward_name_or_path", type=str, help="Path to the reward model or data.")
    parser.add_argument("--data_path", type=str, help="Path to the input data.")
    parser.add_argument("--split_out", type=str)
    parser.add_argument("--turn", type=int)
    parser.add_argument("--mode", type=str, help='Refine by `gemma` or `PRM` or `R1_7B`')
    parser.add_argument("--analyze_template", type=str, default="<analyze>\nLet's analyze the Paragraph {cur_step} step by step: ")
    parser.add_argument("--verify_template", type=str, default="<verify>\nLet's use python code to find any potential error:\n```python\n")
    parser.add_argument("--output_template", type=str, default="<output>\n**Judgement**: $\\boxed")
    # parser.add_argument("--controller_addr", type=str, help="Controller address (required when --gen is used).")
    return parser.parse_args()

args = parse_args()
print_args(args, 
    program_name="_final_refine_PRM",
    version=version)

#####################################################           模型VLLM加载             ########################################################

def initialize_vllm(model_path):
    if 'gemma' in model_path:
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        model = LLM(model=model_path, gpu_memory_utilization=0.90, enable_chunked_prefill=True, max_model_len=16384)
        tokenizer = AutoProcessor.from_pretrained(model_path)
        return model, tokenizer
    else:
        if '_32b_' in model_path:
            model = LLM(model=model_path, gpu_memory_utilization=0.90, enable_chunked_prefill=True, max_model_len=16384)
        elif 'R1' in model_path:
            model = LLM(model=model_path, gpu_memory_utilization=0.99, enable_chunked_prefill=True, max_model_len=32768)
        else:
            model = LLM(model=model_path, gpu_memory_utilization=0.90, enable_chunked_prefill=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

if args.mode == 'gemma':
    model, tokenizer = initialize_vllm("/cpfs02/user/liurunze/hf_models/models--google--gemma-3-12b-it")
elif args.mode == 'PRM':
    model, tokenizer = initialize_vllm(args.reward_name_or_path)
elif args.mode == 'R1_7B':
    model, tokenizer = initialize_vllm("/cpfs02/user/liurunze/hf_models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B")

#####################################################           数据集(分布式)加载             ########################################################


random.seed(int(time.time()))  # 使用当前时间作为种子

# 根据 split 构造分布式处理
def get_shuffled_folders(directory):
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    random.shuffle(folders)
    return folders

target_list = get_shuffled_folders(args.data_path)

for data_path in target_list:
    # 获取文件夹名称
    folder_name = os.path.basename(data_path)
    if args.mode == 'gemma':
        save_path = os.path.join(args.split_out, folder_name) + '_gemma'
    elif args.mode == 'R1_7B':
        save_path = os.path.join(args.split_out, folder_name) + '_R1_7B'
    elif args.mode == 'PRM':
        save_path = os.path.join(args.split_out, folder_name) + '_verify'

    # 检查是否存在同名文件夹
    if not os.path.exists(save_path):
        # 如果不存在，创建文件夹并执行后续逻辑
        try:
            os.makedirs(save_path)
            print(f"创建文件夹: {save_path}")
        except Exception as e:
            print(f"Error: {e}")
            continue
    else:
        # 如果存在，检查文件夹是否为空且创建时间是否超过 5 分钟
        if not os.listdir(save_path):  # 检查文件夹是否为空
            creation_time = os.path.getctime(save_path)  # 获取文件夹创建时间
            current_time = time.time()
            if (current_time - creation_time) > TIME_LIMIT:  # 检查是否超过 5 分钟
                # 如果满足条件，重新创建文件夹
                os.makedirs(save_path, exist_ok=True)
                print(f"重新创建文件夹: {save_path}")
            else:
                # 如果不满足条件，跳过当前文件夹
                print(f"跳过文件夹: {save_path} (未超过 5 分钟)")
                continue
        else:
            # 如果文件夹不为空，跳过当前文件夹
            print(f"跳过文件夹: {save_path} (文件夹不为空)")
            continue
    
    # 启用心跳机制
    stop_event.clear()      # 清除停止标志
    thread = threading.Thread(target=heart_beat_worker, args=(save_path,))
    thread.daemon = True
    thread.start()
    timestamped_print("Heartbeat thread started. Main thread continues...")
    # 执行处理逻辑
    data = load_hf(os.path.join(args.data_path, folder_name), 'huggingface')
    print(data)
    data_new = data.to_list()


    # 如果轮次不是0的话判定是否需要Refine，并且根据上一轮的Refine结果调整本轮steps
    if args.turn > 0:
        def find_value_below_threshold(data):
            value_list = [v[0] for v in data[0]['value']]
            for idx, val in enumerate(value_list):
                if val < 0.5:
                    return idx
            return None
        
        idx = find_value_below_threshold(data_new)
        if idx is None:
            print('无需Refine')
            print(type(data_new))
            print(type(Dataset.from_list(data_new)))
            (Dataset.from_list(data_new)).save_to_disk(save_path)  # 保存数据集
            print(f"数据集已保存到: {save_path}")
            # 处理完成后停止线程
            stop_event.set()      # 设置停止标志
            thread.join(timeout=5)  # 等待线程结束（超时5秒）

            # 可添加线程状态监控
            if thread.is_alive():
                timestamped_print("Warning: Heartbeat thread did not exit cleanly!", 'ERROR')
            else:
                timestamped_print("Heartbeat thread exited successfully")
            
            continue
        else:
            origin_step_list = data_new[0]['steps'][:idx]
            refine_step_list = data_new[0]['refine'].split('\n\n')
            data_new[0]['steps'] = origin_step_list + refine_step_list

    sample = deepcopy(data_new)[0]

    data_input = sample['steps']
    data_input[0] = sample['problem'] + '\n' + data_input[0]
    if data_input and data_input[-1] == '':
        data_input.pop()
    if args.mode == 'PRM':
        message = {
            'conversation': [
                {'role': 'system', 'content': 'You are a math teacher. Your task is to review and critique the paragraphs in solution step by step.'}
            ]
        }
        for j1 in range(len(data_input)):
            line = {'content': data_input[j1], 'role': 'user'}
            message['conversation'].append(line)
            line = {'content': '', 'role': 'assistant'}
            message['conversation'].append(line)
    elif args.mode == 'gemma':
        message = {
            'conversation': [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]}
            ]
        }
        for j1 in range(len(data_input)):
            if j1 == 0:
                origin_prompt=  f"The following is a math problem and my solution. Your task is to review and critique the paragraphs in solution step by step. Pay attention that you should not solve the problem and give the final answer. All of your task is to critique. Output your judgement of whether the paragraph is correct in the form of `\\boxed{{Yes|No}}` at the end of each paragraph verification:\n\n[Math Problem]\n\n{sample['problem']}\n\n[Solution]\n\n<paragraph_1>\n{sample['steps'][0]}\n</paragraph_1>"
                line = {'content': [{"type": "text", "text": origin_prompt}], 'role': 'user'}
                message['conversation'].append(line)
                line = {'content': [{"type": "text", "text": ''}], 'role': 'assistant'}
                message['conversation'].append(line)
            else:
                line = {'content': [{"type": "text", "text": f'<paragraph_{j1+1}>\n' + data_input[j1] + f'\n</paragraph_{j1+1}>'}], 'role': 'user'}
                message['conversation'].append(line)
                line = {'content': [{"type": "text", "text": ''}], 'role': 'assistant'}
                message['conversation'].append(line)
    elif args.mode == 'R1_7B':
        message = {
            'conversation': [
                {"role": "system", "content": "You are a helpful and harmless assistant. You should think step-by-step."},
            ]
        }
        for j1 in range(len(data_input)):
            if j1 == 0:
                origin_prompt=  f"The following is a math problem and my solution. Your task is to review and critique the paragraphs in solution step by step. Pay attention that you should not solve the problem and give the final answer. All of your task is to critique. Output your judgement of whether the paragraph is correct in the form of `\\boxed{{Yes|No}}` at the end of each paragraph verification:\n\n[Math Problem]\n\n{sample['problem']}\n\n[Solution]\n\n<paragraph_1>\n{sample['steps'][0]}\n</paragraph_1>"
                line = {'content': origin_prompt, 'role': 'user'}
                message['conversation'].append(line)
                line = {'content': '', 'role': 'assistant'}
                message['conversation'].append(line)
            else:
                line = {'content': f'<paragraph_{j1+1}>\n' + data_input[j1] + f'\n</paragraph_{j1+1}>', 'role': 'user'}
                message['conversation'].append(line)
                line = {'content': '', 'role': 'assistant'}
                message['conversation'].append(line)
    else:
        raise ValueError('invalid mode')
    
    print(message)

    #####################################################           模型测试           ########################################################
    try:
        conversation = message['conversation']
        step_scores = []
        code_executor = CodeExecutor()
        cur_step = 0
        start = time.perf_counter()
        for step_index, mm in enumerate(conversation):
            role = mm.get('role', '').lower()
            if role == 'user' or role == 'system':
                continue
            # 生成当前步骤之前的所有消息路径
            paths = conversation[:step_index]
            cur_step += 1
            if args.mode == 'gemma':
                # 构造 prompt
                prompt = tokenizer.apply_chat_template(paths, tokenize=False, add_generation_prompt=True)
                # tokenized_prompt = tokenizer.tokenize(prompt)
                sampling_params = SamplingParams(
                    n=1,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=64,
                    max_tokens=2048,    # Maximum number of tokens to generate
                    logprobs=20,       # Number of log probabilities to return
                )
                analyze_start = f"Let's verify the correctness of Paragraph {cur_step} step by step: "
                cprint(prompt+analyze_start, f'paragraph {cur_step} request 1')
                output = model.generate(prompt+analyze_start, sampling_params, use_tqdm=False)[0].outputs[0]
                cprint(output.text, 'output')
                
                # 将生成的响应添加到消息历史中，以便下一步生成时包含上下文
                conversation[step_index] = {
                    'role': 'assistant',
                    'content':  [{"type": "text", "text": output.text}]
                }
                if 'boxed{No}' in output.text:
                    step_scores.append([0.0])
                else:
                    step_scores.append([1.0])
            elif args.mode == 'R1_7B':
                # 构造 prompt
                prompt = tokenizer.apply_chat_template(paths, tokenize=False, add_generation_prompt=True)
                tokenized_prompt = tokenizer.tokenize(prompt)
                sampling_params = SamplingParams(
                    n=1,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=40,
                    max_tokens=32768,    # Maximum number of tokens to generate
                )
                analyze_start = f"<think>\nLet's verify the correctness of Paragraph {cur_step} step by step: "
                cprint(prompt+analyze_start, f'paragraph {cur_step} request 1')
                output = model.generate(prompt+analyze_start, sampling_params, use_tqdm=False)[0].outputs[0]
                cprint(output.text, 'output')
                
                # 将生成的响应添加到消息历史中，以便下一步生成时包含上下文
                conversation[step_index] = {
                    'role': 'assistant',
                    'content': output.text.split('</think>')[0]
                }
                if 'boxed{No}' in output.text:
                    step_scores.append([0.0])
                else:
                    step_scores.append([1.0])
            elif args.mode == 'PRM':
                context = {"cur_step": cur_step}
                analyze_start = args.analyze_template.format(**context)
                verify_start = args.verify_template.format(**context)
                output_start = args.output_template.format(**context)

                # 构造 prompt
                prompt = tokenizer.apply_chat_template(paths, tokenize=False, add_generation_prompt=True)
                tokenized_prompt = tokenizer.tokenize(prompt)

                num_beams = 1
                temperature = 0.6
                top_p = 0.95
                top_k = 20
                repetition_penalty=1.0
                
                # 两阶段生成
                beam_scores = []
                beam_outputs = []
                # 第一阶段
                # 先输出analyze部分, 获取多个状态+分数
                sampling_params = SamplingParams(
                    n=num_beams,
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,  # Controls randomness
                    top_p=top_p,        # Nucleus sampling
                    top_k=top_k,
                    stop=['</analyze>\n'],  # 设置停止字符串
                    include_stop_str_in_output=True,  # 结果中包含停止字符串
                    max_tokens=MAX_TOKENS,    # Maximum number of tokens to generate
                    logprobs=top_k,       # Number of log probabilities to return
                )
                cprint(prompt+analyze_start, f'paragraph {cur_step} request 1')
                outputs_1 = model.generate(prompt+analyze_start, sampling_params, use_tqdm=False)[0].outputs

                cur_prompts = [analyze_start+output1.text+output_start for output1 in outputs_1]
                analyze_paths = []
                analyze_scores = []
                for idx, cur_prompt in enumerate(cur_prompts):
                    sampling_params = SamplingParams(
                        repetition_penalty=repetition_penalty,
                        temperature=temperature,  # Controls randomness
                        top_p=top_p,        # Nucleus sampling
                        top_k=top_k,
                        stop=['</output>\n'],  # 设置停止字符串
                        include_stop_str_in_output=True,  # 结果中包含停止字符串
                        max_tokens=20,    # Maximum number of tokens to generate
                        logprobs=top_k,       # Number of log probabilities to return
                    )
                    cprint(prompt + cur_prompt, f'paragraph {cur_step} request 1 - output {idx}')
                    out = model.generate(prompt + cur_prompt, sampling_params, use_tqdm=False)[0].outputs[0]
                    analyze_path = cur_prompt + out.text
                    analyze_paths.append(analyze_path)
                    analyze_scores.append(get_reward_score(out))

                # analyze选择策略
                analyze_paths, analyze_scores = select_analyze_path(analyze_paths, analyze_scores)

                # 第二阶段
                for analyze_path, analyze_score in zip(analyze_paths, analyze_scores):
                    for _ in range(1):
                        cur_prompts = [analyze_path.split('<output>')[0] + verify_start]
                        out_nodes = []
                        cur_time = 0
                        error_flag = False
                        while len(cur_prompts) > 0:
                            tokenized_prompt = tokenizer.tokenize(cur_prompts[0])
                            max_tokens = MAX_TOKENS - len(tokenized_prompt)
                            if max_tokens > 0 and cur_time < 3:
                                sampling_params = SamplingParams(
                                    repetition_penalty=repetition_penalty,
                                    temperature=temperature,  # Controls randomness
                                    top_p=top_p,        # Nucleus sampling
                                    top_k=top_k,
                                    stop=['\n```\n', '</output>\n'],  # 设置停止字符串
                                    include_stop_str_in_output=True,  # 结果中包含停止字符串
                                    max_tokens=max_tokens,    # Maximum number of tokens to generate
                                    logprobs=top_k,       # Number of log probabilities to return
                                )
                                cprint(prompt + cur_prompts[0], f'paragraph {cur_step} request {cur_time + 2}')
                                origin_outputss_2 = model.generate([prompt + cur_prompt for cur_prompt in cur_prompts], sampling_params, use_tqdm=False)
                            else:
                                error_flag = True
                                break
                            cur_time += 1
                            new_prompts = []
                            # origin_outputss_2是多个pot路径，每个理论上只有一个pot输出
                            for cur_prompt, origin_outputs in zip(cur_prompts, origin_outputss_2):
                                origin_output = origin_outputs.outputs[0]
                                # origin_output是单个pot输出
                                if origin_output.text.endswith('</output>\n'):
                                    origin_outputs.outputs[0].text = cur_prompt+origin_output.text
                                    out_nodes.append(origin_outputs)
                                else:
                                    code_output = code_executor.execute(cur_prompt+origin_output.text)
                                    code_content = f"[Code Output]\n\n```\n{code_output}\n```\n"
                                    new_prompts.append(cur_prompt+origin_output.text+code_content)

                            cur_prompts = new_prompts

                        if error_flag:
                            print(f'paragraph {cur_step} request {cur_time + 2} error')
                            beam_outputs.append(analyze_path)
                            beam_scores.append(analyze_score)
                        else:
                            origin_outputss_2 = out_nodes
                            outputss_2 = [element.outputs for element in origin_outputss_2]
                            for outputs_2 in outputss_2:
                                output2 = outputs_2[0]
                                beam_output = output2.text
                                print(beam_output)
                                beam_outputs.append(beam_output)
                                beam_scores.append(get_reward_score(output2))

                # 选择得分最接近平均值的输出
                selected_output = select_closest_to_mean(beam_outputs, beam_scores)
                # 将生成的响应添加到消息历史中，以便下一步生成时包含上下文
                conversation[step_index] = {
                    'role': 'assistant',
                    'content': selected_output
                }
                step_scores.append(beam_scores)
                # if '<output>\n**Judgement**: $\\boxed{No}$\n</output>' in selected_output:
                #     break

        end = time.perf_counter()
        data_new[0]['time'] = end - start
        data_new[0]['value'] = step_scores
        data_new[0]['conversation'] = conversation
        print(type(data_new))
        print(type(Dataset.from_list(data_new)))
        (Dataset.from_list(data_new)).save_to_disk(save_path)  # 保存数据集
        print(f"数据集已保存到: {save_path}")
    except Exception as e:
        traceback.print_exc()
    
    # 处理完成后停止线程
    stop_event.set()      # 设置停止标志
    thread.join(timeout=5)  # 等待线程结束（超时5秒）

    # 可添加线程状态监控
    if thread.is_alive():
        timestamped_print("Warning: Heartbeat thread did not exit cleanly!", 'ERROR')
    else:
        timestamped_print("Heartbeat thread exited successfully")
