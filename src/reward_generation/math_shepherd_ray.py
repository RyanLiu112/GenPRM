# math_shepherd_ray.py

import ssl
context = ssl.create_default_context()
context.check_hostname = False

import argparse
import copy
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
import random
import time
from typing import List, Optional

import numpy as np
import ray
from ray.util import ActorPool
from datetime import datetime
import threading

from math_shepherd_utils import (
    load_data, load_json, save_json, is_file_exists, check_correctness, format_query, post_process_act,
    assign_tasks, check_question_finished, get_current_save_idx, check_process_cnt, check_lock_timeout, create_empty_file
)
from utils.vllm_worker import VLLMRemoteCaller
from utils.parse_utils import parse_ground_truth, extract_answer

os.environ['VLLM_USE_V1'] = '1'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

stop_event = threading.Event()

def heart_beat_worker(chosen_idxs, save_dir, lock_dir):
    while not stop_event.is_set():
        for (i, j) in chosen_idxs:
            if stop_event.is_set():  # Check stop event more frequently
                print('==============heartbeat worker exit==============')
                return
            lock_file_path = os.path.join(save_dir, f"{lock_dir}/question_{i}_{j}.lock")
            try:
                if is_file_exists(lock_file_path):
                    os.utime(lock_file_path)
                    print(f"Update lock file {lock_file_path}.")
                else:
                    create_empty_file(lock_file_path)
                    print(f"Create lock file {lock_file_path}.")
            except Exception as e:
                print(f"Error in heartbeat worker: {str(e)}")
        # Use shorter sleep intervals with stop event checks
        for _ in range(60):  # 60 * 5 = 300 seconds total
            if stop_event.is_set():
                print('==============heartbeat worker exit==============')
                return
            time.sleep(5)
    print('==============heartbeat worker exit==============')


class Generator:
    def __init__(self, args, llm):
        self.args = args
        self.llm = llm

    def generate_step_by_step_paths(self, problem, num_samples_list, temperature_list):
        messages = format_query(self.args.cot_prompt, problem, 'Step 1: ')

        inference_paths = []
        assert len(num_samples_list) == len(temperature_list)
        for n, temperature in zip(num_samples_list, temperature_list):  # Generate inference paths for each sample size and temperature
            temp_paths, finish_reasons = self.llm.generate_fastchat(messages, n=n, temperature=temperature)
            for idx, (path, finish_reason) in enumerate(zip(temp_paths, finish_reasons)):
                if finish_reason != 'stop':
                    step_list = ['<INVALID>']
                else:
                    step_list = post_process_act(path)
                    if len(step_list) == 1:
                        step_list = ['<INVALID>']
                inference_paths.append(step_list)
            print(f"Generated {len(temp_paths)} inference paths for n={n}, temperature={temperature}.")

        return inference_paths

    def process_inference_paths(self, paths, expected_answer, correct_paths, incorrect_paths, invalid_idxs):
        for path_idx, path in enumerate(paths):
            final_step = path[-1]
            if final_step == '<INVALID>':
                incorrect_paths.append(path)
                invalid_idxs.append(1)
            # elif '\\boxed' not in final_step:
            #     incorrect_paths.append(path)
            #     invalid_idxs.append(1)
            else:
                if check_correctness(final_step, expected_answer, data_name='math'):
                    correct_paths.append(path)
                else:
                    incorrect_paths.append(path)
                    invalid_idxs.append(0)
        return correct_paths, incorrect_paths, invalid_idxs

    def evaluate_inference_step(self, problem, extracted_groundtruth, inference_steps: List[str], step_idx: int, num_mc: int=25) -> float:
        print(f'Sample {num_mc} times for the monte carlo score.')
        if step_idx == len(inference_steps) - 1:  # Last step, no completion required, just check the answer
            generated_response = inference_steps[step_idx].strip()
            if check_correctness(generated_response, extracted_groundtruth, data_name='math'):  # Check correctness of the last step
                return 1.0  # Correct answer, Monte Carlo score = 1
            else:
                return 0.0  # Incorrect answer, Monte Carlo score = 0
        else:
            # Generate num_mc_samples completions for the current step
            assistant_content = "\n\n".join(inference_steps[:step_idx + 1]) + f"\n\nStep {step_idx + 2}: "
            # assistant_content = "\n\n".join(inference_steps[:step_idx + 1]) + f"\n\n"
            messages = format_query(self.args.cot_prompt, problem, assistant_content)

            correct_count = 0
            generated_response, finish_reasons = self.llm.generate_fastchat(messages, n=num_mc, temperature=0.7)  # Generate response

            for idx in range(num_mc):
                final_step = generated_response[idx]
                if check_correctness(final_step, extracted_groundtruth, data_name='math'):  # Check correctness of the response
                    correct_count += 1

            # Calculate Monte Carlo score based on the number of correct completions
            monte_carlo_score = correct_count / num_mc

            return monte_carlo_score

    def evaluate_inference_paths(self, problem, extracted_groundtruth, inference_paths, data, start_path_idx=0, num_mc=25):
        all_scores = []
        if start_path_idx > 0:
            for idx in range(start_path_idx):
                all_scores.append(data['monte_carlo_scores'][idx])

        for idx, path in enumerate(inference_paths):
            if idx < start_path_idx:
                continue
            path_monte_carlo_scores = []
            steps = copy.deepcopy(path)
            print(f'Processing path {idx} with {len(steps)} steps.')
            for step_idx in range(len(steps)):
                score = self.evaluate_inference_step(problem, extracted_groundtruth, steps, step_idx, num_mc)
                path_monte_carlo_scores.append(score)
            all_scores.append(copy.deepcopy(path_monte_carlo_scores))
            data['monte_carlo_scores'] = all_scores

        return all_scores

    def evaluate_problem(self, data):
        # not only monte carlo
        if not self.args.only_monte_carlo:
            problem = data['problem']
            solution = data['solution']
            extracted_groundtruth = data['extracted_groundtruth']
            meta_data = data['meta_data']

            inference_paths = self.generate_step_by_step_paths(problem, self.args.num_samples_list, self.args.temperature_list)
            random.shuffle(inference_paths)
            print(f'Randomly shuffle the inference paths.')

            correct_paths, incorrect_paths, invalid_idxs = [], [], []
            correct_paths, incorrect_paths, invalid_idxs = self.process_inference_paths(inference_paths, extracted_groundtruth, correct_paths, incorrect_paths, invalid_idxs)

            half_sample = self.args.num_paths // 2  # 16 / 2 = 8
            metrics = {'correct_num': 0, 'incorrect_num': 0, 'invalid_num': 0, 'base_score': 0.0}
            if self.args.round > 0:
                q_idx = data['idx']
                last_data_path = os.path.join(self.args.save_dir, f'question_{q_idx}', f'record_{self.args.round - 1}.json')
                if is_file_exists(last_data_path):
                    last_data = load_json(last_data_path)
                    last_metrics = last_data['metrics']
                    metrics = last_metrics
                else:
                    print(f'Last round data not found for question {q_idx}.')
                    return {}
            # metrics['correct_num'] += len(correct_paths)
            # metrics['incorrect_num'] += len(incorrect_paths)
            # metrics['invalid_num'] += sum(invalid_idxs)
            # metrics['base_score'] = 1.0 * metrics['correct_num'] / (metrics['correct_num'] + metrics['incorrect_num'])
            # data['metrics'] = metrics

            try_cnt = 0
            while 0 <= len(correct_paths) < half_sample:
                more_inference_paths = self.generate_step_by_step_paths(problem, num_samples_list=[16], temperature_list=[0.7])
                random.shuffle(more_inference_paths)
                correct_paths, incorrect_paths, invalid_idxs = self.process_inference_paths(
                    more_inference_paths, extracted_groundtruth, correct_paths, incorrect_paths, invalid_idxs
                )
                try_cnt += 1
                if try_cnt >= self.args.num_try:
                    break
            try_cnt = 0
            while 0 <= len(incorrect_paths) - sum(invalid_idxs) < half_sample:
                more_inference_paths = self.generate_step_by_step_paths(problem, num_samples_list=[16], temperature_list=[0.7])
                random.shuffle(more_inference_paths)
                correct_paths, incorrect_paths, invalid_idxs = self.process_inference_paths(
                    more_inference_paths, extracted_groundtruth, correct_paths, incorrect_paths, invalid_idxs
                )
                try_cnt += 1
                if try_cnt >= self.args.num_try:
                    break

            # Total paths to sample
            total_correct = len(correct_paths)
            total_incorrect = len(incorrect_paths)

            # Sample from correct paths
            if total_correct >= half_sample:
                correct_idx = np.random.choice(np.arange(total_correct), half_sample, replace=False)
                sample_correct = [correct_paths[idx] for idx in correct_idx]
                remaining = self.args.num_paths - len(sample_correct)  # 16 - 8 = 8
                valid_idx = [idx for idx in np.arange(total_incorrect) if invalid_idxs[idx] == 0]
                incorrect_idx = np.random.choice(valid_idx, min(remaining, len(valid_idx)), replace=False)
                sample_incorrect = [incorrect_paths[idx] for idx in incorrect_idx]
            else:
                correct_idx = np.arange(total_correct)
                sample_correct = correct_paths
                remaining = self.args.num_paths - len(sample_correct)  # 16 - len(sample_correct) > 8
                valid_idx = [idx for idx in np.arange(total_incorrect) if invalid_idxs[idx] == 0]
                incorrect_idx = np.random.choice(valid_idx, min(remaining, len(valid_idx)), replace=False)
                sample_incorrect = [incorrect_paths[idx] for idx in incorrect_idx]

            # Ensure that we do not exceed the total_sample
            if len(sample_correct) + len(sample_incorrect) < self.args.num_paths:
                additional = self.args.num_paths - len(sample_correct) - len(sample_incorrect)  # If still not enough, sample additional paths from the larger group
                if total_correct >= total_incorrect:
                    remaining_correct_idx = np.array([idx for idx in range(total_correct) if idx not in correct_idx])
                    additional_correct_idx = np.random.choice(
                        remaining_correct_idx, min(additional, len(remaining_correct_idx)), replace=False
                    )
                    additional_sample_correct = [correct_paths[idx] for idx in additional_correct_idx]
                    sample_correct.extend(additional_sample_correct)
                else:
                    remaining_incorrect_idx = np.array([idx for idx in range(total_incorrect) if idx not in incorrect_idx and invalid_idxs[idx] == 0])
                    additional_incorrect_idx = np.random.choice(remaining_incorrect_idx, min(additional, len(remaining_incorrect_idx)), replace=False)
                    additional_sample_incorrect = [incorrect_paths[idx] for idx in additional_incorrect_idx]
                    sample_incorrect.extend(additional_sample_incorrect)

            # Step 3: Combine the sampled paths and Ensure that we have exactly 16 paths
            if len(sample_correct) + len(sample_incorrect) > self.args.num_paths:
                if len(sample_correct) >= len(sample_incorrect):
                    sample_correct = sample_correct[:(self.args.num_paths - len(sample_incorrect))]
                else:
                    sample_incorrect = sample_incorrect[:(self.args.num_paths - len(sample_correct))]
            print(f'Collect {len(sample_correct)} correct paths and {len(sample_incorrect)} incorrect paths.')
            sampled_paths = sample_correct + sample_incorrect

            metrics['correct_num'] += len(correct_paths)
            metrics['incorrect_num'] += len(incorrect_paths)
            metrics['invalid_num'] += sum(invalid_idxs)
            metrics['base_score'] = 1.0 * metrics['correct_num'] / (metrics['correct_num'] + metrics['incorrect_num'])
            data['metrics'] = metrics

            if len(incorrect_paths) == 0:
                data['flag'] = 'easy'
            elif len(correct_paths) == 0:
                data['flag'] = 'hard'
            elif not self.args.close_monte_carlo:
                data['steps'] = sampled_paths
                monte_carlo_scores = self.evaluate_inference_paths(
                    problem=problem,
                    extracted_groundtruth=extracted_groundtruth,
                    inference_paths=sampled_paths,
                    data=data,
                    num_mc=self.args.num_mc_samples
                )
                data['monte_carlo_scores'] = monte_carlo_scores
                data['flag'] = 'medium'
            else:
                data['steps'] = sampled_paths
                data['monte_carlo_scores'] = [
                    [-1 for _ in step_list]
                    for step_list in sampled_paths
                ]
                data['flag'] = 'medium'

            return data
        else: # only monte carlo
            problem = data['problem']
            solution = data['solution']
            extracted_groundtruth = data['extracted_groundtruth']
            # meta_data = data['meta_data']
            if 'metrics' in data:
                if data['metrics']['base_score'] >= 0.9:
                    monte_carlo_id = 0
                elif data['metrics']['base_score'] >= 0.1:
                    monte_carlo_id = 1
                else:
                    monte_carlo_id = 2
            else:
                monte_carlo_id = 2

            completor_num = data['metrics']['correct_num'] + data['metrics']['incorrect_num'] + data['metrics']['invalid_num']
            print(f'Sample {completor_num} times for the completor score.')
            inference_paths = self.generate_step_by_step_paths(problem, [completor_num], [self.args.monte_carlo_temperature_list[monte_carlo_id]])
            # random.shuffle(inference_paths)

            correct_paths, incorrect_paths, invalid_idxs = [], [], []
            correct_paths, incorrect_paths, invalid_idxs = self.process_inference_paths(inference_paths, extracted_groundtruth, correct_paths, incorrect_paths, invalid_idxs)

            metrics = {'correct_num': 0, 'incorrect_num': 0, 'invalid_num': 0, 'base_score': 0.0}
            metrics['correct_num'] += len(correct_paths)
            metrics['incorrect_num'] += len(incorrect_paths)
            # metrics['invalid_num'] += sum(invalid_idxs)
            metrics['base_score'] = 1.0 * metrics['correct_num'] / (metrics['correct_num'] + metrics['incorrect_num'])
            data['completor_score'] = metrics['base_score']
            
            # 添加majority vote判断
            # 统计所有答案的出现次数
            answer_counts = {}
            for path in inference_paths:
                final_answer = extract_answer(path[-1], data_name='math')
                if final_answer:
                    if final_answer not in answer_counts:
                        answer_counts[final_answer] = 0
                    answer_counts[final_answer] += 1
            
            # 找出出现次数最多的答案
            if answer_counts:
                majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
                majority_count = answer_counts[majority_answer]
                is_majority = majority_answer == extracted_groundtruth
                data['is_majority'] = is_majority
                completor_info = {}
                completor_info['majority_answer'] = majority_answer
                completor_info['majority_count'] = majority_count
                completor_info['answer_distribution'] = answer_counts
                data['completor_info'] = completor_info
                print(f'Completor score: {data["completor_score"]:.3f}, '
                          f'Majority answer: {majority_answer} ({majority_count}/{len(inference_paths)}), '
                          f'Is majority: {is_majority}')
                
            if data['completor_score'] > 0:
                if data['completor_score'] >= 0.9:
                    monte_carlo_id = 0
                elif data['completor_score'] >= 0.1:
                    monte_carlo_id = 1
                else:
                    monte_carlo_id = 2
                monte_carlo_scores = self.evaluate_inference_paths(
                    problem=problem,
                    extracted_groundtruth=extracted_groundtruth,
                    inference_paths=data['steps'],
                    data=data,
                    num_mc=completor_num
                )
                data['monte_carlo_scores'] = monte_carlo_scores
            else:
                print('No effective completor answer, skip!.')
            
            return data




@ray.remote
class RemoteGenerator(Generator):
    def __init__(self, args, llm):
        super().__init__(args, llm)


def process_dataset(args, raw_test_ds, actor_pool, question2id, question_mask):
    results = []
    # for i in range(len(raw_test_ds)):
    #     if i < start_idx or i >= end_idx:
    #         question_mask[i] = False
    #     if args.task_name == "MATH":
    #         level = str(raw_test_ds[i]["level"].split(" ")[-1])
    #         if level not in args.levels:
    #             question_mask[i] = False
    test_ds, chosen_dict, chosen_idxs, save_dir, lock_dir = assign_tasks(
        raw_test_ds, question_mask, args.num_paths, args.save_dir, args.lock_dir, args.round, args.eager, args.batch_size, args.max_time
    )
    if len(test_ds) == 0:
        sys.exit(0)
    
    print('open thread to update lock file')
    stop_event.clear()
    thread = threading.Thread(target=heart_beat_worker, args=(chosen_idxs, save_dir, lock_dir), daemon=True)
    thread.start()

    res_q = actor_pool.map_unordered(lambda p, x: p.evaluate_problem.remote(x), test_ds)
    
    start_time = time.time()
    last_time = start_time

    for i, (data) in enumerate(res_q):
        q_idx = question2id[data["problem"]]
        question_path = os.path.join(args.save_dir, f"question_{q_idx}")
        os.makedirs(question_path, exist_ok=True)

        try:
            idx = chosen_dict[q_idx][0]
            if not is_file_exists(os.path.join(question_path, f"record_{idx}.json")):
                chosen_dict[q_idx].pop(0)
            else:
                print(f"Prepare to save but file exists: {os.path.join(question_path, f'record_{idx}.json')}")
                idx = get_current_save_idx()
        except Exception as e:
            print(f"Error: {e}")
            idx = get_current_save_idx()
        save_json(data, os.path.join(question_path, f"record_{idx}.json"))

        temp_time = time.time()
        delta_time = temp_time - last_time
        total_time = temp_time - start_time
        time_str = datetime.now().strftime("%y-%m-%d %H:%M:%S")
        last_time = temp_time

        try:
            success_flag, flag, file_cnt = check_question_finished(question_path, args.num_paths, args.save_dir, args.lock_dir, args.round, detailed=True)
            check_lock_timeout(raw_test_ds, args.save_dir, args.lock_dir, args.round, args.max_time)
            cnt = check_process_cnt(raw_test_ds, args.save_dir)
            total = len(raw_test_ds)
            print(
                f"[{time_str}]   Cnt: {i + 1:>3} / {len(test_ds):>3}  |  Q: {q_idx:>3}  |  Idx: {idx:>3}  |  "
                f"Del T: {delta_time:>6.1f}s  |  Tot T: {total_time:>7.1f}s  |  Avg T: {total_time / (i + 1):>6.1f}s/it  |  "
                f"Pct: {cnt:>5} / {total:>5} = {cnt / total * 100:.2f}%"
            )
        except Exception as e:
            print(
                f"[{time_str}]   Cnt: {i + 1:>3} / {len(test_ds):>3}  |  Q: {q_idx:>3}  |  Idx: {idx:>3}  |  "
                f"Del T: {delta_time:>6.1f}s  |  Tot T: {total_time:>7.1f}s  |  Avg T: {total_time / (i + 1):>6.1f}s/it  |  "
            )
            print(f"Error: {e}")
    
    # Only stop heartbeat after all tasks are complete
    print('==============debug==============')
    stop_event.set()
    thread.join(timeout=10)
    if thread.is_alive():
        print("Warning: Heartbeat thread did not exit cleanly!")
    else:
        print("Heartbeat thread exited successfully")

    return results

def print_args(args: argparse.Namespace, 
              program_name: str = None,
              version: str = None,
              show_version: bool = True) -> None:
    '''
    print the args settings
    '''
    args_dict = {k: v for k, v in vars(args).items() if not k.startswith('_')}
    
    max_len = max(len(str(k)) for k in args_dict.keys())
    sep = '-' * (max_len + 20)
    
    output = []
    if program_name:
        output.append(f"\n\033[1;36m{program_name}\033[0m")
        
    if version and show_version:
        output.append(f"\033[1;34mVersion:\033[0m \033[1;33m{version}\033[0m")
    
    output.append(f"\033[1;35m{sep}\033[0m")
    
    for k, v in sorted(args_dict.items()):
        key = f"\033[1;32m{k:>{max_len}}\033[0m"
        val = f"\033[1;37m{str(v)}\033[0m"
        output.append(f"{key} : {val}")
    
    output.append(f"\033[1;35m{sep}\033[0m\n")
    
    print('\n'.join(output))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_path', '-b', type=str, default=root_dir, help="Path to the JSON file containing problem-answer pairs.")
    parser.add_argument('--data_dir', type=str, default='data', help="Path to the JSON file containing problem-answer pairs.")
    # parser.add_argument('--origin_data_dir', type=str, default='data', help="Copy of Path to the JSON file containing problem-answer pairs.")
    parser.add_argument('--origin_model_path', type=str, help="(Only used in montecarlo)The base model's Path")
    parser.add_argument('--special_data_dir', action="store_true")
    parser.add_argument('--data_name', '-d', type=str, default='math')
    parser.add_argument('--close_monte_carlo', action="store_true", help="Close the monte carlo score.")
    parser.add_argument('--only_monte_carlo', action="store_true", help="Only calculate the monte carlo score.")
    parser.add_argument('--monte_carlo_list', type=list, default=[32, 64, 128], help="Number of completion.")
    parser.add_argument('--monte_carlo_temperature_list', type=list, default=[0.7, 0.7, 0.7], help="Temperature for sampling completions.")
    parser.add_argument('--split', type=str, default='train-test', help="Path to the JSON file containing problem-answer pairs.")
    parser.add_argument('--model_path', '-m', type=str, help="Path to the pre-trained model.")
    parser.add_argument('--num_paths', type=int, default=16, help="Number of inference paths to collect per problem.")
    parser.add_argument('--num_samples_list', type=list, default=[16, 16], help="Number of samples to generate per problem.")
    parser.add_argument('--temperature_list', type=list, default=[0.7, 0.7], help="Temperature for sampling completions.")
    parser.add_argument('--num_mc_samples', type=int, default=25, help="Number of completions to generate per inference step.")
    parser.add_argument('--save_dir', type=str, default='_output', help="Path to save the output JSON file.")
    parser.add_argument('--add_sleep', '-s', action='store_true', help="Reverse the order of inference steps.")
    parser.add_argument('--num_try', type=int, default=126, help="Number of completions to generate per inference step.")

    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--add_step_prompt", action="store_true")
    parser.add_argument("--apply_chat_template", type=bool, default=True)
    parser.add_argument("--serve_type", type=str, default="fastchat", choices=["vllm_model", "vllm_api", "fastchat", "sgl_api"])
    parser.add_argument("--cot_prompt", type=str, default="")
    parser.add_argument("--llm_step_tag", type=str, default="\nStep ")

    # parallel config
    parser.add_argument("--controller_addr", type=str, default="http://0.0.0.0:10014")
    parser.add_argument("--worker_addr", type=str, default="http://0.0.0.0:10082")
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--local", action="store_true", default=False)
    parser.add_argument("--disable_ray", action="store_true")
    parser.add_argument("--round", '-r', type=int, default=0)
    parser.add_argument("--lock_dir", type=str, default="lock_dir")
    parser.add_argument("--eager", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--max_time", type=int, default=0)
    parser.add_argument("--loop", type=int, default=0)

    args = parser.parse_args()
    # patch
    # args.batch_size = args.batch_size ** 2
    if args.only_monte_carlo and args.close_monte_carlo:
        print("Error: only_monte_carlo and close_monte_carlo cannot be set at the same time.")
        sys.exit(1)

    print_args(args, 
              program_name="Math_shepherd",
              version="1.1.0")
    
    cot_prompt_dict = {
        'llama_official': """Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.""",
        # 'llama': """Please reason step by step, and put your final answer within \\boxed{}.""",
        # 'ministral': """Please reason step by step, and put your final answer within \\boxed{}.""",
        # 'gemma': """Please reason step by step, and put your final answer within \\boxed{}.""",
        # 'qwen': """Please reason step by step, and put your final answer within \\boxed{}.""",
        'skywork-o1': """You are Skywork-o1, a thinking model developed by Skywork AI, specializing in solving complex problems involving mathematics, coding, and logical reasoning through deep thought. When faced with a user's request, you first engage in a lengthy and in-depth thinking process to explore possible solutions to the problem. After completing your thoughts, you then provide a detailed explanation of the solution process in your response.""",
        'qwq': """You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.""",
        'default': """Please reason step by step, and put your final answer within \\boxed{}.""",
    }
    if 'llama-3' in args.model_path.lower() and 'skywork-o1' not in args.model_path.lower():  # TODO: Check llama
        # args.cot_prompt = cot_prompt_dict['llama_official']
        args.cot_prompt = cot_prompt_dict['default']
    elif 'ministral' in args.model_path.lower():
        args.cot_prompt = cot_prompt_dict['default']
    elif 'gemma' in args.model_path.lower():
        args.cot_prompt = cot_prompt_dict['default']
    elif 'qwen' in args.model_path.lower():
        args.cot_prompt = cot_prompt_dict['default']
    elif 'skywork-o1-open-llama' in args.model_path.lower():
        args.cot_prompt = cot_prompt_dict['skywork-o1']
        args.add_step_prompt = False  # TODO: Check
    elif 'qwq' in args.model_path.lower():
        args.cot_prompt = cot_prompt_dict['qwq']
        args.add_step_prompt = False  # TODO: Check
    else:
        args.cot_prompt = cot_prompt_dict['default']

    args.origin_save_dir = args.save_dir
    if args.only_monte_carlo:
        args.save_dir = os.path.join(args.base_path, args.origin_save_dir, 'monte_carlo_processed', f'{args.data_name}_{args.split}_{args.model_path.split("/")[-1].split("--")[-1]}_monte_carlo')
    else:
        args.save_dir = os.path.join(args.base_path, args.origin_save_dir, f'{args.data_name}_{args.split}_{args.model_path.split("/")[-1].split("--")[-1]}_PRM-Data')
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, args.lock_dir), exist_ok=True)

    llm = VLLMRemoteCaller(args)
    raw_test_ds = load_data(args)
    question2id = {problem_inst["problem"]: i for i, problem_inst in enumerate(raw_test_ds)}
    question_mask = [True] * len(raw_test_ds)

    actor_pool = ActorPool([RemoteGenerator.remote(args, llm) for _ in range(args.num_worker)])

    while True:
        process_dataset(args, raw_test_ds, actor_pool, question2id, question_mask)


if __name__ == "__main__":
    main()
