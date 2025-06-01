#####################################################           import packeges and args             ########################################################

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
import argparse
import json
import random
import time
import threading
import traceback
from utils.util import *
from copy import *
from datasets import Dataset, load_from_disk
from prm_evaluation.genprm_inference import GenPRM, CodeExecutor

os.environ['VLLM_USE_V1'] = '0'

version = 'v2.3'

TIME_LIMIT = 300  # set time limit
stop_event = threading.Event()


def heart_beat_worker(file_path):
    start_time = time.time()

    while not stop_event.is_set():
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

        for _ in range(6):
            if stop_event.is_set():
                timestamped_print("Heartbeat worker exiting...")
                return
            time.sleep(5)


def parse_args():
    parser = argparse.ArgumentParser(description="Process data with optional generation config.")
    parser.add_argument("--reward_name_or_path", type=str, help="Path to the reward model or data.")
    parser.add_argument("--data_path", type=str, help="Path to the input data.")
    parser.add_argument("--split_out", type=str, help="Path to the output data.")
    parser.add_argument("--analyze", action='store_true', help='analyze or not')
    parser.add_argument("--verify", action='store_true', help='verify or not')
    parser.add_argument("--execute", action='store_true', help='execute or not')
    parser.add_argument("--analyze_template", type=str, default="<analyze>\nLet's analyze the Paragraph {cur_step} step by step: ")
    parser.add_argument("--verify_template", type=str, default="<verify>\nLet's use python code to find any potential error:\n```python\n")
    parser.add_argument("--output_template", type=str, default="<output>\n**Judgement**: $\\boxed")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--idd", type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print_args(
        args,
        program_name="prm_evaluate",
        version=version
    )

    #####################################################           model load with VLLM             ########################################################

    genprm = GenPRM(args.reward_name_or_path, args.tensor_parallel_size)

    #####################################################           load splited dataset             ########################################################

    random.seed(int(time.time()))


    def get_shuffled_folders(directory):
        folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
        random.shuffle(folders)
        return folders


    target_list = get_shuffled_folders(args.data_path)

    for data_path in target_list:
        folder_name = os.path.basename(data_path)
        save_path = os.path.join(args.split_out, folder_name)

        if args.analyze:
            save_path += '_analyze'
        if args.verify:
            save_path += '_verify'
        if args.execute:
            save_path += '_execute'

        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
                timestamped_print(f"create folder: {save_path}")
            except Exception as e:
                timestamped_print(f"Error: {e}", 'ERROR')
                continue
        else:
            if not os.listdir(save_path):
                creation_time = os.path.getctime(save_path)
                current_time = time.time()
                if (current_time - creation_time) > TIME_LIMIT:
                    os.makedirs(save_path, exist_ok=True)
                    timestamped_print(f"create folder again: {save_path}", 'ERROR')
                else:
                    timestamped_print(f"skip: {save_path} (not exceed time limit)")
                    continue
            else:
                timestamped_print(f"skip: {save_path} (not empty)")
                continue

        stop_event.clear()
        thread = threading.Thread(target=heart_beat_worker, args=(save_path,))
        thread.daemon = True
        thread.start()
        timestamped_print("Heartbeat thread started. Main thread continues...")

        with open(os.path.join(args.data_path, folder_name, 'sample.json'), 'r') as f:
            data_new = json.load(f)
        sample = deepcopy(data_new)
        data_input = sample['steps']
        data_input[0] = sample['problem'] + '\n' + data_input[0]
        if data_input and data_input[-1] == '':
            data_input.pop()
        if args.analyze or args.verify:
            message = {
                'conversation': [
                    {'role': 'system', 'content': 'You are a math teacher. Your task is to review and critique the paragraphs in solution step by step.'}
                ]
            }
        else:
            message = {
                'conversation': [
                    {'role': 'system', 'content': 'You are a math teacher. Your task is to review and critique the paragraphs in solution directly. Output your judgement in the format of `\\boxed{Yes}` if the paragraph is correct, or `\\boxed{No}` if the paragraph is incorrect.'}
                ]
            }
        for j1 in range(len(data_input)):
            line = {'role': 'user', 'content': data_input[j1]}
            message['conversation'].append(line)
            line = {'content': '', 'role': 'assistant'}
            message['conversation'].append(line)

        timestamped_print(message)

        #####################################################           prm evaluate           ########################################################
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

                paths = conversation[:step_index]
                cur_step += 1

                outputs, reward = genprm.inference(
                    messages=paths,
                    majority_num=1,
                    cur_step=cur_step,
                    analyze=args.analyze,
                    verify=args.verify,
                    execute=args.execute,
                    time_limit=3,
                    max_tokens=2048,
                    code_executor=code_executor,
                    analyze_template=args.analyze_template,
                    verify_template=args.verify_template,
                    output_template=args.output_template,
                    logging=True
                )

                conversation[step_index] = {
                    'role': 'assistant',
                    'content': outputs[0]
                }
                step_scores.append(reward)

            end = time.perf_counter()
            data_new['time'] = end - start
            data_new['value'] = step_scores
            data_new['conversation'] = conversation

            timestamped_print(type(data_new))
            with open(os.path.join(save_path, f'result.json'), 'w') as f:
                json.dump(data_new, f, ensure_ascii=False, indent=4)
            timestamped_print(f"dataset has been saved to: {save_path}")
        except Exception as e:
            traceback.print_exc()

        stop_event.set()
        thread.join(timeout=5)

        if thread.is_alive():
            timestamped_print("Warning: Heartbeat thread did not exit cleanly!", 'ERROR')
        else:
            timestamped_print("Heartbeat thread exited successfully")
