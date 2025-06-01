#####################################################           import packeges and args             ########################################################

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
import argparse
import os
import re
import random
import time
import threading
import traceback
from utils.util import *
from copy import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from datasets import Dataset, load_from_disk
from utils.util import *

os.environ['VLLM_USE_V1'] = '0'

version = 'v1.0'

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


def split_at_last_boxed(text):
    last_boxed_match = None
    for match in re.finditer(r'\\boxed\{', text):
        last_boxed_match = match

    if not last_boxed_match:
        return (text, None)

    start_pos = last_boxed_match.start()

    last_period = text.rfind('.', 0, start_pos)

    if last_period == -1:
        return (text[:start_pos], text[start_pos:])
    else:
        return (text[:last_period + 1], text[last_period + 1:])


def parse_args():
    parser = argparse.ArgumentParser(description="Process data with optional generation trigger.")
    parser.add_argument("--model_path", type=str, help="Path to the model.")
    parser.add_argument("--data_path", type=str, help="Path to the input data.")
    parser.add_argument("--split_out", type=str, help="Path to the output data.")
    return parser.parse_args()


args = parse_args()
print_args(
    args,
    program_name="policy refine with critique",
    version=version
)


#####################################################           model load with VLLM             ########################################################

def initialize_vllm(model_path):
    if 'gemma' in model_path:
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        model = LLM(model=model_path, gpu_memory_utilization=0.90, enable_chunked_prefill=True, max_model_len=16384)
        tokenizer = AutoProcessor.from_pretrained(model_path)
        return model, tokenizer
    else:
        if '_32b_' in model_path:
            model = LLM(model=model_path, gpu_memory_utilization=0.90, enable_chunked_prefill=True, max_model_len=16384)
        else:
            model = LLM(model=model_path, gpu_memory_utilization=0.90, enable_chunked_prefill=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer


model, tokenizer = initialize_vllm(args.model_path)

#####################################################           load dataset             ########################################################


random.seed(int(time.time()))


def get_shuffled_folders(directory):
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    random.shuffle(folders)
    return folders


target_list = get_shuffled_folders(args.data_path)

for data_path in target_list:
    folder_name = os.path.basename(data_path)
    save_path = os.path.join(args.split_out, folder_name) + '_PRM'

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
                timestamped_print(f"create folder again: {save_path}")
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
    data = load_from_disk(os.path.join(args.data_path, folder_name))
    timestamped_print(data)
    data_new = data.to_list()
    sample = deepcopy(data_new)[0]

    data_input = sample['steps']
    data_input[0] = sample['problem'] + '\n' + data_input[0]

    #####################################################          policy refine           ########################################################
    try:
        def find_value_below_threshold(data):
            value_list = [v for v in data[0]['value']]
            for idx, val in enumerate(value_list):
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


        idx = find_value_below_threshold(data_new)
        if idx is not None:
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
                sampling_params = SamplingParams(
                    n=1,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    max_tokens=16384,  # Maximum number of tokens to generate
                )
            elif 'gemma' in args.model_path:
                critic_content = "There might be some problem in this paragraph of your reasoning, please rethink and refine your answer:\n" + \
                                 '>' + sample['steps'][idx] + '\n\n' + \
                                 extract_after_first_colon(sample['conversation'][2 * idx + 2]['content'])
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                    {"role": "user", "content": [{"type": "text", "text": sample['problem'] + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."}]},
                    {"role": "assistant", "content": [{"type": "text", "text": '\n\n'.join(sample['steps'])}]},
                    {"role": "user", "content": [{"type": "text", "text": critic_content}]},
                ]
                sampling_params = SamplingParams(
                    n=1,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=64,
                    max_tokens=16384,  # Maximum number of tokens to generate
                )
            else:
                raise ValueError('invalid model_path')

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            cprint(idx, 'idx')
            cprint(sample['conversation'][2 * idx + 2]['content'], 'critic content')
            cprint(messages, 'messages')
            cprint(prompt, 'prompt')
            output = model.generate(prompt, sampling_params, use_tqdm=False)[0].outputs[0]
            data_new[0]['refine'] = output.text
        else:
            data_new[0]['refine'] = '\n\n'.join(data_new[0]['steps'])

        timestamped_print(type(data_new))
        timestamped_print(type(Dataset.from_list(data_new)))
        (Dataset.from_list(data_new)).save_to_disk(save_path)
        timestamped_print(f"dataset has been saved to: {save_path}")
    except Exception as e:
        traceback.print_exc()

    stop_event.set()
    thread.join(timeout=5)

    if thread.is_alive():
        timestamped_print("Warning: Heartbeat thread did not exit cleanly!", 'ERROR')
    else:
        timestamped_print("Heartbeat thread exited successfully")
