import argparse
import json
from pathlib import Path
from copy import *
from datetime import datetime
from colorama import Fore, init
from math_verify import parse, verify

# initialize colorama log
init()


def check_correctness(response : str, answer: str) -> bool:
    return verify(
        parse(response), 
        parse(f"\\boxed{{{answer}}}"),
    )

def load_json(filepath: str):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        timestamped_print(f"\tUTILS: Successfully loaded JSON from {filepath}")
        return data
    except FileNotFoundError:
        timestamped_print(f"\tUTILS: ERROR - File not found: {filepath}", "ERROR")
        raise
    except json.JSONDecodeError as e:
        timestamped_print(f"\tUTILS: ERROR - Error decoding JSON from {filepath}: {e}", "ERROR")
        raise
    except Exception as e:
        timestamped_print(f"\tUTILS: ERROR - An unexpected error occurred loading {filepath}: {e}", "ERROR")
        raise

def save_json(data, filepath: str, indent=4):
    output_path_obj = Path(filepath)
    try:
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_obj, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        timestamped_print(f"\tUTILS: Successfully saved JSON to {output_path_obj}")
    except Exception as e:
        timestamped_print(f"\tUTILS: ERROR - An unexpected error occurred saving to {output_path_obj}: {e}", "ERROR")
        raise

def cprint(s, start):
    if not isinstance(s, str):
        s = str(s)

    print(f"{'*' * 40}")
    print(f"Start: {start}")
    print(f"{'-' * 40}")

    print(s.replace('\n', '\\n'))

    print(f"{'-' * 40}")
    print(f"End: {start}")
    print(f"{'*' * 40}\n")


def timestamped_print(message, level="INFO"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    color = {
        "ERROR": Fore.RED,
        "WARNING": Fore.YELLOW,
        "INFO": Fore.GREEN
    }.get(level, Fore.WHITE)
    print(f"{Fore.CYAN}[{now}]{Fore.RESET} {color}[{level}]{Fore.RESET} {message}")


def build_prompt(messages, tokenizer):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if prompt.endswith(f"{tokenizer.eos_token}\n"):
        prompt = prompt[:-len(f"{tokenizer.eos_token}\n")]
    elif prompt.endswith(tokenizer.eos_token):
        prompt = prompt[:-len(tokenizer.eos_token)]
    return prompt


def print_args(
    args: argparse.Namespace,
    program_name: str = None,
    version: str = None,
    show_version: bool = True
) -> None:
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