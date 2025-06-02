"""
This file is largely borrowed from OpenR (https://github.com/openreasoner/openr)
"""

import copy
import traceback
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import requests

from utils.infer_fns import (
    _math_shepherd_infer_fn,
    _skywork_infer_fn,
    _rlhflow_mistral_infer_fn,
    _rlhflow_deepseek_infer_fn,
    _erprm_infer_fn,
    _pqm_infer_fn,
    _qwen_infer_fn,
)


def get_prm_special_tokens(model_name, tokenizer):
    step_tag_id, returned_token_ids = None, None
    if 'math-shepherd' in model_name.lower():
        prm_step_tag = "ки"
        good_tag, bad_tag = "+", "-"
        step_tag_id = tokenizer.encode(prm_step_tag)[-1]  # 12902
        returned_token_ids = tokenizer.encode(f"{good_tag} {bad_tag}")[1:]  # [648, 387]
    elif 'qwen2.5-math' in model_name.lower():  # models--Skywork--Skywork-o1-Open-PRM-Qwen-2.5-7B
        prm_step_tag = "<extra_0>"
        step_tag_id = tokenizer.encode(prm_step_tag)[0]
        returned_token_ids = []
    elif 'skywork' in model_name.lower():  # models--Skywork--Skywork-o1-Open-PRM-Qwen-2.5-7B
        prm_step_tag = "\n"
        step_tag_id = tokenizer.encode(prm_step_tag)[-1]
        returned_token_ids = []
    elif 'mistral-data' in model_name.lower():  # models--RLHFlow--Llama3.1-8B-PRM-Mistral-Data
        good_tag, bad_tag = "+", "-"
        good_tag_id = tokenizer.encode(good_tag)[-1]
        bad_tag_id = tokenizer.encode(bad_tag)[-1]
        returned_token_ids = [good_tag_id, bad_tag_id]
    elif 'deepseek-data' in model_name.lower():  # models--RLHFlow--Llama3.1-8B-PRM-Deepseek-Data
        good_tag, bad_tag = "+", "-"
        good_tag_id = tokenizer.encode(good_tag)[-1]
        bad_tag_id = tokenizer.encode(bad_tag)[-1]
        returned_token_ids = [good_tag_id, bad_tag_id]
    elif "llama3.1-math-prm" in model_name.lower():
        prm_step_tag = "ки"
        good_tag, bad_tag = "+", "-"
        step_tag_id = tokenizer.encode(f" {prm_step_tag}")[-1]
        good_tag_id = tokenizer.encode(f" {good_tag}")[-1]
        bad_tag_id = tokenizer.encode(f" {bad_tag}")[-1]
        returned_token_ids = [good_tag_id, bad_tag_id]
    elif 'pqm' in model_name.lower():  # models--RLHFlow--Llama3.1-8B-PRM-Deepseek-Data
        prm_step_tag = "[PRM]"
        step_tag_id = tokenizer.encode(prm_step_tag, add_special_tokens=False)[-1]
    else:
        raise ValueError("Model path: {} not recognized".format(model_name))
    return step_tag_id, returned_token_ids


def get_infer_fn(model_path, rm_serve_type='fastchat'):
    if "math-shepherd" in model_path.lower():
        return _math_shepherd_infer_fn
    elif "qwen2.5-math" in model_path.lower():
        return _qwen_infer_fn
    elif "skywork" in model_path.lower():
        return _skywork_infer_fn
    elif "mistral-data" in model_path.lower():
        return _rlhflow_mistral_infer_fn
    elif "deepseek-data" in model_path.lower():
        return _rlhflow_deepseek_infer_fn
    elif "llama3.1-math-prm" in model_path.lower():
        return _erprm_infer_fn
    elif "pqm" in model_path.lower():
        return _pqm_infer_fn
    else:
        raise ValueError("Model path: {} not recognized".format(model_path))


@dataclass
class RewardModelBaseConfig:
    prm_step_tag: str
    format_str: str  # a format string that takes in question and answer need to have {question} and {answer} in the string

    rm_serve_type: str
    step_tag_id: int
    returned_token_ids: List[int]


class RewardModelCallingFunction:

    def __init__(self, config: RewardModelBaseConfig):
        self.config = config
        self.prm_step_tag = config.prm_step_tag
        self.format_str = config.format_str

    def __call__(
        self,
        question_answer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        model_names: List[str],
    ) -> Union[List[int], List[List[int]]]:
        raise NotImplementedError

    def replace_step_tag(self, answer: str):
        if self.prm_step_tag not in answer:
            answer += f" {self.prm_step_tag}"
        splits = answer.split(f" {self.prm_step_tag}")
        splits = [s.strip() for s in splits]
        response = f" {self.prm_step_tag}".join([s for s in splits if s != ""])
        response += f" {self.prm_step_tag}"
        return response


class DummyRewardModelCaller(RewardModelCallingFunction):
    # a dummy rm caller that always return 0

    def __init__(self, config: RewardModelBaseConfig):
        super().__init__(config)

    def __call__(
        self,
        question_answer_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        model_names: List[str],
    ) -> Union[List[int], List[List[int]]]:

        def fn(s):
            steps = s.split(self.prm_step_tag)
            steps = [s for s in steps if s.strip() != ""]
            return list(range(len(steps)))

        if isinstance(question_answer_pairs[0], str):
            return fn(
                self.format_str.format(
                    question=question_answer_pairs[0],
                    answer=self.replace_step_tag(question_answer_pairs[1]),
                )
            )
        else:
            return [
                fn(
                    self.format_str.format(
                        question=s[0],
                        answer=self.replace_step_tag(s[1]),
                    )
                ) for s in question_answer_pairs
            ]


@dataclass
class RemoteRewardModelConfig(RewardModelBaseConfig):
    model_name: str
    controller_addr: str
    multi_gpu: bool


def _reward_inference_fastchat(input_str, model_name, controller_addr="http://localhost:10014", multi_gpu=False, timeout=0):
    if multi_gpu:
        ret = requests.post(controller_addr + "/get_worker_address", json={"model": model_name})
        worker_addr = ret.json()["address"]
        if not worker_addr:
            raise ValueError("Value Model name {} does not exist.".format(model_name))
    else:
        worker_addr = "http://0.0.0.0:10081"

    headers = {"User-Agent": "FastChat Client"}
    gen_params = {"input_str": input_str}
    try:
        if timeout > 0:
            response = requests.post(worker_addr + "/worker_reward_inference", headers=headers, json=gen_params, stream=True, timeout=timeout)
        else:
            response = requests.post(worker_addr + "/worker_reward_inference", headers=headers, json=gen_params, stream=True)
        results = response.json()
        reward = results["reward"]
    except Exception as e:
        for i in range(len(input_str)):
            print(f'input_str {i}: {input_str[i]}')
        error_info = traceback.format_exc()
        print(f'Error in _reward_inference_fastchat: {error_info}')
        traceback.print_exc()

    return reward


class RMRemoteCaller(RewardModelCallingFunction):

    def __init__(self, config: RemoteRewardModelConfig, tokenizer):
        self.model_name = config.model_name
        self.controller_addr = config.controller_addr
        self.tokenizer = tokenizer

        self.prm_step_tag = config.prm_step_tag
        self.step_tag_id = config.step_tag_id
        self.returned_token_ids = config.returned_token_ids

        self.multi_gpu = config.multi_gpu

        super().__init__(config)

    def process_input(self, qa_pairs, verbose, legal_action=[]):
        if verbose and legal_action:
            print('*' * 8, 'rm_call.py: start legal action', '*' * 8)
            print('*' * 8, legal_action[0]["raw_action"], '*' * 8)
            print('*' * 8, legal_action[0]["action"], '*' * 8)
            print('*' * 8, legal_action[0]["messages"], '*' * 8)
            print('*' * 8, legal_action[0]["stop_str"], '*' * 8)
            print('*' * 8, legal_action[0]["finish_reason"], '*' * 8)
            print('*' * 8, 'rm_call.py: end legal action', '*' * 8)
        if isinstance(qa_pairs[0], str):
            raise ValueError("The input of PRM should be a list of tuples")
        if 'skywork' in self.model_name.lower():
            temp_qa_pairs = copy.deepcopy(qa_pairs)
            for i in range(len(temp_qa_pairs)):
                raw_splits = temp_qa_pairs[i][1].split(f" ки\n")
                splits = []
                for s in raw_splits:
                    temp = s.replace("\n", " ").strip()
                    if temp:
                        splits.append(temp)
                if verbose:
                    print('*' * 8, 'rm_call.py: start', '*' * 8)
                    print('*' * 8, qa_pairs[i][0], '*' * 8)
                    print('*' * 8, qa_pairs[i][1], '*' * 8)
                    print('*' * 8, splits, '*' * 8)
                    print('*' * 8, 'rm_call.py: end', '*' * 8)
                if len(splits) == 1:
                    answer = splits[0]
                else:
                    answer = f"\n".join(splits)
                temp_qa_pairs[i] = (temp_qa_pairs[i][0], answer)
            return temp_qa_pairs
        elif 'qwen2.5-math' in self.model_name.lower():
            conversations = []
            temp_qa_pairs = copy.deepcopy(qa_pairs)
            for i in range(len(temp_qa_pairs)):
                conversation = [
                    {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                    {"role": "user", "content": temp_qa_pairs[i][0]},
                ]
                assistant_content = ""
                raw_splits = temp_qa_pairs[i][1].split(f" ки\n")
                for j in range(len(raw_splits)):
                    if raw_splits[j].strip() == "":
                        continue
                    text = raw_splits[j].strip()
                    assistant_content += f"{text}<extra_0>"
                conversation.append({"role": "assistant", "content": assistant_content})
                conversations.append(conversation)
            return conversations
        elif 'mistral-data' in self.model_name.lower() or 'deepseek-data' in self.model_name.lower():
            conversations = []
            temp_qa_pairs = copy.deepcopy(qa_pairs)
            for i in range(len(temp_qa_pairs)):
                conversation = []
                raw_splits = temp_qa_pairs[i][1].split(f" ки\n")
                for j in range(len(raw_splits)):
                    if raw_splits[j].strip() == "":
                        continue
                    if j == 0:
                        text = f"{temp_qa_pairs[i][0]} {raw_splits[j].strip()}"
                    else:
                        text = raw_splits[j].strip()
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "<|reserved_special_token_0|>", "role": "assistant"})  # 128002
                conversations.append(conversation)
            return conversations
        else:
            input_str = []
            for i in range(len(qa_pairs)):
                answer = self.replace_step_tag(qa_pairs[i][1])
                if 'llama3.1-math-prm' in self.model_name.lower():
                    answer = answer.replace(" ки\n", " ки")
                elif 'pqm' in self.model_name.lower():
                    answer = answer.replace(" ки", " [PRM]")  # Each step ends with: " [PRM]\n"
                format_str = self.format_str.format(question=qa_pairs[i][0], answer=answer)
                input_str.append(format_str)
                if verbose:
                    print('*' * 8, 'rm_call.py: start', '*' * 8)
                    print('*' * 8, qa_pairs[i][0], '*' * 8)
                    print('*' * 8, qa_pairs[i][1], '*' * 8)
                    print('*' * 8, format_str, '*' * 8)
                    print('*' * 8, 'rm_call.py: end', '*' * 8)
            return input_str

    def __call__(
        self,
        qa_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        verbose: Optional[bool] = False,
        local: Optional[bool] = False,
        legal_action: Optional[List[str]] = [],
        process: Optional[bool] = True,
        timeout: Optional[int] = 0,
    ) -> Union[List[int], List[List[int]]]:
        if process:
            input_str = self.process_input(qa_pairs, verbose=verbose, legal_action=legal_action)
        else:
            input_str = qa_pairs

        if local:
            infer_fn = get_infer_fn(self.model_name, rm_serve_type='fastchat')
            return infer_fn(input_str)

        return _reward_inference_fastchat(
            input_str=input_str, model_name=self.model_name, controller_addr=self.controller_addr, timeout=timeout
        )