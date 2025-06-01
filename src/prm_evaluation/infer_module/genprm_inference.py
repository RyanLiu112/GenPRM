import re
import math
import io
import signal
from copy import *
from utils.util import *
from contextlib import redirect_stdout
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
REPETITION_PENALTY = 1.0
version = 'v1.0'


class timeout:
    """timeout context manager"""

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
    """code executor"""

    def __init__(self):
        self.namespace = {}  # indicate the global namespace for exec
        self.code_pattern = re.compile(r'```python\s*(.*?)\s*```', re.DOTALL)

    def execute(self, text):
        # extract code block
        try:
            code_block = self.code_pattern.findall(text)[-1].strip()
        except Exception as e:
            actual = f"Code format error: No code found."
            return actual

        # execute code block
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


class GenPRM:
    def __init__(self, model_path, tensor_parallel_size):
        if tensor_parallel_size > 1:
            os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

        # Load the model and tokenizer
        timestamped_print(f"Loading model from {model_path}", level="INFO")
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            enable_chunked_prefill=True,
            max_model_len=16384
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        timestamped_print(f"GenPRM loaded successfully", level="INFO")

    def get_reward_score(self, out):
        '''calculate the reward score'''
        generated_text = out.text
        logprobs = out.logprobs
        tokens = out.token_ids
        token_logprobs = logprobs

        # find the position of Yes/No token
        boxed_match = re.search(r'(Yes|No)\}', generated_text, re.IGNORECASE)
        yes_token = self.tokenizer.encode('Yes')[-1]
        no_token = self.tokenizer.encode('No')[-1]

        if boxed_match:
            decision = boxed_match.group(1).capitalize()
            if decision == "Yes":
                yes_index = len(tokens) - 1 - tokens[::-1].index(yes_token)
                yes_logprob = token_logprobs[yes_index][yes_token].logprob
                # convert logprob to probability
                yes_prob = math.exp(yes_logprob)  # e^log(prob) = prob

                # find the position of 'No' token
                try:
                    no_logprob = token_logprobs[yes_index][no_token].logprob
                    no_prob = math.exp(no_logprob)
                except KeyError:
                    # set 'No' probability to the minimum logprob of the remaining 4 logprobs
                    min_logprob = min(v.logprob for k, v in token_logprobs[yes_index].items())
                    no_prob = math.exp(min_logprob)

                # calculate softmax value
                softmax_denominator = yes_prob + no_prob
                if softmax_denominator == 0:
                    softmax_yes = 0.5  # in case of division by zero, assign neutral score
                else:
                    softmax_yes = yes_prob / softmax_denominator

                return softmax_yes

            elif decision == "No":
                no_index = len(tokens) - 1 - tokens[::-1].index(no_token)
                no_logprob = token_logprobs[no_index][no_token].logprob
                # convert logprob to probability
                no_prob = math.exp(no_logprob)  # e^log(prob) = prob

                # find the position of 'Yes' token
                try:
                    yes_logprob = token_logprobs[no_index][yes_token].logprob
                    yes_prob = math.exp(yes_logprob)
                except KeyError:
                    # set 'Yes' probability to the minimum logprob of the remaining 4 logprobs
                    min_logprob = min(v.logprob for k, v in token_logprobs[no_index].items())
                    yes_prob = math.exp(min_logprob)

                # calculate softmax value
                softmax_denominator = yes_prob + no_prob
                if softmax_denominator == 0:
                    softmax_yes = 0.5  # in case of division by zero, assign neutral score
                else:
                    softmax_yes = yes_prob / softmax_denominator

                return softmax_yes
        else:
            # return neutral score if no decision found
            timestamped_print("No boxed{Yes/No} found in the output", level="WARNING")
            return 0.5

    def inference(
        self,
        messages,
        majority_num=1,
        cur_step=1,
        analyze=True,
        verify=True,
        execute=True,
        time_limit=3,
        max_tokens=2048,
        code_executor=None,
        analyze_template="<analyze>\nLet's analyze the Paragraph {cur_step} step by step: ",
        verify_template="<verify>\nLet's use python code to find any potential error:\n```python\n",
        output_template="<output>\n**Judgement**: $\\boxed",
        logging=True
    ):
        '''
        messages: the input messages
        majority_num: the number of majority votes
        cur_step: the current step index (start from 1)
        analyze: whether to analyze the input
        verify: whether to verify the input
        execute: whether to execute the code
        time_limit: the time limit for code execution
        max_tokens: the maximum tokens for the output
        analyze_template: the template for analyze start
        verify_template: the template for verify start
        output_template: the template for output start
        logging: whether to log the process
        '''
        output_paths = []
        reward_scores = []
        for i in range(majority_num):
            # perform inference
            output_path, reward_score = self._single_inference(
                messages,
                cur_step=cur_step,
                analyze=analyze,
                verify=verify,
                execute=execute,
                time_limit=time_limit,
                max_tokens=max_tokens,
                code_executor=code_executor,
                analyze_template=analyze_template,
                verify_template=verify_template,
                output_template=output_template,
                logging=logging
            )

            output_paths.append(output_path)
            reward_scores.append(reward_score)

        return output_paths, sum(reward_scores) / len(reward_scores)

    def _single_inference(
        self,
        messages,
        cur_step=1,
        analyze=True,
        verify=True,
        execute=True,
        time_limit=3,
        max_tokens=2048,
        code_executor=None,
        analyze_template="<analyze>\nLet's analyze the Paragraph {cur_step} step by step: ",
        verify_template="<verify>\nLet's use python code to find any potential error:\n```python\n",
        output_template="<output>\n**Judgement**: $\\boxed",
        logging=True
    ):
        context = {"cur_step": cur_step}
        analyze_start = analyze_template.format(**context)
        verify_start = verify_template.format(**context)
        output_start = output_template.format(**context)
        # Prepare the input
        prompt = build_prompt(messages, self.tokenizer)

        # Generate the output
        # Stage 1
        if analyze:
            sampling_params = SamplingParams(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                stop=['</analyze>\n'],
                include_stop_str_in_output=True,
                max_tokens=max_tokens,
                logprobs=20,  # Number of log probabilities to return
                repetition_penalty=REPETITION_PENALTY
            )
            if logging:
                cprint(prompt + analyze_start, f'paragraph {cur_step} request 1')
            output1 = self.model.generate(prompt + analyze_start, sampling_params=sampling_params, use_tqdm=False)[0].outputs[0]
            if verify:
                cur_prompt = analyze_start + output1.text + verify_start  # generate <verify> if verify is True
            else:
                cur_prompt = analyze_start + output1.text + output_start  # directly generate <output> if verify is False

        elif verify:
            cur_prompt = verify_start
        else:
            cur_prompt = output_start

        # Stage 2
        cur_prompts = [cur_prompt]
        out_nodes = []
        cur_time = 0
        while len(cur_prompts) > 0:
            tokenized_prompt = self.tokenizer.tokenize(cur_prompts[0])
            left_tokens = max_tokens - len(tokenized_prompt)
            if left_tokens > 0 and cur_time < time_limit:
                if verify and execute:
                    sampling_params = SamplingParams(
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        top_k=TOP_K,
                        stop=['\n```\n', '</output>\n'],  # set the stop string
                        include_stop_str_in_output=True,  # include the stop string in the output
                        max_tokens=left_tokens,  # Maximum number of tokens to generate
                        logprobs=20,  # Number of log probabilities to return
                        repetition_penalty=REPETITION_PENALTY,
                    )
                else:
                    # not execute
                    sampling_params = SamplingParams(
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        top_k=TOP_K,
                        stop=['</output>\n'],  # set the stop string
                        include_stop_str_in_output=True,  # include the stop string in the output
                        max_tokens=left_tokens,  # Maximum number of tokens to generate
                        logprobs=20,  # Number of log probabilities to return
                        repetition_penalty=REPETITION_PENALTY,
                    )
                if logging:
                    cprint(prompt + cur_prompts[0], f'paragraph {cur_step} request {cur_time + 2}')
                output2 = self.model.generate(prompt + cur_prompts[0], sampling_params, use_tqdm=False)[0].outputs[0]
            else:
                # if the time limit is reached, or the left tokens are not enough
                if analyze:
                    # degrade into analyze mode
                    cur_prompts = [analyze_start + output1.text.split('</analyze>')[0] + '</analyze>\n' + output_start]
                else:
                    # enter the output mode
                    cur_prompts = [cur_prompts[0] + '</verify>\n' + output_start]
                tokenized_prompt = self.tokenizer.tokenize(cur_prompts[0])
                left_tokens = 20
                sampling_params = SamplingParams(
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    stop=['</output>\n'],  # set the stop string
                    include_stop_str_in_output=True,  # include the stop string in the output
                    max_tokens=left_tokens,  # Maximum number of tokens to generate
                    logprobs=20,  # Number of log probabilities to return
                    repetition_penalty=REPETITION_PENALTY,
                )
                if logging:
                    cprint(prompt + cur_prompts[0], f'paragraph {cur_step} request {cur_time + 2}')
                output2 = self.model.generate(prompt + cur_prompts[0], sampling_params, use_tqdm=False)[0].outputs[0]

            cur_time += 1
            new_prompts = []
            if output2.text.endswith('</output>\n'):
                output2.text = cur_prompts[0] + output2.text
                out_nodes.append(output2)
            else:
                if execute:
                    # execute the code
                    code_output = code_executor.execute(cur_prompts[0] + output2.text)
                    code_content = f"[Code Output]\n\n```\n{code_output}\n```\n"
                    new_prompts.append(cur_prompts[0] + output2.text + code_content)
                else:
                    new_prompts.append(cur_prompts[0] + output2.text + '[Code Output]\n\n```\n')

            cur_prompts = new_prompts

        output2 = out_nodes[0]

        # extract the Probability of Yes token as the reward score
        reward_score = self.get_reward_score(output2)

        return output2.text, reward_score
