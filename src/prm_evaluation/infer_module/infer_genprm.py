import re
import math
import io
import signal
from openai import OpenAI
from transformers import AutoTokenizer
from contextlib import redirect_stdout
from copy import *
from typing import List, Any
from utils.util import timestamped_print, cprint


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

class Reward_Service:
    def __init__(self, model_path: str, url: str):
        # Load the model and tokenizer
        self.client = OpenAI(
            api_key="dummy", 
            base_url=url  # "http://127.0.0.1:8001/v1"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        timestamped_print(f"VLLM model loaded successfully", level="INFO")
    
    def build_prompt(self, messages: List) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt
    
    def inference(self, 
        prompt: str, 
        config: dict, 
        cur_step: int, 
        analyze: bool, 
        verify: bool,
        execute: bool,
        time_limit: int,
        code_executor: CodeExecutor,
        analyze_template="<analyze>\nLet's analyze the Paragraph {cur_step} step by step: ",
        verify_template="<verify>\nLet's use python code to find any potential error:\n```python\n",
        output_template="<output>\n**Judgement**: $\\boxed",
        ):
        """
        "n": int,
        "temperature": float,
        "max_tokens": int,
        "top_p": float,
        "top_k": int,
        "logprobs": int,
        "stop": List,
        "include_stop_str_in_output": Bool,
        """
        if 'max_tokens' not in config:
            timestamped_print('Please set up the `max_tokens` in config.', 'ERROR')
            raise ValueError('Please set up the `max_tokens` in config.')
        
        max_tokens = config['max_tokens']

        context = {"cur_step": cur_step}
        analyze_start = analyze_template.format(**context)
        verify_start = verify_template.format(**context)
        output_start = output_template.format(**context)
        if analyze:
            config['stop'] = ['</analyze>\n']
            config['include_stop_str_in_output'] = True
            prompt_len = len(
                self.tokenizer.encode(prompt + analyze_start)
            )
            cprint(prompt_len, 'prompt_len')
            config['max_tokens'] -= prompt_len
            if  config['max_tokens'] < 0:
                timestamped_print('The max_tokens is not enough to generate the analyze prompt.', 'ERROR')
                return "", 0.5

            cprint(prompt + analyze_start, f'paragraph {cur_step} request 1')
            request_results_1 = self.client.completions.create(
                model="reward",
                prompt=prompt + analyze_start,
                extra_body=config
            )
            texts_1 = self.get_text(request_results_1)
            if verify:
                cur_prompts = [analyze_start + text + verify_start for text in texts_1]  # generate <verify> if verify is True
            else:
                cur_prompts = [analyze_start + text + output_start for text in texts_1]  # directly generate <output> if verify is False

        elif verify:
            cur_prompts = [verify_start for i in range(config['n'])]
            texts_1 = ["" for i in range(config['n'])]
        else:
            cur_prompts = [output_start for i in range(config['n'])]
            texts_1 = ["" for i in range(config['n'])]
        
        final_texts = []
        final_results = []
        for cur_prompt, text1 in zip(cur_prompts, texts_1):
            in_nodes = [cur_prompt]
            out_nodes = []
            cur_time = 0
            while len(in_nodes) > 0:
                tokenized_prompt = self.tokenizer.encode(prompt + in_nodes[0])
                left_tokens = max_tokens - len(tokenized_prompt)
                if left_tokens > 0 and cur_time < time_limit:
                    if verify and execute:
                        config['n'] = 1
                        config['stop'] = ['\n```\n', '</output>\n']
                        config['include_stop_str_in_output'] = True
                        config['max_tokens'] = left_tokens
                    else:
                        # not execute
                        config['n'] = 1
                        config['stop'] = ['</output>\n']
                        config['include_stop_str_in_output'] = True
                        config['max_tokens'] = left_tokens
                else:
                    # if the time limit is reached, or the left tokens are not enough
                    if analyze:
                        # degrade into analyze mode
                        in_nodes = [analyze_start + text1.split('</analyze>')[0] + '</analyze>\n' + output_start]
                    else:
                        # enter the output mode
                        in_nodes = [in_nodes[0] + '</verify>\n' + output_start]
                    
                    left_tokens = 20
                    config['n'] = 1
                    config['stop'] = ['</output>\n']
                    config['include_stop_str_in_output'] = True
                    config['max_tokens'] = left_tokens

                cprint(prompt + in_nodes[0], f'paragraph {cur_step} request {cur_time + 2}')
                request_results_2 = self.client.completions.create(
                    model="reward",
                    prompt=prompt + in_nodes[0],
                    extra_body=config
                )
                texts_2 = self.get_text(request_results_2)

                cur_time += 1
                new_prompts = []
                if texts_2[0].endswith('</output>\n') or cur_time >= time_limit:
                    texts_2[0] = in_nodes[0] + texts_2[0]
                    request_results_2.choices[0].text = texts_2[0]
                    out_nodes.append(texts_2[0])
                else:
                    if execute:
                        # execute the code
                        code_output = code_executor.execute(in_nodes[0] + texts_2[0])
                        code_content = f"[Code Output]\n\n```\n{code_output}\n```\n"
                        new_prompts.append(in_nodes[0] + texts_2[0] + code_content)
                    else:
                        new_prompts.append(in_nodes[0] + texts_2[0] + '[Code Output]\n\n```\n')

                in_nodes = new_prompts
        
            final_texts.append(out_nodes[0])
            final_results.append(request_results_2)
        
        # extract the Probability of Yes token as the reward score
        rewards = [self.get_reward(result) for result in final_results]
        
        return final_texts, rewards
    
    def get_text(self, request_results: Any) -> List:
        # Extract text from the request results
        text_results = [
            choice.text
            for choice in request_results.choices
        ]

        return text_results
    
    def get_finish_reason(self, request_results: Any) -> List:
        # Extract finish reason from the request results
        finish_reasons = [
            choice.finish_reason
            for choice in request_results.choices
        ]

        return finish_reasons
    
    def get_logprobs(self, request_results: Any) -> List:
        # Extract logit probabilities from the request results
        logprobs = [
            choice.logprobs
            for choice in request_results.choices
        ]

        return logprobs
    
    def get_reward(self, request_results: Any):
        """
        Extracts the actual probabilities of "Yes" and "No" at the position
        of the last occurring "Yes" or "No" token.

        If one is found and the other is not in top_logprobs, the other's probability
        is estimated as 1 - sum_of_top_logprobs_probabilities.

        Returns:
            A tuple (prob_yes, prob_no), or None if neither "Yes" nor "No" is found.
            Probabilities are actual probabilities (0 to 1), not logprobs.
            Elements in the tuple can be None if a probability could not be determined.
        """
        completion_result = request_results
        choices = []
        if hasattr(completion_result, 'choices'):
            choices = completion_result.choices
        elif isinstance(completion_result, dict) and 'choices' in completion_result:
            choices = completion_result['choices']
        else:
            # print("Error: 'choices' attribute or key not found in the input.")
            return 0.5

        if not choices:
            return 0.5

        for choice_idx, choice_data in enumerate(choices): # Typically n=1, so one choice
            logprobs_data = None
            if hasattr(choice_data, 'logprobs'): logprobs_data = choice_data.logprobs
            elif isinstance(choice_data, dict) and 'logprobs' in choice_data: logprobs_data = choice_data['logprobs']
            if not logprobs_data: continue

            tokens, token_logprobs_list, top_logprobs_list = [], [], []
            if hasattr(logprobs_data, 'tokens'): tokens = logprobs_data.tokens
            elif isinstance(logprobs_data, dict) and 'tokens' in logprobs_data: tokens = logprobs_data['tokens']
            if hasattr(logprobs_data, 'token_logprobs'): token_logprobs_list = logprobs_data.token_logprobs
            elif isinstance(logprobs_data, dict) and 'token_logprobs' in logprobs_data: token_logprobs_list = logprobs_data['token_logprobs']
            if hasattr(logprobs_data, 'top_logprobs'): top_logprobs_list = logprobs_data.top_logprobs
            elif isinstance(logprobs_data, dict) and 'top_logprobs' in logprobs_data: top_logprobs_list = logprobs_data['top_logprobs']

            if not (tokens and token_logprobs_list and top_logprobs_list and \
                    len(tokens) == len(token_logprobs_list) == len(top_logprobs_list)):
                # print(f"Warning: Tokens/logprobs/top_logprobs missing or mismatched in choice {choice_idx}.")
                continue

            for i in range(len(tokens) - 1, -1, -1): # Iterate backwards
                token_text_stripped = tokens[i].strip()

                prob_yes_at_this_pos: Optional[float] = None
                prob_no_at_this_pos: Optional[float] = None

                if token_text_stripped == "Yes" or token_text_stripped == "No":
                    actual_token_generated = token_text_stripped
                    prob_actual_token = math.exp(token_logprobs_list[i])

                    current_top_logprobs_dict: Dict[str, float] = top_logprobs_list[i]
                    sum_probs_in_top_logprobs = 0.0

                    found_yes_in_top = False
                    found_no_in_top = False

                    for top_token, top_logprob in current_top_logprobs_dict.items():
                        top_token_stripped = top_token.strip()
                        current_top_token_prob = math.exp(top_logprob)
                        sum_probs_in_top_logprobs += current_top_token_prob

                        if top_token_stripped == "Yes":
                            prob_yes_at_this_pos = current_top_token_prob
                            found_yes_in_top = True
                        elif top_token_stripped == "No":
                            prob_no_at_this_pos = current_top_token_prob
                            found_no_in_top = True

                    # Ensure the actually generated token's probability is correctly assigned
                    if actual_token_generated == "Yes":
                        prob_yes_at_this_pos = prob_actual_token
                        found_yes_in_top = True # It was generated, so it's "found"
                    elif actual_token_generated == "No":
                        prob_no_at_this_pos = prob_actual_token
                        found_no_in_top = True # It was generated

                    # Estimate probability for the one not found in top_logprobs (if any)
                    # This assumes that 1 - sum_probs_in_top_logprobs is the probability mass for "all other tokens"
                    # And we are assigning this entire mass to "Yes" or "No" if it wasn't in top_logprobs.
                    # This is a strong assumption.
                    remaining_prob_mass = max(0.0, 1.0 - sum_probs_in_top_logprobs) # Ensure non-negative

                    if not found_yes_in_top and actual_token_generated == "No": # We are looking for Yes's estimated prob
                        # If "Yes" was not the generated token and not in top K, estimate it.
                        # If "Yes" *was* the generated token, its prob is already set.
                        prob_yes_at_this_pos = remaining_prob_mass

                    if not found_no_in_top and actual_token_generated == "Yes": # We are looking for No's estimated prob
                        prob_no_at_this_pos = remaining_prob_mass

                    # If neither was generated nor in top-k, this estimation logic might assign remaining_prob_mass to both.
                    # This part needs careful thought based on what "relative probability" truly means.
                    # The current logic: if "Yes" was generated, its prob is known. If "No" wasn't in top-k, its prob is estimated.
                    # If "No" was generated, its prob is known. If "Yes" wasn't in top-k, its prob is estimated.

                    # Refined logic for estimation:
                    # If "Yes" was generated, prob_yes_at_this_pos is its actual prob.
                    #   If "No" was also in top_logprobs, prob_no_at_this_pos is set.
                    #   If "No" was NOT in top_logprobs, estimate prob_no_at_this_pos = remaining_prob_mass
                    # Symmetric for "No" being generated.

                    if actual_token_generated == "Yes":
                        if not found_no_in_top: # "No" was not in top_logprobs
                            prob_no_at_this_pos = remaining_prob_mass
                    elif actual_token_generated == "No":
                        if not found_yes_in_top: # "Yes" was not in top_logprobs
                            prob_yes_at_this_pos = remaining_prob_mass

                    # Final check: if one is still None after generation and top-k search,
                    # it means it wasn't generated, wasn't in top-k.
                    # The above logic should have assigned remaining_prob_mass.
                    # If for some reason one is still None (e.g. both Yes/No not in top_k and neither generated - which is impossible here)
                    # we can assign remaining_prob_mass if the other is already set.
                    # However, this case is complex if both are missing from top_k.
                    # The most straightforward is what's done: actual prob for the generated one,
                    # then lookup for the other in top_k, then estimate if not in top_k.

                    return prob_yes_at_this_pos

        return 0.5 # Neither "Yes" nor "No" found in any choice's tokens
