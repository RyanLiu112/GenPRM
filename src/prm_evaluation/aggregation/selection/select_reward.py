import os
import sys
import numpy as np
from collections import Counter
from typing import List, Any, Tuple
from transformers import AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from aggregation_util import count_tokens

def select_function(
    args: Any,
    steps: List[List[Any]],
    rewards: List[List[float]],
    extracted_answers: List[str],
    correctness: List[bool], 
    conversations: List[List[dict]],
    tokenizer: AutoTokenizer,
) -> Tuple[str, bool, int]:
    """
    """
    if args.prm_strategy == "last":
        last_rewards = [reward_seq[-1] for reward_seq in rewards]
        selected_index = np.argmax(last_rewards)
    elif args.prm_strategy == "average":
        avg_rewards = [np.mean(reward_seq) for reward_seq in rewards]
        selected_index = np.argmax(avg_rewards)
    elif args.prm_strategy == "min":
        min_rewards = [np.min(reward_seq) for reward_seq in rewards]
        selected_index = np.argmax(min_rewards)
    else:
        raise ValueError(f"Unknown strategy: {args.prm_strategy}")
    
    select_correctness = correctness[selected_index]
    
    token_cost_policy = sum(
        count_tokens("\n\n".join(steps[i]), tokenizer) 
        for i in range(len(steps))
    )
    token_cost_reward = sum(
        count_tokens(tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True), tokenizer)
        for conversation in conversations
	)
    token_cost = token_cost_policy + token_cost_reward
    
    return args.prm_strategy, select_correctness, token_cost