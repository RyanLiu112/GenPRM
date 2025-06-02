import os
import sys
import numpy as np
from collections import Counter
from typing import List, Any, Tuple, Dict

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
    tokenizer: Any,
) -> Tuple[str, bool, int]:
    """
    """
    answer_groups: Dict[str, Dict[str, Any]] = {}
    
    for i, answer in enumerate(extracted_answers):
        if answer not in answer_groups:
            answer_groups[answer] = {
                'indices': [],
                'rewards': [],
                'correctness': []
            }
        
        answer_groups[answer]['indices'].append(i)
        answer_groups[answer]['rewards'].append(rewards[i])
        answer_groups[answer]['correctness'].append(correctness[i])
    
    answer_scores = {}
    for answer, group in answer_groups.items():
        if args.prm_strategy == "last":
            scores = [reward_seq[-1] for reward_seq in group['rewards']]
        elif args.prm_strategy == "average":
            scores = [np.mean(reward_seq) for reward_seq in group['rewards']]
        elif args.prm_strategy == "min":
            scores = [np.min(reward_seq) for reward_seq in group['rewards']]
        else:
            raise ValueError(f"Unknown strategy: {args.prm_strategy}")
        
        answer_scores[answer] = np.sum(scores)
    
    selected_answer = max(answer_scores.items(), key=lambda x: x[1])[0]
    
    selected_group = answer_groups[selected_answer]
    if args.prm_strategy == "last":
        best_in_group_idx = np.argmax([r[-1] for r in selected_group['rewards']])
    elif args.prm_strategy == "average":
        best_in_group_idx = np.argmax([np.mean(r) for r in selected_group['rewards']])
    elif args.prm_strategy == "min":
        best_in_group_idx = np.argmax([np.min(r) for r in selected_group['rewards']])
    
    original_index = selected_group['indices'][best_in_group_idx]
    select_correctness = selected_group['correctness'][best_in_group_idx]
    
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