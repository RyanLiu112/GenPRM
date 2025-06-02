import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from typing import List, Any, Tuple
from aggregation_util import count_tokens


def select_function(
    args: Any,
    steps: List[List[Any]],
    rewards: List[List[float]],
    extracted_answers: List[Any],
    correctness: List[bool], 
    conversations: List[List[dict]],
    tokenizer: Any,
) -> Tuple[str, bool, int]:
    """
    """
    select_correctness = correctness[0]
    token_cost = count_tokens("\n\n".join(steps[0]), tokenizer)

    return "pass@1", select_correctness, token_cost