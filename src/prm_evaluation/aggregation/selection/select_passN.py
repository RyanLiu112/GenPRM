import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from typing import List, Any, Tuple
from aggregation_util import count_tokens


def has_correct_answer(correctness: List[bool]) -> bool:
    """Check if any step has a correct answer."""
    return any(correctness)

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
    select_correctness = has_correct_answer(correctness)
    token_cost = sum(
        [count_tokens("\n\n".join(steps[i]), tokenizer) for i in range(len(steps))]
    )

    return "pass@N", select_correctness, token_cost