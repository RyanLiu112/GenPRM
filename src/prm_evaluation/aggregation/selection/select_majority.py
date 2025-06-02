import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)
from collections import Counter
from typing import List, Any, Tuple
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
    answer_counts = Counter(extracted_answers)
    most_common_answer, _ = answer_counts.most_common(1)[0]
    answer_indices = [i for i, ans in enumerate(extracted_answers) if ans == most_common_answer]
    select_correctness = any(correctness[i] for i in answer_indices)

    token_cost = sum(
        [count_tokens("\n\n".join(steps[i]), tokenizer) for i in range(len(steps))]
    )

    return "majority", select_correctness, token_cost