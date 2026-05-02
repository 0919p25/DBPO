import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from math_reward import math_reward_fn


def _extract_valid_response_length(extra_info: dict, solution_str: str) -> int:
    """Prefer real completion token count from rollout metadata; fallback to char length."""
    if not isinstance(extra_info, dict):
        return len(solution_str)

    candidate_keys = [
        "valid_response_length",
        "completion_tokens",
        "response_tokens",
        "output_tokens",
        "generated_tokens",
    ]

    for key in candidate_keys:
        value = extra_info.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return int(value)

    usage = extra_info.get("usage")
    if isinstance(usage, dict):
        value = usage.get("completion_tokens")
        if isinstance(value, (int, float)) and value > 0:
            return int(value)

    return len(solution_str)



def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> float:
    """
    VERL 标准的逐条打分接口
    """
    if extra_info is None:
        extra_info = {}

    valid_length = _extract_valid_response_length(extra_info, solution_str)
    
    try:
        from rewards_types import RewardConfig
        config = RewardConfig()


        difficulty = float(extra_info.get('difficulty', 1.0)) if extra_info else 1.0
        gt_with_difficulty = {'answer': ground_truth, 'difficulty': difficulty}

        score = math_reward_fn(
            solution_str=solution_str,
            ground_truth=gt_with_difficulty,
            num_tokens=-1,
            base_budget=config.base_budget,
            valid_response_length=valid_length,
            ignore_think_token=config.ignore_think_token,
            reward_config=config,
            return_delta_score=False,
        )
        return float(score)
    except Exception as e:
        print(f"[Reward Warning] 打分异常: {e}")
        return -1.0
