"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to 
validate answers when necessary.
"""
from typing import List, Union
import re



from rewards_types import RewardConfig, RewardFn, RewardInput, RewardOutput, RewardType
from utils import extract_answer, grade_answer_sympy, grade_answer_mathd
import random
import numpy as np

import math 



THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"


class RewardMathFn(RewardFn):
    """
    Reward function for evaluating mathematical answers.

    This class implements the __call__ method to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __call__(self, input: RewardInput, ignore_think_token = False) -> RewardOutput:
        assert input.problem_type == RewardType.MATH, \
            "Invalid problem type: expected 'MATH', but got '{}'".format(input.problem_type)
        
        problem = input.problem
        model_response = input.model_response
        
        # ==========================================
        # 模块 A：格式检查 (Format Check)
        # ==========================================
        # 1. 检查思维链格式 (Think Format)
        has_think_start = THOUGHT_DELIMITER_START in model_response
        has_think_end = THOUGHT_DELIMITER_END in model_response
        if ignore_think_token:
            has_think_format = True  # 直接豁免检查
        else:
            has_think_format = has_think_start and has_think_end
        
        # 2. 检查答案 Box 格式 (Box Format)
        # 提取 <think> 之后的内容作为作答区
        model_solution = model_response.split(THOUGHT_DELIMITER_END)[1] if has_think_end else model_response
        
        # 检查文本中是否显式包含了 \boxed 标签 (这是最直观的格式判断)
        has_boxed_tag = "\\boxed" in model_solution
        
        # 尝试提取最终答案
        model_answer = extract_answer(model_solution)
        has_valid_extraction = model_answer is not None

        # 只有显式包含 \boxed 且能成功提取，才算 Box 格式完全正确
        has_box_format = has_boxed_tag and has_valid_extraction

        # 3. 格式检查：只惩罚不奖励，格式正确不加分
        if not (has_think_format and has_box_format):
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False)

        # ==========================================
        # 模块 B：正确性打分 (Correctness Reward)
        # ==========================================
        # Process the ground truth(s)
        ground_truths = input.ground_truth.get("answer", None)
        if ground_truths is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, (str, float, int)):
            ground_truths = [ground_truths]

        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)

        if not processed_ground_truths:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if is_correct:
                return RewardOutput(reward=self.config.correct_reward, is_correct=True)

        # 答案错误：格式正确也不给分，消除保底分
        return RewardOutput(reward=0.0, is_correct=False)

def get_delta_score(num_tokens: int, used_tokens: int):
    # Stddev = num_tokens/5
    # Calculate z-score based on how far used_tokens deviates from target (num_tokens)
    z_score = (used_tokens - num_tokens) / (500)
    # Simple Gaussian function that peaks at 1.0 when used_tokens matches target
    delta_score = math.exp(-z_score**2 / 2)
    return max(0.1, delta_score)

def get_delta_score_linear(num_tokens: int, used_tokens: int, alpha = 1/3000):
    # z_score = abs(used_tokens - num_tokens) / (num_tokens/2)
    z_score = abs(used_tokens - num_tokens) * alpha
    
    delta_score = 1 - z_score
    # return max(0, min(1, delta_score))
    return delta_score - 1

def get_delta_score_linear_both(num_tokens: int, used_tokens: int, alpha = 0.002):
    # If used_tokens is negative, we have to setup maximum budget constraint
    if num_tokens < 0:
        beta = alpha

        delta = used_tokens - abs(num_tokens)
        sc = 0
        if delta < 0:
            sc = beta * delta * -1
        else:
            sc = alpha * delta * -1

        # Clip sc to [-1, 1]
        sc = max(-1, min(1, sc))
        return (sc + 1)/2
    else:
        return get_delta_score_linear(num_tokens, used_tokens, alpha)

def get_delta_score_sigmoid(num_tokens: int, used_tokens: int, alpha = 0.01):
    delta = abs(num_tokens) - used_tokens
    if delta < 0:
        delta = delta*alpha
        sigma_score = 1 / (1 + math.exp(-delta))
    else:
        delta = delta*alpha
        sigma_score = 1 / (1 + math.exp(-delta))
        sigma_score += 0.1 # Small bonus
    return max(0, min(1, sigma_score))

def get_binary_score(num_tokens: int, used_tokens: int):
    if used_tokens > num_tokens:
        return 0.0
    else:
        return 1.0

def get_delta_score_normalized(B_dyn: int, T: int, alpha: float = 0.5, beta: float = 0.4, r_correct: float = 1.0, max_penalty_ratio: float = 0.5, tolerance: float = 0.1):
    """
    对称惩罚：偏离 target 两侧都扣分，target ±10% 宽容区间内不惩罚。
    B_dyn: 目标 token 预算
    T: 模型实际消耗的 token 数量
    alpha: 超出预算时的惩罚系数
    beta: 低于预算时的惩罚系数（短思考惩罚）
    r_correct: 正确答案的奖励值
    max_penalty_ratio: 最大惩罚比例限制
    tolerance: target 附近的宽容比例（默认 10%）
    """
    target = abs(B_dyn)
    if target == 0:
        return 0.0

    lower = target * (1 - tolerance)
    upper = target * (1 + tolerance)

    if lower <= T <= upper:
        return 0.0
    elif T < lower:
        pct_short = (lower - T) / target
        S_delta = -beta * pct_short
    else:
        pct_exceeded = (T - upper) / target
        S_delta = -alpha * pct_exceeded

    return max(S_delta, -max_penalty_ratio * r_correct)

def gpqa_reward_fn(solution_str: str, ground_truth: Union[str, List[str]], enable_llm = False, num_tokens = -1, valid_response_length = -1):
    reward_config = RewardConfig()
    reward_config.use_math_orm = enable_llm
    def get_model_choice(res: str) -> str:
        for i in range(len(res) - 1, -1, -1):
            ch = res[i]
            if ch in ("A", "B", "C", "D"):
                prev_ch = res[i - 1] if i > 0 else " "
                next_ch = res[i + 1] if i + 1 < len(res) else " "
                if (not prev_ch.isalpha()) and (not next_ch.isalpha()):
                    return ch
        return ""
    # ...existing code...
    model_choice = get_model_choice(solution_str)
    if model_choice == ground_truth:
        return 1.0
    else:
        return 0.0

def math_reward_fn(solution_str: str, ground_truth: Union[str, List[str], dict], num_tokens = -1, base_budget: int = -1, valid_response_length = -1, ignore_think_token = False, reward_config: RewardConfig = None, return_delta_score = False):
    if reward_config is None:
        reward_config = RewardConfig()

    difficulty = 1.0
    actual_ground_truth = ground_truth

    if isinstance(ground_truth, dict):
        difficulty = float(ground_truth.get("difficulty", 1.0))
        actual_ground_truth = ground_truth.get("answer", ground_truth)

    # base_budget is a training hyperparameter, not a per-sample label.
    if base_budget == -1:
        base_budget = reward_config.base_budget
    if base_budget != -1:
        num_tokens = int(base_budget)

    reward_fn = RewardMathFn(reward_config)
    reward_response = reward_fn(RewardInput(problem=solution_str, problem_type=RewardType.MATH, model_response=solution_str, ground_truth={"answer": actual_ground_truth}), ignore_think_token=ignore_think_token)

    if not reward_config.linear_reward and not reward_config.sigmoid_reward:
        return reward_response.reward

    if num_tokens != -1:
        raw_multiplier = reward_config.difficulty_growth_factor ** (difficulty - 1.0)
        M = min(raw_multiplier, reward_config.max_budget_multiplier)
        B_dyn = int(num_tokens * M)

        T = float(valid_response_length)
        S_delta = 0.0

        if reward_response.is_correct:
            # 当动态预算已达到最大生成长度时，高难度题目只要答对即可，不施加 token 惩罚
            if reward_config.use_normalized_penalty and B_dyn < reward_config.max_response_length:
                S_delta = get_delta_score_normalized(
                    B_dyn=B_dyn,
                    T=T,
                    alpha=reward_config.alpha,
                    beta=reward_config.beta,
                    r_correct=reward_config.correct_reward,
                    max_penalty_ratio=reward_config.max_penalty_ratio,
                    tolerance=reward_config.tolerance,
                )
            else:
                if reward_config.sigmoid_reward:
                    S_delta = get_delta_score_sigmoid(-B_dyn, T, reward_config.alpha)
                else:
                    S_delta = get_delta_score_linear_both(-B_dyn, T, reward_config.alpha)
            final_score = reward_response.reward + S_delta
        else:
            final_score = reward_response.reward

        if return_delta_score:
            return final_score, S_delta
        else:
            return final_score
    else:
        return reward_response.reward

def majority_at_k(generations: List[str], ground_truths: Union[str, List[str]], k: int = -1, problem: str = "", enable_llm: bool = False, ignore_think_token: bool = False, shuffle: bool = False) -> float:
    """
    """
    if not isinstance(ground_truths, list) and not isinstance(ground_truths, np.ndarray):
        ground_truths = [ground_truths]
    processed_ground_truths = []
    for truth in ground_truths:
        truth = str(truth)
        if "\\boxed" in truth:
            processed_truth = extract_answer(truth)
            if processed_truth is not None:
                processed_ground_truths.append(processed_truth)
        else:
            processed_ground_truths.append(truth)
    if k > 0 and k < len(generations):
        if shuffle:
            generations_copy = generations.copy()
            random.shuffle(generations_copy)
            generations = generations_copy[:k]
        else:
            generations = generations[:k]
    
    processed_answers = []
    for gen in generations:
        if ignore_think_token:
            gen = re.sub(r'<think>.*?</think>', '', gen, flags=re.DOTALL)
        
        if "\\boxed" in gen:
            extracted = extract_answer(gen)
            if extracted is not None:
                processed_answers.append(extracted)
        else:
            processed_answers.append(gen)
    
    answer_clusters = []
    cluster_counts = []
    
    for answer in processed_answers:
        found_cluster = False
        
        for i, cluster_representative in enumerate(answer_clusters):
            if grade_answer_mathd(answer, cluster_representative) or grade_answer_sympy(answer, cluster_representative):
                cluster_counts[i] += 1
                found_cluster = True
                break

        if not found_cluster:
            answer_clusters.append(answer)
            cluster_counts.append(1)

    if not answer_clusters:
        return 0.0
    
    max_count_index = cluster_counts.index(max(cluster_counts))
    final_answer = answer_clusters[max_count_index]
    for truth in processed_ground_truths:
        if grade_answer_mathd(final_answer, truth) or grade_answer_sympy(final_answer, truth):
            return 1.0
    return 0.0

if __name__ == "__main__":
    reward_config = RewardConfig()
    reward = RewardMathFn(reward_config)
    input = RewardInput(problem="Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$.", problem_type=RewardType.MATH, model_response="<think> I am omniscient. </think> The answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}.", ground_truth={"answer": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"]})
    output = reward(input)
    print(output)