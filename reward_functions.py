# reward_functions.py
import re
import math
from typing import List, Any, Callable
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

def compute_accuracy_reward(completions: List[Any], **kwargs) -> List[float]:
    """
    Computes an accuracy reward by comparing the model's output to the ground truth solution.
    Returns a list of rewards (1.0 for correct, 0.0 for incorrect, 0.5 for unparsed).
    """
    output_texts = [comp[0]["content"] for comp in completions]
    rewards: List[float] = []
    ground_truths = kwargs.get("solution")
    if ground_truths is None:
        return [0.5] * len(completions)
    for text, solution in zip(output_texts, ground_truths):
        parsed_solution = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if parsed_solution:
            parsed_output = parse(
                text,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            reward_value = float(verify(parsed_output, parsed_solution))
        else:
            reward_value = 0.5
            print("Warning: Could not parse ground truth solution:", solution)
        rewards.append(reward_value)
    return rewards

def compute_format_reward(completions: List[Any], **kwargs) -> List[float]:
    """
    Checks if the output follows the required format with <think> and <answer> tags.
    Returns 1.0 if format is correct, else 0.0.
    """
    required_pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    output_texts = [comp[0]["content"] for comp in completions]
    return [1.0 if re.match(required_pattern, text, re.DOTALL | re.MULTILINE) else 0.0 for text in output_texts]

def compute_reasoning_steps_reward(completions: List[Any], **kwargs) -> List[float]:
    """
    Rewards the output based on the number of reasoning step indicators found.
    The reward is proportional to the count (max capped at 1.0).
    """
    reasoning_pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    output_texts = [comp[0]["content"] for comp in completions]
    return [min(1.0, len(re.findall(reasoning_pattern, text, re.MULTILINE)) / 3) for text in output_texts]

def get_cosine_scaled_reward_function(
    min_wrong: float = -0.5,
    max_wrong: float = -0.1,
    min_correct: float = 0.8,
    max_correct: float = 1.0,
    max_length: int = 1000,
) -> Callable:
    """
    Returns a function that computes a cosine scaled reward based on the output length.
    """
    def cosine_scaled_reward(completions: List[Any], solutions: List[str], accuracy_rewards: List[float], **kwargs) -> List[float]:
        output_texts = [comp[0]["content"] for comp in completions]
        rewards: List[float] = []
        for text, solution, acc_reward in zip(output_texts, solutions, accuracy_rewards):
            text_length = len(text)
            progress_ratio = text_length / max_length
            cosine_value = math.cos(progress_ratio * math.pi)
            if acc_reward > 0.5:
                lower_bound = min_correct
                upper_bound = max_correct
            else:
                lower_bound = max_wrong
                upper_bound = min_wrong
            reward_value = lower_bound + 0.5 * (upper_bound - lower_bound) * (1.0 + cosine_value)
            rewards.append(float(reward_value))
        return rewards
    return cosine_scaled_reward

def get_repetition_penalty_reward_function(ngram_size: int = 3, max_penalty: float = -0.1) -> Callable:
    """
    Returns a function that penalizes repeated n-grams in the output.
    """
    if max_penalty > 0:
        raise ValueError("max_penalty must be negative")
    
    def generate_ngrams(text: str, n: int) -> List[tuple]:
        words = text.lower().split()
        # Ensure total ngrams is at least 1 to avoid division by zero.
        if len(words) < n:
            return []
        return list(zip(*[words[i:] for i in range(n)]))
    
    def repetition_penalty(completions: List[Any], **kwargs) -> List[float]:
        output_texts = [comp[0]["content"] for comp in completions]
        rewards: List[float] = []
        for text in output_texts:
            if not text or len(text.split()) < ngram_size:
                rewards.append(0.0)
                continue
            ngrams = set(generate_ngrams(text, ngram_size))
            total_ngrams = len(text.split()) - ngram_size + 1
            repetition_ratio = 1 - (len(ngrams) / total_ngrams) if total_ngrams > 0 else 0
            rewards.append(repetition_ratio * max_penalty)
        return rewards
    return repetition_penalty
