import math
from typing import List, Any, Callable, Tuple
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

def _verify_math_expression(response_text: str, true_value_text: str) -> float:
    """
    Parse and verify a math response against its ground truth.
    Returns 1.0 if correct, 0.5 if ground truth parsing fails, 0.0 otherwise.
    """
    try:
        # Parse ground truth
        gt_config = LatexExtractionConfig()
        parsed_true = parse(
            true_value_text,
            extraction_mode="first_match",
            extraction_config=[gt_config],
        )

        if parsed_true:
            # Parse response with normalization
            norm_cfg = NormalizationConfig(
                nits=False,
                malformed_operators=False,
                basic_latex=True,
                equations=True,
                boxed="all",
                units=True,
            )
            resp_cfg = LatexExtractionConfig(
                normalization_config=norm_cfg,
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
            parsed_response = parse(
                response_text,
                extraction_mode="first_match",
                extraction_config=[resp_cfg],
            )
            score_value = float(verify(parsed_response, parsed_true))
        else:
            print(f"Warning: GT parsing failed for '{true_value_text}'. Score=0.5")
            score_value = 0.5

    except Exception as e:
        print(f"Warning: Error verifying '{response_text}' vs '{true_value_text}': {e}")
        score_value = 0.0

    return score_value


def _check_format_tags(text: str) -> bool:
    """
    Check for a strict <think>...</think><answer>...</answer> structure.
    """
    start_think = "<think>"
    end_think = "</think>"
    start_ans = "<answer>"
    end_ans = "</answer>"

    if not text.startswith(start_think):
        return False
    try:
        te = text.index(end_think)
        as_ = text.index(start_ans, te + len(end_think))
        ae = text.index(end_ans, as_ + len(start_ans))
    except ValueError:
        return False

    # No content between </think> and <answer>
    if text[te + len(end_think):as_].strip():
        return False
    # No content after </answer>
    if text[ae + len(end_ans):].strip():
        return False

    return True


def _count_reasoning_indicators(text: str) -> int:
    """
    Count step indicators (e.g., 'Step N:', '1.', '-', transition words).
    """
    count = 0
    lines = text.splitlines()
    bullets = {"-", "*"}
    trans = {"first,", "second,", "next,", "finally,"}

    for line in lines:
        s = line.strip()
        # "Step N:" pattern
        if s.startswith("Step ") and ":" in s:
            p = s.split(":", 1)[0].split(" ")
            if len(p) == 2 and p[1].isdigit():
                count += 1
                continue
        # "N." pattern
        if "." in s:
            p = s.split(".", 1)
            if p[0].isdigit() and (len(p) == 1 or p[1].startswith(" ")):
                count += 1
                continue
        # bullet
        if s and s[0] in bullets and (len(s) == 1 or s[1].isspace()):
            count += 1
            continue

    # transition words
    for w in text.lower().split():
        if w in trans:
            count += 1

    return count


def _calculate_single_cosine_reward(
    text_len: int,
    length_limit: int,
    acc_score: float,
    low_bad: float,
    high_bad: float,
    low_good: float,
    high_good: float,
) -> float:
    """
    Cosine-scaled reward based on response length and accuracy.
    Short correct answers → closer to high_good;
    Long incorrect answers → closer to low_bad.
    """
    progress = (text_len / length_limit) if length_limit > 0 else 0.0
    cosine_val = math.cos(progress * math.pi)
    is_correct = acc_score > 0.5

    if is_correct:
        min_bound = low_good
        max_bound = high_good
    else:
        min_bound = low_bad
        max_bound = high_bad

    reward_range = max_bound - min_bound
    scaled = 0.5 * reward_range * (1.0 + cosine_val)
    return float(min_bound + scaled)


def _extract_ngrams_helper(text: str, n: int) -> List[Tuple[str, ...]]:
    """
    Build a list of n-gram tuples from the text.
    """
    words = text.lower().split()
    if len(words) < n:
        return []
    result = []
    for i in range(len(words) - n + 1):
        result.append(tuple(words[i : i + n]))
    return result


def _calculate_single_repetition_penalty(text: str, ngram: int, penalty_value: float) -> float:
    """
    Negative penalty proportional to repeated n-grams.
    """
    words = text.lower().split()
    if len(words) < ngram or penalty_value > 0:
        return 0.0
    all_ngrams = _extract_ngrams_helper(text, ngram)
    if not all_ngrams:
        return 0.0
    unique = len(set(all_ngrams))
    fraction_unique = unique / len(all_ngrams)
    repetition_ratio = 1.0 - fraction_unique
    return repetition_ratio * penalty_value


def evaluate_accuracy(completions: List[Any], **kwargs) -> List[float]:
    """
    Score each response against its ground truth in kwargs['solution'].
    """
    try:
        responses = [item[0]["content"] for item in completions]
    except Exception:
        print("Error: Unexpected format in evaluate_accuracy.")
        return [0.0] * len(completions)

    ground_truths = kwargs.get("solution")
    if ground_truths is None:
        print("Warning: No 'solution' provided for accuracy.")
        return [0.0] * len(completions)
    if len(responses) != len(ground_truths):
        print("Warning: Mismatch responses vs solutions.")
        return [0.0] * len(completions)

    return [_verify_math_expression(resp, truth) for resp, truth in zip(responses, ground_truths)]


def evaluate_format(completions: List[Any], **kwargs) -> List[float]:
    """
    Check <think>…</think><answer>…</answer> format; 1.0 if correct, else 0.0.
    """
    try:
        responses = [item[0]["content"] for item in completions]
    except Exception:
        print("Error: Unexpected format in evaluate_format.")
        return [0.0] * len(completions)

    return [1.0 if _check_format_tags(text) else 0.0 for text in responses]


def evaluate_reasoning_steps(completions: List[Any], **kwargs) -> List[float]:
    """
    Reward based on count of reasoning indicators (normalized by 3, capped at 1.0).
    """
    try:
        responses = [item[0]["content"] for item in completions]
    except Exception:
        print("Error: Unexpected format in evaluate_reasoning_steps.")
        return [0.0] * len(completions)

    scores = []
    for text in responses:
        cnt = _count_reasoning_indicators(text)
        scores.append(min(1.0, cnt / 3.0))
    return scores


def create_cosine_reward_func(
    low_bad: float = -0.5,
    high_bad: float = -0.1,
    low_good: float = 0.8,
    high_good: float = 1.0,
    length_limit: int = 1000,
) -> Callable[[List[Any], List[str], List[float]], List[float]]:
    """
    Factory returning a function to compute cosine rewards.
    """
    def cosine_reward(
        responses: List[Any],
        solution: List[str],
        **kwargs
    ) -> List[float]:
        texts = [item[0]["content"] for item in responses]
        if not (len(texts) == len(solution)):
            print("Warning: Mismatch in lengths for cosine rewards.")
            return []

        return [
            _calculate_single_cosine_reward(
                len(text),
                length_limit,
                _verify_math_expression(text, truth),
                low_bad,
                high_bad,
                low_good,
                high_good,
            )
            for text, truth in zip(texts, solution)
        ]

    return cosine_reward


def create_repetition_penalty_func(
    ngram: int = 3,
    penalty_value: float = -0.1
) -> Callable[[List[Any]], List[float]]:
    """
    Factory returning a function to compute repetition penalties.
    """
    if penalty_value > 0:
        raise ValueError(f"Penalty must be non-positive, got {penalty_value}")

    def rep_penalty(
        responses: List[Any],
        **kwargs
    ) -> List[float]:
        texts = [item[0]["content"] for item in responses]
        return [
            _calculate_single_repetition_penalty(text, ngram, penalty_value)
            for text in texts
        ]

    return rep_penalty