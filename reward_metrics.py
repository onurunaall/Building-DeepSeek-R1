import re
import math
from typing import List, Any, Callable
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

def evaluate_accuracy(outputs: List[Any], **kwargs) -> List[float]:
    """
    Compare each model response to the correct answer and score its accuracy.
    Returns 1.0 if the answer is right, 0.0 if it's wrong or if ground truth parsing fails.
    """

    # Get the text content from each output.
    responses = [item[0]["content"] for item in outputs]
    scores: List[float] = []

    # Get the ground truth answers from the provided arguments.
    ground_truths = kwargs.get("solution")

    if ground_truths is None:
        # If we don't have ground truth answers, score as incorrect.
        print("Warning: No ground truth 'solution' provided for accuracy evaluation.")
        return [0.0] * len(outputs)

    for response, true_value in zip(responses, ground_truths):
        parsed_true = parse(true_value, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])

        if parsed_true:
            # Parse the model's response with detailed settings.
            parsed_response = parse(
                response,
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

            # Compare the parsed response with the true answer.
            try:
                score_value = float(verify(parsed_response, parsed_true))
            except Exception as e:
                print(f"Warning: Error during verification for response '{response}' against truth '{true_value}': {e}")
                score_value = 0.0 

        else:
            score_value = 0.5
            print(f"Warning: Ground truth parsing failed for '{true_value}'. Assigning accuracy score 0.0.")

        scores.append(score_value)

    return scores
    
def evaluate_format(outputs: List[Any], **kwargs) -> List[float]:
    """
    Check if each response is formatted with the proper <think> and <answer> tags.
    Returns 1.0 if the format is right, 0.0 if it's not.
    """
    
    # This regex defines the format we expect.
    format_regex = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    responses = [item[0]["content"] for item in outputs]
    
    # For each response, check if it matches the expected format.
    return [
        1.0 if re.match(format_regex, text, re.DOTALL | re.MULTILINE) else 0.0
        for text in responses
    ]

def evaluate_reasoning_steps(outputs: List[Any], **kwargs) -> List[float]:
    """
    Look for clues in the text that show the model is reasoning through its answer.
    Returns a score (capped at 1.0) based on how many step markers it finds.
    """
    # Define a pattern for common markers of reasoning (such as "Step 1:" or "First,")
    
    step_pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    responses = [item[0]["content"] for item in outputs]
    
    # Count the markers and scale the score.
    return [min(1.0, len(re.findall(step_pattern, text, re.MULTILINE)) / 3) for text in responses]

def create_cosine_reward_func(
    low_bad: float = -0.5,
    high_bad: float = -0.1,
    low_good: float = 0.8,
    high_good: float = 1.0,
    length_limit: int = 1000,
):
    """
    Build and return a reward function that uses a cosine curve to scale rewards based on the response length.
    """
    def cosine_reward(responses: List[Any], truth_list: List[str], accuracy_scores: List[float], **kwargs) -> List[float]:
        texts = [item[0]["content"] for item in responses]
        reward_scores: List[float] = []
        
        for text, truth, acc_score in zip(texts, truth_list, accuracy_scores):
            text_len = len(text)
            
            # See how far along the text is relative to our maximum length.
            progress = text_len / length_limit
            
            cosine_val = math.cos(progress * math.pi)  # This gives us a smooth scaling factor.
            
            # Choose reward bounds based on whether the answer was accurate.
            if acc_score > 0.5:
                low_bound = low_good
                high_bound = high_good
            else:
                low_bound = high_bad
                high_bound = low_bad
                
            # Compute the reward based on our cosine scaling.
            reward_value = low_bound + 0.5 * (high_bound - low_bound) * (1.0 + cosine_val)
            reward_scores.append(float(reward_value))
            
        return reward_scores
    return cosine_reward

def create_repetition_penalty_func(ngram: int = 3, penalty_value: float = -0.1) -> Callable:
    """
    Build and return a function that applies a penalty when a response has too many repeated n-grams.
    """
    if penalty_value > 0:
        raise ValueError("Penalty value must be negative.")

    def extract_ngrams(text: str, n: int) -> List[tuple]:
        # Convert the text to lowercase and split it into words.
        words = text.lower().split()
        
        # If there aren't enough words, return an empty list.
        if len(words) < n:
            return []
            
        # Return a list of n-grams (tuples of consecutive words).
        return list(zip(*[words[i:] for i in range(n)]))
    
    def repetition_penalty(responses: List[Any], **kwargs) -> List[float]:
        texts = [item[0]["content"] for item in responses]
        penalties: List[float] = []
        
        for text in texts:
            # If the text is too short, there's nothing to penalize.
            if not text or len(text.split()) < ngram:
                penalties.append(0.0)
                continue
            
            unique_ngrams = set(extract_ngrams(text, ngram))
            total = len(text.split()) - ngram + 1
            
            # Calculate how much repetition there is.
            repeat_ratio = 1 - (len(unique_ngrams) / total) if total > 0 else 0
            penalties.append(repeat_ratio * penalty_value)
            
        return penalties
    return repetition_penalty
