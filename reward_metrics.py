import math
from typing import List, Any, Callable, Tuple
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

def _verify_math_expression(response_text: str, true_value_text: str) -> float:
    """
    Helper function to parse and verify a single mathematical response against its ground truth.
    Returns 1.0 if correct, 0.5 if ground truth parsing fails, 0.0 if incorrect/error.
    Configurations are created explicitly for readability.
    """
    try:
        ground_truth_extraction_config = LatexExtractionConfig()

        # Attempt to parse the ground truth solution first
        parsed_true = parse(true_value_text,
                            extraction_mode="first_match",
                            extraction_config=[ground_truth_extraction_config])
        
        if parsed_true:
            # Define the normalization settings for parsing the model's response
            response_normalization_config = NormalizationConfig(nits=False,
                                                                malformed_operators=False,
                                                                basic_latex=True,
                                                                equations=True,
                                                                boxed="all",
                                                                units=True)

            # Define the extraction settings for the model's response, using the normalization settings
            response_extraction_config = LatexExtractionConfig(normalization_config=response_normalization_config,
                                                               boxed_match_priority=0,
                                                               try_extract_without_anchor=False)

            parsed_response = parse(response_text,
                                    extraction_config=[response_extraction_config],
                                    extraction_mode="first_match")

           
            # Verify the parsed response against the parsed ground truth
            # The verify function should return True (1.0) or False (0.0)
            score_value = float(verify(parsed_response, parsed_true))

        else:
            print(f"Warning: Ground truth parsing failed for '{true_value_text}'. Assigning neutral accuracy score 0.5.")
            score_value = 0.5

    except Exception as e:
        print(f"Warning: Error during accuracy verification for response '{response_text}' against truth '{true_value_text}': {e}")
        score_value = 0.0

    return score_value


def _check_format_tags(text: str) -> bool:
    """
    Helper function to check for <think>...</think><answer>...</answer> structure
    without using regular expressions.
    """
    # Define the required tags
    start_think_tag = "<think>"
    end_think_tag = "</think>"
    start_answer_tag = "<answer>"
    end_answer_tag = "</answer>"

    # Check if the text starts with the opening think tag
    if not text.startswith(start_think_tag):
        return False

    # Find the positions of the essential tags in order
    try:
        # Find the end of the think block
        think_end_pos = text.index(end_think_tag)

        # Find the start of the answer block *after* the think block ends
        answer_start_pos = text.index(start_answer_tag, think_end_pos + len(end_think_tag))

        # Find the end of the answer block *after* the answer block starts
        answer_end_pos = text.index(end_answer_tag, answer_start_pos + len(start_answer_tag))
    
    except ValueError:
        return False

    # Extract the substring between the end of the think tag and the start of the answer tag
    content_between = text[think_end_pos + len(end_think_tag) : answer_start_pos]
    if content_between.strip():
        return False

    # Extract the substring after the end of the answer tag
    content_after = text[answer_end_pos + len(end_answer_tag):]
    if content_after.strip():
        return False

    return True


def _count_reasoning_indicators(text: str) -> int:
    """
    Helper function to count occurrences of reasoning step indicators
    without using regular expressions. Counts indicators line-by-line and word-by-word.
    """
    indicator_count = 0
    lines = text.splitlines()

    line_start_bullet_indicators = {"-", "*"}
    word_indicators = {"first,", "second,", "next,", "finally,"} # Transition words

    for line in lines:
        stripped_line = line.strip()

        if stripped_line.startswith("Step ") and ":" in stripped_line:
            parts = stripped_line.split(":", 1)[0].split(" ") # Get ["Step", "N"] part
            
            if len(parts) == 2 and parts[0] == "Step" and parts[1].isdigit():
                indicator_count += 1
                
                # Found indicator, no need to check other patterns
                continue 

        # Check for "N." pattern at the start (e.g., "1.", "25.")
        if "." in stripped_line:
            parts = stripped_line.split(".", 1)
            
            if parts[0].isdigit() and line.startswith(stripped_line):
                 # Check if it's likely a list item (ends after dot or followed by space)
                 if len(parts) == 1 or not parts[1] or parts[1][0].isspace():
                      indicator_count += 1
                      continue 

        # Check for bullet point indicators at the start (e.g., "- ", "* ")
        if stripped_line:
            first_char = stripped_line[0]
            
            if first_char in line_start_bullet_indicators:
                
                if len(stripped_line) == 1 or (len(stripped_line) > 1 and stripped_line[1].isspace()):
                    indicator_count += 1
                    continue

    # Check for transition words globally 
    words = text.lower().split() 
    for word in words:
        if word in word_indicators:
            indicator_count += 1

    return indicator_count


def _calculate_single_cosine_reward(text_len: int, 
                                    length_limit: int,
                                    acc_score: float,
                                    low_bad: float,
                                    high_bad: float,
                                    low_good: float,
                                    high_good: float) -> float:
    """ Helper function to calculate the cosine scaled reward for one item. """
    progress = (text_len / length_limit) if length_limit > 0 else 0.0

    cosine_val = math.cos(progress * math.pi)

    # Determine the min/max reward bounds based on whether the accuracy score indicates correctness
    # Assuming 0.5 is neutral/fail, >0.5 is pass
    is_correct = acc_score > 0.5 
                                        
    min_reward_bound = low_good if is_correct else high_bad
    max_reward_bound = high_good if is_correct else low_bad

    # Interpolate between min and max reward bounds using the cosine value
    # (1.0 + cosine_val) scales from 0 (at max length) to 2 (at zero length)
    # This means short correct answers get closer to max_reward_bound,
    # and short incorrect answers get closer to min_reward_bound (which is high_bad, less penalty)
    # While long incorrect answers get closer to low_bad (more penalty)
    reward_range = max_reward_bound - min_reward_bound
    scaled_value = 0.5 * reward_range * (1.0 + cosine_val)
    final_reward = min_reward_bound + scaled_value

    return float(final_reward)


def _extract_ngrams_helper(text: str, n: int) -> List[Tuple[str, ...]]:
    """
    Helper function to generate n-grams from text without complex list comprehensions.
    """
    words = text.lower().split()
    if len(words) < n:
        return []

    ngrams_list: List[Tuple[str]] = []
    last_start_index = len(words) - n
    for i in range(last_start_index + 1):
        ngram_tuple = tuple(words[i : i + n])
        ngrams_list.append(ngram_tuple)

    return ngrams_list


def _calculate_single_repetition_penalty(text: str, ngram: int, penalty_value: float) -> float:
    """ Helper function to calculate the repetition penalty for one item. """
    words = text.lower().split()
    if not text or len(words) < ngram:
        return 0.0

    # Extract all n-grams from the text using the helper function
    all_ngrams_list = _extract_ngrams_helper(text, ngram) # Use the already split words if efficient
    total_ngram_count = len(all_ngrams_list)

    # If no n-grams were generated (should not happen with check above, but safety first)
    if total_ngram_count == 0:
        return 0.0

    # Find the number of unique n-grams by converting the list to a set
    unique_ngram_count = len(set(all_ngrams_list))

    # Calculate the fraction of n-grams that are unique
    fraction_unique = unique_ngram_count / total_ngram_count

    # The repetition ratio is 1 minus the fraction that are unique
    # E.g., if all ngrams are unique, fraction_unique=1, repetition_ratio=0
    # E.g., if all ngrams are the same, fraction_unique=1/total, repetition_ratio=1-(1/total) -> approaches 1
    repetition_ratio = 1.0 - fraction_unique

    # Calculate the final penalty (which is negative or zero)
    # More repetition (higher ratio) results in a larger penalty (closer to penalty_value)
    final_penalty = repetition_ratio * penalty_value # penalty_value is negative

    return final_penalty


# --- Main Reward Functions ---

def evaluate_accuracy(outputs: List[Any], **kwargs) -> List[float]:
    """
    Compare each model response to the correct answer and score its accuracy.
    Uses a helper function for parsing and verification of each item.
    Returns 1.0 if correct, 0.5 if ground truth parsing fails, 0.0 if incorrect/error.
    """
    # Get the text content from each output dictionary. Assumes format [{"content": "..."}]
    try:
        responses = [item[0]["content"] for item in outputs]
    except (IndexError, KeyError, TypeError):
         print("Error: Unexpected format for 'outputs' in evaluate_accuracy. Expected List[List[Dict[str, Any]]].")
         # Depending on desired robustness, could return [], raise error, or return default scores
         return [0.0] * len(outputs) # Fallback: return 0.0 for all if format is wrong

    # Initialize an empty list to store the score for each response.
    scores: List[float] = []

    # Get the ground truth answers from the keyword arguments.
    ground_truths = kwargs.get("solution")

    # Check if ground truth solutions were actually provided.
    if ground_truths is None:
        # If not, we cannot calculate accuracy. Return 0.0 for all.
        print("Warning: No ground truth 'solution' provided for accuracy evaluation.")
        return [0.0] * len(outputs)

    # Ensure we have the same number of responses and ground truths.
    if len(responses) != len(ground_truths):
         print(f"Warning: Mismatch between number of responses ({len(responses)}) and solutions ({len(ground_truths)}). Cannot calculate accuracy.")
         # Handle mismatch - here returning 0.0 for all as a fallback
         return [0.0] * len(outputs) # Or maybe raise an error?

    # Iterate through each response and its corresponding ground truth.
    for response_text, true_value_text in zip(responses, ground_truths):
        # Call the helper function to get the score for this pair.
        score = _verify_math_expression(response_text, true_value_text)
        # Append the calculated score to our list of scores.
        scores.append(score)

    # Return the list of calculated accuracy scores.
    return scores


def evaluate_format(outputs: List[Any], **kwargs) -> List[float]:
    """
    Check if each response is formatted with the proper <think> and <answer> tags.
    Uses a helper function for checking format without regex.
    Returns 1.0 if the format is right, 0.0 if it's not.
    """
    # Extract the text content from each response item.
    try:
        responses = [item[0]["content"] for item in outputs]
    except (IndexError, KeyError, TypeError):
         print("Error: Unexpected format for 'outputs' in evaluate_format. Expected List[List[Dict[str, Any]]].")
         return [0.0] * len(outputs) # Fallback

    # Prepare a list to hold the format scores (1.0 or 0.0).
    format_scores: List[float] = []

    # Loop through each response text.
    for text in responses:
        # Use the helper function to check if the format is valid.
        is_format_correct = _check_format_tags(text)
        # Assign a score: 1.0 for correct format, 0.0 otherwise.
        score = 1.0 if is_format_correct else 0.0
        # Add the score to our list.
        format_scores.append(score)

    # Return the list of format scores.
    return format_scores


def evaluate_reasoning_steps(outputs: List[Any], **kwargs) -> List[float]:
    """
    Look for clues in the text that show the model is reasoning through its answer.
    Uses a helper function to count indicators without regex.
    Returns a score (capped at 1.0) based on how many step markers it finds (normalized by 3).
    """
    # Extract the text content from each response.
    try:
        responses = [item[0]["content"] for item in outputs]
    except (IndexError, KeyError, TypeError):
         print("Error: Unexpected format for 'outputs' in evaluate_reasoning_steps. Expected List[List[Dict[str, Any]]].")
         return [0.0] * len(outputs) # Fallback

    # Initialize a list to store the reasoning scores.
    reasoning_scores: List[float] = []

    # Process each response text individually.
    for text in responses:
        # Count the number of reasoning indicators using the helper function.
        indicator_count = _count_reasoning_indicators(text)
        # Calculate the score based on the count, normalizing by 3.0 (float division).
        # Cap the score at a maximum of 1.0 using min().
        score = min(1.0, indicator_count / 3.0)
        # Append the score to the results list.
        reasoning_scores.append(score)

    # Return the list containing scores for each response.
    return reasoning_scores


# --- Factory Functions (Returning Reward Calculation Functions) ---

def create_cosine_reward_func(
    low_bad: float = -0.5, high_bad: float = -0.1,
    low_good: float = 0.8, high_good: float = 1.0,
    length_limit: int = 1000,
) -> Callable[[List[Any], List[str], List[float]], List[float]]: # Added input types to Callable hint
    """
    Factory to create a cosine reward function with specific parameters baked in.
    The returned function calculates rewards based on response length and accuracy.
    """
    # Define the function that will be returned.
    # It uses the parameters passed to the factory (low_bad, high_good, etc.)
    def actual_cosine_reward_calculator(
        responses: List[Any], # Expected: List[List[Dict[str, Any]]]
        solution: List[str], # The ground truth solutions
        accuracy_rewards: List[float], # Pre-calculated accuracy scores
        **kwargs # Allow for extra arguments from trainer
    ) -> List[float]:
        """ This inner function performs the actual calculation for a batch. """
        # Extract the text content from the responses.
        try:
            texts = [item[0]["content"] for item in responses]
        except (IndexError, KeyError, TypeError):
            print("Error: Unexpected format for 'responses' in cosine reward calculator.")
            return [] # Return empty if format is wrong

        # Initialize a list for the final reward scores for this batch.
        reward_scores: List[float] = []

        # Ensure input lists have the same length for safe zipping
        if not (len(texts) == len(solution) == len(accuracy_rewards)):
             print("Warning: Mismatch in lengths for cosine reward inputs. Returning empty list.")
             # Decide on error handling: could raise error, return zeros, or empty list.
             return [] # Returning empty list here

        # Iterate through each response, its solution, and its pre-calculated accuracy score.
        for text, _, acc_score in zip(texts, solution, accuracy_rewards): # Solution text isn't used directly here
            # Get the length of the current response text.
            text_len = len(text)
            # Calculate the reward for this single item using the top-level helper.
            # Pass all necessary parameters, including those baked into the factory closure.
            single_reward = _calculate_single_cosine_reward(
                text_len, length_limit, acc_score,
                low_bad, high_bad, low_good, high_good
            )
            # Append the calculated reward to the list.
            reward_scores.append(single_reward)

        # Return the list of calculated scores for the batch.
        return reward_scores

    # Return the calculation function created above.
    return actual_cosine_reward_calculator


def create_repetition_penalty_func(
    ngram: int = 3, penalty_value: float = -0.1
) -> Callable[[List[Any]], List[float]]: # Added input types to Callable hint
    """
    Factory to create a repetition penalty function with specific ngram size and penalty value.
    The returned function penalizes responses with repeated n-grams.
    """
    # Ensure the penalty value is not positive, as it should penalize.
    if penalty_value > 0:
        # Raise an error immediately if the configuration is invalid.
        raise ValueError(f"Penalty value must be non-positive, but got {penalty_value}")

    # Define the function that will be returned, using 'ngram' and 'penalty_value' from the factory scope.
    def actual_repetition_penalty_calculator(
        responses: List[Any], # Expected: List[List[Dict[str, Any]]]
        **kwargs # Allow for extra arguments from trainer
    ) -> List[float]:
        """ This inner function calculates the repetition penalty for a batch. """
        # Extract the content of each response.
        try:
            texts = [item[0]["content"] for item in responses]
        except (IndexError, KeyError, TypeError):
             print("Error: Unexpected format for 'responses' in repetition penalty calculator.")
             return [] # Return empty if format is wrong

        # Initialize a list to store the penalties (negative or zero values) for each response.
        penalties: List[float] = []

        # Calculate the penalty for each text response in the batch.
        for text in texts:
            # Use the top-level helper function to calculate the penalty for this text.
            # Pass the specific ngram size and penalty value captured by the factory closure.
            single_penalty = _calculate_single_repetition_penalty(text, ngram, penalty_value)
            # Append the calculated penalty to the list.
            penalties.append(single_penalty)

        # Return the list of penalties for the batch.
        return penalties

    # Return the calculation function defined above.
    return actual_repetition_penalty_calculator
