def compute_accuracy_reward(completions: list, **kwargs) -> list:
    """
    Computes an accuracy reward by comparing the model's output to the ground truth solution.
    Returns a list of rewards (1.0 for correct, 0.0 for incorrect, 0.5 for unparsed).
    """
    output_texts = [comp[0]["content"] for comp in completions]
    rewards: list = []
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
