# settings.py

import os
from dataclasses import dataclass, field
from typing import List

MODEL_REF: str = "Qwen/Qwen2.5-0.5B-Instruct"

RL_OUTPUT_DIR: str = os.path.join("data", "Qwen-RL-training")
FT_OUTPUT_DIR: str = os.path.join("data", "Qwen-FT-training")

SYSTEM_TEMPLATE: str = (
    "This is a dialogue between a User and an Assistant. The user poses a query, "
    "and the Assistant resolves it. The Assistant first outlines its thought "
    "process and then gives an answer. The reasoning process and answer are "
    "enclosed within <think> </think> and <answer> </answer> tags, respectively."
)

@dataclass
class RLTrainingSettings:
    # Which reward metrics to use during RL training
    metric_identifiers: List[str] = field(default_factory=lambda: ["accuracy", "format"])

    # Cosine-reward parameters for incorrect responses
    cosine_low_bad: float = -0.5
    cosine_high_bad: float = -0.1

    # Cosine-reward parameters for correct responses
    cosine_low_good: float = 0.8
    cosine_high_good: float = 1.0

    # Max length for scaling cosine reward
    cosine_length_limit: int = 1000

    # N‑gram size for repetition penalty
    repetition_ngram: int = 3

    # Negative penalty value for repeated phrases
    repetition_penalty_value: float = -0.1
