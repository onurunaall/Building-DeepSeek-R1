import os
from dataclasses import dataclass, field
from typing import Optional, List

MODEL_REF: str = "Qwen/Qwen2.5-0.5B-Instruct"

RL_OUTPUT_DIR: str = os.path.join("data", "Qwen-RL-training")
FT_OUTPUT_DIR: str = os.path.join("data", "Qwen-FT-training")

SYSTEM_TEMPLATE: str = (
    "This is a dialogue between a User and an Assistant. The user poses a query, and the Assistant resolves it. "
    "The Assistant first outlines its thought process and then gives an answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively."
)

@dataclass
class RLTrainingSettings:
    # List of reward metric identifiers to be used during reinforcement training
    metric_identifiers: List[str] = field(default_factory=lambda: ["accuracy", "format"])
  
    # Cosine-based reward parameters for incorrect responses
    cosine_low_bad: float = -0.5
    cosine_high_bad: float = -0.1
  
    # Cosine-based reward parameters for correct responses
    cosine_low_good: float = 0.8
    cosine_high_good: float = 1.0
  
    # Maximum length for scaling the cosine reward
    cosine_length_limit: int = 1000
  
    # N-gram setting for repetition penalty computation
    repetition_ngram: int = 3
  
    # Maximum penalty (as a negative value) for repeated phrases
    repetition_penalty_value: float = -0.1

@dataclass
class ModelSetupSettings:
    model_reference: str = MODEL_REF
    revision: Optional[str] = "main"
    weight_dtype: Optional[str] = "bfloat16"
    remote_code_allowed: bool = True
  
    # Implementation of the attention mechanism
    attention_type: Optional[str] = "flash_attention_2"
