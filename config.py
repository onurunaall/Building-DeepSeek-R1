import os
from dataclasses import dataclass, field
from typing import Optional, List

BASE_MODEL_IDENTIFIER: str = "Qwen/Qwen2.5-0.5B-Instruct"

GRPO_OUTPUT_DIRECTORY: str = os.path.join("data", "Qwen-GRPO-training")
SFT_OUTPUT_DIRECTORY: str = os.path.join("data", "Qwen-SFT-training")

DEFAULT_SYSTEM_PROMPT: str = (
    "A conversation between a User and an Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in its mind and then provides the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively."
)

@dataclass
class GRPOTrainingConfig:
    # List of reward function names to use during GRPO training
    reward_function_names: List[str] = field(default_factory=lambda: ["accuracy", "format"])
    
    # Cosine reward parameters for incorrect answers
    cosine_min_incorrect: float = -0.5
    cosine_max_incorrect: float = -0.1
    
    # Cosine reward parameters for correct answers
    cosine_min_correct: float = 0.8
    cosine_max_correct: float = 1.0
    
    # Maximum length considered for scaling the cosine reward
    cosine_max_length: int = 1000
    
    # N-gram size for applying repetition penalty
    ngram_size_for_repetition: int = 3
    
    # Maximum penalty value for repeated n-grams (negative value)
    repetition_max_penalty: float = -0.1

@dataclass
class ModelInitializationConfig:
    model_identifier: str = BASE_MODEL_IDENTIFIER
    model_revision: Optional[str] = "main"
    model_dtype: Optional[str] = "bfloat16"
    allow_remote_code: bool = True
    
    # Type of attention mechanism to use
    attention_implementation: Optional[str] = "flash_attention_2"
