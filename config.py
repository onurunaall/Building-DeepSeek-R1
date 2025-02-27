# config.py
import os
from dataclasses import dataclass, field
from typing import Optional, List

# Global configuration constants with descriptive names and type annotations
BASE_MODEL_NAME: str = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR_GRPO: str = os.path.join("data", "Qwen-GRPO-training")
OUTPUT_DIR_SFT: str = os.path.join("data", "Qwen-SFT-training")
CONVERSATION_SYSTEM_PROMPT: str = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively."
)

@dataclass
class GRPOConfigArgs:
    # List of reward function names to be used during GRPO training
    reward_function_names: List[str] = field(default_factory=lambda: ["accuracy", "format"])
    # Parameters for the cosine scaled reward function
    cosine_min_wrong: float = -0.5
    cosine_max_wrong: float = -0.1
    cosine_min_correct: float = 0.8
    cosine_max_correct: float = 1.0
    cosine_max_length: int = 1000
    # Parameters for repetition penalty reward
    ngram_size_for_repetition: int = 3
    repetition_max_penalty: float = -0.1

@dataclass
class ModelInitializationConfig:
    model_identifier: str = BASE_MODEL_NAME
    model_revision: Optional[str] = "main"
    model_dtype: Optional[str] = "bfloat16"
    allow_remote_code: bool = True
    attention_implementation: Optional[str] = "flash_attention_2"
