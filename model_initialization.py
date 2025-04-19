# model_initialization.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
from settings import MODEL_REF

def setup_tokenizer() -> AutoTokenizer:
    """
    Initialize and return the tokenizer for the base model.
    Ensures that the pad token is defined, using the EOS token if necessary.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REF, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def setup_model() -> Tuple[AutoModelForCausalLM, torch.device]:
    """
    Initialize the base model and transfer it to the appropriate device (GPU/CPU).
    Returns both the model and the device it is assigned to.
    """
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REF, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device
