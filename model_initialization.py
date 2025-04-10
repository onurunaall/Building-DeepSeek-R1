import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
from settings import MODEL_REF

def setup_tokenizer() -> AutoTokenizer:
    """
    Initialize and return the tokenizer for the base model.
    - Ensures that the pad token is defined, using the EOS token if necessary.
    """

    tokenizer_obj = AutoTokenizer.from_pretrained(MODEL_REF, trust_remote_code=True, padding_side="right") #

    # Check if a pad token exists; if not, assign the EOS token as the pad token. #
    if tokenizer_obj.pad_token is None: #
        tokenizer_obj.pad_token = tokenizer_obj.eos_token #
    return tokenizer_obj #

def setup_model() -> Tuple[AutoModelForCausalLM, torch.device]:
    """
    Initialize the base model and transfer it to the appropriate device (GPU/CPU).
    - Returns both the model and the device it is assigned to.
    """

    # Load the model from the specified reference with a chosen data type. #
    model_obj = AutoModelForCausalLM.from_pretrained(MODEL_REF, trust_remote_code=True, torch_dtype=torch.bfloat16) #

    # Select GPU if available; otherwise, default to CPU. #
    device_used = torch.device("cuda" if torch.cuda.is_available() else "cpu") #
    model_obj.to(device_used) #

    return model_obj, device_used #
