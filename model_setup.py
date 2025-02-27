# model_setup.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
from config import BASE_MODEL_NAME

def initialize_tokenizer() -> AutoTokenizer:
    """
    Loads the tokenizer for the base model and ensures the pad token is set.
    """
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def initialize_model() -> Tuple[AutoModelForCausalLM, torch.device]:
    """
    Loads the base model and moves it to GPU if available.
    Returns the model and the device.
    """
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device

def perform_inference_test(input_text: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: torch.device) -> str:
    """
    Runs a simple inference test on the model using the provided input text.
    Returns the generated output as a string.
    """
    conversation = [
        {"role": "system", "content": "You are Qwen, a helpful assistant."},
        {"role": "user", "content": input_text}
    ]
    prompt_text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt_text, return_tensors="pt").to(device)
    generated_ids = model.generate(**input_ids, max_new_tokens=100, do_sample=True, temperature=0.7)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return output_text
