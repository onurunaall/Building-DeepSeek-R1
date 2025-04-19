# model_inference.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
from settings import RL_OUTPUT_DIR, SYSTEM_TEMPLATE

def load_saved_model(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    """
    Load a saved model and its tokenizer from the given path.
    Returns the model, tokenizer, and the device (GPU or CPU).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, tokenizer, device

def get_model_response(
    user_input: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device
) -> str:
    """
    Generate a response by concatenating SYSTEM_TEMPLATE and user_input,
    then slicing off the prompt tokens from the model's output.
    """
    prompt_text = SYSTEM_TEMPLATE + "\n" + user_input
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7
    )

    # Slice off the prompt tokens to get only the generated reply
    gen_tokens = output_tokens[0][input_len:]
    response = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    return response

if __name__ == "__main__":
    # Adjust to FT_OUTPUT_DIR if you want the fine‑tuned model
    model_dir = RL_OUTPUT_DIR
    model, tokenizer, device = load_saved_model(model_dir)

    sample_query = "how are you?"
    reply = get_model_response(sample_query, model, tokenizer, device)

    print(f"Input: {sample_query}")
    print(f"Response: {reply}")
