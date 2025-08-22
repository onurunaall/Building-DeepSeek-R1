
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
    Generate a model response using the training-consistent chat template.
    """

    # Format messages in chat-style structure
    messages = [{"role": "system", "content": SYSTEM_TEMPLATE}, {"role": "user", "content": user_input}]

    # Tokenize messages with generation prompt
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Move tensors to the correct device (CPU/GPU)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    # Track prompt length for slicing later
    prompt_length = inputs["input_ids"].shape[1]

    # Generate response tokens
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
    )

    # Remove prompt tokens, keep only generated part
    generated_tokens = output_tokens[0][prompt_length:]

    # Decode tokens to string response
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response
if __name__ == "__main__":
    model_dir = RL_OUTPUT_DIR
    model, tokenizer, device = load_saved_model(model_dir)

    sample_query = "how are you?"
    reply = get_model_response(sample_query, model, tokenizer, device)

    print(f"Input: {sample_query}")
    print(f"Response: {reply}")
