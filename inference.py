# inference.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
from config import OUTPUT_DIR_GRPO  # Or change to OUTPUT_DIR_SFT if needed

def load_trained_model(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    """
    Loads a trained model and its tokenizer from the specified path.
    Returns the model, tokenizer, and device.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

def run_inference_test(input_text: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: torch.device) -> str:
    """
    Runs a test inference using the loaded model and returns the generated response.
    """
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_text}
    ]
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    input_tensor = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(**input_tensor, max_new_tokens=200, do_sample=True, temperature=0.7)
    response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response_text

if __name__ == "__main__":
    chosen_model_path = OUTPUT_DIR_GRPO  # Change to OUTPUT_DIR_SFT if you prefer that model
    loaded_model, loaded_tokenizer, device_used = load_trained_model(chosen_model_path)
    test_query = "how are you?"
    result_response = run_inference_test(test_query, loaded_model, loaded_tokenizer, device_used)
    print(f"Test Input: {test_query}")
    print(f"Generated Response: {result_response}")
