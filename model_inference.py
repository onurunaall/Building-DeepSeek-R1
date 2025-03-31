import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
from settings import RL_OUTPUT_DIR, FT_OUTPUT_DIR

def load_saved_model(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
    """
    Load a saved model and its tokenizer from the given path.
    Returns the model, tokenizer, and the device (GPU or CPU).
    """
  
    # Setup the tokenizer with appropriate configuration
    tokenizer_instance = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="right")
  
    if tokenizer_instance.pad_token is None:
        tokenizer_instance.pad_token = tokenizer_instance.eos_token

    model_instance = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    
    device_used = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance.to(device_used)
  
    return model_instance, tokenizer_instance, device_used

def get_model_response(user_input: str, model_instance: AutoModelForCausalLM, tokenizer_instance: AutoTokenizer, device_used: torch.device) -> str:
    """
    Generate a response from the model using a simple dialogue template.
    Returns the generated text.
    """
    # Build the conversation structure with system and user messages
    dialogue = [
        {"role": "system", "content": "You are a friendly assistant."},
        {"role": "user", "content": user_input}
    ]
  
    # Prepare the prompt text using the tokenizer's chat function
    prompt_text = tokenizer_instance.apply_chat_template(dialogue, tokenize=False, add_generation_prompt=True)
    input_data = tokenizer_instance(prompt_text, return_tensors="pt").to(device_used)
  
    # Generate a response from the model
    output_tokens = model_instance.generate(**input_data, max_new_tokens=200, do_sample=True, temperature=0.7)
    response_text = tokenizer_instance.decode(output_tokens[0], skip_special_tokens=True)
  
    return response_text

if __name__ == "__main__":
    # Use the RL output directory; adjust if using the fine-tuned model
    model_dir = RL_OUTPUT_DIR
    model, tokenizer, device = load_saved_model(model_dir)
  
    sample_query = "how are you?"
    reply = get_model_response(sample_query, model, tokenizer, device)
  
    print(f"Input: {sample_query}")
    print(f"Response: {reply}")
