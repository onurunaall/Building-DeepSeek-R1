import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
from settings import MODEL_REF

def setup_tokenizer() -> AutoTokenizer:
    """
    Initialize the tokenizer for the base model.
    """
  
    tokenizer_obj = AutoTokenizer.from_pretrained(MODEL_REF, trust_remote_code=True, padding_side="right")
  
    if tokenizer_obj.pad_token is None:
        tokenizer_obj.pad_token = tokenizer_obj.eos_token
      
    return tokenizer_obj

def setup_model() -> Tuple[AutoModelForCausalLM, torch.device]:
    """
    Initialize the base model and move it to GPU if available.
    Returns the model and the assigned device.
    """
    model_obj = AutoModelForCausalLM.from_pretrained(MODEL_REF, trust_remote_code=True, torch_dtype=torch.bfloat16)
  
    device_used = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_obj.to(device_used)
  
    return model_obj, device_used

def test_model_inference(user_text: str, model_obj: AutoModelForCausalLM, tokenizer_obj: AutoTokenizer, device_used: torch.device) -> str:
    """
    Perform a test inference using a simple dialogue.
    Returns the generated output as a string.
    """
    dialogue = [{"role": "system", "content": "You are Qwen, a helpful assistant."},
                {"role": "user", "content": user_text}]
  
    prompt = tokenizer_obj.apply_chat_template(dialogue, tokenize=False, add_generation_prompt=True)
  
    tokenized_input = tokenizer_obj(prompt, return_tensors="pt").to(device_used)
  
    generated_output = model_obj.generate(**tokenized_input, max_new_tokens=100, do_sample=True, temperature=0.7)
  
    output_string = tokenizer_obj.decode(generated_output[0], skip_special_tokens=True)
    return output_string
