import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
from settings import MODEL_REF

def setup_tokenizer() -> AutoTokenizer:
    """
    Initialize and return the tokenizer for the base model.
    - Ensures that the pad token is defined, using the EOS token if necessary.
    """

    tokenizer_obj = AutoTokenizer.from_pretrained(MODEL_REF, trust_remote_code=True, padding_side="right")
    
    # Check if a pad token exists; if not, assign the EOS token as the pad token.
    if tokenizer_obj.pad_token is None:
        tokenizer_obj.pad_token = tokenizer_obj.eos_token
    return tokenizer_obj

def setup_model() -> Tuple[AutoModelForCausalLM, torch.device]:
    """
    Initialize the base model and transfer it to the appropriate device (GPU/CPU).
    - Returns both the model and the device it is assigned to.
    """
    
    # Load the model from the specified reference with a chosen data type.
    model_obj = AutoModelForCausalLM.from_pretrained(MODEL_REF, trust_remote_code=True, torch_dtype=torch.bfloat16)
    
    # Select GPU if available; otherwise, default to CPU.
    device_used = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_obj.to(device_used)
    
    return model_obj, device_used

def test_model_inference(user_text: str, model_obj: AutoModelForCausalLM, tokenizer_obj: AutoTokenizer, device_used: torch.device) -> str:
    """
    Perform a test inference with the model using a simple dialogue.
    - Constructs a conversation, tokenizes it, generates output, and decodes the result.
    - Returns the generated text.
    """
    # Build a dialogue structure with a system and a user message.
    dialogue = [
        {"role": "system", "content": "You are Qwen, a helpful assistant."},
        {"role": "user", "content": user_text}
    ]
    
    # Prepare the prompt from the dialogue using the chat template.
    prompt = tokenizer_obj.apply_chat_template(dialogue, tokenize=False, add_generation_prompt=True)
    
    # Tokenize the prompt and ensure the tensor is on the correct device.
    tokenized_input = tokenizer_obj(prompt, return_tensors="pt").to(device_used)
    
    # Generate a response from the model with specific generation parameters.
    generated_output = model_obj.generate(**tokenized_input, max_new_tokens=100, do_sample=True, temperature=0.7)
    
    # Decode the output tokens to a strings.
    output_string = tokenizer_obj.decode(generated_output[0], skip_special_tokens=True)
    
    return output_string
