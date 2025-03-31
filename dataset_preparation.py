from datasets import load_dataset
from typing import Any, Dict
from settings import SYSTEM_TEMPLATE

def convert_example_to_dialogue(raw_example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a raw dataset record into a dialogue format.
    Wrap the math problem with a system instruction and a user message.
    """
  
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user", "content": raw_example["problem"]},
        ]
    }

def load_and_format_math_data() -> Dict[str, Any]:
    """
    Load the math problem dataset and apply dialogue formatting.
    Returns a dictionary with 'train' and 'test' parts.
    """
    raw_data = load_dataset("AI-MO/NuminaMath-TIR", "default", split=['train', 'test'])
    split_data: Dict[str, Any] = {"train": raw_data[0], "test": raw_data[1]}
    
    # Convert each entry into our dialogue format
    for part in split_data:
        split_data[part] = split_data[part].map(convert_example_to_dialogue)
      
        # Remove the 'messages' column if it exists
        if "messages" in split_data[part].column_names:
            split_data[part] = split_data[part].remove_columns("messages")
          
    return split_data

def check_dataset_integrity(formatted_data: Dict[str, Any]) -> None:
    """
    Verify that the dataset has the required fields and correct dialogue structure.
    """
    required_keys = ["problem", "prompt"]
    for section in ['train', 'test']:
        print(f"\nChecking integrity of '{section}' data:")
      
        keys = formatted_data[section].column_names
        missing = [key for key in required_keys if key not in keys]
      
        if missing:
            print(f"Alert: Missing keys: {missing}")
        else:
            print("✓ All necessary keys are present")
          
        # Verify dialogue structure in the first record
        first_record = formatted_data[section][0]
        dialogue = first_record.get("prompt", [])
      
        if len(dialogue) >= 2 and dialogue[0].get("role") == "system" and dialogue[1].get("role") == "user":
            print("✓ Dialogue structure is valid")
        else:
            print("Alert: Dialogue structure is invalid")
