# data_prep.py
from datasets import load_dataset
from typing import Any, Dict
from config import CONVERSATION_SYSTEM_PROMPT

def format_conversation(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a raw dataset example into a conversation format.
    Wraps the math problem with a system prompt and a user message.
    """
    return {
        "prompt": [
            {"role": "system", "content": CONVERSATION_SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ]
    }

def load_and_prepare_math_dataset() -> Dict[str, Any]:
    """
    Loads the math dataset and applies the conversation formatting.
    Returns a dictionary with 'train' and 'test' splits.
    """
    raw_dataset = load_dataset("AI-MO/NuminaMath-TIR", "default", split=['train', 'test'])
    dataset_splits: Dict[str, Any] = {"train": raw_dataset[0], "test": raw_dataset[1]}
    
    for split in dataset_splits:
        dataset_splits[split] = dataset_splits[split].map(format_conversation)
        # Remove any extra 'messages' column if it exists
        if "messages" in dataset_splits[split].column_names:
            dataset_splits[split] = dataset_splits[split].remove_columns("messages")
    return dataset_splits

def validate_prepared_dataset(dataset: Dict[str, Any]) -> None:
    """
    Checks that the dataset has all required fields and correct conversation format.
    """
    required_fields = ["problem", "prompt"]
    for split in ['train', 'test']:
        print(f"\nValidating {split} split:")
        column_names = dataset[split].column_names
        missing_fields = [field for field in required_fields if field not in column_names]
        if missing_fields:
            print(f"Warning: Missing fields: {missing_fields}")
        else:
            print("✓ All required fields are present")
        sample = dataset[split][0]
        conversation = sample.get("prompt", [])
        if len(conversation) >= 2 and conversation[0].get("role") == "system" and conversation[1].get("role") == "user":
            print("✓ Conversation format is correct")
        else:
            print("Warning: Conversation format is incorrect")
