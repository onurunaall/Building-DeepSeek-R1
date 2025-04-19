from datasets import load_dataset
from typing import Any, Dict
from settings import SYSTEM_TEMPLATE

def convert_example_to_dialogue(raw_example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a raw dataset record into a dialogue format.
    Preserves 'problem' and 'solution', and wraps the math problem
    with a system instruction and a user message under 'prompt'.
    """
    return {
        "problem": raw_example["problem"],
        "solution": raw_example["solution"],
        "prompt": [
            {"role": "system", "content": SYSTEM_TEMPLATE},
            {"role": "user",   "content": raw_example["problem"]},
        ],
    }

def load_and_format_math_data() -> Dict[str, Any]:
    """
    Load the math problem dataset and apply dialogue formatting.
    Returns a dict with 'train' and 'test' datasets, each containing
    'problem', 'solution', and 'prompt' columns.
    """
    raw_data = load_dataset("AI-MO/NuminaMath-TIR", "default", split=['train', 'test'])
    split_data: Dict[str, Any] = {"train": raw_data[0], "test": raw_data[1]}

    for part in split_data:
        # Map each record to include only the fields we want
        split_data[part] = split_data[part].map(convert_example_to_dialogue)
        # Remove legacy 'messages' column if present
        if "messages" in split_data[part].column_names:
            split_data[part] = split_data[part].remove_columns("messages")

    return split_data

def check_dataset_integrity(formatted_data: Dict[str, Any]) -> None:
    """
    Verify that each split has 'prompt' and 'solution' columns,
    and that the first record's 'prompt' follows [system, user] roles.
    """
    required_keys = ["prompt", "solution"]
    for section in ['train', 'test']:
        print(f"\nChecking integrity of '{section}' split:")
        cols = formatted_data[section].column_names
        missing = [k for k in required_keys if k not in cols]
        if missing:
            print(f"Alert: Missing keys: {missing}")
        else:
            print("✓ All necessary keys are present")

        first = formatted_data[section][0]
        dialogue = first.get("prompt", [])
        if (
            isinstance(dialogue, list)
            and len(dialogue) >= 2
            and dialogue[0].get("role") == "system"
            and dialogue[1].get("role") == "user"
        ):
            print("✓ Dialogue structure is valid")
        else:
            print("Alert: Dialogue structure is invalid")
