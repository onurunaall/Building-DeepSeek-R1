from datasets import load_dataset, Dataset
from typing import Any, Dict, Tuple
from settings import SYSTEM_TEMPLATE

def _convert_to_dialogue_format(raw_example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a single raw dataset record into a structured dialogue format.

    This function preserves the original 'problem' and 'solution' fields
    while creating a new 'prompt' field. The 'prompt' is a list of
    dictionaries representing a conversation, starting with a system
    instruction and followed by the user's problem.

    Args:
        raw_example: A dictionary representing one row from the raw dataset.

    Returns:
        A formatted dictionary with 'problem', 'solution', and 'prompt' fields.
    """
    return {"problem": raw_example["problem"],
            "solution": raw_example["solution"],
            "prompt": [{"role": "system", "content": SYSTEM_TEMPLATE},
                       {"role": "user", "content": raw_example["problem"]},
                      ],
           }

def load_and_format_math_data() -> Dict[str, Dataset]:
    """
    Loads the NuminaMath-TIR dataset and applies the dialogue formatting.

    This is the main data loading function. It fetches the raw dataset from
    Hugging Face Hub, applies the `_convert_to_dialogue_format` to every
    example in both the train and test splits, and cleans up any unnecessary
    columns.

    Returns:
        A dictionary containing the formatted 'train' and 'test' datasets.
    """
    print("Loading raw 'AI-MO/NuminaMath-TIR' dataset...")
    # Load both 'train' and 'test' splits from the hub
    raw_data = load_dataset("AI-MO/NuminaMath-TIR", "default", split=['train', 'test'])
    split_data: Dict[str, Dataset] = {"train": raw_data[0], "test": raw_data[1]}

    print("Applying dialogue format to train and test splits...")
    for split_name in split_data:
        # Apply the conversion function to each example in the dataset
        formatted_dataset = split_data[split_name].map(_convert_to_dialogue_format)

        # Remove the old 'messages' column if it exists to keep the dataset clean
        if "messages" in formatted_dataset.column_names:
            formatted_dataset = formatted_dataset.remove_columns("messages")

        split_data[split_name] = formatted_dataset

    print("Dataset loading and formatting complete.")
    return split_data

def check_dataset_integrity(formatted_data: Dict[str, Dataset]) -> None:
    """
    Performs a basic verification of the formatted dataset's structure.

    This function checks two things:
    1.  That the required 'prompt' and 'solution' columns exist.
    2.  That the 'prompt' field follows the expected [system, user] structure.

    Args:
        formatted_data: The dictionary of formatted datasets to check.
    """
    required_columns = ["prompt", "solution"]

    for split_name, dataset in formatted_data.items():
        print(f"\nVerifying integrity of '{split_name}' split:")
        current_columns = dataset.column_names
        missing_columns = [col for col in required_columns if col not in current_columns]

        if missing_columns:
            print(f"  [Alert] Missing required columns: {missing_columns}")
        else:
            print("  ✓ All required columns are present.")

        # Check the structure of the first example's dialogue
        first_example = dataset[0]
        dialogue = first_example.get("prompt", [])
        is_valid_structure = (
            isinstance(dialogue, list) and
            len(dialogue) >= 2 and
            dialogue[0].get("role") == "system" and
            dialogue[1].get("role") == "user"
        )

        if is_valid_structure:
            print("  ✓ Dialogue structure appears valid.")
        else:
            print("  [Alert] The dialogue structure of the first example is invalid.")

def split_seed_refine(train_dataset: Dataset,
                      seed_frac: float = 0.1,
                      seed: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Splits a training dataset into a 'seed' set and a 'refine' set.

    This is used in the two-stage fine-tuning experiment, where a small
    fraction of the data is used for initial tuning.

    Args:
        train_dataset: The full training dataset to split.
        seed_frac: The fraction of the data to allocate to the seed set.
        seed: A random seed for reproducibility.

    Returns:
        A tuple containing the (seed_set, refine_set).
    """
    # Use the built-in train_test_split method. The 'test' split becomes our seed set.
    split = train_dataset.train_test_split(test_size=seed_frac, seed=seed)

    seed_set = split["test"]
    refine_set = split["train"]

    return seed_set, refine_set
