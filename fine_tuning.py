import torch
from functools import partial
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from dataset_preparation import load_and_format_math_data
from settings import MODEL_REF, FT_OUTPUT_DIR


def _create_tokenized_dataset(example: Dict[str, Any],
                              tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
    Applies the tokenizer’s chat template to a single training example.

    This function formats the dialogue and prepares it with the necessary
    'input_ids', 'attention_mask', and 'labels' for supervised fine-tuning.

    Args:
        example: Dictionary containing 'prompt' (list of message dicts) and 'solution' (str)
        tokenizer: The tokenizer to use for encoding

    Returns:
        Dictionary with 'input_ids', 'attention_mask', and 'labels' tensors

    Raises:
        ValueError: If example is missing required fields
        RuntimeError: If tokenization fails
    """
    if not isinstance(example.get("prompt"), list):
        raise ValueError("Example must contain 'prompt' field with list of message dictionaries")
    if not isinstance(example.get("solution"), str):
        raise ValueError("Example must contain 'solution' field with string content")
    if not example["solution"].strip():
        raise ValueError("Solution field cannot be empty")

    # Construct the full conversation, including the assistant's solution
    full_conversation = example["prompt"] + [{"role": "assistant", "content": example["solution"]}]

    max_length = getattr(tokenizer, 'model_max_length', None)
    if max_length is None or max_length > 100000:
        max_length = 2048

    try:
        tokenized_inputs = tokenizer.apply_chat_template(
            full_conversation,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to tokenize conversation: {e}")

    # Verify expected keys exist
    required_keys = ["input_ids", "attention_mask"]
    missing_keys = [key for key in required_keys if key not in tokenized_inputs]
    if missing_keys:
        raise RuntimeError(f"Tokenizer output missing expected keys: {missing_keys}")

    return {
        "input_ids": tokenized_inputs["input_ids"].squeeze(),
        "attention_mask": tokenized_inputs["attention_mask"].squeeze(),
        "labels": tokenized_inputs["input_ids"].squeeze().clone(),
    }


def _create_tokenized_dataset_efficient(example: Dict[str, Any],
                                        tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
    ALTERNATIVE IMPLEMENTATION: More memory-efficient version without max_length padding.

    This version doesn't pad to max_length during preprocessing, allowing for
    dynamic padding during training which is more memory efficient.
    """
    if not isinstance(example.get("prompt"), list):
        raise ValueError("Example must contain 'prompt' field with list of message dictionaries")
    if not isinstance(example.get("solution"), str):
        raise ValueError("Example must contain 'solution' field with string content")

    full_conversation = example["prompt"] + [{"role": "assistant", "content": example["solution"]}]

    max_length = getattr(tokenizer, 'model_max_length', 2048)
    if max_length > 100000:
        max_length = 2048

    try:
        tokenized_inputs = tokenizer.apply_chat_template(
            full_conversation,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to tokenize conversation: {e}")

    return {
        "input_ids": tokenized_inputs["input_ids"].squeeze(),
        "attention_mask": tokenized_inputs["attention_mask"].squeeze(),
        "labels": tokenized_inputs["input_ids"].squeeze().clone(),
    }


def run_ft_training(input_model_path: str = MODEL_REF,
                    output_dir: str = FT_OUTPUT_DIR,
                    train_dataset=None,
                    use_efficient_tokenization: bool = True) -> None:
    """
    Runs a full supervised fine-tuning (SFT) loop.
    """
    print("--- Starting Baseline: Full Supervised Fine-Tuning ---")
    print(f"Model: {input_model_path}")
    print(f"Output Directory: {output_dir}")
    print(f"Efficient tokenization: {use_efficient_tokenization}")

    # Step 1: Load Dataset
    if train_dataset is None:
        print("No dataset provided, loading 'NuminaMath-TIR'...")
        try:
            math_data = load_and_format_math_data()
            ft_dataset = math_data["train"]
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
    else:
        ft_dataset = train_dataset

    print(f"Loaded {len(ft_dataset)} training examples.")

    # Step 2: Tokenizer
    try:
        ft_tokenizer = AutoTokenizer.from_pretrained(
            input_model_path,
            trust_remote_code=True,
            padding_side="right"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")

    if ft_tokenizer.pad_token is None:
        ft_tokenizer.pad_token = ft_tokenizer.eos_token
        print(f"Set pad_token to eos_token: {ft_tokenizer.eos_token}")

    if use_efficient_tokenization:
        tokenization_function = partial(
            _create_tokenized_dataset_efficient,
            tokenizer=ft_tokenizer
        )
        print("Using memory-efficient tokenization (dynamic padding)")
    else:
        tokenization_function = partial(
            _create_tokenized_dataset,
            tokenizer=ft_tokenizer
        )
        print("Using standard tokenization (max-length padding)")

    print("Tokenizing dataset...")
    try:
        tokenized_ds = ft_dataset.map(
            tokenization_function,
            remove_columns=ft_dataset.column_names,
            desc="Tokenizing examples"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to tokenize dataset: {e}")

    print(f"Tokenized dataset size: {len(tokenized_ds)}")

    # Step 3: Training Config
    training_config = {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "bf16": True,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 2,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 2,
        "seed": 42,
        "report_to": "none",
    }

    ft_training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        **training_config
    )

    # Step 4: Load Model
    print("Loading base model...")
    try:
        ft_model = AutoModelForCausalLM.from_pretrained(
            input_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    total_params = sum(p.numel() for p in ft_model.parameters())
    trainable_params = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
    print(f"Model loaded: {total_params:,} total parameters, {trainable_params:,} trainable")

    # Step 5: Trainer
    try:
        ft_trainer = SFTTrainer(
            model=ft_model,
            train_dataset=tokenized_ds,
            processing_class=ft_tokenizer,
            args=ft_training_args
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize SFTTrainer: {e}")

    print("Starting fine-tuning...")
    try:
        ft_trainer.train()
        print("Fine-tuning training completed successfully.")
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")

    # Step 6: Save Model
    print(f"Saving fine-tuned model and tokenizer to {output_dir}...")
    try:
        ft_trainer.save_model(output_dir)
        ft_tokenizer.save_pretrained(output_dir)
        print("Model and tokenizer saved successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to save model: {e}")

    print("--- Baseline SFT Finished ---")


def test_tokenization(model_path: str = MODEL_REF) -> None:
    """
    Utility function to verify tokenization works correctly.
    """
    print("Testing tokenization...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        test_example = {
            "prompt": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            "solution": "2+2 equals 4."
        }

        result1 = _create_tokenized_dataset(test_example, tokenizer)
        result2 = _create_tokenized_dataset_efficient(test_example, tokenizer)

        print("Standard tokenization: SUCCESS")
        print(f"Input IDs shape: {result1['input_ids'].shape}")
        print(f"Attention mask shape: {result1['attention_mask'].shape}")

        print("Efficient tokenization: SUCCESS")
        print(f"Input IDs shape: {result2['input_ids'].shape}")
        print(f"Attention mask shape: {result2['attention_mask'].shape}")

        print("All tokenization tests passed!")

    except Exception as e:
        print(f"Tokenization test failed: {e}")
        raise


if __name__ == "__main__":
    test_tokenization()
    run_ft_training()