import os
import torch
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from dataset_preparation import load_and_format_math_data, split_seed_refine
from settings import MODEL_REF, FT_OUTPUT_DIR

def _create_tokenized_dataset(example, tokenizer):
    """
    Applies the tokenizer's chat template to a single training example.
    """
    # Construct the full conversation, including the assistant's solution
    full_conversation = example["prompt"] + [{"role": "assistant", "content": example["solution"]}]

    # Apply the tokenizer's built-in chat template
    tokenized_inputs = tokenizer.apply_chat_template(full_conversation,
                                                     truncation=True,
                                                     padding="max_length",
                                                     return_tensors="pt")

    return {
        "input_ids": tokenized_inputs["input_ids"].squeeze(),
        "attention_mask": tokenized_inputs["attention_mask"].squeeze(),
        "labels": tokenized_inputs["input_ids"].squeeze().clone(),
    }


def run_seed_ft_training(base_model_path: str = MODEL_REF,
                         seed_frac: float = 0.1,
                         output_dir: str = None,
                         train_dataset=None):
    """
    Runs supervised fine-tuning on a small 'seed' subset of the dataset.

    This function first splits the training data, then fine-tunes the model
    only on the smaller seed portion. The remaining data is returned for a
    subsequent training stage.

    Args:
        base_model_path: The identifier of the base model to fine-tune.
        seed_frac: The fraction of the training data to use as the seed set.
        output_dir: Directory to save the seed-tuned model.
        train_dataset: An optional pre-loaded dataset to be split.

    Returns:
        The remaining 'refine' portion of the dataset for subsequent use.
    """
    print("--- Starting Experiment Stage 1: Seed Fine-Tuning ---")
    print(f"Base Model: {base_model_path}")

    # Step 1: Prepare Directories
    if output_dir is None:
        output_dir = os.path.join(FT_OUTPUT_DIR, "seed")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output Directory: {output_dir}")

    if train_dataset is None:
        print("No dataset provided, loading 'NuminaMath-TIR'...")
        math_data = load_and_format_math_data()
        full_train_dataset = math_data["train"]
    else:
        full_train_dataset = train_dataset

    # Step 2: Split Dataset into Seed and Refine Sets
    print(f"Splitting dataset into {seed_frac*100}% seed and {100-seed_frac*100}% refine sets...")
    seed_set, refine_set = split_seed_refine(full_train_dataset, seed_frac=seed_frac)
    print(f"Seed set size: {len(seed_set)}, Refine set size: {len(refine_set)}")

    # Step 3: Initialize Tokenizer and Prepare Seed Dataset
    tokenizer = AutoTokenizer.from_pretrained(base_model_path,
                                              trust_remote_code=True,
                                              padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create a partial function for tokenization
    tokenization_function = partial(_create_tokenized_dataset, tokenizer=tokenizer)

    print("Tokenizing seed dataset...")
    tokenized_ds = seed_set.map(tokenization_function,
                                remove_columns=seed_set.column_names)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        seed=42,
        report_to="none",
    )

    model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                                 trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16)

    # Initialize and Execute the Trainer on the Seed Set
    trainer = SFTTrainer(model=model,
                       train_dataset=tokenized_ds,
                       tokenizer=tokenizer,
                       args=training_args)

    print("Starting seed training...")
    trainer.train()
    print("Seed fine-tuning completed successfully.")

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Seed-fine-tuned model saved at {output_dir}")

    # Return the Unused Portion of the Data
    print("--- Seed Fine-Tuning Finished ---")
    return refine_set


if __name__ == "__main__":
    run_seed_ft_training()
