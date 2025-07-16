# fine_tuning_seed.py

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)
from trl import SFTTrainer
from dataset_preparation import load_and_format_math_data, split_seed_refine
from settings import MODEL_REF, FT_OUTPUT_DIR

def run_seed_ft_training(base_model_path: str = MODEL_REF, seed_frac: float = 0.1, output_dir: str = None):
    """
    Run supervised fine-tuning on a small 'seed' subset of the math dataset.
    Splits the full train set using seed_frac, then fine-tunes only on that subset.
    """
    # Determine output directory for seed FT
    if output_dir is None:
        output_dir = os.path.join(FT_OUTPUT_DIR, "seed")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting seed SFT (fraction={seed_frac}) from {base_model_path}")
    print(f"Saving seed-fine-tuned model to: {output_dir}")

    # 1. Load and split dataset
    math_data = load_and_format_math_data()
    full_train = math_data["train"]
    seed_set, refine_set = split_seed_refine(full_train, seed_frac=seed_frac)

    # 2. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Tokenization function
    def tokenize_fn(example):
        text = "".join(m["content"] for m in example["prompt"])
        text += example["solution"]
        toks = tokenizer(text, truncation=True, padding="max_length")
        toks["labels"] = toks["input_ids"].copy()
        return toks

    # 4. Apply tokenization
    tokenized_ds = seed_set.map(tokenize_fn, remove_columns=seed_set.column_names)

    # 5. Training arguments
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
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        dataloader_num_workers=2,
        seed=42,
        bf16=True,
        push_to_hub=False,
        gradient_checkpointing=True,
        report_to="none")

    # 6. Load model (fixed torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

    # 7. Initialize trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        args=training_args
    )

    # 8. Train
    trainer.train()
    print("Seed SFT completed.")

    # 9. Save artifacts
    tokenizer.save_pretrained(output_dir)
    trainer.save_model(output_dir)
    print(f"Seed-fine-tuned model saved at {output_dir}")
    return refine_set


if __name__ == "__main__":
    run_seed_ft_training()