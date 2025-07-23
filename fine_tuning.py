# fine_tuning.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from dataset_preparation import load_and_format_math_data
from settings import MODEL_REF, FT_OUTPUT_DIR

def run_ft_training(input_model_path: str = MODEL_REF, output_dir: str = None, train_dataset=None) -> None:
    """
    Run the fine-tuning training loop using our math dataset.
    Loads the RL‑trained base model, tokenizes prompts+solutions,
    and fine‑tunes via SFTTrainer.
    """
    # Use default output dir if not specified
    if output_dir is None:
        output_dir = FT_OUTPUT_DIR
        
    print(f"Starting fine-tuning using model from: {input_model_path}")
    print(f"Output directory: {output_dir}")

    # Load dataset if not provided
    if train_dataset is None:
        math_data = load_and_format_math_data()
        ft_dataset = math_data["train"]
    else:
        ft_dataset = train_dataset

    # Load tokenizer
    ft_tokenizer = AutoTokenizer.from_pretrained(
        input_model_path, trust_remote_code=True, padding_side="right"
    )
    if ft_tokenizer.pad_token is None:
        ft_tokenizer.pad_token = ft_tokenizer.eos_token

    # Tokenization function: flatten prompt messages and append solution
    def tokenize_fn(example):
        conv = ""
        for msg in example["prompt"]:
            conv += msg["content"]
        conv += example["solution"]
        tok = ft_tokenizer(conv, truncation=True, padding="max_length")
        tok["labels"] = tok["input_ids"].copy()
        return tok

    # Apply tokenization and remove original columns
    tokenized_ds = ft_dataset.map(
        tokenize_fn,
        remove_columns=ft_dataset.column_names,
    )

    # Set up training arguments
    ft_training_args = TrainingArguments(
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
        report_to="none",
    )

    # Load model
    ft_model = AutoModelForCausalLM.from_pretrained(
        input_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    )

    # Initialize and run SFTTrainer
    ft_trainer = SFTTrainer(
        model=ft_model,
        train_dataset=tokenized_ds,
        tokenizer=ft_tokenizer,
        args=ft_training_args,
    )

    ft_trainer.train()
    print("Fine-tuning training completed.")

    # Save artifacts
    ft_tokenizer.save_pretrained(output_dir)
    ft_trainer.save_model(output_dir)
    print(f"Fine-tuned model and tokenizer stored at {output_dir}")


if __name__ == "__main__":
    run_ft_training()
