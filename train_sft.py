# train_sft.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from config import BASE_MODEL_NAME, OUTPUT_DIR_SFT

def execute_sft_training() -> None:
    """
    Executes Supervised Fine-Tuning (SFT) training using a curated dataset.
    """
    # Load the SFT dataset (Bespoke-Stratos-17k)
    sft_dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", "default", split="train")
    
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set up SFT training arguments
    sft_training_args = TrainingArguments(
        output_dir=OUTPUT_DIR_SFT,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
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
    
    # Load the base model for SFT training
    sft_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16)
    
    # Initialize SFT Trainer
    sft_trainer = SFTTrainer(
        model=sft_model,
        train_dataset=sft_dataset,
        tokenizer=tokenizer,
        args=sft_training_args,
    )
    
    # Start SFT training
    sft_trainer.train()
    print("SFT training is complete.")
    
    # Save the fine-tuned model and tokenizer
    tokenizer.save_pretrained(OUTPUT_DIR_SFT)
    sft_trainer.save_model(OUTPUT_DIR_SFT)
    print(f"SFT trained model and tokenizer saved to {OUTPUT_DIR_SFT}")

if __name__ == "__main__":
    execute_sft_training()
