import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from settings import MODEL_REF, FT_OUTPUT_DIR, RL_OUTPUT_DIR

def run_ft_training(input_model_path: str = MODEL_REF) -> None:
    """
    Run the fine-tuning training loop with our curated dataset.
    This function loads the dataset, sets up the model and tokenizer from the
    specified input_model_path (defaulting to MODEL_REF), and starts training.
    """
    print(f"Starting fine-tuning using model from: {input_model_path}")

    ft_dataset = load_dataset("HuggingFaceH4/Bespoke-Stratos-17k", "default", split="train")

    # Load tokenizer from the specified input model path
    ft_tokenizer = AutoTokenizer.from_pretrained(input_model_path, trust_remote_code=True, padding_side="right")

    # Make sure the tokenizer has a pad token; if not, use the EOS token.
    if ft_tokenizer.pad_token is None:
        ft_tokenizer.pad_token = ft_tokenizer.eos_token

    ft_training_args = TrainingArguments(
        output_dir=FT_OUTPUT_DIR,
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

    ft_model = AutoModelForCausalLM.from_pretrained(input_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

    ft_trainer = SFTTrainer(model=ft_model,
                            train_dataset=ft_dataset,
                            tokenizer=ft_tokenizer,
                            args=ft_training_args)

    ft_trainer.train()
    print("Fine-tuning training completed.")

    ft_tokenizer.save_pretrained(FT_OUTPUT_DIR)
    ft_trainer.save_model(FT_OUTPUT_DIR)

    print(f"Fine-tuned model and tokenizer stored at {FT_OUTPUT_DIR}")

if __name__ == "__main__":
    run_ft_training()
