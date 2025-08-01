import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from dataset_preparation import load_and_format_math_data
from settings import MODEL_REF, FT_OUTPUT_DIR

def create_tokenized_dataset(example, tokenizer):
    """
    Applies the tokenizer's chat template to a single example.
    """
    # Combine the prompt and solution into a single conversation
    full_conversation = example["prompt"] + [{"role": "assistant", "content": example["solution"]}]

    # Apply the tokenizer's built-in chat template
    tokenized_inputs = tokenizer.apply_chat_template(full_conversation,
                                                     truncation=True,
                                                     padding="max_length",
                                                     return_tensors="pt")
    
    # For SFT, the labels are the input_ids
    return {
        "input_ids": tokenized_inputs["input_ids"].squeeze(),
        "attention_mask": tokenized_inputs["attention_mask"].squeeze(),
        "labels": tokenized_inputs["input_ids"].squeeze().clone(),
    }


def run_ft_training(input_model_path: str = MODEL_REF,
                    output_dir: str = FT_OUTPUT_DIR,
                    train_dataset=None) -> None:
    """
    Runs a full supervised fine-tuning (SFT) loop on a given model.
    """
    print(f"Starting full fine-tuning using model from: {input_model_path}")
    print(f"Output will be saved to: {output_dir}")

    # Step 1: Load the dataset
    if train_dataset is None:
        math_data = load_and_format_math_data()
        ft_dataset = math_data["train"]
    else:
        ft_dataset = train_dataset

    # Step 2: Load tokenizer and prepare the dataset
    ft_tokenizer = AutoTokenizer.from_pretrained(input_model_path,
                                                 trust_remote_code=True,
                                                 padding_side="right")
    if ft_tokenizer.pad_token is None:
        ft_tokenizer.pad_token = ft_tokenizer.eos_token

    tokenized_ds = ft_dataset.map(lambda x: create_tokenized_dataset(x, ft_tokenizer),
                                  remove_columns=ft_dataset.column_names)

    # Step 3: Configure training arguments
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
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        seed=42,
        report_to="none",
        push_to_hub=False,
    )

    # Step 4: Load the base model
    ft_model = AutoModelForCausalLM.from_pretrained(input_model_path,
                                                    trust_remote_code=True,
                                                    torch_dtype=torch.bfloat16)

    # Step 5: Initialize and run the SFTTrainer
    ft_trainer = SFTTrainer(model=ft_model,
                            train_dataset=tokenized_ds,
                            tokenizer=ft_tokenizer,
                            args=ft_training_args)

    print("Starting training...")
    ft_trainer.train()
    print("Fine-tuning training completed.")

    # Step 6: Save the final model and tokenizer
    ft_trainer.save_model(output_dir)
    ft_tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model and tokenizer saved at {output_dir}")


if __name__ == "__main__":
    run_ft_training()
