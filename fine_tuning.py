import torch
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from dataset_preparation import load_and_format_math_data
from settings import MODEL_REF, FT_OUTPUT_DIR

def _create_tokenized_dataset(example, tokenizer):
    """
    Applies the tokenizer's chat template to a single training example.

    This function formats the dialogue and prepares it with the necessary
    'input_ids', 'attention_mask', and 'labels' for supervised fine-tuning.
    """
    
    # Construct the full conversation, including the assistant's solution
    full_conversation = example["prompt"] + [{"role": "assistant", "content": example["solution"]}]

    tokenized_inputs = tokenizer.apply_chat_template(full_conversation,
                                                     truncation=True,
                                                     padding="max_length",
                                                     return_tensors="pt")

    return {
        "input_ids": tokenized_inputs["input_ids"].squeeze(),
        "attention_mask": tokenized_inputs["attention_mask"].squeeze(),
        "labels": tokenized_inputs["input_ids"].squeeze().clone(),
    }

def run_ft_training(input_model_path: str = MODEL_REF,
                    output_dir: str = FT_OUTPUT_DIR,
                    train_dataset=None) -> None:
    """
    Runs a full supervised fine-tuning (SFT) loop.

    This serves as the baseline training method, performing a full-parameter
    update on the entire training dataset.

    Args:
        input_model_path: The identifier of the base model to fine-tune.
        output_dir: The directory where the final trained model will be saved.
        train_dataset: An optional, pre-loaded Hugging Face dataset to use.
    """
    print("--- Starting Baseline: Full Supervised Fine-Tuning ---")
    print(f"Model: {input_model_path}")
    print(f"Output Directory: {output_dir}")

    # Step 1: Load the Training Dataset
    # If a dataset isn't provided, load the default math problems dataset.
    if train_dataset is None:
        print("No dataset provided, loading 'NuminaMath-TIR'...")
        math_data = load_and_format_math_data()
        ft_dataset = math_data["train"]
    else:
        ft_dataset = train_dataset
    print(f"Loaded {len(ft_dataset)} training examples.")

    # Step 2: Initialize Tokenizer and Prepare Dataset
    ft_tokenizer = AutoTokenizer.from_pretrained(input_model_path,
                                                 trust_remote_code=True,
                                                 padding_side="right")
                        
    if ft_tokenizer.pad_token is None:
        # Ensure a padding token is set, which is required for SFT.
        ft_tokenizer.pad_token = ft_tokenizer.eos_token

    # Create a version of our tokenization function with the tokenizer argument pre-filled.
    tokenization_function = partial(_create_tokenized_dataset, tokenizer=ft_tokenizer)

    print("Tokenizing dataset...")
    tokenized_ds = ft_dataset.map(tokenization_function,
                                  remove_columns=ft_dataset.column_names)

    # Step 3: Configure Training Parameters
    # These arguments define the training regimen (learning rate, batch size, etc.).
    ft_training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2, # Effective batch size = 8 * 2 = 16
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

    # Step 4: Load the Language Model
    print("Loading base model...")
    ft_model = AutoModelForCausalLM.from_pretrained(input_model_path,
                                                    trust_remote_code=True,
                                                    torch_dtype=torch.bfloat16)

    # Step 5: Initialize and Execute the Trainer
    ft_trainer = SFTTrainer(model=ft_model,
                            train_dataset=tokenized_ds,
                            tokenizer=ft_tokenizer,
                            args=ft_training_args)

    print("Starting fine-tuning...")
    ft_trainer.train()
    print("Fine-tuning training completed successfully.")

    print(f"Saving fine-tuned model and tokenizer to {output_dir}...")
    ft_trainer.save_model(output_dir)
    ft_tokenizer.save_pretrained(output_dir)
    print("--- Baseline SFT Finished ---")


if __name__ == "__main__":
    run_ft_training()
