import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from dataset_preparation import load_and_format_math_data, split_seed_refine
from settings import MODEL_REF, FT_OUTPUT_DIR

def create_tokenized_dataset(example, tokenizer):
    """
    Applies the tokenizer's chat template to a single example.
    """
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


def run_seed_ft_training(base_model_path: str = MODEL_REF,
                         seed_frac: float = 0.1,
                         output_dir: str = None,
                         train_dataset=None):
    """
    Runs supervised fine-tuning on a small 'seed' subset of the dataset.
    """
    # Step 1: Determine output directory and load the full dataset
    if output_dir is None:
        output_dir = os.path.join(FT_OUTPUT_DIR, "seed")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting seed SFT (fraction={seed_frac}) from {base_model_path}")
    print(f"Saving seed-fine-tuned model to: {output_dir}")

    if train_dataset is None:
        math_data = load_and_format_math_data()
        full_train_dataset = math_data["train"]
    else:
        full_train_dataset = train_dataset

    # Split the dataset into a small seed set and a larger refine set
    seed_set, refine_set = split_seed_refine(full_train_dataset, seed_frac=seed_frac)

    # Step 2: Load tokenizer and prepare the seed dataset
    tokenizer = AutoTokenizer.from_pretrained(base_model_path,
                                              trust_remote_code=True,
                                              padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_ds = seed_set.map(lambda x: create_tokenized_dataset(x, tokenizer),
                                remove_columns=seed_set.column_names)

    # Step 3: Configure training arguments
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
        push_to_hub=False
    )

    # Step 4: Load the base model
    model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                                 trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16)

    # Step 5: Initialize and run the SFTTrainer
    trainer = SFTTrainer(model=model,
                         train_dataset=tokenized_ds,
                         tokenizer=tokenizer,
                         args=training_args)

    print("Starting seed training...")
    trainer.train()
    print("Seed SFT completed.")

    # Step 6: Save the trained model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Seed-fine-tuned model saved at {output_dir}")

    # Return the unused portion of the data for the next stage
    return refine_set


if __name__ == "__main__":
    run_seed_ft_training()
