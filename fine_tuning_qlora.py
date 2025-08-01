import torch
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, default_data_collator
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer

def _create_tokenized_dataset(example, tokenizer):
    """
    Applies the tokenizer's chat template to a single training example.

    This function formats the dialogue and prepares it with the necessary
    'input_ids', 'attention_mask', and 'labels' for supervised fine-tuning.
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


def run_qlora_fine_tuning(base_model_path: str,
                            refine_dataset,
                            output_dir: str):
    """
    Performs a full QLoRA fine-tuning loop.

    Args:
        base_model_path: The identifier of the base model to fine-tune.
        refine_dataset: The Hugging Face dataset to use for training.
        output_dir: The directory where the final trained adapters will be saved.
    """
    print("--- Starting Experiment: QLoRA Fine-Tuning ---")
    print(f"Base Model: {base_model_path}")
    print(f"Output Directory: {output_dir}")

    # Step 1: Initialize Tokenizer and Prepare Dataset
    tokenizer = AutoTokenizer.from_pretrained(base_model_path,
                                              trust_remote_code=True,
                                              padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create a partial function for tokenization to use with .map()
    tokenization_function = partial(_create_tokenized_dataset, tokenizer=tokenizer)

    print("Tokenizing dataset for QLoRA...")
    tokenized_dataset = refine_dataset.map(tokenization_function,
                                           remove_columns=refine_dataset.column_names)

    # Step 2: Configure 4-bit Quantization (BitsAndBytes)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Step 3: Configure Low-Rank Adaptation (LoRA)
    lora_config = LoraConfig(
        r=8, # The rank of the update matrices
        lora_alpha=32, # The scaling factor for the LoRA parameters
        target_modules=["q_proj", "v_proj"], # Target the query and value projections in attention
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Step 4: Load the Quantized Model and Apply LoRA Adapters
    print("Loading base model with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                                 quantization_config=quant_config,
                                                 device_map="auto",
                                                 trust_remote_code=True)

    # Prepare the model for k-bit training and apply the LoRA configuration
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    print("LoRA adapters applied.")

    # Step 5: Configure Training Arguments
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
        report_to="none",
    )

    # Step 6: Initialize and Execute the SFTTrainer
    trainer = SFTTrainer(model=model,
                         train_dataset=tokenized_dataset,
                         tokenizer=tokenizer,
                         args=training_args,
                         data_collator=default_data_collator)

    print("Starting QLoRA training...")
    trainer.train()
    print("QLoRA fine-tuning completed successfully.")

    print(f"Saving LoRA adapters and tokenizer to {output_dir}...")
    trainer.save_model(output_dir) # This saves only the trained adapters, not the full model
    tokenizer.save_pretrained(output_dir)
    print("--- QLoRA Finished ---")
