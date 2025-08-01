import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, default_data_collator
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer

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


def run_qlora_fine_tuning(base_model_path: str,
                            refine_dataset,
                            output_dir: str):
    """
    Performs parameter-efficient fine-tuning (PEFT) using QLoRA.
    """
    print(f"Starting QLoRA fine-tuning on: {base_model_path}")

    # Step 1: Load tokenizer and prepare the dataset
    tokenizer = AutoTokenizer.from_pretrained(base_model_path,
                                              trust_remote_code=True,
                                              padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_dataset = refine_dataset.map(lambda x: create_tokenized_dataset(x, tokenizer),
                                           remove_columns=refine_dataset.column_names)

    # Step 2: Configure 4-bit quantization
    quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.bfloat16,
                                      bnb_4bit_use_double_quant=True)

    # Step 3: Load the quantized model
    model = AutoModelForCausalLM.from_pretrained(base_model_path,
                                                 quantization_config=quant_config,
                                                 device_map="auto",
                                                 trust_remote_code=True)
                                
    model = prepare_model_for_kbit_training(model)

    # Step 4: Configure and apply LoRA adapters
    lora_config = LoraConfig(r=8,
                             lora_alpha=32,
                             target_modules=["q_proj", "v_proj"],
                             lora_dropout=0.05,
                             bias="none",
                             task_type="CAUSAL_LM")
                                
    model = get_peft_model(model, lora_config)

    # Step 5: Configure training arguments
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

    # Step 6: Initialize and run the SFTTrainer
    trainer = SFTTrainer(model=model,
                         train_dataset=tokenized_dataset,
                         tokenizer=tokenizer,
                         args=training_args,
                         data_collator=default_data_collator)

    print("Starting QLoRA training...")
    trainer.train()
    print("QLoRA fine-tuning completed.")

    # Step 7: Save the final model and tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"QLoRA fine-tuned model saved at {output_dir}")
