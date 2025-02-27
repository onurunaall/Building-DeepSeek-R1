def execute_sft_training() -> None:
    """
    Executes Supervised Fine-Tuning (SFT) training using a curated dataset.
    """
    sft_dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", "default", split="train")
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
    
    sft_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16)
    
    sft_trainer = SFTTrainer(
        model=sft_model,
        train_dataset=sft_dataset,
        tokenizer=tokenizer,
        args=sft_training_args,
    )
    
    sft_trainer.train()
    print("SFT training is complete.")
    
    tokenizer.save_pretrained(OUTPUT_DIR_SFT)
    sft_trainer.save_model(OUTPUT_DIR_SFT)
    print(f"SFT trained model and tokenizer saved to {OUTPUT_DIR_SFT}")
