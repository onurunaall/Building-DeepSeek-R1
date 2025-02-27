def execute_grpo_training() -> None:
    """
    Executes the GRPO (reinforcement learning) training process.
    Loads data, initializes the model, sets up reward functions, and trains the model.
    """
    math_dataset = load_and_prepare_math_dataset()
    validate_prepared_dataset(math_dataset)
    
    language_model, compute_device = initialize_model()
    text_tokenizer = initialize_tokenizer()
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR_GRPO,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
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
    grpo_config = GRPOConfig(**training_args.to_dict())
    
    grpo_config_args = GRPOConfigArgs()
    reward_functions = register_reward_functions(grpo_config_args)
    
    callbacks = [SimpleLoggingCallback()]
    
    grpo_trainer = GRPOTrainer(
        model=language_model,
        reward_funcs=reward_functions,
        args=grpo_config,
        train_dataset=math_dataset["train"],
        eval_dataset=math_dataset["test"],
        callbacks=callbacks,
    )
    
    grpo_trainer.train()
    print("GRPO training is complete.")
    
    text_tokenizer.save_pretrained(OUTPUT_DIR_GRPO)
    grpo_trainer.save_model(OUTPUT_DIR_GRPO)
    print(f"GRPO trained model and tokenizer saved to {OUTPUT_DIR_GRPO}")
