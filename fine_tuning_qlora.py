# fine_tuning_qlora.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, default_data_collator
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer
from dataset_preparation import load_and_format_math_data
from settings import FT_OUTPUT_DIR, MODEL_REF

def run_qlora_fine_tuning(base_model_path: str, refine_dataset, output_dir: str,):
    print(f"Starting QLoRA fine-tuning on: {base_model_path}")

    # 2. tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. model in 4-bit
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quant_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = prepare_model_for_kbit_training(model)

    # 4. apply LoRA
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # 5. tokenize examples
    def tokenize_fn(example):
        text = "".join(m["content"] for m in example["prompt"])
        text += example["solution"]
        tokens = tokenizer(text, truncation=True, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = refine_dataset.map(tokenize_fn,remove_columns=refine_dataset.column_names)

    # 6. training args
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

    # 7. trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=default_data_collator)
        
    trainer.train()

    tokenizer.save_pretrained(output_dir)
    trainer.save_model(output_dir)
    print(f"QLoRA fine-tuned model saved at {output_dir}")

if __name__ == "__main__":
    run_qlora_fine_tuning()