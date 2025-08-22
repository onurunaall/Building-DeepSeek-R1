import torch
import json
from functools import partial
from typing import Dict, Any, List, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, default_data_collator
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _create_tokenized_dataset(example: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
    Applies the tokenizer's chat template to a single training example.

    This function formats the dialogue and prepares it with the necessary
    'input_ids', 'attention_mask', and 'labels' for supervised fine-tuning.

    Args:
        example: Dictionary containing 'prompt' (list of message dicts) and 'solution' (str)
        tokenizer: The tokenizer to use for encoding
        
    Returns:
        Dictionary with 'input_ids', 'attention_mask', and 'labels' tensors
        
    Raises:
        ValueError: If example is missing required fields
        RuntimeError: If tokenization fails
    """
    if not isinstance(example.get("prompt"), list):
        raise ValueError("Example must contain 'prompt' field with list of message dictionaries")
    
    if not isinstance(example.get("solution"), str):
        raise ValueError("Example must contain 'solution' field with string content")
    
    if not example["solution"].strip():
        raise ValueError("Solution field cannot be empty")

    full_conversation = example["prompt"] + [{"role": "assistant", "content": example["solution"]}]

    max_length = getattr(tokenizer, 'model_max_length', None)
    if max_length is None or max_length > 100000:
        max_length = 2048

    try:
        tokenized_inputs = tokenizer.apply_chat_template(
            full_conversation,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to tokenize conversation: {e}")

    required_keys = ["input_ids", "attention_mask"]
    missing_keys = []
    for key in required_keys:
        if key not in tokenized_inputs:
            missing_keys.append(key)
    
    if missing_keys:
        raise RuntimeError(f"Tokenizer output missing expected keys: {missing_keys}")

    return {
        "input_ids": tokenized_inputs["input_ids"].squeeze(),
        "attention_mask": tokenized_inputs["attention_mask"].squeeze(),
        "labels": tokenized_inputs["input_ids"].squeeze().clone(),
    }


def create_qlora_config(
    model_size_gb: float = None,
    task_complexity: str = "medium",
    target_modules: List[str] = None
) -> LoraConfig:
    """
    Create optimized LoRA configuration based on model size and task.

    Args:
        model_size_gb: Approximate model size in GB for parameter selection
        task_complexity: "simple", "medium", or "complex" 
        target_modules: List of target module names
        
    Returns:
        Optimized LoraConfig
    """
    if model_size_gb is None:
        r = 8
    elif model_size_gb < 1:
        r = 4 if task_complexity == "simple" else 8
    elif model_size_gb < 7:
        r = 8 if task_complexity == "simple" else 16
    else:
        r = 16 if task_complexity == "simple" else 32

    lora_alpha = 2 * r

    dropout_map = {
        "simple": 0.05,
        "medium": 0.1, 
        "complex": 0.15
    }
    lora_dropout = dropout_map.get(task_complexity, 0.1)

    logger.info(f"Creating LoRA config: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
    logger.info(f"Target modules: {target_modules}")

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        target_modules="all-linear",
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    


def run_qlora_fine_tuning(
    base_model_path: str,
    refine_dataset,
    output_dir: str,
    auto_find_target_modules: bool = True,
    custom_target_modules: Optional[List[str]] = None,
    task_complexity: str = "medium"
) -> None:
    """
    Performs a full QLoRA fine-tuning loop with intelligent target module detection.

    Args:
        base_model_path: The identifier of the base model to fine-tune
        refine_dataset: The Hugging Face dataset to use for training
        output_dir: Directory where the trained adapters will be saved
        auto_find_target_modules: Whether to automatically detect target modules
        custom_target_modules: Manual override for target modules
        task_complexity: "simple", "medium", or "complex" for LoRA config optimization
    """
    logger.info("--- Starting Experiment: Enhanced QLoRA Fine-Tuning ---")
    logger.info(f"Base Model: {base_model_path}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Auto-detect target modules: {auto_find_target_modules}")
    logger.info(f"Task complexity: {task_complexity}")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side="right"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    # Tokenize dataset
    tokenization_function = partial(_create_tokenized_dataset, tokenizer=tokenizer)

    logger.info("Tokenizing dataset for QLoRA...")
    try:
        tokenized_dataset = refine_dataset.map(
            tokenization_function,
            remove_columns=refine_dataset.column_names,
            desc="Tokenizing for QLoRA"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to tokenize dataset: {e}")

    logger.info(f"Tokenized dataset size: {len(tokenized_dataset)}")

    # Configure quantization
    try:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("4-bit quantization config created successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to create quantization config: {e}")

    # Load quantized model
    logger.info("Loading base model with 4-bit quantization...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        logger.info("Model loaded successfully with 4-bit quantization")
    except Exception as e:
        raise RuntimeError(f"Failed to load quantized model: {e}")

    if hasattr(model, 'get_memory_footprint'):
        memory_mb = model.get_memory_footprint() / 1024 / 1024
        logger.info(f"Model memory footprint: {memory_mb:.2f} MB")

    # Determine target modules
    if custom_target_modules:
        target_modules = custom_target_modules
        logger.info(f"Using custom target modules: {target_modules}")
    elif auto_find_target_modules:
        target_modules = find_target_modules(model)
        logger.info(f"Auto-detected target modules: {target_modules}")
    else:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        logger.info(f"Using default target modules: {target_modules}")

    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    model_size_gb = (total_params * 4) / (1024**3)
    logger.info(f"Estimated model size: {model_size_gb:.2f} GB ({total_params:,} parameters)")

    # Create LoRA config
    lora_config = create_qlora_config(
        model_size_gb=model_size_gb,
        task_complexity=task_complexity,
        target_modules=target_modules
    )

    # Prepare model and apply LoRA
    try:
        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)
        
        logger.info("Applying LoRA adapters...")
        model = get_peft_model(model, lora_config)
        
        model.print_trainable_parameters()
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_percentage = 100 * trainable_params / total_params
        logger.info(f"Trainable: {trainable_params:,} / Total: {total_params:,} ({trainable_percentage:.4f}%)")
        
    except Exception as e:
        raise RuntimeError(f"Failed to apply LoRA adapters: {e}")

    # Configure training
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        warmup_ratio=0.03,
        weight_decay=0.001,
        max_grad_norm=1.0,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        dataloader_num_workers=0,
        remove_unused_columns=True,
        report_to="none",
        seed=42,
    )

    # Initialize trainer
    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=tokenized_dataset,
            processing_class=tokenizer,
            args=training_args,
            data_collator=default_data_collator,
            max_seq_length=2048,
        )
        logger.info("SFTTrainer initialized successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize SFTTrainer: {e}")

    # Execute training
    logger.info("Starting QLoRA training...")
    try:
        trainer.train()
        logger.info("QLoRA fine-tuning completed successfully")
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")

    # Save results
    logger.info(f"Saving LoRA adapters and tokenizer to {output_dir}...")
    try:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        training_info = {
            "base_model": base_model_path,
            "target_modules": target_modules,
            "lora_config": {
                "r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "lora_dropout": lora_config.lora_dropout,
                "task_type": lora_config.task_type
            },
            "model_size_gb": model_size_gb,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_percentage": 100 * trainable_params / total_params
        }
        
        with open(f"{output_dir}/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        logger.info("QLoRA adapters and training info saved successfully")
        
    except Exception as e:
        raise RuntimeError(f"Failed to save model: {e}")

    logger.info("--- QLoRA Fine-tuning Finished Successfully ---")


def test_target_module_detection(model_path: str) -> None:
    """Test function to verify target module detection works correctly."""
    logger.info(f"Testing target module detection for {model_path}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        
        target_modules = find_target_modules(model)
        
        logger.info("Target module detection test successful")
        logger.info(f"Detected modules: {target_modules}")
        
        existing_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module_name = name.split(".")[-1]
                if module_name in target_modules:
                    existing_modules.append(module_name)
        
        logger.info(f"Verified existing modules: {list(set(existing_modules))}")
        
        if not existing_modules:
            logger.warning("No target modules found in model - this will cause LoRA to fail")
        else:
            logger.info("Target modules verified to exist in model")
            
    except Exception as e:
        logger.error(f"Target module detection test failed: {e}")
        raise


if __name__ == "__main__":
    test_model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    test_target_module_detection(test_model_path)