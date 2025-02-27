# train_grpo.py
import os
import torch
import logging
from typing import List
from transformers import TrainingArguments, TrainerCallback, TrainerControl, TrainerState
from trl import GRPOTrainer, GRPOConfig
from config import OUTPUT_DIR_GRPO, GRPOConfigArgs
from data_prep import load_and_prepare_math_dataset, validate_prepared_dataset
from model_setup import initialize_model, initialize_tokenizer
from reward_functions import (
    compute_accuracy_reward,
    compute_format_reward,
    compute_reasoning_steps_reward,
    get_cosine_scaled_reward_function,
    get_repetition_penalty_reward_function
)

# Set up simple logging
logger = logging.getLogger(__name__)

class SimpleLoggingCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs) -> None:
        if state.global_step % args.logging_steps == 0:
            current_loss = state.log_history[-1].get('loss', None) if state.log_history else None
            current_lr = state.log_history[-1].get('learning_rate', None) if state.log_history else None
            logger.info(f"Step {state.global_step}: Loss = {current_loss}, Learning Rate = {current_lr}")

def register_reward_functions(config_args: GRPOConfigArgs) -> List:
    """
    Registers and returns a list of reward functions based on configuration arguments.
    """
    reward_function_registry = {
        "accuracy": compute_accuracy_reward,
        "format": compute_format_reward,
        "reasoning_steps": compute_reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward_function(
            min_wrong=config_args.cosine_min_wrong,
            max_wrong=config_args.cosine_max_wrong,
            min_correct=config_args.cosine_min_correct,
            max_correct=config_args.cosine_max_correct,
            max_length=config_args.cosine_max_length,
        ),
        "repetition_penalty": get_repetition_penalty_reward_function(
            ngram_size=config_args.ngram_size_for_repetition,
            max_penalty=config_args.repetition_max_penalty,
        ),
    }
    reward_functions_list: List = []
    for func_name in config_args.reward_function_names:
        if func_name not in reward_function_registry:
            raise ValueError(f"Reward function '{func_name}' not found in registry.")
        reward_functions_list.append(reward_function_registry[func_name])
    return reward_functions_list

def execute_grpo_training() -> None:
    """
    Executes the GRPO (reinforcement learning) training process.
    Loads data, initializes the model, sets up reward functions, and trains the model.
    """
    # Load and validate dataset
    math_dataset = load_and_prepare_math_dataset()
    validate_prepared_dataset(math_dataset)
    
    # Initialize model and tokenizer
    language_model, compute_device = initialize_model()
    text_tokenizer = initialize_tokenizer()
    
    # Set up training arguments
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
    
    # Register reward functions using our config arguments
    grpo_config_args = GRPOConfigArgs()
    reward_functions = register_reward_functions(grpo_config_args)
    
    # Set up logging callback
    callbacks = [SimpleLoggingCallback()]
    
    # Initialize GRPO Trainer
    grpo_trainer = GRPOTrainer(
        model=language_model,
        reward_funcs=reward_functions,
        args=grpo_config,
        train_dataset=math_dataset["train"],
        eval_dataset=math_dataset["test"],
        callbacks=callbacks,
    )
    
    # Start training
    grpo_trainer.train()
    print("GRPO training is complete.")
    
    # Save model and tokenizer
    text_tokenizer.save_pretrained(OUTPUT_DIR_GRPO)
    grpo_trainer.save_model(OUTPUT_DIR_GRPO)
    print(f"GRPO trained model and tokenizer saved to {OUTPUT_DIR_GRPO}")

if __name__ == "__main__":
    execute_grpo_training()
