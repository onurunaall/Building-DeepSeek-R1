import os
import torch
import logging
from typing import List
from transformers import TrainingArguments, TrainerCallback, TrainerControl, TrainerState
from trl import GRPOTrainer, GRPOConfig
from settings import RL_OUTPUT_DIR, RLTrainingSettings
from dataset_preparation import load_and_format_math_data, check_dataset_integrity
from model_initialization import setup_model, setup_tokenizer
from reward_metrics import evaluate_accuracy, evaluate_format, evaluate_reasoning_steps, create_cosine_reward_func, create_repetition_penalty_func

logger = logging.getLogger(__name__)

class BasicLogger(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Every few steps, log the current loss and learning rate.
        if state.global_step % args.logging_steps == 0:
            if state.log_history:
                loss_val = state.log_history[-1].get('loss', None)
                lr_val = state.log_history[-1].get('learning_rate', None)
            else:
                None
            
            logger.info(f"Step {state.global_step}: Loss = {loss_val}, LR = {lr_val}")

def compile_reward_metrics(training_settings: RLTrainingSettings) -> List:
    """
    Based on our training settings, put together a list of reward functions we want to use.
    """
    
    metric_registry = {
        "accuracy": evaluate_accuracy,
        "format": evaluate_format,
        "reasoning_steps": evaluate_reasoning_steps,
        "cosine": create_cosine_reward_func(low_bad=training_settings.cosine_low_bad,
                                            high_bad=training_settings.cosine_high_bad,
                                            low_good=training_settings.cosine_low_good,
                                            high_good=training_settings.cosine_high_good,
                                            length_limit=training_settings.cosine_length_limit
                                           ),
        "repetition_penalty": create_repetition_penalty_func(ngram=training_settings.repetition_ngram,
                                                             penalty_value=training_settings.repetition_penalty_value
                                                            )
    }
    
    compiled_metrics: List = []
    
    # For each metric name in our settings, add the corresponding function.
    for metric_id in training_settings.metric_identifiers:
        if metric_id not in metric_registry:
            raise ValueError(f"Metric '{metric_id}' not available in registry.")
            
        compiled_metrics.append(metric_registry[metric_id])
        
    return compiled_metrics

def run_rl_training() -> None:
    """
    This function runs our reinforcement learning training loop.
    It loads the data, sets up the model, compiles reward functions, and starts training.
    """
    
    math_data = load_and_format_math_data()
    check_dataset_integrity(math_data)
    lang_model, compute_device = setup_model()
    text_tokenizer = setup_tokenizer()
    
    training_params = TrainingArguments(output_dir=RL_OUTPUT_DIR,
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
                                        report_to="none"
                                       )
    

    rl_config = GRPOConfig(**training_params.to_dict())
    
    training_settings = RLTrainingSettings()
    reward_metrics_list = compile_reward_metrics(training_settings)
    
    callbacks = [BasicLogger()]
    

    rl_trainer = GRPOTrainer(model=lang_model,
                             reward_funcs=reward_metrics_list,
                             args=rl_config,
                             train_dataset=math_data["train"],
                             eval_dataset=math_data["test"],
                             callbacks=callbacks
                            )
    
    # Start the training process.
    rl_trainer.train()
    print("Reinforcement learning training completed.")
    
    # Save our trained model and tokenizer.
    text_tokenizer.save_pretrained(RL_OUTPUT_DIR)
    rl_trainer.save_model(RL_OUTPUT_DIR)
    print(f"Model and tokenizer saved at {RL_OUTPUT_DIR}")

if __name__ == "__main__":
    run_rl_training()
