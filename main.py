# main.py
"""
Runs three experiments to compare fine-tuning methods:
  1. Baseline: Full-parameter SFT on entire dataset
  2. QLoRA: Parameter-efficient fine-tuning on entire dataset  
  3. Two-Stage: Seed SFT (10%) + QLoRA on remainder (90%)
"""

import os
import csv
from settings import MODEL_REF, RL_OUTPUT_DIR, FT_OUTPUT_DIR
from reinforcement_training import run_rl_training
from fine_tuning import run_ft_training
from fine_tuning_seed import run_seed_ft_training
from fine_tuning_qlora import run_qlora_fine_tuning
from dataset_preparation import load_and_format_math_data
from model_inference import load_saved_model, get_model_response
from reward_metrics import evaluate_accuracy, evaluate_format, evaluate_reasoning_steps

def evaluate_model(method: str, model_dir: str, test_ds, writer: csv.writer):
    """
    Evaluate `model_dir` on `test_ds`, write one row to csv `writer`.
    """
    print(f"[eval] {method}")
    model, tok, dev = load_saved_model(model_dir)

    outputs = []
    solutions = []
    
    for ex in test_ds:
        # Only pass the user question — get_model_response will add SYSTEM_TEMPLATE
        reply = get_model_response(ex["problem"], model, tok, dev)
        outputs.append([{"content": reply}])
        solutions.append(ex["solution"])

    # Calculate metrics
    accuracy_scores = evaluate_accuracy(outputs, solution=solutions)
    format_scores = evaluate_format(outputs)
    reasoning_scores = evaluate_reasoning_steps(outputs)
    
    # Average the scores
    acc = sum(accuracy_scores) / len(outputs)
    fmt = sum(format_scores) / len(outputs)
    rea = sum(reasoning_scores) / len(outputs)

    writer.writerow([method, acc, fmt, rea])
    print(f"  accuracy={acc:.4f}  format={fmt:.4f}  reasoning={rea:.4f}")

def main_pipeline():
    print("\n== Loading and Preparing Data ==")
    datasets = load_and_format_math_data()
    train_ds = datasets["train"]
    test_ds = datasets["test"]

    # 1. Reinforcement learning (provides base checkpoint)
    print("\n== RL training ==")
    run_rl_training()
    base_ckpt = RL_OUTPUT_DIR

    # Define output directories
    baseline_dir = os.path.join(FT_OUTPUT_DIR, "full_sft")
    qlora_full_dir = os.path.join(FT_OUTPUT_DIR, "qlora_full")
    seed_dir = os.path.join(FT_OUTPUT_DIR, "seed")
    qlora_two_stage_dir = os.path.join(FT_OUTPUT_DIR, "qlora_two_stage")
    
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(qlora_full_dir, exist_ok=True)
    os.makedirs(qlora_two_stage_dir, exist_ok=True)

    # Experiment 1: Full SFT (Baseline)
    print("\n== Experiment 1: Full-dataset Supervised FT ==")
    run_ft_training(
        input_model_path=base_ckpt, 
        output_dir=baseline_dir, 
        train_dataset=train_ds
    )

    # Experiment 2: QLoRA on Full Dataset
    print("\n== Experiment 2: QLoRA on Full Dataset ==")
    run_qlora_fine_tuning(
        base_model_path=base_ckpt,
        refine_dataset=train_ds,  # Use the full training set
        output_dir=qlora_full_dir,
    )

    # Experiment 3: Two-Stage Seed + QLoRA
    print("\n== Experiment 3: Two-Stage Seed FT + QLoRA ==")
    print("  Stage 3a: Seed (10%) supervised FT")
    refine_ds = run_seed_ft_training(
        base_model_path=base_ckpt,
        seed_frac=0.10,
        output_dir=seed_dir,
        train_dataset=train_ds  # Pass the full train set to be split
    )
    
    print("  Stage 3b: QLoRA fine-tuning on refine set")
    run_qlora_fine_tuning(
        base_model_path=seed_dir,
        refine_dataset=refine_ds,
        output_dir=qlora_two_stage_dir,
    )

    # Evaluation
    print("\n== Evaluation on held-out test split ==")
    metrics_csv = os.path.join(FT_OUTPUT_DIR, "all_metrics.csv")
    
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "accuracy", "format", "reasoning"])
        
        evaluate_model("full_sft", baseline_dir, test_ds, writer)
        evaluate_model("qlora_full_sft", qlora_full_dir, test_ds, writer)
        evaluate_model("two_stage_qlora", qlora_two_stage_dir, test_ds, writer)

    print(f"\nAll metrics saved to {metrics_csv}")
    print("Pipeline finished.")


if __name__ == "__main__":
    main_pipeline()
