# main.py
"""
Runs every stage end‑to‑end:
  1. RL training (optional)
  2. Full‑dataset supervised FT   → data/Qwen-FT-training
  3. Seed‑subset supervised FT    → data/Qwen-FT-training/seed
  4. QLoRA on refine set          → data/Qwen-FT-training/qlora
  5. Evaluation on held‑out test  → data/Qwen-FT-training/all_metrics.csv
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
    baseline_dir = FT_OUTPUT_DIR                        # full‑dataset SFT
    seed_dir = os.path.join(FT_OUTPUT_DIR, "seed")      # seed SFT
    qlora_dir = os.path.join(FT_OUTPUT_DIR, "qlora")    # two‑stage QLoRA
    metrics_csv = os.path.join(FT_OUTPUT_DIR, "all_metrics.csv")
    os.makedirs(FT_OUTPUT_DIR, exist_ok=True)

    # 1. Reinforcement learning (comment out if you want to skip)
    print("\n== RL training ==")
    run_rl_training() # outputs to RL_OUTPUT_DIR
    base_ckpt = RL_OUTPUT_DIR # later stages start here

    # 2. Full‑dataset supervised FT (baseline)
    print("\n== Full‑dataset supervised FT ==")
    run_ft_training(input_model_path=base_ckpt) # writes to baseline_dir

    # 3. Seed subset FT (10 %)
    print("\n== Seed (10 %) supervised FT ==")
    refine_ds = run_seed_ft_training(
        base_model_path=base_ckpt,
        seed_frac=0.10,
        output_dir=seed_dir,
    )

    # 4. QLoRA on the remaining 90 %
    print("\n== QLoRA fine‑tuning on refine set ==")
    run_qlora_fine_tuning(
        base_model_path=seed_dir,
        refine_dataset=refine_ds,
        output_dir=qlora_dir,
    )

    # 5. Evaluation
    print("\n== Evaluation on held‑out test split ==")
    test_ds = load_and_format_math_data()["test"]
    
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "accuracy", "format", "reasoning"])
        
        evaluate_model("baseline_full_sft", baseline_dir, test_ds, writer)
        evaluate_model("seed_ft", seed_dir, test_ds, writer)
        evaluate_model("two_stage_qlora", qlora_dir, test_ds, writer)

    print(f"\nAll metrics saved to {metrics_csv}")
    print("Pipeline finished.")


if __name__ == "__main__":
    main_pipeline()