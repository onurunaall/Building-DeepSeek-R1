# run_two_stage_ft_qlora.py

import os
import csv
from settings import MODEL_REF, FT_OUTPUT_DIR
from fine_tuning_seed import run_seed_ft_training
from fine_tuning_qlora import run_qlora_fine_tuning
from dataset_preparation import load_and_format_math_data
from model_inference import load_saved_model, get_model_response
from reward_metrics import evaluate_accuracy, evaluate_format, evaluate_reasoning_steps

def evaluate_model(
    method_name: str,
    model_dir: str,
    test_dataset,
    csv_writer: csv.writer
):
    """
    Run inference on test_dataset using the saved model at model_dir,
    compute average accuracy, format, and reasoning scores,
    and write a row to csv_writer.
    """
    print(f"Evaluating {method_name}...")
    model, tokenizer, device = load_saved_model(model_dir)

    outputs = []
    solutions = []
    for example in test_dataset:
        # Build the prompt from system+user roles
        prompt_text = "".join(m["content"] for m in example["prompt"])
        # Generate the model's reply
        reply = get_model_response(prompt_text, model, tokenizer, device)
        outputs.append([{"content": reply}])
        solutions.append(example["solution"])

    # Calculate metrics
    acc_scores = evaluate_accuracy(outputs, solution=solutions)
    fmt_scores = evaluate_format(outputs)
    reason_scores = evaluate_reasoning_steps(outputs)

    # Compute averages
    avg_acc = sum(acc_scores) / len(acc_scores) if acc_scores else 0.0
    avg_fmt = sum(fmt_scores) / len(fmt_scores) if fmt_scores else 0.0
    avg_reason = sum(reason_scores) / len(reason_scores) if reason_scores else 0.0

    # Write to CSV
    csv_writer.writerow([method_name, avg_acc, avg_fmt, avg_reason])

def run_two_stage_ft_qlora(
    base_model_path: str = MODEL_REF,
    seed_fraction: float = 0.1,
    root_output_dir: str = FT_OUTPUT_DIR,
):
    """
    Two-stage pipeline:
      1. Supervised FT on a small 'seed' subset.
      2. QLoRA fine-tuning on the seed-fine-tuned model using the full dataset.
      3. Evaluation and metrics saving.
    """
    seed_dir = os.path.join(root_output_dir, "seed")
    qlora_dir = os.path.join(root_output_dir, "qlora")

    # Stage 1: Seed SFT
    print(f"Stage 1: Seed SFT (fraction={seed_fraction})")
    run_seed_ft_training(
        base_model_path=base_model_path,
        seed_frac=seed_fraction,
        output_dir=seed_dir,
    )

    # Stage 2: QLoRA
    print("Stage 2: QLoRA fine-tuning")
    run_qlora_fine_tuning(
        base_model_path=seed_dir,
        output_dir=qlora_dir,
    )

    print("Two-stage FT→QLoRA pipeline completed.")
    print(f"  Seed model at:   {seed_dir}")
    print(f"  QLoRA model at:  {qlora_dir}")

    # Stage 3: Evaluation
    print("Stage 3: Evaluation on held-out test set")
    test_dataset = load_and_format_math_data()["test"]
    metrics_file = os.path.join(root_output_dir, "two_stage_metrics.csv")
    os.makedirs(root_output_dir, exist_ok=True)

    with open(metrics_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["method", "accuracy", "format", "reasoning"])
        evaluate_model("seed_ft", seed_dir, test_dataset, writer)
        evaluate_model("two_stage_qlora", qlora_dir, test_dataset, writer)

    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    run_two_stage_ft_qlora()
