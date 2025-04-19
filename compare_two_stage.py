"""
Read the metrics CSV produced by run_two_stage_ft_qlora.py,
plot bar charts for each metric, and print a short numeric comparison.

Usage:
    python compare_two_stage.py \
        --csv_path data/Qwen-FT-training/two_stage_metrics.csv \
        --out_dir reports
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_metric(df: pd.DataFrame, metric: str, out_dir: str) -> None:
    """
    Create a simple bar chart comparing `metric` for each method.
    The plot is saved as <metric>.png in out_dir.
    """
    plt.figure()
    plt.bar(df["method"], df[metric])
    plt.ylabel(metric)
    plt.title(f"{metric} comparison")
    fname = os.path.join(out_dir, f"{metric}.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"[saved] {fname}")

def main(csv_path: str, out_dir: str) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    for metric in ["accuracy", "format", "reasoning"]:
        if metric in df.columns:
            plot_metric(df, metric, out_dir)
        else:
            print(f"[warning] metric '{metric}' not found in CSV.")

    print("\nNumeric comparison:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        default=os.path.join("data", "Qwen-FT-training", "two_stage_metrics.csv"),
        help="Path to the CSV produced by run_two_stage_ft_qlora.py",
    )
    parser.add_argument(
        "--out_dir",
        default="reports",
        help="Directory to save the plots",
    )
    args = parser.parse_args()

    main(args.csv_path, args.out_dir)
