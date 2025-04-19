import os
from settings import MODEL_REF, FT_OUTPUT_DIR
from fine_tuning_seed import run_seed_ft_training
from fine_tuning_qlora import run_qlora_fine_tuning

def run_two_stage_ft_qlora(
    base_model_path: str = MODEL_REF,
    seed_fraction: float = 0.1,
    root_output_dir: str = FT_OUTPUT_DIR,
):
    """
    Two‑stage pipeline:
      1. Supervised FT on a small 'seed' subset (seed_fraction of data).
      2. QLoRA fine‑tuning on the seed‑fine‑tuned model using the full dataset.
    """
    seed_dir = os.path.join(root_output_dir, "seed")
    qlora_dir = os.path.join(root_output_dir, "qlora")

    # Stage 1: seed SFT
    print(f"Stage 1: Seed SFT (fraction={seed_fraction})")
    run_seed_ft_training(
        base_model_path=base_model_path,
        seed_frac=seed_fraction,
        output_dir=seed_dir,
    )

    # Stage 2: QLoRA on seed‐fine‐tuned model
    print("Stage 2: QLoRA fine‑tuning")
    run_qlora_fine_tuning(
        base_model_path=seed_dir,
        output_dir=qlora_dir,
    )

    print("Two‑stage FT→QLoRA pipeline completed.")
    print(f"  Seed model at:   {seed_dir}")
    print(f"  QLoRA model at:  {qlora_dir}")

if __name__ == "__main__":
    run_two_stage_ft_qlora()
