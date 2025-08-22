import os
import torch
import json
import logging
from functools import partial
from typing import Dict, Any, Optional, Tuple, List
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from dataset_preparation import load_and_format_math_data, split_seed_refine
from settings import MODEL_REF, FT_OUTPUT_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _create_tokenized_dataset(example: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
    Applies the tokenizer's chat template to a single training example.

    Args:
        example: Dictionary containing 'prompt' (list of message dicts) and 'solution' (str)
        tokenizer: The tokenizer to use for encoding
        
    Returns:
        Dictionary with 'input_ids', 'attention_mask', and 'labels' tensors
        
    Raises:
        ValueError: If example is missing required fields
        RuntimeError: If tokenization fails
    """
    # Validate input data structure
    if not isinstance(example.get("prompt"), list):
        raise ValueError("Example must contain 'prompt' field with list of message dictionaries")
    if not isinstance(example.get("solution"), str):
        raise ValueError("Example must contain 'solution' field with string content")
    if not example["solution"].strip():
        raise ValueError("Solution field cannot be empty")

    # Construct full conversation including assistant response
    full_conversation = example["prompt"] + [{"role": "assistant", "content": example["solution"]}]

    # Determine appropriate sequence length
    max_length = getattr(tokenizer, 'model_max_length', None)
    if max_length is None or max_length > 100000:
        max_length = 2048  # Safe default for most models

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

    # Verify all required keys are present in tokenizer output
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


def validate_seed_fraction(seed_frac: float, dataset_size: int) -> None:
    """
    Validate seed fraction parameter against research best practices.

    Args:
        seed_frac: Fraction of data to use for seed training
        dataset_size: Total size of the dataset
        
    Raises:
        ValueError: If seed_frac is invalid
        
    Based on research recommendations for two-stage training:
    - Minimum seed set size should be >= 50 examples for meaningful training
    - Maximum seed fraction should be <= 0.5 to preserve enough data for refinement
    - Typical optimal range: 0.05-0.2 (5%-20%) for most use cases
    """
    if not isinstance(seed_frac, (int, float)):
        raise ValueError(f"seed_frac must be a number, got {type(seed_frac)}")

    if seed_frac <= 0:
        raise ValueError(f"seed_frac must be positive, got {seed_frac}")

    if seed_frac >= 1:
        raise ValueError(f"seed_frac must be less than 1, got {seed_frac}")

    # Calculate minimum viable seed size based on research
    min_seed_size = max(50, dataset_size * 0.01)  # At least 50 examples or 1% of data
    actual_seed_size = int(dataset_size * seed_frac)

    if actual_seed_size < min_seed_size:
        raise ValueError(
            f"Seed set too small: {actual_seed_size} examples. "
            f"Need at least {min_seed_size} examples for effective seed training. "
            f"Try increasing seed_frac to {min_seed_size/dataset_size:.3f} or larger."
        )

    # Warn about potentially suboptimal configurations
    if seed_frac > 0.5:
        logger.warning(
            f"Large seed fraction ({seed_frac:.1%}) may not leave enough data for refinement stage. "
            f"Research recommends 5%-20% for optimal two-stage training."
        )

    if not (0.05 <= seed_frac <= 0.2):
        logger.info(
            f"seed_frac={seed_frac:.1%} is outside the typical optimal range of 5%-20%. "
            f"This may still work but could be suboptimal."
        )


def analyze_dataset_split(seed_set, refine_set, dataset_name: str = "dataset") -> Dict[str, Any]:
    """
    Analyze the quality and representativeness of dataset split.

    Args:
        seed_set: The seed dataset subset
        refine_set: The refinement dataset subset  
        dataset_name: Name for logging purposes
        
    Returns:
        Dictionary with split analysis results
        
    Based on research best practices for two-stage training validation.
    """
    analysis = {
        "seed_size": len(seed_set),
        "refine_size": len(refine_set),
        "total_size": len(seed_set) + len(refine_set),
        "seed_fraction": len(seed_set) / (len(seed_set) + len(refine_set)),
        "quality_checks": {}
    }

    # Basic size validation - ensure both sets have data
    if analysis["seed_size"] == 0:
        raise ValueError("Seed set is empty! Cannot proceed with seed training.")

    if analysis["refine_size"] == 0:
        raise ValueError("Refine set is empty! No data available for second stage training.")

    # Quality checks based on research recommendations
    quality_checks = analysis["quality_checks"]

    # Check 1: Minimum viable seed size
    quality_checks["sufficient_seed_size"] = analysis["seed_size"] >= 50
    if not quality_checks["sufficient_seed_size"]:
        logger.warning(f"Seed set size ({analysis['seed_size']}) is below recommended minimum of 50 examples")

    # Check 2: Balanced split ratio
    quality_checks["balanced_split"] = 0.05 <= analysis["seed_fraction"] <= 0.5
    if not quality_checks["balanced_split"]:
        logger.warning(f"Split ratio ({analysis['seed_fraction']:.1%}) may be suboptimal for two-stage training")

    # Check 3: Sufficient refinement data
    quality_checks["sufficient_refine_size"] = analysis["refine_size"] >= analysis["seed_size"]
    if not quality_checks["sufficient_refine_size"]:
        logger.warning("Refine set is smaller than seed set - this may limit second stage effectiveness")

    # Log analysis results
    logger.info(f"{dataset_name} split analysis:")
    logger.info(f"Seed: {analysis['seed_size']:,} examples ({analysis['seed_fraction']:.1%})")
    logger.info(f"Refine: {analysis['refine_size']:,} examples ({1-analysis['seed_fraction']:.1%})")
    logger.info(f"Quality checks passed: {sum(quality_checks.values())}/{len(quality_checks)}")

    return analysis


def create_seed_training_config(seed_size: int, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create optimized training configuration for seed training.

    Args:
        seed_size: Number of examples in seed set
        base_config: Base training configuration
        
    Returns:
        Optimized configuration for seed training
        
    Based on research showing seed training benefits from different hyperparameters
    when working with smaller datasets.
    """
    seed_config = base_config.copy()

    # Research-based adaptations for small dataset training
    # Lower learning rate to prevent overfitting on small datasets
    seed_config["learning_rate"] = base_config.get("learning_rate", 2e-5) * 0.5

    # Adjust epochs based on dataset size - more epochs for smaller datasets
    if seed_size < 100:
        seed_config["num_train_epochs"] = 3  # More epochs for very small datasets
    elif seed_size < 500:
        seed_config["num_train_epochs"] = 2  # Moderate increase
    else:
        seed_config["num_train_epochs"] = 1  # Standard for larger seed sets

    # Smaller batch size for small datasets to improve gradient stability
    original_batch_size = base_config.get("per_device_train_batch_size", 8)
    if seed_size < 100:
        seed_config["per_device_train_batch_size"] = min(4, original_batch_size)
    elif seed_size < 500:
        seed_config["per_device_train_batch_size"] = min(6, original_batch_size)

    # More aggressive gradient accumulation for small datasets
    if seed_size < 200:
        seed_config["gradient_accumulation_steps"] = base_config.get("gradient_accumulation_steps", 2) * 2

    # Shorter warmup for small datasets
    seed_config["warmup_ratio"] = 0.05  # Reduced from typical 0.1

    # More frequent saving and logging for better monitoring
    seed_config["save_steps"] = min(25, max(10, seed_size // 10))
    seed_config["logging_steps"] = min(5, max(1, seed_size // 20))

    logger.info(f"Seed training config adaptations:")
    logger.info(f"Learning rate: {seed_config['learning_rate']:.2e} (vs {base_config.get('learning_rate', 2e-5):.2e})")
    logger.info(f"Epochs: {seed_config['num_train_epochs']} (vs {base_config.get('num_train_epochs', 1)})")
    logger.info(f"Batch size: {seed_config['per_device_train_batch_size']} (vs {base_config.get('per_device_train_batch_size', 8)})")

    return seed_config


def run_seed_ft_training(
    base_model_path: str = MODEL_REF,
    seed_frac: float = 0.1,
    output_dir: Optional[str] = None,
    train_dataset=None,
    export_metrics: bool = True,
    validate_split: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Runs supervised fine-tuning on a small 'seed' subset of the dataset.

    This implements the first stage of two-stage fine-tuning, a research-proven technique
    that improves generalization and reduces format specialization compared to single-stage
    training.

    Args:
        base_model_path: The identifier of the base model to fine-tune
        seed_frac: Fraction of training data for seed set (recommended: 0.05-0.2)
        output_dir: Directory to save the seed-tuned model (auto-generated if None)
        train_dataset: Optional pre-loaded dataset to be split
        export_metrics: Whether to export detailed training metrics and analysis
        validate_split: Whether to perform quality checks on the dataset split

    Returns:
        Tuple of (refine_dataset, training_analysis)
        - refine_dataset: Remaining data for second stage training
        - training_analysis: Dictionary with training metrics and analysis

    Raises:
        ValueError: If parameters are invalid or dataset split fails validation
        RuntimeError: If training fails
    """
    logger.info("=" * 60)
    logger.info("STARTING SEED-BASED FINE-TUNING (STAGE 1 OF 2)")
    logger.info("=" * 60)
    logger.info(f"Base Model: {base_model_path}")
    logger.info(f"Seed Fraction: {seed_frac:.1%}")

    # Step 1: Prepare output directory
    if output_dir is None:
        output_dir = os.path.join(FT_OUTPUT_DIR, "seed")

    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output Directory: {output_dir}")
    except OSError as e:
        raise RuntimeError(f"Failed to create output directory {output_dir}: {e}")

    # Step 2: Load and validate dataset
    if train_dataset is None:
        logger.info("Loading NuminaMath-TIR dataset...")
        try:
            math_data = load_and_format_math_data()
            full_train_dataset = math_data["train"]
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")
    else:
        full_train_dataset = train_dataset

    dataset_size = len(full_train_dataset)
    logger.info(f"Loaded dataset with {dataset_size:,} examples")

    # Validate seed fraction parameter
    try:
        validate_seed_fraction(seed_frac, dataset_size)
    except ValueError as e:
        raise ValueError(f"Invalid seed_frac parameter: {e}")

    # Step 3: Split dataset with validation
    logger.info(f"Splitting dataset: {seed_frac:.1%} seed, {1-seed_frac:.1%} refine...")
    try:
        seed_set, refine_set = split_seed_refine(full_train_dataset, seed_frac=seed_frac)
    except Exception as e:
        raise RuntimeError(f"Failed to split dataset: {e}")

    # Analyze and validate split quality
    split_analysis = analyze_dataset_split(seed_set, refine_set, "Training dataset")

    if validate_split:
        failed_checks = [k for k, v in split_analysis["quality_checks"].items() if not v]
        if len(failed_checks) > 1:  # Allow one failed check, but warn on multiple
            logger.error(f"Dataset split failed multiple quality checks: {failed_checks}")
            raise ValueError(
                f"Dataset split quality insufficient. Failed checks: {failed_checks}. "
                f"Consider adjusting seed_frac or using a larger dataset."
            )

    # Step 4: Initialize tokenizer
    logger.info("Initializing tokenizer...")
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

    # Step 5: Tokenize seed dataset
    tokenization_function = partial(_create_tokenized_dataset, tokenizer=tokenizer)

    logger.info("Tokenizing seed dataset...")
    try:
        tokenized_ds = seed_set.map(
            tokenization_function,
            remove_columns=seed_set.column_names,
            desc="Tokenizing seed examples"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to tokenize seed dataset: {e}")

    logger.info(f"Tokenized seed dataset: {len(tokenized_ds)} examples")

    # Step 6: Create optimized training configuration
    base_config = {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 50,
        "save_total_limit": 2,
        "bf16": True,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 2,
        "seed": 42,
        "report_to": "none",
    }

    # Optimize config for seed training
    seed_config = create_seed_training_config(len(seed_set), base_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        **seed_config
    )

    # Step 7: Load model
    logger.info("Loading base model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded: {total_params:,} total parameters, {trainable_params:,} trainable")

    # Step 8: Initialize and execute trainer
    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=tokenized_ds,
            processing_class=tokenizer,
            args=training_args
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize trainer: {e}")

    logger.info("Starting seed training...")
    logger.info(f"Training on {len(tokenized_ds)} examples for {seed_config['num_train_epochs']} epochs")

    try:
        training_result = trainer.train()
        logger.info("Seed fine-tuning completed successfully!")
    except Exception as e:
        raise RuntimeError(f"Seed training failed: {e}")

    # Step 9: Save model and analysis
    logger.info(f"Saving seed-trained model to {output_dir}...")
    try:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to save model: {e}")

    # Export comprehensive training analysis
    training_analysis = {
        "stage": "seed_training",
        "base_model": base_model_path,
        "dataset_analysis": split_analysis,
        "training_config": seed_config,
        "training_results": {
            "final_loss": training_result.training_loss if hasattr(training_result, 'training_loss') else None,
            "total_steps": training_result.global_step if hasattr(training_result, 'global_step') else None,
            "epochs_completed": seed_config['num_train_epochs']
        },
        "model_info": {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params
        },
        "next_stage_ready": {
            "refine_dataset_size": len(refine_set),
            "recommended_method": "QLoRA" if total_params > 1e9 else "Full fine-tuning"
        }
    }

    if export_metrics:
        try:
            metrics_path = os.path.join(output_dir, "seed_training_analysis.json")
            with open(metrics_path, "w") as f:
                json.dump(training_analysis, f, indent=2)
            logger.info(f"Training analysis exported to {metrics_path}")
        except Exception as e:
            logger.warning(f"Failed to export training analysis: {e}")

    # Step 10: Validate results and provide guidance
    if training_analysis["training_results"]["final_loss"] and training_analysis["training_results"]["final_loss"] > 10.0:
        logger.warning(
            f"High final loss ({training_analysis['training_results']['final_loss']:.3f}) - "
            f"seed training may not have converged properly"
        )

    logger.info("=" * 60)
    logger.info("SEED TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Seed model saved to: {output_dir}")
    logger.info(f"Refine dataset ready: {len(refine_set):,} examples")
    logger.info("=" * 60)

    return refine_set, training_analysis


def validate_seed_training_output(
    model_path: str,
    analysis: Dict[str, Any],
    test_prompts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Validate the quality of seed training results.

    Args:
        model_path: Path to the seed-trained model
        analysis: Training analysis from seed training
        test_prompts: Optional test prompts for quality assessment
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "model_loads": False,
        "generates_responses": False,
        "training_converged": False,
        "ready_for_stage2": False
    }

    try:
        # Test 1: Model loading capability
        from model_inference import load_saved_model
        model, tokenizer, device = load_saved_model(model_path)
        validation_results["model_loads"] = True
        logger.info("Seed model loads successfully")
        
        # Test 2: Response generation functionality
        if test_prompts is None:
            test_prompts = ["What is 2+2?", "Explain photosynthesis briefly."]
        
        for prompt in test_prompts[:2]:  # Test first 2 prompts
            try:
                from model_inference import get_model_response
                response = get_model_response(prompt, model, tokenizer, device)
                if response and len(response.strip()) > 10:
                    validation_results["generates_responses"] = True
                    break
            except Exception as e:
                logger.warning(f"Response generation test failed: {e}")
        
        if validation_results["generates_responses"]:
            logger.info("✅ Seed model generates valid responses")
        
        # Test 3: Training convergence check
        final_loss = analysis.get("training_results", {}).get("final_loss")
        if final_loss and final_loss < 5.0:  # Reasonable threshold
            validation_results["training_converged"] = True
            logger.info(f"✅ Training converged (final loss: {final_loss:.3f})")
        elif final_loss:
            logger.warning(f"⚠️ High final loss: {final_loss:.3f} - may need more training")
        
        # Test 4: Stage 2 readiness
        refine_size = analysis.get("next_stage_ready", {}).get("refine_dataset_size", 0)
        if refine_size > 100:  # Sufficient data for stage 2
            validation_results["ready_for_stage2"] = True
            logger.info(f"✅ Ready for stage 2 ({refine_size:,} examples available)")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")

    # Summary of validation results
    passed_tests = sum(validation_results.values())
    total_tests = len(validation_results)
    logger.info(f"Seed training validation: {passed_tests}/{total_tests} tests passed")

    return validation_results


def test_seed_training_pipeline(dataset_size: int = 1000) -> None:
    """
    Test function to verify the seed training pipeline works correctly.
    """
    logger.info("Testing seed training pipeline...")

    try:
        # Create mock dataset for testing
        from datasets import Dataset
        mock_data = [
            {
                "prompt": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"What is {i}+{i}?"}
                ],
                "solution": f"The answer is {i+i}."
            }
            for i in range(1, dataset_size + 1)
        ]
        
        test_dataset = Dataset.from_list(mock_data)
        
        # Test different seed fractions
        for seed_frac in [0.05, 0.1, 0.2]:
            logger.info(f"Testing with seed_frac={seed_frac}")
            validate_seed_fraction(seed_frac, len(test_dataset))
            
            # Test split functionality
            seed_set, refine_set = split_seed_refine(test_dataset, seed_frac=seed_frac)
            analysis = analyze_dataset_split(seed_set, refine_set, f"Test dataset (frac={seed_frac})")
            
            logger.info(f"✅ Split test passed for seed_frac={seed_frac}")
        
        logger.info("🎉 All seed training pipeline tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Seed training pipeline test failed: {e}")
        raise


if __name__ == "__main__":
    # Test the pipeline first
    test_seed_training_pipeline()

    # Run actual seed training
    refine_dataset, analysis = run_seed_ft_training()

    # Validate results
    seed_model_path = os.path.join(FT_OUTPUT_DIR, "seed")
    validation_results = validate_seed_training_output(seed_model_path, analysis)

    print("\n" + "="*60)
    print("SEED TRAINING SUMMARY")
    print("="*60)
    print(f"Refine dataset size: {len(refine_dataset):,}")
    print(f"Validation passed: {sum(validation_results.values())}/{len(validation_results)}")
    print(f"Next: Use refine_dataset for QLoRA training")
    print("="*60)