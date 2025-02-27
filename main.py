# main.py
from train_grpo import execute_grpo_training
from train_sft import execute_sft_training
from inference import load_trained_model, run_inference_test
from config import OUTPUT_DIR_GRPO

def main_pipeline() -> None:
    """
    Main function to run the entire training and inference pipeline.
    It executes GRPO training, then SFT training, and finally runs an inference test.
    """
    print("Starting GRPO training...")
    execute_grpo_training()
    
    print("Starting SFT training...")
    execute_sft_training()
    
    print("Running inference test on GRPO trained model...")
    model_instance, tokenizer_instance, device_instance = load_trained_model(OUTPUT_DIR_GRPO)
    sample_input = "how are you?"
    generated_output = run_inference_test(sample_input, model_instance, tokenizer_instance, device_instance)
    print("Inference Result:")
    print(generated_output)

if __name__ == "__main__":
    main_pipeline()
