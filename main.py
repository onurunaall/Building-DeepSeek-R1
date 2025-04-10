from reinforcement_training import run_rl_training
from fine_tuning import run_ft_training
from model_inference import load_saved_model, get_model_response
from settings import RL_OUTPUT_DIR, FT_OUTPUT_DIR

def main_pipeline() -> None:
    """
    This is the main pipeline that runs our whole process.
    It first does reinforcement learning training, then fine-tunes the RL-trained model,
    and finally runs an inference test using the final fine-tuned model.
    """
    print("Starting reinforcement learning training...")
    run_rl_training()

    print("Starting fine-tuning training on the RL-trained model...")
    # Pass the RL output directory as the input model path for fine-tuning
    run_ft_training(input_model_path=RL_OUTPUT_DIR)

    print("Running inference test on the final fine-tuned model...")

    # Load the saved Fine-Tuned model and its tokenizer
    model_instance, tokenizer_instance, device_instance = load_saved_model(FT_OUTPUT_DIR)

    sample_input = "how are you?"

    # Get the model's response for our sample input
    generated_output = get_model_response(sample_input, model_instance, tokenizer_instance, device_instance)

    print("Inference Result:")
    print(generated_output)

if __name__ == "__main__":
    main_pipeline()
