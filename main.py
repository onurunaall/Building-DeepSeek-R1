from reinforcement_training import run_rl_training
from fine_tuning import run_ft_training
from model_inference import load_saved_model, get_model_response
from settings import RL_OUTPUT_DIR

def main_pipeline() -> None:
    """
    This is the main pipeline that runs our whole process.
    It first does reinforcement learning training, then fine-tuning training,
    and finally runs an inference test using the RL-trained model.
    """
    print("Starting reinforcement learning training...")
    run_rl_training()
    
    print("Starting fine-tuning training...")
    run_ft_training()
    
    print("Running inference test on the RL-trained model...")
    
    # Load the saved RL-trained model and its tokenizer
    model_instance, tokenizer_instance, device_instance = load_saved_model(RL_OUTPUT_DIR)
    
    sample_input = "how are you?"

    # Get the model's response for our sample input
    generated_output = get_model_response(sample_input, model_instance, tokenizer_instance, device_instance)
    
    print("Inference Result:")
    print(generated_output)

if __name__ == "__main__":
    main_pipeline()
