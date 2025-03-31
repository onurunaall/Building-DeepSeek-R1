# My Experimental RL Language Model

This project is an experimental implementation of a reinforcement learning (RL) and fine-tuning (SFT) pipeline for training a language model. It’s built on a small base model and is designed to encourage better reasoning and more structured responses. The code is intentionally kept simple and unpolished as it’s mainly a research and learning experiment.

## Installation
Clone this repository and install the required packages:
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
pip install -r requirements.txt
```
### Usage
Run the complete training and inference pipeline with:
```bash 
python main.py
```
This command will execute the reinforcement learning training loop, run the fine-tuning stage test the trained model with a sample input.

### File Structure
```bash
project/
├── main.py                # Runs the entire pipeline (training and inference)
├── settings.py            # Configuration settings for the project
├── model_inference.py     # Code for loading and running model inference
├── reinforcement_training.py  # RL training code
├── fine_tuning.py         # Fine-tuning (SFT) training code
├── reward_metrics.py      # Definitions of reward functions
├── dataset_preparation.py # Dataset loading and preprocessing code
└── README.md              # This file
```

### Diagrams and Drawings
I’ve included a few simple, hand-drawn diagrams to help explain the training process and data flow. These diagrams are intentionally basic to keep the focus on the code rather than on professional graphics. They cover topics such as:

The overall training pipeline (data → RL training → SFT → inference)
A simplified flowchart of the reinforcement learning loop
A basic diagram of the file structure
Feel free to take these drawings as a starting point, or modify them to suit your own understanding and style.

### Notes
This project is experimental and not intended for production use.

### License
This project is licensed under the MIT License.

##### Inspiration
This project is inspired by FareedKhan-dev/train-deepseek-r1. This is my own implementation based on that repository.
