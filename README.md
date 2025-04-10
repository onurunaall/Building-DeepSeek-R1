# My Experimental RL Language Model Fine Tuning

This repository contains my experimental playground for combining Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT) on a smaller language model (specifically, Qwen 0.5B Instruct). It’s built on a small base model and is designed to encourage better reasoning and more structured responses. My main goal here was to see if I could nudge the model towards better reasoning, particularly for math problems, and encourage it to structure its answers more clearly by showing its thought process first. 

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
This script handles loading data, setting up the model, running both training phases, and doing a quick inference check at the end.

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

#### main.py: The main script that runs the different stages (RL, SFT, inference).
settings.py: Contains most configuration settings like model names, output paths, and training parameters.
model_initialization.py: Code to load the initial base model and tokenizer.
dataset_preparation.py: Loads and formats the datasets (NuminaMath-TIR, Bespoke-Stratos-17k) into the required dialogue structure.
reward_metrics.py: Defines all the custom reward functions used during RL (accuracy, format, reasoning steps, length penalty, repetition penalty).
reinforcement_training.py: Handles the RL training loop using GRPOTrainer.
fine_tuning.py: Handles the SFT training loop using SFTTrainer.
model_inference.py: Code for loading a saved (trained) model and running inference.
requirements.txt: Lists the Python dependencies.
README.md: This file!
Some Notes & Caveats
Experimental: This is definitely experimental code. It's not optimized for speed or memory and hasn't been rigorously tested for production use. Use it as a learning resource or a starting point.
Training Flow: The main.py currently runs RL, then SFT (on the base model, not the RL-tuned one), then tests the RL model. You might want to modify this flow depending on your goals.
Hardware: Training LLMs, even smaller ones, requires significant compute resources (GPU recommended). The settings use bfloat16 to help manage memory.

### Diagrams and Drawings
I’ve included a few simple, hand-drawn diagrams to help explain the training process and data flow. These diagrams are intentionally basic to keep the focus on the code rather than on professional graphics. They cover topics such as:

The overall training pipeline (data → RL training → SFT → inference)
A simplified flowchart of the reinforcement learning loop
A basic diagram of the file structure
Feel free to take these drawings as a starting point, or modify them to suit your own understanding and style.

### License
This project is licensed under the MIT License.

##### Inspiration
This project is inspired by FareedKhan-dev/train-deepseek-r1. This is my own implementation based on that repository.



# Experimenting with RL & SFT for Better LLM Reasoning

This repository contains my experimental playground for combining Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT) on a smaller language model (specifically, Qwen 0.5B Instruct). 

Think of this as a learning exercise rather than a polished product. It's built using Python and leans heavily on the fantastic Hugging Face libraries (`transformers`, `trl`, `datasets`).

## What I Tried to Achieve

* **Structured Output:** Train the model to use `<think>` and `<answer>` tags to separate its reasoning steps from the final solution[cite: 1]. This makes it easier to see *how* it got there.
* **Improved Reasoning (especially Math):** Use RL with custom rewards to incentivize accurate answers for math problems sourced from the `NuminaMath-TIR` dataset. This involved using a math verification library (`math_verify` [cite: 1]) to check correctness.
* **Reward Engineering:** Beyond just accuracy, I built rewards to encourage good formatting[cite: 1], penalize excessive repetition[cite: 1], and even factor in the length of the response relative to its accuracy[cite: 1]. The idea was to balance correctness with conciseness and clarity.
* **Combining RL and SFT:** Explore how RL (using TRL's `GRPOTrainer`) and SFT (using TRL's `SFTTrainer` on the `Bespoke-Stratos-17k` dataset) could potentially work together or complement each other, although the main pipeline currently tests the RL model output.

## Getting Started

### Installation

1.  Clone this repository:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  Install the necessary Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    *(You'll likely want to do this in a virtual environment).*

### Running the Pipeline

To run the default sequence (RL training -> SFT training -> Inference test on RL model), execute:
```bash



Inspiration
This project was inspired by and adapted from initial ideas explored in FareedKhan-dev/train-deepseek-r1. This is my own implementation based on concepts from that repository.

License
Licensed under the MIT License.
