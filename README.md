# My Experimental RL Language Model Fine Tuning

This repository contains my experimental playground for combining Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT) on a smaller language model (specifically, Qwen 0.5B Instruct). It's built on a small base model and is designed to encourage better reasoning and more structured responses. My main goal here was to see if I could nudge the model towards better reasoning, particularly for math problems, and encourage it to structure its answers more clearly by showing its thought process first. 

### What I Tried to Achieve
* **Structured Output:** Train the model to use `<think>` and `<answer>` tags to separate its reasoning steps from the final solution. This makes it easier to see *how* it got there.
* **Improved Reasoning (especially Math):** Use RL with custom rewards to incentivize accurate answers for math problems sourced from the `NuminaMath-TIR` dataset. This involved using a math verification library (`math_verify`) to check correctness.
* **Reward Engineering:** Beyond just accuracy, I built rewards to encourage good formatting, penalize excessive repetition, and even factor in the length of the response relative to its accuracy. The idea was to balance correctness with conciseness and clarity.
* **Combining RL and SFT:** Explore how RL (using TRL's `GRPOTrainer`) and SFT (using TRL's `SFTTrainer` on the `NuminaMath-TIR` dataset) could potentially work together or complement each other. The pipeline uses the RL-trained model as a starting checkpoint for the subsequent SFT stages.
  
### Installation
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
├── main.py 
├── settings.py    
├── model_inference.py
├── model_initialization.py     
├── reinforcement_training.py  
├── fine_tuning.py
├── fine_tuning_seed.py
├── fine_tuning_qlora.py         
├── reward_metrics.py      
├── dataset_preparation.py
├── plot_results.py 
└── README.md              

```
*main.py: The main script that runs the different stages (RL, SFT, inference).*
*settings.py: Contains most configuration settings like model names, output paths, and training parameters.*
*model_initialization.py: Code to load the initial base model and tokenizer.*
*dataset_preparation.py: Loads and formats the datasets (NuminaMath-TIR) into the required dialogue structure.*
*reward_metrics.py: Defines all the custom reward functions used during RL (accuracy, format, reasoning steps, length penalty, repetition penalty).*
*reinforcement_training.py: Handles the RL training loop using GRPOTrainer.*
*fine_tuning.py: Handles the SFT training loop using SFTTrainer.*
*fine_tuning_seed.py: Handles seed-based partial dataset fine-tuning.*
*fine_tuning_qlora.py: Handles QLoRA parameter-efficient fine-tuning.*
*model_inference.py: Code for loading a saved (trained) model and running inference.*
*plot_results.py: Generates comparison plots from evaluation metrics.*

### Some Notes & Caveats
Experimental: This is definitely experimental code. It's not optimized for speed or memory and hasn't been rigorously tested for production use. I am using it as a learning resource or a starting point.

Hardware: Training LLMs, even smaller ones, requires significant compute resources (GPU recommended). The settings use bfloat16 to help manage memory.

### License
This project is licensed under the MIT License.

##### Inspiration
*This project is inspired by FareedKhan-dev/train-deepseek-r1. Check it out for sure. This is my own implementation based on that repository.*
