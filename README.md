# Building-DeepSeek-R1

## What This Project Does

This project trains a language model to solve math problems with clear, step-by-step reasoning. Two training methods are used:
- **GRPO Training:** A reinforcement learning step where the model learns to give good answers based on custom reward functions.
- **SFT Training:** A supervised fine-tuning step where the model learns from high-quality, curated examples.

After training, the model is tested by providing a sample question and evaluating its response.

## Purpose

The goal is to build a reasoning-enhanced language model by:
- Learning from math problems using reinforcement learning (GRPO).
- Refining the model using supervised fine-tuning (SFT) with curated data.
- Testing the model through inference.

## File Overview

- **config.py:**  
  Contains global settings and configuration classes.  
  _What it does:_ Stores the model name, output directories, and training parameters in one centralized place.  
  _Why:_ To allow easy updates to settings without modifying multiple files.

- **data_prep.py:**  
  Loads and formats the dataset.  
  _What it does:_ Converts raw math problems into a standardized conversation format (with system and user messages) and validates the data.  
  _Why:_ To ensure the training data is consistent and correctly formatted.

- **model_setup.py:**  
  Initializes the model and tokenizer.  
  _What it does:_ Loads the base model and tokenizer and provides a function to perform a test inference.  
  _Why:_ To guarantee that both training and inference use the same model initialization process.

- **reward_functions.py:**  
  Implements reward functions for training.  
  _What it does:_ Contains functions to compute rewards based on accuracy, format correctness, reasoning steps, cosine scaling, and repetition penalty.  
  _Why:_ These reward functions guide the reinforcement learning process.

- **train_grpo.py:**  
  Runs the GRPO training loop (reinforcement learning).  
  _What it does:_ Loads the dataset, initializes the model, registers reward functions, and trains the model using GRPO.  
  _Why:_ To teach the model to generate correct and well-formatted answers.

- **train_sft.py:**  
  Runs the SFT training loop (supervised fine-tuning).  
  _What it does:_ Loads a curated dataset, fine-tunes the model, and saves the refined model.  
  _Why:_ To further improve the model using high-quality examples.

- **inference.py:**  
  Provides functions to load a trained model and run inference.  
  _What it does:_ Loads the saved model and tokenizer, and generates responses for sample inputs.  
  _Why:_ To check the performance of the trained model.

- **main.py:**  
  The main script that runs the complete pipeline.  
  _What it does:_ Executes GRPO training, then SFT training, and finally tests inference with the trained model.  
  _Why:_ To run the entire training and testing process with one command.
