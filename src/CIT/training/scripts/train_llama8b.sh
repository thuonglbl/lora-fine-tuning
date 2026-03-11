#!/bin/bash
cd ../
# Define variables
BASE_MODEL_NAME="unsloth/Meta-Llama-3.1-8B"
PATH_TRAIN_QUESTIONS="./src/CIT/training/data/split_balance_zeros/urls_formatted/training_data_without_facultative_urls_transformed.jsonl"
R=16
ALPHA=32
LORA_DROPOUT=0
RUN_NAME="CIT_llama3.1-balanced_split_new_urls"
BATCH_SIZE=1
NUM_EPOCHS=2
TRAINING_OUTPUT_DIR="./src/CIT/training/models/chekpoints"
OUTPUT_MODEL_PATH="./src/CIT/training/models/final_models/ft_05.5_new_urls"

# Run the training script with the arguments
python training_script.py \
  --base_model_name "$BASE_MODEL_NAME" \
  --path_train_questions "$PATH_TRAIN_QUESTIONS" \
  --r "$R" \
  --alpha "$ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --run_name "$RUN_NAME" \
  --batch_size "$BATCH_SIZE" \
  --num_epochs "$NUM_EPOCHS" \
  --training_output_dir "$TRAINING_OUTPUT_DIR" \
  --output_model_path "$OUTPUT_MODEL_PATH"