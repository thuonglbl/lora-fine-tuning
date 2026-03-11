#!/bin/bash
cd ../
# Define variables
WAITING_TIME=12000 # delay in seconds
PATH_ALL_QUESTIONS="./src/CIT/training/data/split_balance_zeros/urls_formatted/run_22.5/new_retrieved_context/all_data_transformed.jsonl"
NUM_FOLDS=5
TRAINING_OUTPUT_DIR="./src/CIT/training/models/cv/data/22.5"
OUTPUT_MODELS_DIR="./src/CIT/training/models/cv/ft_22.5"


echo "Waiting for $WAITING_TIME seconds before starting the training script..."
sleep $WAITING_TIME

python perform_cross_validation_training.py \
  --path_all_questions "$PATH_ALL_QUESTIONS" \
  --num_folds "$NUM_FOLDS" \
  --training_output_dir "$TRAINING_OUTPUT_DIR" \
  --output_models_dir "$OUTPUT_MODELS_DIR"

echo "Cross-validation training completed."