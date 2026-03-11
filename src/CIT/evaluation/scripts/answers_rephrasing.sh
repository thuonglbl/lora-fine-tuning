#!/bin/bash

INPUT_PATH="./src/CIT/evaluation/QA_generation/answers_clean/checked_answers_format.jsonl"
OUPUT_PATH="./src/CIT/evaluation/QA_generation/answers_clean/rephrased_answers.jsonl"
MODEL_NAME="llama3.3"
NUM_WORKERS=2

python ../QA_generation/rephrase_answers.py \
    --input_path "$INPUT_PATH" \
    --output_path "$OUPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --num_workers "$NUM_WORKERS"
