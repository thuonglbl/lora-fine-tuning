#!/bin/bash

DIRECTORY="./src/CIT/documents/confluence_json_without_root_with_titles"
QUESTIONS_WITH_URLS_PATH="all"
MODEL_NAME="llama3.3"
ANSWERS_PATH="./src/CIT/evaluation/QA_generation/synthetic_answers/llama3.3_answers.jsonl"

python ../QA_generation/answer_questions.py \
    --directory "$DIRECTORY" \
    --questions_with_urls_path "$QUESTIONS_WITH_URLS_PATH" \
    --model_name "$MODEL_NAME" \
    --answers_path "$ANSWERS_PATH"
