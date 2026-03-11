# This script matches questions and answers from two different files based on their IDs.
#!/bin/bash
REPHRASED_QUESTIONS_PATH="./src/CIT/evaluation/QA_generation/answers_clean/rephrased_questions.jsonl"
REPHRASED_ANSWERS_PATH="./src/CIT/evaluation/QA_generation/answers_clean/rephrased_answers.jsonl"
OUTPUT_PATH="./src/CIT/evaluation/QA_generation/answers_clean/matched_questions_answers.jsonl"

python ../QA_generation/match_questions_answers.py \
    --rephrased_questions_path "$REPHRASED_QUESTIONS_PATH" \
    --rephrased_answers_path "$REPHRASED_ANSWERS_PATH" \
    --output_path "$OUTPUT_PATH"