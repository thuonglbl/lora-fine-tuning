#!/bin/bash
cd ./src/CIT/evaluation
python compute_url_precision_recall.py \
    --directory "../documents/confluence_json_without_root_with_titles" \
    --embedding_model "sentence-transformers/all-MiniLM-L6-v2" \
    --top_k 6 \
    --threshold 0.1 \
    --num_predict_tokens 2000 \
    --questions_with_urls_path "./src/CIT/training/data/split_balance_zeros/urls_formatted/test_data_transformed.jsonl" \
    --model_name "user_ft0505_sd_urls" \
    --answers_path "./src/CIT/evaluation/results/balanced_split/answers/ft_sd_urls/user_ft0505_sd_urls_run2.jsonl" \
    --answers_already_computed "false" \
    --output_path "./src/CIT/evaluation/results/balanced_split/precision_recall/ft_sd_urls/user_ft0505_sd_urls_run2.json" \