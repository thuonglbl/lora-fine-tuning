#!/bin/bash
cd ../


##########################################################
########evaluate the base model###########################
##########################################################
NB_RUNS=1
MODEL_NAME="random_urls_model_baseline"
TOP_K=1
SPLITS_FOLDER="./src/CIT/training/models/cv/data/27.5_2epochs"
ALWAYS_DO_RETRIEVAL="True"
ADD_EXTERNAL_LINKS_DOCS="False"
ANSWERS_ALREADY_COMPUTED="False"

for ((run=1; run<=NB_RUNS; run++)); do
  echo "Run $run of $NB_RUNS"
  TEST_ANSWERS_FOLDER_BASE="./src/CIT/training/models/cv/data/test_answers/random_urls_model_baseline/top_k${TOP_K}/run${run}"
  SCORES_FOLDER_BASE="./src/CIT/training/models/cv/scores/random_urls_model_baseline/top_k${TOP_K}/run${run}"
  # Run the evaluation script
  python base_cv_eval.py \
    --model_name "$MODEL_NAME" \
    --top_k "$TOP_K" \
    --splits_folder "$SPLITS_FOLDER" \
    --test_answers_folder "$TEST_ANSWERS_FOLDER_BASE" \
    --scores_folder "$SCORES_FOLDER_BASE" \
    --always_do_retrieval "$ALWAYS_DO_RETRIEVAL" \
    --add_external_links_docs "$ADD_EXTERNAL_LINKS_DOCS" \
    --answers_already_computed "$ANSWERS_ALREADY_COMPUTED"
    echo "Run $run completed."
done




