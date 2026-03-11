#!/bin/bash
cd ../


##########################################################
########evaluate the base model###########################
##########################################################

MODEL_NAME="llama3.1:8b"
SPLITS_FOLDER="./src/CIT/evaluation/proof_urls_as_proxy/data/answers_to_compute"
ALWAYS_DO_RETRIEVAL="True"
ADD_EXTERNAL_LINKS_DOCS="False"
ANSWERS_ALREADY_COMPUTED="False"
NOT_CITING_SOURCE="True"
TEST_ANSWERS_FOLDER_BASE="./src/CIT/evaluation/proof_urls_as_proxy/data/answers_to_compute/test_answers"
SCORES_FOLDER_BASE="./src/CIT/evaluation/proof_urls_as_proxy/data/answers_to_compute/scores"


# Run the evaluation script
python base_cv_eval.py \
  --model_name "$MODEL_NAME" \
  --splits_folder "$SPLITS_FOLDER" \
  --test_answers_folder "$TEST_ANSWERS_FOLDER_BASE" \
  --scores_folder "$SCORES_FOLDER_BASE" \
  --always_do_retrieval "$ALWAYS_DO_RETRIEVAL" \
  --add_external_links_docs "$ADD_EXTERNAL_LINKS_DOCS" \
  --answers_already_computed "$ANSWERS_ALREADY_COMPUTED"\
  --not_citing_source "$NOT_CITING_SOURCE"





