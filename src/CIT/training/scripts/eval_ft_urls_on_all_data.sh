#!/bin/bash
cd ../
PIPELINE_NAME="28.5_with_at_least_1_url"

#Train the base model
WAITING_TIME=1 # delay in seconds
PATH_ALL_QUESTIONS="./src/CIT/training/data/split_balance_zeros/urls_formatted/run_22.5/new_retrieved_context/data_with_at_least_one_url.jsonl"
#PATH_ALL_QUESTIONS="./src/CIT/training/data/samples/sample.jsonl"
NUM_FOLDS=5
NUM_EPOCHS=2
RUN_NAME="cv_${PIPELINE_NAME}"
TRAINING_OUTPUT_DIR="./src/CIT/training/models/cv/data/${PIPELINE_NAME}"
OUTPUT_MODELS_DIR="./src/CIT/training/models/cv/ft_${PIPELINE_NAME}"


#countdown to start the training script
for ((i=WAITING_TIME; i>0; i--)); do
  echo -ne "Starting in $i seconds...\r"
  sleep 1
done


sleep 2
#create the ollama models with the finetuned models

OUTPUT_MODELFILES_DIR="./src/CIT/training/models/cv/modelfiles/ft_${PIPELINE_NAME}"
if false; then
python create_ollama_models.py \
  --output_modelfiles_dir "$OUTPUT_MODELFILES_DIR" \
  --output_models_dir "$OUTPUT_MODELS_DIR"

sleep 2
fi


#evaluate the finetuned models
PIPELINE_NAME="02.06_with_at_least_1_url_eval_all_data"
SPLITS_FOLDER="./src/CIT/training/models/cv/data/27.5_2epochs" #use the splits from a CV where all data is used, relevant with splits of CV 28.5_with_at_least_1_url, no cheating
TEST_ANSWERS_FOLDER="./src/CIT/training/models/cv/data/test_answers/ft_${PIPELINE_NAME}/ft"
SCORES_FOLDER="./src/CIT/training/models/cv/scores/ft_${PIPELINE_NAME}/ft"
ALWAYS_DO_RETRIEVAL="True"
ADD_EXTERNAL_LINKS_DOCS="False"
ANSWERS_ALREADY_COMPUTED="False"

if true; then
python perform_cross_validation_evaluation.py \
  --splits_folder "$SPLITS_FOLDER" \
  --models_folder "$OUTPUT_MODELS_DIR" \
  --test_answers_folder "$TEST_ANSWERS_FOLDER" \
  --scores_folder "$SCORES_FOLDER" \
  --always_do_retrieval "$ALWAYS_DO_RETRIEVAL" \
  --add_external_links_docs "$ADD_EXTERNAL_LINKS_DOCS" \
  --answers_already_computed "$ANSWERS_ALREADY_COMPUTED"

sleep 2
fi


##########################################################
########evaluate the base model###########################
##########################################################
MODEL_NAME="llama3.1:8b"
TEST_ANSWERS_FOLDER_BASE="./src/CIT/training/models/cv/data/test_answers/ft_${PIPELINE_NAME}/base"
SCORES_FOLDER_BASE="./src/CIT/training/models/cv/scores/ft_${PIPELINE_NAME}/base"

if false; then
python base_cv_eval.py \
  --model_name "$MODEL_NAME" \
  --splits_folder "$TRAINING_OUTPUT_DIR" \
  --test_answers_folder "$TEST_ANSWERS_FOLDER_BASE" \
  --scores_folder "$SCORES_FOLDER_BASE" \
  --always_do_retrieval "$ALWAYS_DO_RETRIEVAL" \
  --add_external_links_docs "$ADD_EXTERNAL_LINKS_DOCS" \
  --answers_already_computed "$ANSWERS_ALREADY_COMPUTED"
fi

echo "Base model evaluation completed."
sleep 2