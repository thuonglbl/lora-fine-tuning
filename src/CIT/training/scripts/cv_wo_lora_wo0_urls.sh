#!/bin/bash
cd ../
WAITING_TIME=21000 # delay in seconds

####################################################################
###############Train the base model#################################
####################################################################
PIPELINE_NAME="wo_lora_r128_a16_1ep_wo0_urls"
PATH_ALL_QUESTIONS="./src/CIT/training/data/split_balance_zeros/urls_formatted/run_22.5/new_retrieved_context/data_with_at_least_one_url.jsonl"
#PATH_SAMPLE_QUESTIONS="./src/CIT/training/data/samples/sample.jsonl"
NUM_FOLDS=5
NUM_EPOCHS=1
#lora parameters
RANK=128
ALPHA=16
RUN_NAME="cv_${PIPELINE_NAME}"
TRAINING_OUTPUT_DIR="./src/CIT/training/models/cv/data/${PIPELINE_NAME}"
OUTPUT_MODELS_DIR="./src/CIT/training/models/cv/ft_${PIPELINE_NAME}"

#countdown to start the training script
for ((i=WAITING_TIME; i>0; i--)); do
  echo -ne "Starting in $i seconds...\r"
  sleep 1
done


if true; then

  python perform_cross_validation_training.py \
    --path_all_questions "$PATH_ALL_QUESTIONS" \
    --num_folds "$NUM_FOLDS" \
    --r "$RANK" \
    --alpha "$ALPHA" \
    --num_epochs "$NUM_EPOCHS" \
    --run_name "$RUN_NAME" \
    --training_output_dir "$TRAINING_OUTPUT_DIR" \
    --output_models_dir "$OUTPUT_MODELS_DIR"
fi
echo "Cross-validation training completed."

sleep 1
#create the ollama models with the finetuned models

OUTPUT_MODELFILES_DIR="./src/CIT/training/models/cv/modelfiles/ft_${PIPELINE_NAME}"
if true; then
python create_ollama_models.py \
  --output_modelfiles_dir "$OUTPUT_MODELFILES_DIR" \
  --output_models_dir "$OUTPUT_MODELS_DIR"

sleep 1
fi
#evaluate the finetuned models
SPLITS_FOLDER="./src/CIT/training/models/cv/data/27.5_2epochs"
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
