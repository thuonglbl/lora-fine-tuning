#!/bin/bash
cd ../

# Function to check if a directory contains subdirectories
has_subdirs() {
    find "$1" -mindepth 1 -type d | grep -q .
}


ROOT_FOLDER="./src/CIT/training/models/cv/data/test_answers"

# Go through all directories under ROOT_FOLDER
find "$ROOT_FOLDER" -type d | while read -r DIR; do
    if ! has_subdirs "$DIR"; then
        # Replace 'data/test_answers' with 'scores'
        SCORES_FOLDER="${DIR/data\/test_answers/scores}"

        echo "Leaf folder found: $DIR"
        echo "Calling Python with: $SCORES_FOLDER"

        # Call the Python function here
        python compute_quality_from_answers.py \
            --answers_folder "$DIR" \
            --answers_with_quality_folder "$DIR" \
            --scores_folder "$SCORES_FOLDER"
    fi
done



