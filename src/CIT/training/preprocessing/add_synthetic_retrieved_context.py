# this script is used to add synthetic retrieved context to questions and answers.
# It takes a JSONL file containing questions and answers, retrieves the context from the documents, and adds synthetic noise to the context.
# This new data will be used  to train the model.

import argparse
import json
import os

from CIT.evaluation.utils import (
    add_noise_to_context,
    build_context_from_file_paths,
    load_jsonl,
    save_jsonl,
)

parser = argparse.ArgumentParser(
    description="Add synthetic retrieved context to questions."
)
parser.add_argument(
    "--QA_path",
    type=str,
    default="./src/CIT/training/data/split_balance_zeros/urls_formatted/run_22.5/training_data_without_facultative_urls_transformed.jsonl",
    help="Path to the JSONL file containing the questions.",
)
parser.add_argument(
    "--dir_documents",
    type=str,
    default="./src/CIT/documents/confluence_json_without_root_with_titles",
    help="Directory containing the documents.",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="./src/CIT/training/data/split_balance_zeros/urls_formatted/run_22.5/new_retrieved_context/training_data_without_facultative_urls_transformed.jsonl",
    help="Path to the output JSONL file with retreived_context in it",
)

if __name__ == "__main__":
    args = parser.parse_args()
    QA_path = args.QA_path
    dir_documents = args.dir_documents
    output_path = args.output_path

    data = load_jsonl(QA_path)
    print(f"Loaded {len(data)} questions and answers from {QA_path}")


    path_mapping_urls_path=os.path.join(
        dir_documents, "mappings", "mapping_urls_paths.json"
    )
    with open(path_mapping_urls_path, "r") as f:
        map_urls_paths = json.load(f)

    augmented_data = []

    for question in data:
        urls = question["urls"]
        paths = [map_urls_paths[url] for url in urls]
        context = build_context_from_file_paths(paths)
        nb_noise = int(5 - len(urls))
        if nb_noise < 0:
            nb_noise = 0
        synthetic_context = add_noise_to_context(
            context, dir_documents, nb_noise_doc=nb_noise
        )
        question["retrieved_context"] = synthetic_context
        augmented_data.append(question)

    save_jsonl(output_path, augmented_data)
    print(
        f"Saved {len(augmented_data)} questions and answers with synthetic retrieved context to {output_path}"
    )
