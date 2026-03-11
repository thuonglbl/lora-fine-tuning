#script to launch cross_validation, only the training part
import os
from argparse import ArgumentParser

import datasets

from CIT.evaluation.utils import load_jsonl, save_jsonl
from CIT.training.utils import (
    Kfolds,
    build_trainer,
    format_dataset,
    load_model_with_lora,
)

parser = ArgumentParser()
parser.add_argument("--base_model_name", type=str, default="unsloth/Meta-Llama-3.1-8B")
parser.add_argument(
    "--path_all_questions",
    type=str,
    default="./src/CIT/training/data/split_balance_zeros/urls_formatted/run_22.5/new_retrieved_context/all_data_transformed.jsonl",
)
parser.add_argument(
    "--num_folds", type=int, default=5, help="number of folds for cross-validation"
)
parser.add_argument(
    "--r", type=int, default=16, help="rank of the low-rank approximation"
)
parser.add_argument(
    "--alpha",
    type=int,
    default=32,
    help="alpha parameter for the low-rank approximation",
)
parser.add_argument("--lora_dropout", type=float, default=0)
parser.add_argument(
    "--run_name", type=str, default="CIT_CV_llama3.1-20.5", help="name of the run in wandb"
)
parser.add_argument(
    "--batch_size", type=int, default=1, help="batch size for training"
)
parser.add_argument(
    "--num_epochs", type=int, default=1, help="number of epochs for training"
)
parser.add_argument(
    "--training_output_dir",
    type=str,
    default="./src/CIT/training/models/cv/data",
    help="output directory for the splits",
)
parser.add_argument(
    "--output_models_dir",
    type=str,
    default="./src/CIT/training/models/cv/ft_20.5",
    help="output directory for the models",
)


if __name__ == "__main__":
    args = parser.parse_args()
    base_model_name = args.base_model_name
    r = args.r
    alpha = args.alpha
    lora_dropout = args.lora_dropout
    run_name = args.run_name
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    path_all_questions = args.path_all_questions
    num_folds = args.num_folds
    output_dir = args.training_output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_models_dir = args.output_models_dir
    if not os.path.exists(output_models_dir):
        os.makedirs(output_models_dir)

    all_training_set=load_jsonl(path_all_questions)
    print(f"Loaded {len(all_training_set)} questions and answers from {path_all_questions}")
    # Create the Kfolds object
    kfolds = Kfolds(all_training_set, n_splits=num_folds)
    splits = kfolds.splits
    print(f"Created {num_folds} folds for cross-validation")

    # Iterate over the folds
    for i, (train_set,test_set) in enumerate(splits):
        print(f"Fold {i + 1}/{num_folds}")
        # Save the train and validation sets to JSONL files
        path_train = f"{output_dir}/train_fold_{i + 1}.jsonl"
        path_test= f"{output_dir}/test_fold_{i + 1}.jsonl"

        #remove facultative urls from the train set
        for question in train_set:
            if "facultative_urls" in question:
                del question["facultative_urls"]
        save_jsonl(path_train, train_set)
        save_jsonl(path_test, test_set)
        print(f"Saved {len(train_set)} questions and answers to {path_train}")
        print(f"Saved {len(test_set)} questions and answers to {path_test}")

        # Train the model on the current fold
        model, tokenizer = load_model_with_lora(
        base_model_name=base_model_name,
        r=r,
        alpha=alpha,
        lora_dropout=lora_dropout,
    )
        #load the dataset and split it into train and validation sets
        dataset_questions = datasets.load_dataset(
            "json", data_files=path_train, split="train"
        )
        train_dataset, val_dataset = format_dataset(
            dataset_questions,
        ) 
        print(f"end of sentence token: {tokenizer.eos_token}")
        # define training arguments
        trainer = build_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=output_dir,
            run_name=run_name,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )
        # Train the model
        trainer.train()
        # Save the model
        model.save_pretrained(f"{output_models_dir}/model_fold_{i + 1}")
        tokenizer.save_pretrained(f"{output_models_dir}/model_fold_{i + 1}")
        print(f"Saved model for fold {i + 1} to {output_models_dir}/model_fold_{i + 1}")

