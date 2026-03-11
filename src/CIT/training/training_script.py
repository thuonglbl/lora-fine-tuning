import os
from argparse import ArgumentParser

import datasets

from CIT.training.utils import build_trainer, format_dataset, load_model_with_lora

os.environ['UNSLOTH_RETURN_LOGITS'] = '1'


parser = ArgumentParser()
parser.add_argument("--base_model_name", type=str, default="unsloth/Meta-Llama-3.1-8B")
parser.add_argument(
    "--path_train_questions",
    type=str,
    default="./src/CIT/training/data/split_balance_zeros/urls_formatted/run_22.5/new_retrieved_context/training_data_without_facultative_urls_transformed.jsonl",
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
    "--run_name", type=str, default="CIT_llama3.1-4.4", help="name of the run in wandb"
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
    default="./src/CIT/training/models/chekpoints",
    help="output directory for the checkpoints",
)
parser.add_argument(
    "--output_model_path",
    type=str,
    default="./src/CIT/training/models/final_models/cross_validation/ft_4.4",
    help="output directory for the model",
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
    path_train_questions = args.path_train_questions
    output_dir = args.training_output_dir
    output_model_path = args.output_model_path

    #load the model with its lora parameters
    model, tokenizer = load_model_with_lora(
        base_model_name=base_model_name,
        r=r,
        alpha=alpha,
        lora_dropout=lora_dropout,
    )

    #load the dataset and split it into train and validation sets
    dataset_questions = datasets.load_dataset(
        "json", data_files=path_train_questions, split="train"
    )
    train_dataset, val_dataset = format_dataset(
        dataset_questions,
    ) 

    # define training arguments

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_epochs=num_epochs,
        run_name=run_name,
        output_dir=output_dir,
    )

    # start training
    trainer_stats = trainer.train()
    print("Training stats:")
    print(trainer_stats)
    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_model_path)
    print("Model saved to:", output_model_path)
    print("Training finished")
