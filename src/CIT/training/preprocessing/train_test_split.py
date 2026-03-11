# script to split the dataset into train and test sets
# This script takes a JSONL file containing questions and answers, splits it into training and test sets, and saves the resulting datasets to separate JSONL files.
# The split is done in a way that ensures that the same question does not appear in both sets.
# The script also ensures that the training and test sets have a balanced distribution of questions with and without URLs.
# The training set will contain 80% of the questions with URLs and 40% of the questions without URLs, while the test set will contain 20% of the questions with URLs and 60% of the questions without URLs.

from argparse import ArgumentParser

from CIT.evaluation.utils import load_jsonl, save_jsonl
from CIT.training.utils import split_dataset

parser = ArgumentParser()
parser.add_argument(
    "--path_dataset",
    type=str,
    default="./src/CIT/training/data/all_data_copy.jsonl",
)
parser.add_argument(
    "--path_train",
    type=str,
    default="./src/CIT/training/data/split_balance_zeros/training_data.jsonl",
)
parser.add_argument(
    "--path_test",
    type=str,
    default="./src/CIT/training/data/split_balance_zeros/test_data.jsonl",
)
parser.add_argument("--test_size", type=float, default=0.2)
args = parser.parse_args()

if __name__ == "__main__":
    path_dataset = args.path_dataset
    path_train = args.path_train
    path_test = args.path_test
    test_size = args.test_size

    dataset = load_jsonl(path_dataset)
    print(f"Loaded {len(dataset)} questions and answers from {path_dataset}")

    train,test=split_dataset(dataset, test_size=test_size)

    save_jsonl(path_train, train)
    save_jsonl(path_test, test)
    print(f"Saved {len(train)} questions and answers to {path_train}")
    print(f"Saved {len(test)} questions and answers to {path_test}")
    print(f"Train and test datasets saved to {path_train} and {path_test}")
