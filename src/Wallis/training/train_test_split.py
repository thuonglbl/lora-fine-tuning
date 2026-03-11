# split a jsonl file into train and test sets
import json
from argparse import ArgumentParser

from sklearn.model_selection import train_test_split

parser = ArgumentParser()
parser.add_argument(
    "--path_dataset",
    type=str,
    default="../data/intermediate_results/retrieved_qu_1703.jsonl",
)
parser.add_argument(
    "--path_train",
    type=str,
    default="../data/intermediate_results/train/retrieved_qu_1703_train.jsonl",
)
parser.add_argument(
    "--path_test",
    type=str,
    default="../data/intermediate_results/test/retrieved_qu_1703_test.jsonl",
)
parser.add_argument("--test_size", type=float, default=0.2)
args = parser.parse_args()

if __name__ == "__main__":
    path_dataset = args.path_dataset
    path_train = args.path_train
    path_test = args.path_test
    test_size = args.test_size

    dataset = json.load(open(path_dataset, "r"))
    train, test = train_test_split(dataset, test_size=test_size)
    with open(path_train, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False)
    with open(path_test, "w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False)
    print(f"Train and test datasets saved to {path_train} and {path_test}")
