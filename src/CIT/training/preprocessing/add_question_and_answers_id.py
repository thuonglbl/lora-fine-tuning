import argparse
from uuid import uuid4

from CIT.evaluation.utils import load_jsonl, save_jsonl

parser = argparse.ArgumentParser(description="Add unique IDs to questions and answers.")
parser.add_argument(
    "--QA_path",
    type=str,
    default="./src/CIT/training/data/training_data_copy.jsonl",
    help="Path to the JSONL file containing the questions.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    QA_path = args.QA_path
    data = load_jsonl(QA_path)

    map_question_id = {}
    for i in range(len(data)):
        if "question_rephrased_id" in data[i]:
            continue
        else:
            question = data[i]["question"]
            if question not in map_question_id:
                only_question_id = str(uuid4())
                map_question_id[question] = only_question_id
            else:
                only_question_id = map_question_id[question]
            data[i]["question_rephrased_id"] = only_question_id
        if "answer_id" not in data[i]:
            data[i]["answer_id"] = str(uuid4())

    save_jsonl(QA_path, data)
    print(
        f"Saved {len(data)} questions and answers with unique rephrased questions IDs and answers ID to {QA_path}"
    )
