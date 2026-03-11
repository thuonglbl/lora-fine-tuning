# script to match and mix rephrased questions and rephrased answers

import argparse

from ..utils import load_jsonl, save_jsonl

parser = argparse.ArgumentParser(
    description="Match rephrased questions with their rephrased answers."
)
parser.add_argument(
    "--rephrased_answers_path",
    type=str,
    default="all_rephrased_answers.jsonl",
    help="Path to the JSON file containing the rephrased answers.",
)
parser.add_argument(
    "--rephrased_questions_path",
    type=str,
    default="all_rephrased_questions.jsonl",
    help="Path to the JSON file containing the rephrased questions.",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="matched_questions_answers.jsonl",
    help="Path to the output JSON file for matched questions and answers.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    rephrased_answers_path = args.rephrased_answers_path
    rephrased_questions_path = args.rephrased_questions_path
    output_path = args.output_path

    rephrased_answers = load_jsonl(rephrased_answers_path)
    rephrased_questions = load_jsonl(rephrased_questions_path)

    mapping_id_indexes_questions = {}
    for i, question in enumerate(rephrased_questions):
        if question["id"] not in mapping_id_indexes_questions:
            mapping_id_indexes_questions[question["id"]] = [i]
        else:
            mapping_id_indexes_questions[question["id"]].append(i)

    matched_data = []
    for answer in rephrased_answers:
        question_id = answer["id"]
        if question_id in mapping_id_indexes_questions:
            questions_indexes = mapping_id_indexes_questions[question_id]
            for question_index in questions_indexes:
                matched_data.append(
                    {
                        "question": rephrased_questions[question_index]["question"],
                        "answer": answer["answer"],
                        "urls": rephrased_questions[question_index]["urls"],
                        "id": question_id,
                    }
                )

    save_jsonl(output_path, matched_data)
    print(f"Matched {len(matched_data)} questions and answers.")
    print(f"Matched questions and answers saved to {output_path}")
