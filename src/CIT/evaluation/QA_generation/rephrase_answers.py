import argparse
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_ollama import ChatOllama
from tqdm import tqdm

from ..utils import load_jsonl, save_jsonl

parser = argparse.ArgumentParser(
    description="Rephrase questions using a language model."
)
parser.add_argument(
    "--input_path",
    type=str,
    default="./src/CIT/evaluation/QA_generation/answers_clean/checked_answers_format.jsonl",
    help="Path to the JSON file containing the questions, answers with URLs.",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="all_rephrased_answers.jsonl",
    help="Path to the output JSON file for rephrased answers.",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="llama3.1:8b",
    help="Name of the model to use for rephrasing questions.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=2,
    help="Number of worker threads to use for parallel processing.",
)


template_answer = """Here is an answer about a question. Please rephrase the answer in 3 other ways keeping all the infornation of it.
Your task in only to use other words, not to summarize. Do not add or change any information.
At the beginning of the answer there is in many cases lits of sources and URLs. You should copy it in each new rephrased answer.
Your answer should look like this:
1. <rephrased answer 1>\n
2. <rephrased answer 2>\n
3. <rephrased question 3>\n

The original answer is: {answer}"""

no_info_rephrasing = [
    "The necessary data is not available to me.",
    "I am unable to provide the required information.",
    "The requested information is not within my knowledge or possession.",
]


def ask_llm_to_rephrase(question, llm):
    """Function to process a single question in parallel.
    Args:
        question (dict): A dictionary containing the question and its associated URLs.
        llm: The language model to use for rephrasing.
    Returns:
        list: A list of rephrased questions.
    """
    question_id = question["id"]
    new_answers = [question]
    if (
        "Sources" in question["answer"]
    ):  # some sources are quoted and the answer has a specific format
        # Rephrase the question using the language model
        response = llm.invoke(template_answer.format(answer=question["answer"])).content
        rephrased_answers = parse_numbered_answers(response)
    else:  # the answer is I don't have the information', 3 ways to say it
        rephrased_answers = no_info_rephrasing

    for new_answer in rephrased_answers:
        if new_answer.startswith(
            "Here are"
        ):  # the model adds a line saying "Here are the rephrased answers"
            continue
        new_answer = {
            "question": question["question"],
            "answer": new_answer,
            "urls": question["urls"],
            "id": question_id,
            "question_rephrased_id": question["question_rephrased_id"],
            "answer_id": str(uuid.uuid4()),
        }
        new_answers.append(new_answer)
    return new_answers


def parse_numbered_answers(text: str) -> list:
    """
    Parses a string containing multiple numbered answers into a list of answer strings.

    Each answer is expected to start with a numbered prefix like "1. Sources:".

    Args:
        text (str): The raw text containing multiple answers.

    Returns:
        List[str]: A list of individual answer strings.
    """
    # Split the text at lines that start with a number followed by ". Sources:"
    entries = re.split(r"\n(?=\d+\.\sSources:)", text)

    cleaned_entries = [
        re.sub(r"^\d+\.\s+", "", entry.strip()) for entry in entries if entry.strip()
    ]

    return cleaned_entries


if __name__ == "__main__":
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    model_name = args.model_name
    num_workers = args.num_workers

    # Load the QA (there must be a key "question" and a key "answer" in each dict of the jsonl file)
    questions_with_urls = load_jsonl(input_path)
    test = False
    if test:
        questions_with_urls = questions_with_urls[:3]
    llm = ChatOllama(model=model_name, temperature=0)
    all_results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_question = {
            executor.submit(ask_llm_to_rephrase, q, llm): q for q in questions_with_urls
        }

        for future in tqdm(
            as_completed(future_to_question),
            total=len(questions_with_urls),
            desc="Processing answers",
        ):
            try:
                result = future.result()
                all_results.extend(result)
            except Exception as e:
                print(f"Error processing QA: {e}")

    # Save the rephrased questions to a JSONL file

    save_jsonl(output_path, all_results)
    print(f"Rephrased answers saved to {output_path}")
    print(f"Length of questions_with_urls: {len(questions_with_urls)}")
    print(f"Length of rephrased answers: {len(all_results)}")
