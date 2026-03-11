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
    default="all_questions_with_urls.jsonl",
    help="Path to the JSON file containing the questions with URLs.",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="all_rephrased_questions.jsonl",
    help="Path to the output JSON file for rephrased questions.",
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
    default=4,
    help="Number of worker threads to use for parallel processing.",
)


template = """Here is a question about a text. Please rephrase the question in 10 other ways keeping the meaning of it.
Your answer should look like this:
1. <rephrased question 1>\n
2. <rephrased question 2>\n
...
10. <rephrased question 10>\n
The original question is: {question}"""


def ask_llm_to_rephrase(question, llm):
    """Function to process a single question in parallel.
    Args:
        question (dict): A dictionary containing the question and its associated URLs.
        llm: The language model to use for rephrasing.
    Returns:
        list: A list of rephrased questions.
    """

    if "id" not in question:
        question_id = str(uuid.uuid4())
        question["id"] = question_id
    else:
        question_id = question["id"]
        question["id"] = question_id
    new_questions = [question]
    # Rephrase the question using the language model
    response = llm.invoke(template.format(question=question["question"])).content
    rephrased_questions = parse_rephrased_questions(response)
    for new_question in rephrased_questions:
        new_question = {
            "question": new_question,
            "urls": question["urls"],
            "id": question_id,
            "question_rephrased_id": str(uuid.uuid4()),
        }
        new_questions.append(new_question)
    return new_questions


def parse_rephrased_questions(answer):
    """
    Parse the answer from the model and return a list of rephrased questions.
    """
    # Split the answer by lines and remove empty lines
    lines = [line.strip() for line in answer.split("\n") if line.strip()]
    # Extract the rephrased questions from the lines
    rephrased_questions = []
    for line in lines:
        match = re.match(r"^\d+\.\s*(.*)$", line)
        if match:
            rephrased_questions.append(match.group(1))
    return rephrased_questions[:10]  # Return only the first 10 questions


if __name__ == "__main__":
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    model_name = args.model_name
    num_workers = args.num_workers

    # Load the questions with URLs
    questions_with_urls = load_jsonl(input_path)
    llm = ChatOllama(model=model_name, temperature=0)
    all_results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_question = {
            executor.submit(ask_llm_to_rephrase, q, llm): q for q in questions_with_urls
        }

        for future in tqdm(
            as_completed(future_to_question),
            total=len(questions_with_urls),
            desc="Processing questions",
        ):
            try:
                result = future.result()
                all_results.extend(result)
            except Exception as e:
                print(f"Error processing question: {e}")

    # Save the rephrased questions to a JSONL file

    save_jsonl(output_path, all_results)
    print(f"Rephrased questions saved to {output_path}")
    print(f"Length of questions_with_urls: {len(questions_with_urls)}")
    print(f"Length of rephrased questions: {len(all_results)}")
