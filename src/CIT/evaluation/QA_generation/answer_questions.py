# call an llm to answer a question with a perfect retrieved context. These answers are checked manually and modified afterward

import argparse
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_ollama import ChatOllama
from tqdm import tqdm

from ..utils import (
    add_noise_to_context,
    build_context_from_file_paths,
    load_jsonl,
    save_jsonl,
    save_readible_jsonl,
)

parser = argparse.ArgumentParser(
    description="Compute precision and recall for the CIT Knowledge Base."
)

parser.add_argument(
    "--directory",
    type=str,
    default="./src/CIT/documents/confluence_json_without_root_with_titles",
    help="Directory containing the documents to be indexed",
)
parser.add_argument(
    "--questions_with_urls_path",
    type=str,
    default="./src/CIT/evaluation/QA_generation/data/questions_with_urls.jsonl",
    help="Path to the JSON file containing the questions with URLs.",
)

parser.add_argument(
    "--model_name",
    type=str,
    default="llama3.1:8b",
    help="Name of the model to answer questions.",
)
parser.add_argument(
    "--answers_path",
    type=str,
    default="./src/CIT/evaluation/QA_generation/synthetic_answers/llama3.1:8b_answers.jsonl",
    help="Path to the output answers JSON file.",
)


# prompt
template = """You are an assistant working for an internal IT service of a company called Acme Corp. Always answer within the context of the company.
        You work as a RAG. I'll give you tutorials about IT services.
        Answer only with the context I give you, detail the steps the user has to follow to solve his issue.
        Do not invent information that is not in the context so if you cannot answer the question,
        say `I don't have as this information`.
        When you give an information, always cite from which source (title + document url) your answer comes from. The answer should remain concise.
        So the goal is to provide the detailed steps to solve the issue and to cite the source of the information you provide.
        Your answer must respect the following format:\n
        Sources: Title(s) of the document(s)\n
        URL: confluence url(s) of the relevant document(s)
        To solve this issue apply the following steps:\n
        1 Step 1\n
        2 Step 2\n
        3 Step 3\n etc.\n

        \nNow here is the context:\n{retrieved_context}\n\n
        Here is the question:\n{question}\n
        Your answer must respect the following format:\n
        Sources: Title(s) of the document(s)\n
        URL: confluence url(s) of the relevant document(s)
        To solve this issue apply the following steps:\n
        1 Step 1\n
        2 Step 2\n
        3 Step 3\n etc.\n
        Remember to cite your sources, especially the Confluence URLs according to the format I gave you (Title + URL) and to simply say that you don't have the information when wou cannot answer.
         You must absolutely not invent or give infomation that is not in the context. If you don't know the answer, do not cite any source.
         Also I will give you a hint to answer the question. The urls to cite are {urls}. It might be that these URLs are not in the context.
         Then do not cite them. Also do not cite any other urls that I did not give you.
         Also do not repeate the question, and do not mention 'context' or 'retrieved context' in your answer. Instead mention the documents directly.
         
        """


def ask_llm_to_answer(question, llm, map_urls_paths, dir_documents):
    """Function to process a single question in parallel.
    Args:
        question (dict): A dictionary containing the question and its associated URLs.
        llm: The language model to use for answering questions.
    Returns:
        dict: A dictionary containing the question, answer, and associated URLs.
    """
    urls = question["urls"]
    paths_relevant_docs = [map_urls_paths[url] for url in urls]
    relevant_context = build_context_from_file_paths(paths_relevant_docs)
    noise_to_add = int(4 - len(urls))
    synthetic_context = add_noise_to_context(
        relevant_context, dir_documents=dir_documents, nb_noise_doc=noise_to_add
    )

    # Answer the question using the language model
    response = llm.invoke(
        template.format(
            question=question["question"],
            urls=urls,
            retrieved_context=synthetic_context,
        )
    ).content
    return {
        "question": question["question"],
        "answer": response,
        "urls": urls,
        "id": question["id"],
    }


if __name__ == "__main__":
    args = parser.parse_args()
    directory = args.directory
    model_name = args.model_name
    questions_with_urls_path = args.questions_with_urls_path
    answers_path = args.answers_path

    # Load the questions with URLs from the specified path
    if questions_with_urls_path == "all":
        questions_with_urls = load_jsonl(
            "./src/CIT/evaluation/QA_generation/all_questions_with_urls.jsonl"
        )
    else:
        questions_with_urls = load_jsonl(questions_with_urls_path)

    # add ids to the questions if they don't have any
    new_ids = False
    for question in questions_with_urls:
        if "id" not in question:
            question["id"] = str(uuid.uuid4())
            new_ids = True
    if new_ids:
        print("New ids added to the questions")
        if questions_with_urls_path == "all":
            questions_with_urls_path = "./src/CIT/evaluation/QA_generation/all_questions_with_urls.jsonl"
            save_jsonl(questions_with_urls_path, questions_with_urls)
        else:
            save_jsonl(questions_with_urls_path, questions_with_urls)

    # for debugging
    test = False
    if test:
        questions_with_urls = questions_with_urls[:3]

    print(f"Length of questions_with_urls: {len(questions_with_urls)}")
    print(f"Model name: {model_name}")

    # load existing answers if already exists
    if os.path.exists(answers_path):
        existing_answers = load_jsonl(answers_path)
    else:
        existing_answers = []
    print(f"Length of existing answers: {len(existing_answers)}")

    id_answered = set([q["id"] for q in existing_answers])
    questions_to_answer = [q for q in questions_with_urls if q["id"] not in id_answered]
    print(f"Length of questions to answer: {len(questions_to_answer)}")

    llm = ChatOllama(model=model_name, temperature=0, num_ctx=10000)
    num_workers = 4
    # Load the mapping of URLs to file paths
    with open(
        "./src/CIT/documents/mappings/mapping_urls_paths.json", "r"
    ) as f:
        map_urls_paths = json.load(f)

    all_results = existing_answers
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_question = {
            executor.submit(ask_llm_to_answer, q, llm, map_urls_paths, directory): q
            for q in questions_to_answer
        }

        for future in tqdm(
            as_completed(future_to_question),
            total=len(questions_to_answer),
            desc="Processing questions",
        ):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"Error processing question: {e}")

    # Save the answers to a JSONL file
    save_jsonl(answers_path, all_results)
    # Save the answers in a readable format
    readable_path = answers_path.replace(".jsonl", "_readable.jsonl")
    save_readible_jsonl(readable_path, all_results)
    print(f"Answers saved to {answers_path}")
