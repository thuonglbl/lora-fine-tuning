# functions for all tools used for QA generation and QA evaluation (URL metric), as well as retrieval metrics

import json
import logging
import os
import re
import textwrap
from collections import Counter

import numpy as np
from langchain_core.documents import Document
from tqdm import tqdm

from CIT.RAGs.RAG_CIT import RAGv3


def str_to_bool(x):
    if isinstance(x, str):
        if x.lower() == "true":
            return True
        elif x.lower() == "false":
            return False
    return x


def load_jsonl(file_path):
    """Loads a JSONL file and returns a list of JSON objects."""
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def save_jsonl(file_path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        for line in data:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")


def save_readible_jsonl(file_path, data):
    with open(file_path, "w", encoding="utf-8") as file:
        for line in data:
            json_str = json.dumps(
                line, ensure_ascii=False
            )  # Pretty print with indentation
            # Wrap the string to a maximum of 80 characters
            wrapped_str = "\n".join(
                textwrap.wrap(json_str, width=80)
            )  # Break every 8 characters
            file.write(wrapped_str + "\n\n")  # Double newline for readability


def extract_confluence_urls(text):
    """Finds all URLs starting with 'https://confluence.yourcompany.com/'."""
    text = text.replace("](", "] (")
    url_pattern = re.compile(r"https://confluence\.yourcompany\.com/[^\s,\)]+")
    all_matches = url_pattern.findall(text)
    cleaned_matches = [
        match.replace("&src=contextnavpagetreemode", "")
        .replace("?src=contextnavpagetreemode", "")
        .replace("pages/viewpage.action?pageId=", "spaces/CORPORATEITKNOWLEDGEBASE/pages/")
        .strip(",.)];").split("#")[0]
        for match in all_matches
    ]

    return cleaned_matches


def url_harmonization(url, url_to_standard_url_mapping):
    """
    Harmonize a URL to a standard URL.
    Args:
        url: URL to harmonize (str)
        url_to_standard_url_mapping: dictionary with keys as URLs and values as standard URLs
         with template "https://confluence.yourcompany.com/spaces/CORPORATEITKNOWLEDGEBASE/pages/{id}"
    Returns:
        standard_url: standard URL
    """
    if url in url_to_standard_url_mapping:
        return url_to_standard_url_mapping[url]
    else:
        print(f"URL not in mapping: {url}")
        return url


def get_answer_from_rag(graph, config, question):
    """
    Get answers from a RAG model.
    Args:
        graph: RAG model (langcahin graph as defined in ../RAGs/)
        config: RAG configuration
        questions: list of questions: list of dicts with key 'question'
    Returns:
        answers: list of answers (str)
    """
    # Disable logging because to much INFO messages
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    input_message = question["question"]

    input_message = input_message.replace("'", " ").replace("’", " ")

    step = {}
    final_step = {}
    tries = 0
    while (
        "generate" not in final_step
    ) and tries < 2:  # sometimes fails at generating response
        tries += 1

        for step in graph.stream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="updates",
            config=config,
        ):
            pass
        final_step = step

    if "generate" in final_step:
        answer = final_step["generate"]["messages"][0].content
        return answer

    else:
        assert tries >= 2
        print("Error")
        print(f"input_message: {input_message}")
        print(step)
        return "Error"


def compute_precision_recall_1case(question_with_urls):
    """
    Compute precision and recall for one question.
    Args:
        question_with_urls: dictionary with keys  "urls" and "RAG_confluence_urls"
    Returns:
        precision: precision
        recall: recall
        TP: nb of true positives
        FP: number of false positives
        FN: number of false negatives
    """
    if "facultative_urls" in question_with_urls:
        facultative_urls = question_with_urls["facultative_urls"]
    else:
        facultative_urls = []
    if "urls" in question_with_urls:
        expected_urls = set(question_with_urls["urls"])
    else:
        expected_urls = set()
    actual_urls = set(question_with_urls["RAG_confluence_urls"])

    TP = (expected_urls.union(facultative_urls)).intersection(actual_urls)
    TP = len(TP)
    FP = actual_urls - expected_urls.union(facultative_urls)
    FP = len(FP)
    FN = expected_urls - actual_urls
    FN = len(FN)

    if TP + FN == 0:  # nothing to retrieve
        recall = 1
    else:
        recall = TP / (TP + FN)
    if TP + FP == 0:  # no url retrieved
        precision = 1
    else:
        precision = TP / (FP + TP)
    return precision, recall, TP, FP, FN


def compute_mean_precision_recall(questions_with_urls):
    """
    Compute precision and recall for a list of questions.
    Args:
        questions_with_urls: list of dictionaries with keys "question", "urls" and "RAG_confluence_urls"
    Returns:
        precision: mean precision
        recall: mean recall"
    """
    precisions = []
    recalls = []
    TPs = 0
    FPs = 0
    FNs = 0
    for question_with_urls in questions_with_urls:
        precision, recall, TP, FP, FN = compute_precision_recall_1case(
            question_with_urls
        )
        precisions.append(precision)
        recalls.append(recall)
        TPs += TP
        FPs += FP
        FNs += FN

    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    print(f"Mean precision: {mean_precision}")
    print(f"Mean recall: {mean_recall}")

    average_precision = TPs / (TPs + FPs)
    average_recall = TPs / (TPs + FNs)
    print(f"Average precision: {average_precision}")
    print(f"Average recall: {average_recall}")
    return mean_precision, mean_recall, average_precision, average_recall


############################################################################################################
### Retrieval metrics#######################################################################################
############################################################################################################
def retrieve_urls(query: str, compression_retriever, mapping_id_urls):
    """Retrieve documents urls related to a user query to help answer a question.,accordin gto the RAG piepline.
    Meaning adding the most present parent page if present more than 2 times
    Args:
        query (str): user question or query.
    Returns:
        Tuple[str, List[Document]]: retrieved documents."""
    retrieved_docs = compression_retriever.invoke(query)
    retrieved_docs_urls = [doc.metadata["url"] for doc in retrieved_docs]
    # add url of best parent if more than 2 times
    parents_id = Counter([doc.metadata["parent"] for doc in retrieved_docs])
    if parents_id.most_common(1)[0][1] > 2:
        primary_parent_id = parents_id.most_common(1)[0][0]
        sources_id = Counter([doc.metadata["id"] for doc in retrieved_docs])
        if (primary_parent_id in mapping_id_urls) and (
            primary_parent_id not in sources_id
        ):
            retrieved_docs_urls.append(mapping_id_urls[primary_parent_id])

    return retrieved_docs_urls


def compute_retrieval_stats(questions, compression_retriever,mapping_id_urls):
    """
    Compute precision and recall for a list of questions.
    Args:
        questions: list of dictionaries with keys "question", "urls", the question and the urls to retreive
        compression_retriever: compression retriever
    Returns:
        mean precision
        mean recall
        average precision
        average recall


    """
    precisions = []
    recalls = []
    TPs = 0
    FPs = 0
    FNs = 0
    for question in tqdm(questions, total=len(questions)):
        query = question["question"]
        expected_docs_urls = question["urls"]
        retrieved_docs_urls = retrieve_urls(
            query, compression_retriever, mapping_id_urls
        )
        results = {
            "urls": expected_docs_urls,
            "RAG_confluence_urls": retrieved_docs_urls,
        }
        if "facultative_urls" in question:
            results["facultative_urls"] = question["facultative_urls"]
        precision, recall, TP, FP, FN = compute_precision_recall_1case(results)
        precisions.append(precision)
        recalls.append(recall)
        TPs += TP
        FPs += FP
        FNs += FN

    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    average_precision = TPs / (TPs + FPs)
    average_recall = TPs / (TPs + FNs)
    return mean_precision, mean_recall, average_precision, average_recall


def get_non_retrieved_questions(questions, compression_retriever,mapping_id_urls):
    """
    Get the questions that were not retrieved by the retriever.
    Args:
        questions: list of dictionaries with keys "question", "urls"
        compression_retriever: compression retriever
    Returns:
        non_retrieved_questions: list of dictionaries with keys "question", "urls", "RAG_confluence_urls"
    """

    non_retrieved_questions = []
    for question in tqdm(questions, total=len(questions)):
        query = question["question"]
        expected_docs_urls = question["urls"]
        retrieved_docs_urls = retrieve_urls(
            query, compression_retriever, mapping_id_urls
        )
        results = {
            "urls": expected_docs_urls,
            "RAG_confluence_urls": retrieved_docs_urls,
        }
        precision, recall, TP, FP, FN = compute_precision_recall_1case(results)
        if recall != 1:
            non_retrieved_questions.append(
                {
                    "question": query,
                    "urls": expected_docs_urls,
                    "RAG_confluence_urls": retrieved_docs_urls,
                }
            )
    return non_retrieved_questions


##########################################################################################################################
#####Tools to retrieve context, etc##########################################################################################
##########################################################################################################################
def add_parent_content(parents_id, retrieved_docs, mapping_id_paths):
    """
    intermediate function to check if we need to add the most represented parent in the retrieved context
    ARgs:
        - parents_id: list of retrieved documents parents ids
        - retrieved_docs: list of lancgchain_core documents
        - mapping_id_paths: dict mapping ids of documents to their paths

    """
    add_parent = False
    primary_parent_content = None
    if parents_id.most_common(1)[0][1] > 2:
        primary_parent_id = parents_id.most_common(1)[0][0]
        sources_id = Counter([doc.metadata["id"] for doc in retrieved_docs])
        if (primary_parent_id in mapping_id_paths) and (
            primary_parent_id not in sources_id
        ):
            add_parent = True
            #print(os.path.abspath(mapping_id_paths[primary_parent_id]))
            with open(mapping_id_paths[primary_parent_id], "r") as f:
                # print absolute path
                primary_parent_content = json.load(f)
    return add_parent, primary_parent_content


def build_retrieved_context(
    primary_source, retrieved_docs, add_parent, primary_parent_content
):
    """
    Build a full retrieved context by combining:
    - The primary document (full content, title, and URL),
    - Other retrieved secondary documents (chunks with title, URL, and content),
    - Optionally, the parent document (if `add_parent` is True).

    Args:
        primary_source (dict): Dictionary containing the primary document's content, title, and URL.
        retrieved_docs (list): List of retrieved langcahin_core Document instances for secondary documents.
        add_parent (bool): Whether to append the parent document to the context.
        primary_parent_content (dict): Dictionary containing the parent document's content, title, and URL.

    Returns:
        str: A formatted string concatenating all the documents into a single context block.
    """

    template_url = (
            "https://confluence.yourcompany.com/spaces/CORPORATEITKNOWLEDGEBASE/pages/{id}"
        )
    

    primary_source_content = primary_source[
        "content"
    ]  # load the whole content of the primary source
    primary_source_title = primary_source["title"]
    primary_source_url = template_url.format(id=primary_source["id"])
    # Format retrieved context made of full primary document and chunks of secondary documents
    retrieved_context = (
        f"Primary document title: {primary_source_title}\n"
        f"Document URL: {primary_source_url}\n"
        f"Document content: {primary_source_content}"
        + "\nOther documents:\n"
        + "\n".join(
            f"Title: {doc.metadata['original_title']}\n"
            f"URL: {template_url.format(id=doc.metadata['id'])}\n"
            f"Document content: {doc.page_content}"
            for doc in retrieved_docs
            if doc.metadata["source"] != primary_source
        )
    )
    if add_parent:
        retrieved_context += (
            "\nParent document:\n"
            f"Title: {primary_parent_content['title']}\n"
            f"URL: {template_url.format(id=primary_parent_content['id'])}\n"
            f"{primary_parent_content['content']}"
        )

    return retrieved_context


def get_retrieved_context(
    query,
    compression_retriever,
    threshold=0.1,
    mapping_id_paths={},
):
    """
    Get the retrieved context for a question.
    Args:
        question: dictionary with keys "question", "urls"
        compression_retriever: compression retriever
        threshold: threshold for the relevance score of the retrieved documents
        mapping_id_paths: dictionary mapping ids of documents to their paths
    Returns:
        retrieved_context: list of dictionaries with keys "question", "urls", "RAG_confluence_urls"
    """

    # Retrieve documents related to the user query

    retrieved_docs = compression_retriever.invoke(query)
    sources = Counter([doc.metadata["source"] for doc in retrieved_docs])

    parents_id = Counter([doc.metadata["parent"] for doc in retrieved_docs])
    primary_source = Counter(sources).most_common(1)[0][0]

    # if principal parent is present at least 3 times, we load it
    add_parent, primary_parent_content = add_parent_content(
        parents_id, retrieved_docs, mapping_id_paths
    )

    with open(primary_source, "r") as f:
        primary_source = json.load(f)

    # Filter documents based on relevance score by thresholds
    scale = np.max([doc.metadata["relevance_score"] for doc in retrieved_docs])
    real_threhsold = threshold * scale
    retrieved_docs = [
        doc
        for doc in retrieved_docs
        if doc.metadata["relevance_score"] >= real_threhsold
    ]
    retrieved_context = build_retrieved_context(
        primary_source, retrieved_docs, add_parent, primary_parent_content
    )
    return retrieved_context, retrieved_docs


##############################################################################################################################
### Tools to build context from a list of documents ##################################################################################################
##############################################################################################################################
def load_documents_from_file_paths(file_paths):
    """
    Load documents from a list of JSON file paths and create Document instances.

    Args:
        file_paths (list): List of file paths pointing to JSON files,
                           where each file contains a 'content' field and other metadata fields.

    Returns:
        list: A list of Document instances, each containing the page content and associated metadata.
    """
    documents = []
    for file_path in file_paths:
        doc_dict = json.load(open(file_path, "r"))
        metadata = {key: doc_dict[key] for key in doc_dict if key != "content"}
        metadata.update({"source": os.path.abspath(file_path)})
        doc = Document(
            page_content=doc_dict["content"],
            metadata=metadata,
        )
        documents.append(doc)
    return documents


def build_context_from_documents(documents):
    """
    Build a context for a lst of Documents.
    """
    retrieved_context = "\n".join(
        f"Title: {doc.metadata['title']}\n"
        f"URL: {doc.metadata['url']}\n"
        f"Content: {doc.page_content}"
        for doc in documents
    )
    return retrieved_context


def build_context_from_file_paths(file_paths):
    """
    Build a context for a lst of Documents.
    """
    documents = load_documents_from_file_paths(file_paths)
    return build_context_from_documents(documents)


def add_noise_to_context(context, dir_documents, nb_noise_doc=2):
    """
    Add random noise to an existing context by appending content from randomly selected documents.

    Args:
        context (str): The original context string.
        dir_documents (str): Path to the directory containing JSON documents to use as noise.
        nb_noise_doc (int, optional): Number of noise documents to add. Defaults to 2.

    Returns:
        str: The augmented context with additional noise content appended.
    """
    # Load documents from the directory
    file_paths = [
        os.path.join(dir_documents, file) for file in os.listdir(dir_documents) if file.endswith(".json")
    ]
    file_paths = np.random.choice(file_paths, size=nb_noise_doc, replace=False)
    # Load documents from the file paths
    noise_documents = load_documents_from_file_paths(file_paths)
    # Build the noise context
    noise_context = build_context_from_documents(noise_documents)
    # Add noise to the context
    return context + "\n" + noise_context



##################################################################################################################################
### Tools to evaluate a model  ##################################################################################################
##################################################################################################################################

def remove_duplicates_rephrased_questions(questions,id_key="question_rephrased_id"):
    """
    Remove duplicates from a list of questions based on the 'question' field.
    Args:
        questions: list of dictionaries with keys "question", "urls" and "question_rephrased_id" that correspond to the id of the rephrased question
    Returns:
        unique_questions: list of dictionaries with keys "question", "urls"
    """
    seen = set()
    unique_questions = []
    for question in questions:
        question_id = question[id_key]
        if question_id not in seen:
            seen.add(question_id)
            unique_questions.append(question)
    return unique_questions

def load_all_urls_list_from_docs_directory(directory):
    """
    Load all URLs from a directory of JSON documents.
    Args:
        directory: path to the directory containing the documents
    Returns:
        urls: set of URLs
    """
    path_all_urls = directory + "/mappings/all_urls.txt"
    with open(path_all_urls, "r") as f:
        all_urls = f.readlines()
    all_urls = [url.strip() for url in all_urls]
    all_urls_set = set(all_urls)
    return all_urls_set

def load_url_to_standard_url_mapping(directory):
    """
    Load the URL to standard URL mapping from a JSON file.
    Args:
        directory: path to the directory containing the documents
    Returns:
        url_to_standard_url_mapping: dictionary with keys as URLs and values as standard URLs
         with template "https://confluence.yourcompany.com/spaces/CORPORATEITKNOWLEDGEBASE/pages/{id}"
    """
    path_mapping = directory + "/mappings/mapping_original_urls_fixed_urls.json"
    with open(path_mapping, "r") as f:
        url_to_standard_url_mapping = json.load(f)
    return url_to_standard_url_mapping

def answers_and_evaluate(questions_to_evaluate, graph, config, url_to_standard_url_mapping, all_urls_set, answers_save_path, answers_already_computed=False):
    """
    Get answers from a RAG model and evaluate them.
    Args:
        questions_to_evaluate: list of dictionaries with keys "question", "urls"
        graph: RAG model (langcahin graph as defined in ../RAGs/)
        config: RAG configuration
        url_to_standard_url_mapping: dictionary with keys as URLs and values as standard URLs
         with template "https://confluence.yourcompany.com/spaces/CORPORATEITKNOWLEDGEBASE/pages/{id}"
        all_urls_set: set of all urls possible to detect hallucinations
        answers_save_path: path to the file where to save the answers
        answers_already_computed: boolean, if True, the answers are already computed and stored in the questions
    Returns:
        questions_to_evaluate: list of dictionaries with keys "question", "urls", "RAG_answer", "RAG_confluence_urls"
        hallucination_urls: list of URLs that are not in the all urls set
    """
    hallucination_urls = []
    for question in tqdm(questions_to_evaluate):
        if "RAG_answer" in question or answers_already_computed:
            answer = question["RAG_answer"]
        else:
            answer = get_answer_from_rag(graph, config, question)
            question["RAG_answer"] = answer

        # parse urls in the answer
        question["RAG_confluence_urls"] = extract_confluence_urls(answer)
        # put urls in standard format
        question["RAG_confluence_urls"] = list(set([
            url_harmonization(url, url_to_standard_url_mapping)
            for url in question["RAG_confluence_urls"]
        ]))

        question["urls"] = [
            url_harmonization(url, url_to_standard_url_mapping)
            for url in question["urls"]
        ]
        if "facultative_urls" in question:
            question["facultative_urls"] = [
                url_harmonization(url, url_to_standard_url_mapping)
                for url in question["facultative_urls"]
            ]
        skip_metric_computing = False
        # check hallucinations
        for url in question["RAG_confluence_urls"]:
            if url not in all_urls_set:  # hallucination
                print(f"URL not in all urls set: {url}")
                print(f"Answer: {answer}")
                hallucination_urls.append(url)
                recall = 0
                precision = 0
                TP = 0
                FP = 0
                FN = 0
                quality=0
                skip_metric_computing = True
                hallucination = True
                break
        if not skip_metric_computing:  # if no hallucination
            hallucination = False
            precision, recall, TP, FP, FN = compute_precision_recall_1case(question)
            quality = get_answer_quality(precision, recall)

        question["quality"] = quality
        question["precision"] = precision
        question["recall"] = recall
        question["TP"] = TP
        question["FP"] = FP
        question["FN"] = FN
        question["hallucination"] = hallucination

    # save answers with stats
    save_jsonl(answers_save_path, questions_to_evaluate)

    return questions_to_evaluate, hallucination_urls

def save_mean_metrics(questions_to_evaluate,hallucination_urls,scores_path):
    """
    Save the mean metrics in a file.
    Args:
        questions_to_evaluate: list of dictionaries with keys "question", "urls", "RAG_answer", "RAG_confluence_urls"
        scores_path: path to the file where to save the scores
    Returns:
        None
    """
    mean_precision, mean_recall, average_precision, average_recall = (
        compute_mean_precision_recall(questions_to_evaluate)
    )
    # round to 2 decimal places before saving
    precision_decimals=2
    all_qualities=[answer['quality'] for answer in questions_to_evaluate]
    nb_quality_0 = all_qualities.count(0)
    nb_quality_1 = all_qualities.count(1)
    nb_quality_2 = all_qualities.count(2)
    proportion_correct_answers = round(
        (nb_quality_1 + nb_quality_2) / len(questions_to_evaluate), precision_decimals
    )
    proportion_bad_answers = round(
        nb_quality_0 / len(questions_to_evaluate), precision_decimals
    )
    proportion_partially_good_answers = round(
        nb_quality_1 / len(questions_to_evaluate), precision_decimals
    )
    proportion_good_answers = round(
        nb_quality_2 / len(questions_to_evaluate), precision_decimals
    )
    mean_precision = round(mean_precision, precision_decimals)
    mean_recall = round(mean_recall, precision_decimals)
    average_precision = round(average_precision, precision_decimals)
    average_recall = round(average_recall, precision_decimals)
    print(f"NB of hallucinations: {len(hallucination_urls)}")

    # save scores
    with open(scores_path, "w") as f:
        json.dump(
            {
                "proportion_correct_answers": proportion_correct_answers,
                "proportion_bad_answers": proportion_bad_answers,
                "proportion_partially_good_answers": proportion_partially_good_answers,
                "proportion_good_answers": proportion_good_answers,
                "mean_precision": mean_precision,
                "mean_recall": mean_recall,
                "average_precision": average_precision,
                "average_recall": average_recall,
                "nb_hallucinations": len(hallucination_urls),
            },
            f,
        )
    print(f"Scores saved in {scores_path}")
    
def evaluate_rag_model(
    vector_base,
    model_name,
    num_predict_tokens,
    top_k,
    threshold,
    always_do_retrieval,
    add_external_links_docs,
    thread_id,
    questions_with_urls_path,
    answers_path,
    scores_path,
    answers_already_computed,
    not_citing_source=False):
    """
    Evaluate a RAG model on a set of questions with URLs.
    Args:
        vector_base: VectorBase object
        model_name: name of the model to use
        num_predict_tokens: number of tokens to predict
        top_k: number of documents to retrieve
        threshold: threshold for the relevance score of the retrieved documents
        always_do_retrieval: whether to always do retrieval or not
        thread_id: thread id for the RAG model
        questions_with_urls_path: path to the file with questions and URLs
        answers_path: path to the file where to save the answers
        answers_already_computed: whether the answers are already computed or not
    Returns:
        None
    """

    # Build the RAG model
    RAG = RAGv3(
        vector_base,
        model_name=model_name,
        num_predict=num_predict_tokens,
        top_k=top_k,
        threshold=threshold,
        always_do_retrieval=always_do_retrieval,
        thread_id=thread_id,
        add_external_links_docs=add_external_links_docs,
        not_citing_source=not_citing_source,
    )
    print("RAG model built")
    graph = RAG.graph
    config = RAG.config
    # Load the questions with true URLs from the JSONL file
    questions_with_urls = load_jsonl(questions_with_urls_path)

    # there are multiple instances of the same question (but with different rephrased answers as we augmented the dataset), we keep only one
    if answers_already_computed and os.path.exists(answers_path):
        questions_to_evaluate = load_jsonl(answers_path)
    else:
        if answers_already_computed:
            print(f"answers not computed for model {model_name}")
            answers_already_computed = False
        questions_to_evaluate = remove_duplicates_rephrased_questions(
            questions_with_urls, id_key="question_rephrased_id"
        )
    

    print(f"Length of questions_to_evaluate: {len(questions_to_evaluate)}")
    print(f"Model name: {model_name}")


    # load all urls possible to detect hallucinations
    directory=vector_base.directory
    all_urls_set = load_all_urls_list_from_docs_directory(directory)


    # load url to standard url mapping
    url_to_standard_url_mapping = load_url_to_standard_url_mapping(directory)

    #answer, evaluate and save answers
    questions_to_evaluate,hallucination_urls=answers_and_evaluate(questions_to_evaluate,
                                                graph, 
                                                config,
                                                url_to_standard_url_mapping,
                                                all_urls_set, 
                                                answers_path, 
                                                answers_already_computed=answers_already_computed)



    save_mean_metrics(questions_to_evaluate,hallucination_urls,scores_path)


##########################################################################################################################
##########Tools to go from precision, recall to more understandable metrics: good answer, partially good, bad answer######
##########################################################################################################################

def get_answer_quality(precision, recall):
    """
    Get the quality of the answer based on precision and recall.
    Args:
        precision: precision of the answer
        recall: recall of the answer
    Returns:
        quality: int  0 for bad answer, 1 for partially good answer, 2 for perfect answer
    """
    if precision == 1 and recall == 1:
        quality = 2  # perfect answer
    elif precision<=0.5 or recall==0:
        quality = 0
    else:
        quality = 1
    return quality

def get_quality_from_question(question):
    """
    Get the quality of the answer from a question.
    Args:
        question: dictionary with keys "question", "urls", "RAG_answer", "RAG_confluence_urls"
    Returns:
        quality: int  0 for bad answer, 1 for partially good answer, 2 for perfect answer
    """
    precision = question["precision"]
    recall = question["recall"]
    return get_answer_quality(precision, recall)