# Functions to generate QA and evaluate the models performance
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from uuid import uuid4

import numpy as np
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


# prompt used to generate questions and answers
QA_generation_prompt = """
Your task is to write 10 factoid questions and their answer given a context.
Your factoid questions should be about a specific, concise piece of factual information from the context.
Your factoid questions should be formulated in the same style as questions users could ask in a search engine, maximum 9 words.
This means that your factoid questions must not mention something like "according to the passage" or "context" or "according to this article" but on the contrary directly cite what is the context.
Write the questions and the answers in french.
A bad example is "What is the name of the law ?" because we don't know which law you are talking about.
Detail your answer, giving an explanation and always citing from which article or text of law does the answer come from.
Someone should be able to answer the question without the context.
Provide your answer as follows, please do not add any spaces character and absolutely respect this format, do not include anything else in your answer:
Output:::
Question 1: (your factoid question)
Réponse 1: (your answer to the factoid question. Source: Law n°... , Article...)
Question 2: (your factoid question)
Réponse 2: (your answer to the factoid question)
...
Question 10: (your factoid question)
Réponse 10: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::"""

# prompt to rate groundedness of the question
question_groundedness_critique_prompt = """
You will be given a context and a question, both in french.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your short rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """

# prompt to check the relevance of the answer
question_relevance_critique_prompt = """
You will be given a question in french.
Your task is to provide a 'total rating' representing how useful this question can be to someone looking for information or procedure about laws of the canton du Valais in Switzerland.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your short rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """


def create_qa(chunk, llm):
    """
     Generate question and answer pairs from a given chunk of text using a language model.

    Parameters:
    - chunk: A FAISS Document instance containing the text chunk and associated metadata.
             The text content is accessed via `chunk.page_content`, and metadata (e.g., title) via `chunk.metadata`.
    - llm: An instance of an Ollama-hosted language model, used to generate question-answer pairs from the text.

    Returns:
    - A dictionary with the following keys:
        - "title": The title of the source document from which the chunk originates.
        - "generated_questions": A list of question-answer pairs parsed from the model's output.
        - "context": The original text content from the chunk.
    """
    context = chunk.page_content
    prompt = QA_generation_prompt.format(context=context)
    generated_questions = llm.invoke(prompt)
    generated_questions = parse_questions_answers(generated_questions)
    dico_res = {
        "title": chunk.metadata["title"],
        "generated_questions": generated_questions,
        "context": context,
    }

    return dico_res


def parse_questions_answers(generated_questions):
    """Parse the generated questions and answers. Note that it is empirical, based on eobservation of the output of the llm.
    Parameters:
    -generated_questions: text
    Returns:
    - a list of dictionnaries with keys "question" and "answer"
    """
    questions = generated_questions.content.split("Question")
    QA_pairs = []
    for question in questions:
        try:
            if "Réponse" not in question:
                continue
            qu, ans = question.split("Réponse")
            qu = qu[3:].strip(":").strip()
            ans = ans[3:].strip(":").strip()
            QA_pairs.append({"question": qu, "answer": ans})
        except:
            print("Error")
            print(question)
            print("-" * 70)
            continue
    return QA_pairs


def generate_questions_for_chunks_parallel(chunks, llm, max_workers=4):
    """Process chunks in parallel using ThreadPoolExecutor."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(create_qa, chunk, llm): chunk for chunk in chunks}

        for i, future in enumerate(
            tqdm(
                as_completed(futures),
                total=len(chunks),
                desc="Processing Chunks, creating questions",
            )
        ):
            results.append(future.result())

    formatted_results = []
    for result in results:
        for qa in result["generated_questions"]:
            formatted_results.append(
                {
                    "title": result["title"],
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "context": result["context"],
                    "id": str(uuid4()),
                }
            )

    return formatted_results


def grade_generated_questions(generated_questions, llm_groundness, llm_relevance):
    """
    Evaluate the quality of generated questions by scoring their groundedness and relevance using two separate LLMs.

    Parameters:
    - generated_questions: A list of dictionaries, each representing a generated question with at least the keys:
        - "question": The question text.
        - "context": The original text context from which the question was derived.
    - llm_groundness: An Ollama-hosted LLM instance used to evaluate how well the question is grounded in the context.
    - llm_relevance: An Ollama-hosted LLM instance used to evaluate how relevant the question is to the context or topic.

    Returns:
    - The input list `generated_questions`, with each dictionary updated to include:
        - "groundness": A float score (parsed from LLM output) indicating how well the question is supported by the context.
        - "relevance": A float score indicating how relevant the question is.

    Notes:
    - Prompts are dynamically generated using predefined templates: `question_groundedness_critique_prompt` and `question_relevance_critique_prompt`.
    - LLM invocations for groundedness and relevance are executed in parallel using `ThreadPoolExecutor` for efficiency.
    - If parsing the scores from the LLM output fails, a default score of 0 is assigned.
    """
    for question in tqdm(generated_questions, total=len(generated_questions)):
        prompt_groundness = question_groundedness_critique_prompt.format(
            question=question["question"], context=question["context"]
        )
        prompt_relevance = question_relevance_critique_prompt.format(
            question=question["question"]
        )

        def get_groundness():
            return llm_groundness.invoke(prompt_groundness)

        def get_relevance():
            return llm_relevance.invoke(prompt_relevance)

        # Run in parallel
        with ThreadPoolExecutor() as executor:
            future_groundness = executor.submit(get_groundness)
            future_relevance = executor.submit(get_relevance)

            # Get results
            groundness = future_groundness.result()
            relevance = future_relevance.result()

        try:  # parsing the result
            if "Total rating:" in groundness.content:
                groundness = groundness.content.split("Total rating: ")[1]
            else:
                groundness = groundness.content.split("Total rating : ")[1]
            groundness = float(groundness)
        except:
            print("Error")
            print(groundness)
            groundness = 0
        try:
            if "Total rating:" in relevance.content:
                relevance = relevance.content.split("Total rating: ")[1]
            else:
                relevance = relevance.content.split("Total rating : ")[1]
            relevance = float(relevance)
        except:
            print("Error")
            print(relevance)
            relevance = 0
            print("-" * 70)
        question["groundness"] = groundness
        question["relevance"] = relevance
    return generated_questions


def filter_questions(graded_questions, min_groundness=3, min_relevance=3):
    """
    Filter a list of graded questions based on minimum groundedness and relevance scores.

    Parameters:
    - graded_questions: A list of dictionaries, each containing:
        - "groundness": A float score indicating how well the question is grounded in the context.
        - "relevance": A float score indicating how relevant the question is to the context or topic.
    - min_groundness: Minimum acceptable groundedness score (default: 3).
    - min_relevance: Minimum acceptable relevance score (default: 3).

    Returns:
    - A filtered list of questions that meet or exceed both the groundedness and relevance thresholds.
    """
    filtered_questions = []
    for question in graded_questions:
        if (
            question["groundness"] >= min_groundness
            and question["relevance"] >= min_relevance
        ):
            filtered_questions.append(question)
    return filtered_questions


def retrieve(query: str, vector_base, compression_retriever, reranking=True):
    """ "
    Retrieves documents relevant to a query using either a compression-based retriever (with reranking)
    or a standard vector similarity search, and formats the results for further processing or display.

    Args:
        query (str): The input query string used to retrieve relevant documents.
        vector_base: An object containing a FAISS vector store used for similarity search when reranking is disabled.
        compression_retriever: A retriever object that returns documents along with relevance scores, used when reranking is enabled.
        reranking (bool, optional): Whether to use the compression-based retriever with reranking. Defaults to True.

    Returns:
        serialized (str): A string representation of the retrieved documents, formatted for display or logging.
        retrieved_docs (list): A list of tuples, each containing a document and its relevance score.
    """

    if reranking:
        retrieved_docs = compression_retriever.invoke(query)
        serialized = "\n\n".join(
            (f"Source: {doc.page_content}") for doc in retrieved_docs
        )
        cleaned_docs = []
        for doc in retrieved_docs:
            metadata = doc.metadata
            relevance_score = metadata.pop("relevance_score")
            doc.metadata = metadata
            cleaned_docs.append((doc, relevance_score.astype(np.float64)))
        retrieved_docs = cleaned_docs
    else:  # should not be used
        print(
            "WARNING reranking is not used, you sould use it since it improves retrieval al lot"
        )

        retrieved_docs = (
            vector_base.vectorstore.similarity_search_with_relevance_scores(query, k=5)
        )
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nScore: {score}\nContent: {doc.page_content}")
            for doc, score in retrieved_docs
        )
    return serialized, retrieved_docs


def filter_non_retrieved_questions(
    questions, vectorbase, compression_retriever, reranking=True
):
    """Filter out questions that were not retrieved.
    Args:
        questions: list of questions, list of dictionaries with keys "question" and "title": title of the article
        vectorbase: vectorbase object
        compression_retriever: compression retriever object
        reranking: whether to use reranking or not
    Returns:
        filtered_questions: list of questions that were retrieved"""
    filtered_questions = []
    print("Filtering out non retrieved questions")
    for question in tqdm(questions, total=len(questions)):
        query = question["question"]
        doc_origin = question["title"]
        context, retrieved_docs = retrieve(
            query, vectorbase, compression_retriever, reranking=reranking
        )
        if doc_origin in [doc[0].metadata["title"] for doc in retrieved_docs]:
            question["retrieved_context"] = context
            filtered_questions.append(question)
    print(
        f"Retrieved {len(filtered_questions) / len(questions) * 100}% of the questions"
    )
    return filtered_questions


# prompt used to evaluate an answer given by the RAG, considering the context, the question and a reference answer
JUDGE_PROMPT = """
You will be given a user question, a context, a reference answer and a generated answer.
Your task is to provide a 'total rating' scoring how well the generated answer answers the user question regarding the context.
Give your answer as an integer on a scale of 1 to 5, where 1 means that the generated answer is not helpful at all and entirely false,
and 5 means that the answer completely addresses the question and matches the truth.

Here is the scale you should use to build your answer:
1: The generated answer  is terrible: completely irrelevant to the question asked, or false
2: The generated answer is mostly not helpful, there are invented information, meaning not in the context
3: The generated answer is mostly helpful: information given is true or in the context but question is not answered
4: The generated answer is helpful: is true but it lacks some details
5: The generated answer  is excellent: relevant, true, and exhaustive

Provide your feedback as follows, this is very important. The format of your answer must absolutely follow the following template. Do not add anything else after your rating:
Template:
Feedback:::
Evaluation: your short rationale for the rating, as a text
Total rating: (x/5)

Now here are the question, the context the reference answer and the generated answer.

Question: {question}
Context: {context}
Reference answer: {true_answer}
Generated answer: {generated_answer}

Feedback:::
Total rating: """

# load examples of accurate ratings for few shots
#print current working directory
print(os.getcwd())

with open(
    "../data/intermediate_results/eval_few_shots_examples/few_shots_examples.txt",
    "r",
    encoding="utf-8",
) as f:
    few_shots_examples = f.read()

# same with few shots
JUDGE_PROMPT_few_shots = """
You will be given a user question, a context, a reference answer and a generated answer.
Your task is to provide a 'total rating' scoring how well the generated answer answers the user question.
Give your answer as an integer on a scale of 1 to 5, where 1 means that the generated answer is not helpful at all and entirely false,
and 5 means that the answer completely addresses the question and matches the truth.

Here is the scale you should use to build your answer:
1: The generated answer  is terrible: completely irrelevant to the question asked, or false
2: The generated answer is mostly not helpful, there are invented information, meaning not in the context
3: The generated answer is mostly helpful: information given is true or in the context but question is not answered
4: The generated answer is helpful: is true but it lacks some details
5: The generated answer  is excellent: relevant, true, and exhaustive

Here are examples:
{examples}

Provide your feedback as follows, this is very important. The format of your answer must abslutely follow the following template. Do not add anything alse after your rating:
Template:
Feedback:::
Evaluation: your short rationale for the rating, as a text
Total rating: (x/5)

Now here are the question, the context the reference answer and the generated answer.

Question: {question}
Context: {context}
Reference answer: {true_answer}
Generated answer: {generated_answer}

Feedback:::
Total rating: """


def evaluate_sample(sample, llm_judge, few_shots=False):
    """
    Evaluate a single generated answer by comparing it to the ground truth using a judging LLM.

    Args:
        sample (dict): A dictionary containing:
            - "question": The input question.
            - "RAG_answer": The answer generated by the RAG system.
            - "context": The source context used for generation.
            - "answer": The ground-truth reference answer.
        llm_judge: An LLM instance (e.g. via Ollama) used to evaluate the quality of the generated answer.
        few_shots (bool, optional): Whether to use few-shot prompting with examples. Defaults to False.

    Returns:
        sample (dict): The updated sample dictionary with an added rating score:
            - "rating" if few_shots is False.
            - "rating_few_shots" if few_shots is True.
            If the generated answer is invalid ("Error") or parsing fails, the rating is set to NaN.
    """

    question = sample["question"]
    generated_answer = sample["RAG_answer"]
    context = sample["context"]
    if generated_answer == "Error":
        sample["rating"] = np.nan
        return sample
    true_answer = sample["answer"]
    if not few_shots:
        prompt = JUDGE_PROMPT.format(
            question=question,
            true_answer=true_answer,
            generated_answer=generated_answer,
            context=context,
        )
    else:
        prompt = JUDGE_PROMPT_few_shots.format(
            question=question,
            true_answer=true_answer,
            generated_answer=generated_answer,
            context=context,
            examples=few_shots_examples,
        )

    feedback = llm_judge.invoke(prompt)  # call the judge llm

    try:  # parse rating
        pattern = r"Total rating: ?\(?(\d)(?:/5)?\)?"
        match = re.search(pattern, feedback.content)
        if match:
            rating = int(match.group(1))
        else:
            print(feedback.content)
            rating = np.nan
    except Exception as e:
        print(e)
        print("New eroooooooooooooooooooor")

    if few_shots:
        sample["rating_few_shots"] = rating
    else:
        sample["rating"] = rating
    return sample


def evaluate_generated_answers_parallel(
    generated_answers, llm_judge, few_shots=False, num_workers=4
):
    """
    Evaluate a list of generated answers in parallel using a judging LLM and multithreading.

    Args:
        generated_answers (list): A list of sample dictionaries, each containing:
            - "question"
            - "RAG_answer"
            - "context"
            - "answer"
        llm_judge: An LLM instance used to evaluate the quality of the generated answers.
        few_shots (bool, optional): Whether to use few-shot prompting during evaluation. Defaults to False.
        num_workers (int, optional): Number of parallel threads to use. Defaults to 4.

    Returns:
        results (list): A list of evaluated sample dictionaries, each annotated with a rating field
                        ("rating" or "rating_few_shots" depending on the `few_shots` flag).
    """
    results = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_sample = {
            executor.submit(evaluate_sample, sample, llm_judge, few_shots): sample
            for sample in generated_answers
        }

        for future in tqdm(
            as_completed(future_to_sample),
            total=len(generated_answers),
            desc="Evaluating",
        ):
            results.append(future.result())

    return results


def evaluate_generated_answers_from_lists(
    questions, true_answers, generated_answers, llm_judge
):
    answers_good_format = [
        {"question": question, "answer": answer, "truth": true_answer}
        for question, true_answer, answer in zip(
            questions, true_answers, generated_answers
        )
    ]
    return evaluate_generated_answers_parallel(answers_good_format, llm_judge)


def get_answers_from_rag(
    graph,
    config,
    questions,
    output_path,
    existing_answers=[],
    save_every=10,
    recompute_errors=False,
):
    """
    Get answers from a RAG model.
    Args:
        graph: RAG model
        config: RAG configuration
        questions: list of questions i.e dict with keys 'question', 'title' which is is title of the documents fron wich the question comes
    Returns:
        answers: list of answers
    """
    # Disable logging because tomauch INFO messages
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    idx_start = len(existing_answers)
    answers = []
    print(
        f"Getting answers from RAG model (checkpoint saved every {save_every} answers):"
    )

    for i, question in tqdm(enumerate(questions), total=len(questions)):
        if i < idx_start:
            if recompute_errors and existing_answers[i]["RAG_answer"] == "Error":
                pass
            else:
                answers.append(existing_answers[i])
                continue

        only_question = question["question"]
        title = question["title"]
        input_message = f"In the context of: {title}\nQuestion: {only_question}"

        input_message = input_message.replace("'", " ").replace("’", " ")

        step = {}
        final_step = {}
        tries = 0
        while (
            "generate" not in final_step
        ) and tries < 5:  # sometimes fails at generating response
            tries += 1

            for step in graph.stream(
                {"messages": [{"role": "user", "content": input_message}]},
                stream_mode="updates",
                config=config,
            ):
                pass
            final_step = step

        if "generate" in final_step:
            question["RAG_answer"] = final_step["generate"]["messages"][0].content
            answers.append(question)

        else:
            assert tries >= 5
            print("Error")
            print(f"input_message: {input_message}")
            print(step)
            question["RAG_answer"] = "Error"
            answers.append(question)

        if (i + 1) % save_every == 0:  # save checkpoint
            if i < idx_start:
                saved_answers = answers + existing_answers[i + 1 :]
            else:
                saved_answers = answers
            print(f"Saving checkpoint at {output_path}")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(saved_answers, f, ensure_ascii=False)

    return answers
