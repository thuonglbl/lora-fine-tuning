import json
import os
from argparse import ArgumentParser

from langchain_ollama import ChatOllama
from tqdm import tqdm

parser=ArgumentParser()

parser.add_argument(
    "--directory",
    type=str,
    default="./src/CIT/documents/run3/confluence_json",
    help="Directory containing the documents to be indexed",
)
parser.add_argument(
    "--new_directory_path",
    type=str,
    default="./src/CIT/documents/run3/confluence_json_summarized",
    help="Directory to save the summarized documents",
)

parser.add_argument(
    "--model_name",
    type=str,
    default="llama3.1:8b",
    help="Model name to use to summarize the documents",
)


prompt_summ=""" I need you to describe what is in the document. Do not cite the title, or the content directly. Do not tell what the document says, only capture briefly what it deals with. It should be a summary of two short sentences.
Document: {document_content}.
Do not say things like 'Here is a summary', instead just start with the summary.
Summary:"""

if __name__ == "__main__":
    args= parser.parse_args()
    # Load the documents
    documents = []
    for filename in os.listdir(args.directory):
        if filename.endswith(".json"):
            with open(os.path.join(args.directory, filename), "r") as f:
                documents.append(json.load(f))
    # Create the new directory if it does not exist
    if not os.path.exists(args.new_directory_path):
        os.makedirs(args.new_directory_path)
    # Initialize the model
    model = ChatOllama(model=args.model_name, temperature=0)

    # Summarize each document
    for doc in tqdm(documents, desc="Summarizing documents"):
        # Get the content of the document
        content = doc["content"]
        # Summarize the document
        summary = model.invoke(prompt_summ.format(document_content=content)).content
        # Save the summary in the new directory
        filename = os.path.join(args.new_directory_path, f"{doc['title']}.json")
        doc["summary"] = summary
        with open(filename, "w") as f:
            json.dump(doc, f, indent=4)

