from argparse import ArgumentParser
import json
import os

from dynaconf import Dynaconf
from jira import JIRA
from langchain_ollama import ChatOllama
from tqdm import tqdm

from utils import build_doc_from_issue

settings_file_path = "settings.toml"
settings=settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[settings_file_path],
)

parser= ArgumentParser(description="Build documents from JIRA tickets")
parser.add_argument("--model_name", type=str, default="llama3.1:8b", 
                    help="Name of the model to use to generate tutorials from JIRA tickets")
parser.add_argument("--output_dir", type=str, default="./src/CIT/scraping/JIRA/jira_documents",
                    help="Directory to save the generated documents")


if __name__ == "__main__":
    args= parser.parse_args()
    model_name=args.model_name
    print(f"Using model: {model_name}")
    output_dir=args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Output directory: {output_dir}")

    llm = ChatOllama(model=model_name, temperature=0)
    headers = JIRA.DEFAULT_OPTIONS["headers"].copy()
    headers["Authorization"] = f"Bearer {settings.connectors.jira.TOKEN}"
    
    jira = JIRA(server=settings.connectors.jira.URL, options={"headers": headers})
    issues = jira.search_issues(settings.connectors.jira.QUERY,json_result=True,maxResults=1000)
    issues=issues["issues"]

    print(f"Found {len(issues)} tickets")

    documents = []
    for issue in tqdm(issues, desc="Processing issues"):
        try:
            doc = build_doc_from_issue(issue, llm)
            documents.append(doc)
            #print(f"Document for issue {issue['key']} created")
        except Exception as e:
            print(f"Error while processing issue {issue['key']}: {e}")

    print(f"Generated {len(documents)} documents")
    for doc in documents:
        key = doc["id"]
        filename = os.path.join(output_dir, f"{key}.json")
        with open(filename, "w") as f:
            json.dump(doc, f, indent=4)
    print(f"Documents saved in {output_dir}")





