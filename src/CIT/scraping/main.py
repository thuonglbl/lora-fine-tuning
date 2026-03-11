import json
import os
import re
from html.parser import HTMLParser
from io import StringIO

from atlassian import Confluence
from config import settings
from llama_index.core import Document
from markdown import Markdown
from markdownify import markdownify as md
from tqdm import tqdm

from CIT.evaluation.utils import extract_confluence_urls


def add_documents_to_storage(
    docs: list[Document], destination_folder: str, replace: bool = False
):
    """
    Helper to create simple text documents in the storage

    Args:
        docs (list[Document]): documents to be uploaded
        destination_folder (str): folder where to put the documents
        replace (bool, optional): TO replace the file if already exists. Defaults to False.
    """

    for doc in tqdm(docs):
        title = doc.metadata["title"]
        content = f"{doc.metadata}\n\n{doc.text}"
        title = title.replace("/", "-")
        # Upload the file if not existing, or if we want to replace
        if not os.path.exists(f"{destination_folder}/{title}.md") or replace:
            with open(f"{destination_folder}/{title}.md", "w") as f:
                f.write(content)


def load_confluence(
    url: str,
    space: str,
    destination_folder: str = settings.storage.blob.FOLDER_NAME_CONFLUENCE,
    replace: bool = False,
):
    """
    Function to load the confluenc data from a URL/space

    Args:
        url (str): Confluence Base URL
        space (str): Confluence Space Key
    """
    confluence = Confluence(
        url=url,
        username=settings.connectors.confluence.USER,
        token=settings.connectors.confluence.TOKEN,
        cloud=False,
        verify_ssl=True,
    )

    print(space)
    print(settings.connectors.confluence.USER)

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # get all pages in the space, and select the body.export_view to get the page content
    # the limit is 100 pages per request so we need to loop through all the pages
    pages = []
    start_index = 0
    while True:
        pages_temp = confluence.get_all_pages_from_space(
            space=space, expand="body.export_view", limit=100, start=start_index
        )
        if len(pages_temp) == 0:
            break
        pages.extend(pages_temp)
        start_index += 100

    all_urls = []
    mapping_id_paths = {}
    mapping_id_urls = {}
    mapping_original_urls_urls_fixed = {}
    mapping_urls_paths = {}
    mapping_url_id = {}
    mapping_urls_titles = {}
    template_url = (
        "https://confluence.yourcompany.com/spaces/CORPORATEITKNOWLEDGEBASE/pages/{id}"
    )
    template_path = destination_folder + "/{title}.json"

    pages_with_metadata = []

    for page in pages:
        original_title = page["title"]
        title = original_title.replace("/", "-").strip()
        page_url = url + page["_links"]["webui"]
        id = page["id"]
        url_standard = template_url.format(id=id)

        children_titles = confluence.get_child_title_list(id)
        children_ids = confluence.get_child_id_list(id)
        parent = confluence.get_parent_content_title(id)
        parent_id = confluence.get_parent_content_id(id)

        # fill in mappings
        all_urls.append(page_url)
        all_urls.append(url_standard)
        mapping_id_paths[id] = template_path.format(title=title)
        mapping_id_urls[id] = url_standard
        mapping_original_urls_urls_fixed[page_url] = url_standard
        mapping_urls_paths[url_standard] = template_path.format(title=title)
        mapping_url_id[page_url] = id
        mapping_url_id[url_standard] = id
        mapping_urls_titles[url_standard] = title

        content = page["body"]["export_view"]["value"]
        # add page title to the content
        content = title + "\n" + content
        # Transform page into more readable markdown format
        content = md(content)
        # Remove images and convert to plain text
        text_content = unmark(content)

        outgoing_confluence_urls = list(set(extract_confluence_urls(text_content)))

        metadata = {
            "id": id,
            "title": title,
            "original_title": original_title,
            "url": url_standard,
            "url_original": page_url,
            "children": children_ids,
            "children_titles": children_titles,
            "outgoing_confluence_urls": outgoing_confluence_urls,
            "parent": parent_id,
            "parent_title": parent,
            "content": text_content,
        }
        pages_with_metadata.append(metadata)

    for page in pages_with_metadata:
        metadata = page
        outgoing_confluence_urls = metadata["outgoing_confluence_urls"]
        outgoing_page_ids = [
            mapping_url_id[url]
            for url in outgoing_confluence_urls
            if url in mapping_url_id
        ]
        metadata["outgoing_page_ids"] = outgoing_page_ids
        metadata["outgoing_page_paths"] = [
            mapping_id_paths[page_id]
            for page_id in outgoing_page_ids
            if page_id in mapping_id_paths
        ]
        path_page = template_path.format(title=metadata["title"])
        #print(path_page)
        if not os.path.exists(path_page) or replace:
            with open(path_page, "w") as f:
                json.dump(metadata, f, indent=4)

    print(f"Loaded {len(pages)} pages from Confluence space {space}")
    print(f"Storage path: {destination_folder}")

    # save mappings

    if not os.path.exists(f"{destination_folder}/mappings"):
        os.makedirs(f"{destination_folder}/mappings")
    with open(f"{destination_folder}/mappings/mapping_urls_paths.json", "w") as f:
        json.dump(mapping_urls_paths, f, indent=4)
    with open(f"{destination_folder}/mappings/mapping_id_paths.json", "w") as f:
        json.dump(mapping_id_paths, f, indent=4)
    with open(f"{destination_folder}/mappings/mapping_id_urls.json", "w") as f:
        json.dump(mapping_id_urls, f, indent=4)
    with open(
        f"{destination_folder}/mappings/mapping_original_urls_urls_fixed.json", "w"
    ) as f:
        json.dump(mapping_original_urls_urls_fixed, f, indent=4)
    with open(f"{destination_folder}/mappings/mapping_urls_titles.json", "w") as f:
        json.dump(mapping_urls_titles, f, indent=4)
    with open(f"{destination_folder}/mappings/all_urls.txt", "w") as f:
        for url in all_urls:
            f.write(url + "\n")


####################################################
# Helper functions for text extraction and cleaning
####################################################
# HTML Parser to extract text while keeping non-image links
class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = StringIO()

    def handle_data(self, data):
        self.text.write(data)

    def handle_starttag(self, tag, attrs):
        if tag == "a":  # Keep links
            for attr, value in attrs:
                if attr == "href":
                    self.text.write(f" {value} ")

    def get_data(self):
        return self.text.getvalue()


def strip_html(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def remove_images(text):
    """Removes Markdown image syntax but keeps other links."""
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # Remove images
    return text


# Initialize Markdown processor
__md = Markdown(output_format="html")
__md.stripTopLevelTags = False  # Keep raw structure


def unmark(text):
    """Converts Markdown text to plain text, keeping URLs but removing images."""
    text = remove_images(text)  # Remove images first
    html = __md.convert(text)  # Convert Markdown to HTML
    return strip_html(html)  # Extract plain text + URLs


def unmark_list_of_documents(md_folder, txt_folder):
    """
    Converts a list of documents to plain text, keeping URLs but removing images.

    Args:
        docs (list[str]): List of documents path to convert

    Returns:
        list[str]: List of plain text documents
    """
    # List all documents in the md folder
    docs_paths = os.listdir(md_folder)
    docs = [os.path.join(md_folder, doc_path) for doc_path in docs_paths]
    # Create the text folder if it does not exist
    if not os.path.isdir(txt_folder):
        os.makedirs(txt_folder)

    # Convert each document to plain text
    for doc_path in docs:
        with open(doc_path, "r") as f:
            unmarked = unmark(f.read())
        filename = os.path.basename(doc_path).replace(".md", ".txt")
        with open(f"{txt_folder}/{filename}", "w") as f:
            f.write(unmarked)


####################################################
# The main function, the entry point of the script #
####################################################

if __name__ == "__main__":
    # Loading of confluence data
    if settings.connectors.confluence.LOAD:
        load_confluence(
            url=settings.connectors.confluence.URL,
            space=settings.connectors.confluence.SPACE,
            destination_folder=settings.storage.blob.FOLDER_NAME_CONFLUENCE,
            replace=settings.storage.blob.REPLACE_EXISTING,
        )
