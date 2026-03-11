import json

from CIT.evaluation.utils import extract_confluence_urls


def replace_urls_with_titles(text,mapping_urls_titles):
    """
    Replace Confluence URLs in the text with their titles.
    """
    urls = extract_confluence_urls(text)
    titles= []
    good_urls = []
    for url in urls:
        if url not in mapping_urls_titles:
            continue
        title= mapping_urls_titles[url]
        titles.append(title)
        good_urls.append(url)
        text = text.replace(url, f"[{title}]({url})")
    return text,good_urls,titles


def get_summary_from_url(url, mapping_urls_paths):
    if url in mapping_urls_paths:
        path_source =mapping_urls_paths[url]
        with open(path_source, "r") as f:
            source = json.load(f)
        summary = source["summary"]
        return summary
    else:
        print(f"URL {url} not found in mapping_urls_paths.")
        return None