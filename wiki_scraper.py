# langchain & openai imports
from langchain.text_splitter import HTMLHeaderTextSplitter
from langchain_core.documents import Document

# web scraping imports
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

HTML_HEADERS = ["h1", "h2", "h3", "h4", "h5", "h6"]

def get_wiki_page_urls(wiki_url: str) -> list[str]:
    page_urls = []
    wiki = requests.get(wiki_url)

    for link in BeautifulSoup(wiki.content, 'html.parser').find_all('a'):
        href = link.get('href')
        if href and href.startswith("/"):  # Ensure it is a relative link
            full_url = urljoin(wiki_url, href.split("#")[0])
            if full_url.startswith(wiki_url) and full_url not in page_urls:
                page_urls.append(full_url)
                
    return page_urls

def _filter_empty_docs(docs: list[Document]) -> list[Document]:
    return [doc for doc in docs if doc.page_content.strip()]

def _deduplicate_docs(docs: list[Document]) -> list[Document]:
    unique_docs = []
    for doc in docs:
        if doc not in unique_docs:
            unique_docs.append(doc)
    return unique_docs

def get_parsed_wiki_sections(page_urls: list[str], debug: bool = True) -> list[Document]:
    docs = []
    

    def dprint(*args, **kwargs):
        if debug:
            print(*args, **kwargs)

    n_sections = 0

    # Output raw content of all the sites
    dprint(f"Discovered {len(page_urls)} websites")
    for (i, url) in enumerate(page_urls):
        dprint(f"Fetching URL {i + 1}: {url}")
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove the navigation section to avoid redundancy
        for nav in soup.find_all(['nav', 'aside']):  # Adjust based on actual navigation tags
            nav.decompose()

        # Find all header elements that have links
        header_tags = soup.find_all(HTML_HEADERS)
        
        # Initialize text splitter with desired headers
        splitter = HTMLHeaderTextSplitter(headers_to_split_on=HTML_HEADERS)

        for header in header_tags:
            if header.a and header.a.get('href'):
                subroute = header.a['href']
                header_text = header.get_text().strip()
                section_url = url + subroute
                header_section_docs = splitter.split_text(str(header) + ''.join(str(s) for s in header.find_next_siblings()))

                for doc in header_section_docs:                                    
                    n_sections += 1
                    doc.metadata['source'] = section_url
                    doc.page_content = header_text + "\n\n" + doc.page_content
                    docs.append(doc)
                    dprint(f"  [{n_sections}] Extracted section: {section_url}")

    dprint(f"Loaded {len(docs)} documents from {len(page_urls)} websites")

    # Filter out empty documents
    docs = _filter_empty_docs(docs)
    dprint(f"After filtering empty documents, we have {len(docs)} documents")

    # Filter out duplicate documents
    docs = _deduplicate_docs(docs)
    dprint(f"After filtering duplicate documents, we have {len(docs)} documents")

    # Add metadata to the documents
    for doc in docs:
        # add url from metadata to page content
        doc.page_content = f"Auszug von Wiki URL: {doc.metadata['source']}\n\n{doc.page_content}" 
                    
    return docs