import asyncio
import re
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from googlesearch import search
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig
import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from sentence_transformers import SentenceTransformer
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig
import lancedb
from lancedb.pydantic import Vector, LanceModel

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(model_name, cache_folder="huggingface_model").to(device)

database = Path("lanceDB")
database.mkdir(parents=True, exist_ok=True)
db = lancedb.connect(database)

# table_name = "doc"

class Content(LanceModel):
    id: str
    vector: Vector(768)
    source: str
    content: str

def fetch_metadata(url, timeout=5, headers=None):
    headers = headers or {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, timeout=timeout, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        title = (soup.title.string or "").strip()
        desc_tag = soup.find("meta", attrs={"name": "description"})
        description = desc_tag.get("content", "").strip() if desc_tag else ""
        return url, title, description
    except Exception:
        return url, "", ""

def google_search_json(query, initial=20, top_n=5, workers=5):
    urls = [u for u in list(search(query, num_results=initial, lang='en')) if u.startswith(('http://', 'https://'))]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_metadata, url) for url in urls]
        for future in as_completed(futures):
            url, title, desc = future.result()
            if title or desc:
                results.append({"url": url, "title": title, "description": desc})
    return results[:top_n]

def normalize_headings(markdown_text):

    heading_pattern = re.compile(r'^(#{1,6})\s+(.*?)$', re.MULTILINE)
    normalized_markdown = heading_pattern.sub(lambda m: f"## {m.group(2)}", markdown_text)

    return normalized_markdown

async def crawl_url(url):
    try:
        config = CrawlerRunConfig(
            exclude_external_images=True,  
            exclude_all_images=True,       
            exclude_external_links=True,   
            exclude_internal_links=True,   
        )
        
        async with AsyncWebCrawler() as crawler:

            result = await crawler.arun(url=url, config=config, depth=0)
            markdown = normalize_headings(result.markdown)
            markdown = markdown.replace('[', '').replace(']', '')
            
            return url, markdown
    except Exception as e:
        return url, f"Error crawling {url}: {str(e)}"

async def main():

    query = input("Enter your search query: ")

    print(f"Searching Google for: {query}")
    search_results = google_search_json(query, top_n=5)
    
    if not search_results:
        print("No search results found.")
        return
    
    print(f"Found {len(search_results)} results. Converting to markdown...")
    
    tasks = []
    for result in search_results:
        url = result["url"]
        tasks.append(crawl_url(url))
    
    results = await asyncio.gather(*tasks)

    all_markdown = ""
    for url, markdown in results:
        title = next((result["title"] for result in search_results if result["url"] == url), "Unknown Title")
        print("title",title)

        all_markdown += f"## {title}\n\n"
        all_markdown += f"Source: {url}\n\n"
        all_markdown += markdown
        all_markdown += "\n\n" + "-"*80 + "\n\n"

    clean_query = query.replace(" ", "_").replace("-", "_")

    search_data = "search_data"
    os.makedirs(search_data,exist_ok=True)

    filename = f"search_data/{clean_query}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(all_markdown)
        
    print(f"Markdown saved to {filename}")

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "Header 1")])
    chunks = splitter.split_text(all_markdown)
    initial_chunks = [chunk.page_content for chunk in chunks]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50,
    )
    sub_chunks = []
    for chunk in initial_chunks:
        sub_chunks.extend(text_splitter.split_text(chunk) if len(chunk) > 700 else [chunk])
    # [print(f"{chunk}\n{'*'*150}") for chunk in sub_chunks ]

    
    if clean_query not in db.table_names():
        table = db.create_table(clean_query, schema=Content)
    else:
        table = db.open_table(clean_query)

    embeddings = model.encode(sub_chunks).tolist()

    records = [
        Content(
            id=f"{Path(filename)}",
            vector=vec,
            source=str(filename),
            content=text,
        )
        for (vec, text) in zip(embeddings, sub_chunks)
    ]
    table.add(records)

if __name__ == "__main__":
    asyncio.run(main())