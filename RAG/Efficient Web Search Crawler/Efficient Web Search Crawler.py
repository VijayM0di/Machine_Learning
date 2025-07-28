import asyncio
import argparse
import json
import sys
import re
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from googlesearch import search
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig

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
    """
    Convert all heading levels (###, ####, etc.) to ## (h2) headings
    and remove any heading with just a single # (h1)
    """
    # Match any markdown heading (# through ######)
    heading_pattern = re.compile(r'^(#{1,6})\s+(.*?)$', re.MULTILINE)
    
    # Replace all headings with ## level headings
    normalized_markdown = heading_pattern.sub(lambda m: f"## {m.group(2)}", markdown_text)
    
    return normalized_markdown

async def crawl_url(url):
    try:
        # Create configuration to exclude images and ALL links
        config = CrawlerRunConfig(
            exclude_external_images=True,   # Excludes external images
            exclude_all_images=True,        # Excludes all images
            exclude_external_links=True,    # Excludes external links
            exclude_internal_links=True,    # Excludes internal links (prevents further crawling)
        )
        
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url, config=config, depth=0)
            
            # Normalize all headings to ## level
            markdown = normalize_headings(result.markdown)
            
            # Remove any remaining links from the markdown
            markdown = markdown.replace('[', '').replace(']', '')
            
            return url, markdown
    except Exception as e:
        return url, f"Error crawling {url}: {str(e)}"

async def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Search Google and convert top results to markdown")
    parser.add_argument("--query", help="Search query (optional, will prompt if not provided)")
    parser.add_argument("--num", type=int, default=5, help="Number of top results to process (default: 5)")
    args = parser.parse_args()
    
    # Get query from command line args or user input
    query = args.query
    if not query:
        query = input("Enter your search query: ")
    
    num_results = args.num
    
    print(f"Searching Google for: {query}")
    search_results = google_search_json(query, top_n=num_results)
    
    if not search_results:
        print("No search results found.")
        return
    
    print(f"Found {len(search_results)} results. Converting to markdown...")
    
    # Create a list to store all crawling tasks
    tasks = []
    for result in search_results:
        url = result["url"]
        tasks.append(crawl_url(url))
    
    # Run all crawling tasks concurrently
    results = await asyncio.gather(*tasks)
    
    # Combine all results
    all_markdown = ""
    for url, markdown in results:
        # Find the title from search results
        title = next((result["title"] for result in search_results if result["url"] == url), "Unknown Title")
        
        # Add a header with the title and URL using ## instead of #
        all_markdown += f"## {title}\n\n"
        all_markdown += f"Source: {url}\n\n"
        all_markdown += markdown
        all_markdown += "\n\n" + "-"*80 + "\n\n"  # Add separator between articles
    
    # Print a preview
    print("\n--- Preview of Combined Markdown Content ---\n")
    preview_length = 500
    print(all_markdown[:preview_length] + ("..." if len(all_markdown) > preview_length else ""))
    
    # Optionally save to file
    save_option = input("\nDo you want to save the markdown to a file? (y/n): ")
    if save_option.lower() == 'y':
        filename = input("Enter filename (default: search_results.md): ") or "search_results.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(all_markdown)
        print(f"Markdown saved to {filename}")

if __name__ == "__main__":
    asyncio.run(main())