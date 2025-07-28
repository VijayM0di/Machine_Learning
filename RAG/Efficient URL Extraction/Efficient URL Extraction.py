import asyncio
import argparse
import re
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig

def normalize_headings(markdown_text):
    """
    Convert all heading levels (###, ####, etc.) to ## (h2) headings
    """
    # This regex finds all markdown headings (# through ######)
    heading_pattern = re.compile(r'^(#{1,6})\s+(.*?)$', re.MULTILINE)
    
    # Replace all headings with ## level headings
    normalized_markdown = heading_pattern.sub(r'## \2', markdown_text)
    
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
            normalized_markdown = normalize_headings(result.markdown)
            
            return normalized_markdown
    except Exception as e:
        return f"Error crawling {url}: {str(e)}"

async def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Web crawler that converts URLs to markdown with only ## headings")
    parser.add_argument("--url", help="URL to crawl (optional, will prompt if not provided)")
    parser.add_argument("--text-only", action="store_true", help="Output plain text instead of markdown")
    args = parser.parse_args()
    
    # Get URL from command line args or user input
    url = args.url
    if not url:
        url = input("Enter the URL to crawl: ")
    
    print(f"Crawling {url}...")
    
    if args.text_only:
        # If text-only option is selected, use a different approach
        async with AsyncWebCrawler() as crawler:
            config = CrawlerRunConfig(
                exclude_external_images=True,
                exclude_all_images=True,
                exclude_external_links=True,
                exclude_internal_links=True,
            )
            result = await crawler.arun(url=url, config=config, depth=0)
            output = result.text  # Using .text property instead of .markdown
    else:
        # Use markdown with normalized headings
        output = await crawl_url(url)
    
    # Print the output
    if args.text_only:
        print("\n--- Plain Text Content ---\n")
    else:
        print("\n--- Markdown Content (with only ## headings) ---\n")
    print(output)
    
    # Optionally save to file
    save_option = input("\nDo you want to save the output to a file? (y/n): ")
    if save_option.lower() == 'y':
        default_ext = "pdf" if args.text_only else "md"
        filename = input(f"Enter filename (default: output.{default_ext}): ") or f"output.{default_ext}"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Content saved to {filename}")

if __name__ == "__main__":
    asyncio.run(main())