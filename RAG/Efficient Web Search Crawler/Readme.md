# Google Search to Markdown Converter

A command-line tool that searches Google for a query, crawls the top results, and compiles them into a single Markdown document with standardized H2 headings.

## Features

- Performs Google searches and retrieves top results
- Crawls each search result and extracts content
- Converts all content to Markdown format
- Standardizes all headings to H2 level (`##`)
- Preserves title and source URL for each article
- Removes images and links from the crawled content
- Adds separators between different articles
- Processes multiple URLs concurrently for better performance
- Provides a preview of the combined content
- Saves output to a Markdown file

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/google-search-to-markdown.git
   cd google-search-to-markdown
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python search_crawler.py --query "your search query"
```

If you don't provide a query as a command-line argument, you'll be prompted to enter one.

### Command-line Options

- `--query`: The search query (optional, will prompt if not provided)
- `--num`: Number of top search results to process (default: 5)

### Example

```bash
# Search for "markdown guide" and convert top 5 results
python search_crawler.py --query "markdown guide"

# Search for "python async" and convert top 3 results
python search_crawler.py --query "python async" --num 3
```

## Output

After crawling all the search results, the script will:

1. Display a preview of the combined content in the terminal
2. Ask if you want to save it to a file
3. If yes, prompt for a filename (defaults to `search_results.md`)

The output Markdown will have this structure for each article:

```markdown
## Article Title

Source: https://example.com/article

## Subheading from the article
Content from the article...

--------------------------------------------------------------------------------

## Next Article Title
...
```

## How It Works

The script performs these steps:

1. Searches Google using the provided query
2. Extracts metadata (title, description) from the search results
3. Crawls each result URL and converts the content to Markdown
4. Normalizes all heading levels to H2 using regex
5. Combines all articles into a single Markdown document
6. Adds proper attribution with title and source URL

## Requirements

See `requirements.txt` for the full list of dependencies.

## License

MIT