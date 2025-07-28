# Web Page to Markdown Converter

A command-line tool that converts web pages to clean Markdown format with standardized H2 headings.

## Features

- Crawls a single web page and converts it to Markdown
- Standardizes all headings to H2 level (`##`)
- Option to output plain text instead of Markdown
- Excludes all images from the crawled content
- Removes all links to prevent further crawling
- Saves output to MD file

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/webpage-to-markdown.git
   cd webpage-to-markdown
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python webpage_crawler.py --url https://example.com
```

If you don't provide a URL as a command-line argument, you'll be prompted to enter one.

### Command-line Options

- `--url`: The URL to crawl (optional, will prompt if not provided)
- `--text-only`: Output plain text instead of Markdown

### Example

```bash
# Convert a webpage to Markdown with standardized H2 headings
python webpage_crawler.py --url https://en.wikipedia.org/wiki/Markdown

# Get plain text output
python webpage_crawler.py --url https://en.wikipedia.org/wiki/Markdown --text-only
```

## Output

After crawling the URL, the script will:

1. Display the content in the terminal
2. Ask if you want to save it to a file
3. If yes, prompt for a filename (defaults to `output.md`)

## How It Works

The script uses the `crawl4ai` library to:

1. Crawl the specified URL
2. Process the content to exclude images and links
3. Normalize all heading levels (###, ####, etc.) to ## (H2) headings using regex
4. Output the standardized Markdown content

## Requirements

See `requirements.txt` for the full list of dependencies.

## License

MIT