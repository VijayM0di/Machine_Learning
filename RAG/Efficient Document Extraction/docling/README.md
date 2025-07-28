# 🧠 docling – Efficient PDF Content Extraction & Chunking

This submodule is part of a larger repository and is focused on **extracting structured content from PDF files** using the [`docling`](https://github.com/docling-project/docling) library and applying **semantic chunking** with Hugging Face Transformers. It supports extraction of:

- 📝 Raw text and markdown  
- 🖼️ Embedded images and figures  
- 📊 Tables  
- 🔗 Hybrid text chunks optimized for downstream NLP tasks  

---

## 📂 Folder Structure

| File / Folder         | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `1_docling.ipynb`     | A Jupyter notebook demonstrating basic PDF-to-markdown conversion, image and table extraction, and custom OCR pipelines using `docling_core`. |
| `2_hybrid_chunker.py` | A Python script for generating semantically meaningful text chunks using Hugging Face's `AutoTokenizer` and `HybridChunker` from `docling`. |
| `extracted_images/`   | Folder where extracted images and tables are saved.                         |
| `requirements.txt`    | Dependencies required to run all scripts in this folder.                    |

---

## 🚀 Installation

```bash
# Navigate to the docling folder
cd docling

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🛠️ Usage

Run the Jupyter notebook for interactive demos:

```bash
jupyter notebook 1_docling.ipynb
```

Or execute the hybrid chunking script:

```bashs
python 2_hybrid_chunker.py
```

Ensure your input PDFs are properly referenced in the code (update `DOC_SOURCE` if needed).

---

## 📌 Notes

- OCR is handled via different engines (`EasyOCR`, `Tesseract`, `RapidOCR`, etc.).
- Outputs include `.md`, `.txt`, and `.png` formats depending on the script configuration.
- This module is designed for modular use within larger PDF processing or RAG workflows.
- Performance of document conversion depends in the processing power of your machine.
- With GPU: 1 to 5 min and Without GPU: 15 to 30 min

---