# 🧠 Document-based RAG QA System with LanceDB, HuggingFace & Ollama

This project implements a Retrieval-Augmented Generation (RAG) system that extracts content from Documents, chunks and embeds the data into a vector database (LanceDB), and enables question answering using either HuggingFace or Ollama language models.


## 🔥 Features

- 📄 **PDF Parsing**: Extracts rich content (text, tables, images) using `docling`.
- 🖼️ **Image Summarization**: Auto-generates image descriptions using `LLaVA` (`ollama`).
- 🧩 **Intelligent Chunking**: Document segmentation using `markdownsplitter` and `recursivecharactertextsplitter`.
- 🧠 **Embeddings**: Sentence-level embeddings via `all-mpnet-base-v2`.
- 🏛 **Vector Storage**: Stores embeddings efficiently into **LanceDB**.
- 🤖 **Question Answering**:
  - `huggingface_QA.py`: HuggingFace LLaMA for text-based QA.
  - `ollama_QA.py`: Local Ollama models (`mistral:latest`, `llava`) for multimodal QA.

---

- Python 3.10 +
- CUDA-compatible GPU
- `ollama` running locally (`ollama serve`)

 ---


## 🌐 Semantic Web Search to LanceDB Pipeline

This project implements a full pipeline for semantic web search using Google, content crawling, markdown formatting, chunking, and vector storage using `SentenceTransformer` and `LanceDB`.


## 🧠 Use Cases

- 🔍 Intelligent Search Indexing
- 📖 LLM-based Document Q&A (RAG)
- 🧑‍💻 Research Assistance
- 📂 Knowledge Base Creation
- 🤖 Preprocessing for Chatbot Memory

---

## 📦 Features


- 🔎 Google Search
    ⇩
- 🌐 URL Metadata & Validation
    ⇩
- 🕸️ Async Crawling with Crawl4AI
    ⇩
- 📝 Markdown Normalization
    ⇩
- ✂️ Smart Chunking (MarkdownHeader + RecursiveSplitter)
    ⇩
- 🧠 SentenceTransformers Embedding using `all-mpnet-base-v2`
    ⇩
- 🗃️ LanceDB Vector Storage

---

## 🛠️ Requirements

Install the following dependencies:


```bash
pip install -r requirements.txt