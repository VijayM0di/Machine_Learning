# PDF-Extraction-for-LLM 🚀

> An optimized approach for extracting, structuring, and chunking PDF content for improved RAG accuracy with Large Language Models.

## Overview

This repository provides an efficient methodology for PDF text extraction specifically designed to improve Retrieval-Augmented Generation (RAG) performance with Large Language Models (LLMs). Our research demonstrates that naive text chunking methods using popular libraries like LlamaIndex or LangChain often result in poor accuracy (≤70%) and inefficient resource utilization for QA tasks.

## 🔍 The Problem

Traditional PDF extraction libraries are not optimized for LLM-based question answering tasks. Common issues include:

- Poor text ordering and structure preservation
- Loss of formatting and hierarchical information
- Inefficient chunking that breaks contextual continuity
- Metadata loss (page numbers, sections, references)
- Inability to handle mixed content (text, tables, formulas)

## 💡 Our Solution

After extensive evaluation of various techniques and libraries, we've developed an optimized approach that:

1. Converts PDF documents to structured Markdown format
2. Preserves document hierarchy (headings, sections, paragraphs)
3. Maintains essential metadata (page numbers, subsections)
4. Handles tables, images, and formulas efficiently
5. Enables intelligent chunking based on semantic boundaries

## 🛠️ Technology Stack

Our solution leverages the following key technologies:

- **PyMuPDF4LLM**: Converts PDF content to structured Markdown while preserving formatting and hierarchy
- **Docling**: Lightning-fast PDF processing with comprehensive element extraction
- **Marker.pdf**: Alternative high-quality extraction for complex documents
- **Custom chunking algorithms**: Using semantic markers (**, ##, \n\n) for intelligent text segmentation

## ⚡ Performance

Our approach demonstrates significant improvements over traditional methods:

- **Processing Speed**: <1 second per page (60 pages in 55 seconds on 32GB RAM)
- **RAG Accuracy**: Substantial improvement over conventional methods
- **Resource Efficiency**: Optimized for standard hardware configurations

## 🔄 Workflow

1. **PDF Extraction**: Convert PDFs to structured Markdown using PyMuPDF4LLM or Docling
2. **Metadata Preservation**: Store page numbers, sections, and references (IN PROGRESS)
3. **Image Handling**: Extract and store images in a dedicated directory
4. **Intelligent Chunking**: Split text using markdown symbols and semantic boundaries
5. **LLM Integration**: Pass properly chunked text to LLMs for RAG tasks

## 📊 Comparison with Other Methods

| Method | Processing Speed | RAG Accuracy | Preserves Structure | Handles Mixed Content |
|--------|-----------------|--------------|---------------------|----------------------|
| LangChain PDF Loader | Medium | ≤70% | Partial | Limited |
| LlamaIndex Text Loader | Medium | ≤70% | Partial | Limited |
| PyPDF | Fast | Poor | No | No |
| PyPDFPlumber | Slow | Moderate | Partial | Partial |
| **Docling** | **Fast (<1s/page)** | **High** | **Yes** | **Yes** |