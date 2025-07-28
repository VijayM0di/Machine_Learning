import os
import time
import asyncio
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from sentence_transformers import SentenceTransformer
from docx2pdf import convert
from pptxtopdf import convert
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import CrawlerRunConfig

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ollama import chat
import lancedb
import pyarrow as pa
from lancedb.pydantic import Vector, LanceModel

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(model_name, cache_folder="huggingface_model").to(device)

database = Path("/home/qbadmin/Desktop/QAPDF/lanceDB")
database.mkdir(parents=True, exist_ok=True)
db = lancedb.connect(database)
table_name = "doc"

class Content(LanceModel):
    id: str
    vector: Vector(768)
    source: str
    content: str

if table_name not in db.table_names():
    table = db.create_table(table_name, schema=Content)
else:
    table = db.open_table(table_name)

def process_markdown(markdown_text: str, source: str):
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "Header 1")])
    chunks = splitter.split_text(markdown_text)
    initial_chunks = [chunk.page_content for chunk in chunks]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=50,
        # separators=[
        #     "##",
        # ],
    )
    sub_chunks = []
    for chunk in initial_chunks:
        sub_chunks.extend(text_splitter.split_text(chunk) if len(chunk) > 700 else [chunk])
    [print(f"{chunk}\n{'*'*150}") for chunk in sub_chunks ]
    embeddings = model.encode(sub_chunks).tolist()

    records = [
        Content(
            id=f"{Path(source).stem}_{i}",
            vector=vec,
            source=str(source),
            content=text,
        )
        for i, (vec, text) in enumerate(zip(embeddings, sub_chunks))
    ]
    table.add(records)
    _log.info(f"✅ Stored {len(records)} chunks from {source} into LanceDB")

def process_image(image_path: Path) -> tuple[str, str]:
    try:
        prompt = """
        You are an AI assistant. Use the given image to summarize its content with a detailed explanation.

        **Image:**
        [image]

        **Answer:**"""
        start = time.time()
        response = chat(
            model='llava:7b',
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [str(image_path)]
            }]
        )
        end = time.time()
        _log.info(f"Processed {image_path.name} in {end - start:.2f} seconds")
        return image_path.name, response['message']['content']
    except Exception as e:
        _log.error(f"[LLaVA Error] {image_path.name} - {e}")
        return image_path.name, "Image summary not available."

def process_document_file(input_path: str) -> Path:
    input_doc_path = Path(input_path)
    base_path = Path("/home/qbadmin/Desktop/QAPDF")
    converted_files = base_path / "converted_files"
    output_dir = base_path / "output"
    converted_files.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename, ext = os.path.splitext(input_doc_path)
    ext = ext.lower()

    if ext == ".docx":
        convert(str(input_doc_path), str(converted_files))
        pdf_path = converted_files / f"{input_doc_path.stem}.pdf"
    elif ext == ".pptx":
        convert(str(input_doc_path), str(converted_files))
        pdf_path = converted_files / f"{input_doc_path.stem}.pdf"
    elif ext == ".pdf":
        pdf_path = input_doc_path
    else:
        raise ValueError("Unsupported file type. Use .pdf, .pptx, or .docx")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options, device=device)}
    )

    conv_res = doc_converter.convert(str(pdf_path))
    doc_filename = input_doc_path.stem
    table_counter = 0
    picture_counter = 0

    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = output_dir / f"{doc_filename}-table-{table_counter}.png"
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")
        if isinstance(element, PictureItem):
            picture_counter += 1
            element_image_filename = output_dir / f"{doc_filename}-picture-{picture_counter}.png"
            with element_image_filename.open("wb") as fp:
                element.get_image(conv_res.document).save(fp, "PNG")

    md_path = output_dir / f"{doc_filename}-with-images.md"
    conv_res.document.save_as_markdown(md_path, image_mode=ImageRefMode.REFERENCED)

    artifacts_dir = output_dir / f"{doc_filename}-with-images_artifacts"
    image_paths = sorted(artifacts_dir.glob("*.png"))
    image_summaries = {}

    def process_and_store_summary(idx, image_path):
        filename, summary = process_image(image_path)
        return idx, image_path.name, summary

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(process_and_store_summary, idx, path): idx
            for idx, path in enumerate(image_paths)
        }
        for future in as_completed(futures):
            idx, filename, summary = future.result()
            image_summaries[filename] = summary

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        if line.strip().startswith("![") and "](" in line:
            img_path = line.split("](")[-1].replace(")", "").strip()
            img_filename = os.path.basename(img_path)
            summary = image_summaries.get(img_filename)
            if summary:
                updated_lines.append(f"\n> **Image Summary:** {summary.strip()}\n\n")
        updated_lines.append(line)

    final_md_path = output_dir / f"{doc_filename}-with-image-summaries.md"
    with open(final_md_path, "w", encoding="utf-8") as f:
        f.writelines(updated_lines)

    return final_md_path

async def process_url(url: str):
    config = CrawlerRunConfig(
        exclude_external_images=True,
        exclude_all_images=True,
        exclude_external_links=True,
        exclude_internal_links=True,
    )
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, config=config, depth=0)
        markdown = result.markdown
        process_markdown(markdown, source=url)

def process_input(input_path_or_url: str):
    if input_path_or_url.startswith("http://") or input_path_or_url.startswith("https://"):
        asyncio.run(process_url(input_path_or_url))
    else:
        md_path = process_document_file(input_path_or_url)
        with open(md_path, "r", encoding="utf-8") as f:
            markdown = f.read()
        process_markdown(markdown, source=md_path)

if __name__ == "__main__":
    input_path_or_url = input("enter the url or path of doc : ")
    process_input(input_path_or_url)
