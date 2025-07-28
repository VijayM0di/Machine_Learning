from docling.document_converter import DocumentConverter
from transformers import AutoTokenizer
from docling.chunking import HybridChunker

DOC_SOURCE = r"pdfs\budget_speech_2025.pdf"
doc = DocumentConverter().convert(source=DOC_SOURCE).document

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# MAX_TOKENS = 64  # set to a small number for illustrative purposes
MAX_TOKENS = 128  # set to a small number for illustrative purposes

tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
chunker = HybridChunker(
    tokenizer=tokenizer,  
    max_tokens=MAX_TOKENS,  
    merge_peers=True,
)
chunk_iter = chunker.chunk(dl_doc=doc)
chunks = list(chunk_iter)

for i, chunk in enumerate(chunks):
    print(f"=== {i} ===")
    txt_tokens = len(tokenizer.tokenize(chunk.text))
    print(f"chunk.text ({txt_tokens} tokens):\n{repr(chunk.text)}")

    ser_txt = chunker.serialize(chunk=chunk)
    ser_tokens = len(tokenizer.tokenize(ser_txt))
    print(f"chunker.serialize(chunk) ({ser_tokens} tokens):\n{repr(ser_txt)}")
    print()

serialized_chunks = []
serialized_chunks = [chunker.serialize(chunk=chunk) for chunk in chunks]

print(f"Serialized chunks: {len(serialized_chunks)}")