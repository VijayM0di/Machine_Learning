import os
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
import lancedb
from ollama import chat

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

CACHE_DIR = "huggingface_model"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
TABLE_NAME = "doc"
DATABASE_PATH = Path(r"lanceDB")

os.makedirs(CACHE_DIR, exist_ok=True)
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=CACHE_DIR, device=device)
embedding_dim = embedding_model.get_sentence_embedding_dimension()

try:
    db = lancedb.connect(DATABASE_PATH)
    if TABLE_NAME not in db.table_names():
        raise ValueError(f"Table '{TABLE_NAME}' does not exist in the LanceDB at {DATABASE_PATH}.")
    table = db.open_table(TABLE_NAME)

    print(f"Connected to table '{TABLE_NAME}' in LanceDB.")
except Exception as e:
    print(f"Error connecting to LanceDB or table: {e}")
    exit(1)

def query_lancedb(question: str):
    try:
        query_embedding = embedding_model.encode([question])
        search_results = table.search(query_embedding).limit(10).to_pandas()

        if search_results.empty:
            print("No relevant documents found.")
            return []

        return search_results["content"].tolist()
    except Exception as e:
        print(f"Error during LanceDB query: {e}")
        return []

def generate_answer(question: str) -> str:
    retrieved_chunks = query_lancedb(question)
    if not retrieved_chunks:
        return "Sorry, no relevant information found."

    context = "\n".join(retrieved_chunks)

    prompt = f"""You are a knowledgeable assistant. Use only the information from the context below to answer the question clearly and accurately. Do not rely on prior knowledge or make assumptions. Structure the answer logically and keep it focused on the question.

    --- Context ---
    {context}

    --- Question ---
    {question}

    --- Response ---
    """
    
    try:
        response = chat(
            model="mistral:latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        print(f"Error generating response with Ollama: {e}")
        return "An error occurred while generating the answer."


if __name__ == "__main__":
    print("Ask your questions. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("Question: ").strip()
            if question.lower() in {"exit", "quit"}:
                print("Exiting...")
                break

            answer = generate_answer(question)
            print("Answer:", answer)
        except KeyboardInterrupt:
            print("Exiting on interrupt.")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
