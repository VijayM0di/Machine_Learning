import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from chromadb import PersistentClient

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

CACHE_DIR = "huggingface_model"
os.makedirs(CACHE_DIR, exist_ok=True)


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=CACHE_DIR, device=device)
embedding_size = embedding_model.get_sentence_embedding_dimension()


QA_MODEL_NAME = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(QA_MODEL_NAME, cache_dir=CACHE_DIR).to(device)


COLLECTION_NAME = "doc_BIOS"
project_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(project_dir, "chroma_db")

chroma_client = PersistentClient(path=persist_dir)
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
except Exception as e:
    print(f"Error checking/creating collection: {e}")
    exit()


def query_chromadb(question: str, n_results: int = 10):
    try:
        query_embedding = embedding_model.encode([question])
        search_results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        if not search_results or not search_results['documents']:
            print("No relevant documents found.")
            return []
        return search_results['documents'][0]
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        return []

def generate_answer(question: str):
    retrieved_chunks = query_chromadb(question)
    context = "\n".join(retrieved_chunks)

    prompt = f"""
    You are an AI assistant. Use the given context to answer the question accurately and concisely.  

    **Guidelines:**
    - Avoid repetition of sentences or content.  
    - Provide only the required answer from context.  
    - Ensure the content is well-structured and properly formatted.  

    **Context:**
    {context}

    **User Question:**
    {question}

    **Answer:**
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = decoded.split("**Answer:**")[-1].strip()
    return answer


if __name__ == "__main__":
    while True:
        question = input("Question: ")
        if question.strip().lower() in ["exit", "quit"]:
            break
        answer = generate_answer(question)
        print("Answer:", answer)
