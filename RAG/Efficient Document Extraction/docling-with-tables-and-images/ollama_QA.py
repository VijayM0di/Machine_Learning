import os
import torch

from sentence_transformers import SentenceTransformer
from ollama import ProcessResponse, chat, ps, pull
# import ollama  # Import Ollama
import subprocess
from ollama import chat
import os
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

CACHE_DIR = "huggingface_model"
os.makedirs(CACHE_DIR, exist_ok=True)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME,cache_folder=CACHE_DIR,device=device)
embedding_size = embedding_model.get_sentence_embedding_dimension()
COLLECTION_NAME = "doc_BIOS"

# def is_ollama_running():
#     try:
#         subprocess.run(["pgrep", "-f", "ollama"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
#         return True
#     except subprocess.CalledProcessError:
#         return False

project_dir = os.path.dirname(os.path.abspath(__file__))
persist_dir = os.path.join(project_dir, "chroma_db")  # Directory to store ChromaDB data
chroma_client = PersistentClient(path=persist_dir)

try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
except Exception as e:
    print(f"Error checking/creating collection: {e}")
    exit()

def is_ollama_running():
    try:
        result = subprocess.run(
            ["tasklist"], capture_output=True, text=True, check=True
        )
        return "ollama.exe" in result.stdout.lower()  # Adjust if necessary
    except Exception as e:
        print(f"Error checking Ollama process: {e}")
        return False

def query_chromadb(question: str, n_results: int = 10):
    try:
        query_embedding = embedding_model.encode([question])
        search_results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        if not search_results:
            print("No relevant documents found.")
            return []
        return search_results['documents'][0]
    except Exception as e:
        print(f"Error querying chromaDB: {e}")
        return []

def generate_answer(question: str):

    retrieved_chunks = query_chromadb(question)
    if not retrieved_chunks:
        return "Sorry, no relevant information found."
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

    response = chat(model='llama3:latest', messages=[
        {'role': 'user', 'content': prompt},
    ])
    return response['message']['content']
    
if __name__ == "__main__":
    # if not is_ollama_running():
    #     print("Error: Ollama is not running. Please start it with 'ollama serve' and try again.")
    # else:
        while True:
            question = input("Ask a question (or type 'exit' to quit): ").strip()
            if question.lower() == "exit":
                
                print("Exiting...")
                break
            
            answer = generate_answer(question)
            print("\nAnswer:", answer)