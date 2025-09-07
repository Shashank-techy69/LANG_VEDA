import os
import sys
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Configuration ---
DATA_PATH = "data"
DB_FAISS_PATH = "faiss_index"
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def main():
    """
    Main function to handle the ingestion process.
    Loads data, splits it into chunks, creates embeddings, and saves them to a FAISS vector store.
    """
    print("🚀 Starting ingestion process...")

    # --- 1. Load Documents ---
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data directory '{DATA_PATH}' not found.")
        sys.exit(1)

    docs = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".txt"):
            file_path = os.path.join(DATA_PATH, file)
            print(f"  - Loading document: {file_path}")
            loader = TextLoader(file_path, encoding='utf-8')
            docs.extend(loader.load())

    # --- 2. Split Documents into Chunks ---
    print(f"\nSplitting {len(docs)} documents into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(docs)
    print(f"  - Created {len(chunks)} chunks.")

    # --- 3. Create Embeddings and FAISS Index ---
    print(f"\nCreating embeddings using '{MODEL_NAME}' and building FAISS index...")
    embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME, encode_kwargs={'normalize_embeddings': True})
    db = FAISS.from_documents(chunks, embedding_model)

    # --- 4. Save FAISS Index ---
    db.save_local(DB_FAISS_PATH)
    print(f"\n✅ Ingestion complete! FAISS index saved to '{DB_FAISS_PATH}'.")

if __name__ == "__main__":
    main()
