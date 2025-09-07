import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

# Load multilingual embeddings model
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Load all text files
docs = []
for file in os.listdir("data"):
    if file.endswith(".txt"):
        loader = TextLoader(os.path.join("data", file))
        docs.extend(loader.load())

# Convert documents to embeddings
db = FAISS.from_documents(docs, embedding_model.encode, normalize=True)

# Save FAISS index
db.save_local("faiss_index")
print("✅ Multilingual ingestion complete! FAISS index saved.")
