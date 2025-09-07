import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langdetect import detect, DetectorFactory
from dotenv import load_dotenv
from groq import Groq

st.title("📚 LANG_VEDA – Multilingual Chatbot Demo")

# Load environment variables
load_dotenv()

# Enforce consistent langdetect results for reproducibility
DetectorFactory.seed = 0

# --- Configuration ---
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
FAISS_INDEX_PATH = "faiss_index"
GROQ_MODEL_NAME = "llama-3.1-8b-instant" # Using a different active model as others were decommissioned.

@st.cache_resource
def load_vector_db():
    """Loads the FAISS vector database and embedding model."""
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={'normalize_embeddings': True}
    )
    db = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

@st.cache_resource
def get_groq_client():
    """Initializes and returns the Groq client."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    return client

def generate_answer(client, context, query, lang):
    """Generates an answer using the Groq API."""
    system_prompt = f"You are a helpful AI assistant named LANG_VEDA. Answer the user's question directly. Use the provided context to answer if it is relevant. If the context is not relevant, answer using your general knowledge. Provide the answer in the following language: {lang}."
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=GROQ_MODEL_NAME,
        temperature=0.7,
        max_tokens=1024,
    )
    return chat_completion.choices[0].message.content

db = load_vector_db()
client = get_groq_client()

if not os.environ.get("GROQ_API_KEY"):
    st.error("⚠️ GROQ_API_KEY not found. Please add it to your .env file.")

query = st.text_input("Ask your question (any supported language):")

if query:
    try:
        if len(query.strip().split()) > 2:
            lang = detect(query)
        else:
            lang = 'en'
    except Exception:
        lang = 'en'
    st.write(f"*(Detected language: {lang})*")

    docs = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    answer = generate_answer(client, context, query, lang)

    st.subheader("Answer:")
    st.write(answer)
