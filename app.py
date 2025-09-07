import streamlit as st
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect

st.title("📚 LANG_VEDA – Multilingual Chatbot Demo")

# Load embeddings and LLM
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
db = FAISS.load_local("faiss_index", embedding_model.encode, allow_dangerous_deserialization=True)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

query = st.text_input("Ask your question (any supported language):")

if query:
    lang = detect(query)
    st.write(f"Detected language: {lang}")

    docs = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"Answer in {lang} based on context:\n{context}\nQuestion: {query}"
    answer = generate_answer(prompt)

    st.subheader("Answer:")
    st.write(answer)
