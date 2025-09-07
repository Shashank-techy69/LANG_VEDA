from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langdetect import detect

# Load multilingual FAISS embeddings
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
db = FAISS.load_local("faiss_index", embedding_model.encode, allow_dangerous_deserialization=True)

# Load offline LLM
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

while True:
    query = input("User: ")
    if query.lower() in ["exit", "quit"]:
        break

    # Detect language
    lang = detect(query)
    print(f"[Detected language: {lang}]")

    # FAISS search
    docs = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Construct prompt
    prompt = f"Answer in {lang} based on context:\n{context}\nQuestion: {query}"
    answer = generate_answer(prompt)

    print(f"AI: {answer}\n{'-'*50}")
