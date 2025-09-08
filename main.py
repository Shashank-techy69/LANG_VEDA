import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langdetect import detect, DetectorFactory
from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# --- FastAPI App Initialization ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Configuration ---
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
FAISS_INDEX_PATH = "faiss_index"
GROQ_MODEL_NAME = "llama-3.1-8b-instant" # Using a different active model as others were decommissioned.

# Enforce consistent langdetect results for reproducibility
DetectorFactory.seed = 0

# Load multilingual FAISS embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    encode_kwargs={'normalize_embeddings': True}
)
db = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
if not os.environ.get("GROQ_API_KEY"):
    print("⚠️ GROQ_API_KEY not found in .env file. The application might not work.")

def generate_answer(context, query, lang):
    """Generates an answer using the Groq API based on the given context and query."""
    system_prompt = f"You are a helpful AI assistant named LANG_VEDA. Answer the user's question directly. Use the provided context to answer if it is relevant. If the context is not relevant, answer using your general knowledge. Provide the answer in the following language: {lang}."
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    try:
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
    except Exception as e:
        return f"An error occurred with the API call: {e}"

class ChatRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/chat", response_class=JSONResponse)
async def chat_endpoint(request: ChatRequest):
    query = request.query

    try:
        # For short or ambiguous queries, default to English for reliability
        if len(query.strip().split()) > 2:
            lang = detect(query)
        else:
            lang = 'en'
    except Exception as e:
        print(f"Language detection failed: {e}. Defaulting to English.")
        lang = 'en'

    # FAISS search
    docs = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    answer = generate_answer(context, query, lang)
    return {"answer": answer}

def run_cli_chat():
    """Runs the chatbot in a command-line interface loop."""
    print("🤖 LANG_VEDA CLI Chat. Type 'exit' or 'quit' to end.")
    print("-" * 50)
    while True:
        query = input("User: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        try:
            if len(query.strip().split()) > 2:
                lang = detect(query)
            else:
                lang = 'en'
        except Exception as e:
            print(f"Language detection failed: {e}. Defaulting to English.")
            lang = 'en'
        
        print(f"*(Detected language: {lang})*")

        docs = db.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        answer = generate_answer(context, query, lang)
        print(f"AI: {answer}\n{'-'*50}")

if __name__ == "__main__":
    import sys
    import uvicorn

    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        run_cli_chat()
    else:
        print("Starting FastAPI server. Go to http://127.0.0.1:8000")
        print("To chat in the terminal, run: python main.py chat")
        uvicorn.run(app, host="127.0.0.1", port=8000)
